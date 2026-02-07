"""
RDS Data Seeder Lambda.
Handles the ingestion of CSV datasets from S3 into RDS PostgreSQL tables,
generating vector embeddings via Amazon Bedrock Titan v2 for RAG capabilities.
"""

import csv
import json
import os

import boto3
import pg8000.native


def get_db_password():
    """Retrieves the DB password from Secrets Manager using the ARN."""
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=os.environ["DB_SECRET_ARN"])
    return response["SecretString"]


def get_embedding(bedrock_client, text):
    """
    Invokes the Amazon Titan Text Embeddings v2 model via Bedrock.
    Returns a 1024-dimension numerical vector for the provided text.
    """
    body = json.dumps({"inputText": text, "dimensions": 1024})

    response = bedrock_client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
    )

    return json.loads(response.get("body").read())["embedding"]


def lambda_handler(event, context):
    """
    Main entry point for the RDS Seeder.
    Processes a specific S3 CSV file and performs an atomic update on its
    corresponding PostgreSQL table, including vector embedding generation.
    """
    target_file = event["file_name"]
    target_table = event["table_name"]

    # Load the master configuration provided by Terraform as a JSON string
    config = json.loads(os.environ["DB_CONFIG"])

    # Locate the specific table definition from our YAML configuration
    table_conf = next((t for t in config["tables"] if t["name"] == target_table), None)
    if not table_conf:
        raise Exception(f"Configuration for table '{target_table}' not found in YAML.")

    # Initialise service clients
    s3 = boto3.client("s3")
    bedrock = boto3.client("bedrock-runtime")

    # Establish database connection using environment variables
    print(f"Connecting to {os.environ['DB_HOST']}...")
    conn = pg8000.native.Connection(
        password=get_db_password(),
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        port=5432,
        timeout=120,
        tcp_keepalive=True,  # Prevents VPC timeouts on long-running ingestions
    )

    try:
        # Wrap everything in a transaction for atomicity
        conn.run("BEGIN")

        # Ensure pgvector extension is available in the database
        conn.run("CREATE EXTENSION IF NOT EXISTS vector;")

        # 1. TABLE INITIALIZATION
        # To ensure the database matches the latest YAML/CSV schema, we drop
        # the existing table and recreate it. This removes stale data and locks.
        columns = table_conf["columns"]
        col_definitions = ", ".join(
            [f"{name} {dtype}" for name, dtype in columns.items()]
        )

        print(f"Initialising/recreating table: {target_table}")
        conn.run(f"DROP TABLE IF EXISTS {target_table} CASCADE;")
        conn.run(
            f"CREATE TABLE {target_table} (id SERIAL PRIMARY KEY, {col_definitions});"
        )

        # 2. FETCH SOURCE DATA
        print(f"Fetching source data from S3: mock/{target_file}")
        s3_res = s3.get_object(
            Bucket=os.environ["BUCKET_NAME"], Key=f"mock/{target_file}"
        )

        # 'utf-8-sig' handles potential Byte Order Marks (BOM) from Excel CSV exports
        lines = s3_res["Body"].read().decode("utf-8-sig").splitlines()
        reader = csv.DictReader(lines)

        # 3. DYNAMIC INGESTION & EMBEDDING
        embed_cols = table_conf.get("embedding_source_cols", [])

        print(f"Starting ingestion for {target_table}...")
        for i, row in enumerate(reader):
            # Map CSV row data to the columns defined in the YAML
            # 'embedding' is excluded here as it is calculated via Bedrock below
            params = {
                name: row.get(name) for name in columns.keys() if name != "embedding"
            }

            # Semantic Embedding Logic
            # If the schema defines an 'embedding' column and source columns are provided:
            if "embedding" in columns and embed_cols:
                # Construct a descriptive context string for high-quality vector embeddings
                # Concatenating columns specified in the YAML 'embedding_source_cols'
                text_to_embed = ". ".join(
                    [f"{col}: {row.get(col, '')}" for col in embed_cols]
                )

                # Invoke Bedrock to generate a 1024-dimension vector
                vector = get_embedding(bedrock, text_to_embed)

                # Format for pgvector input: [val1, val2, ...]
                params["emb"] = "[" + ",".join(map(str, vector)) + "]"

            # Build Dynamic INSERT statement based on the active parameter map
            col_names = ", ".join(params.keys()).replace("emb", "embedding")
            placeholders = ", ".join([f":{k}" for k in params.keys()])

            conn.run(
                f"INSERT INTO {target_table} ({col_names}) VALUES ({placeholders})",
                **params,
            )

            # Log progress to monitor health and runtime in CloudWatch
            if i % 10 == 0:
                print(f"Successfully processed {i} rows for {target_table}...")

        # Commit only if every row in the file was processed successfully
        conn.run("COMMIT")
        print(f"Ingestion complete for {target_table}. Total rows processed: {i + 1}")
        return {"status": "success", "table": target_table, "rows_processed": i + 1}

    except Exception as e:
        # Roll back the transaction if any row or embedding call fails
        conn.run("ROLLBACK")
        print(f"DATABASE ERROR: {str(e)}")
        raise e
    finally:
        conn.close()
