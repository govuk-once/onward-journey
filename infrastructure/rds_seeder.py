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


def lambda_handler(event, context):
    """
    Main entry point for the RDS Seeder.
    Processes a specific S3 CSV file and performs an atomic update on its
    corresponding PostgreSQL table, including vector embedding generation.
    """
    target_file = event["file_name"]
    target_table = event["table_name"]

    # Initialize service clients
    s3 = boto3.client("s3")
    bedrock = boto3.client("bedrock-runtime")

    # Establish database connection using environment variables
    print(f"Connecting to {os.environ['DB_HOST']}...")
    conn = pg8000.native.Connection(
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        port=5432,
        timeout=120,
        tcp_keepalive=True,
    )

    try:
        # Start transaction manually
        conn.run("BEGIN")

        # Ensure vector support
        conn.run("CREATE EXTENSION IF NOT EXISTS vector;")

        # Nuke the old table (removes locks and old data)
        print(f"Refreshing table: {target_table}")
        conn.run(f"DROP TABLE IF EXISTS {target_table} CASCADE;")

        # Create/re-create the schema
        if target_table == "dept_contacts":
            print(f"Creating schema for {target_table}...")
            conn.run(
                f"""
                CREATE TABLE {target_table} (
                    id SERIAL PRIMARY KEY,
                    uid TEXT,
                    service_name TEXT,
                    department TEXT,
                    phone_number TEXT,
                    topic TEXT,
                    user_type TEXT,
                    tags TEXT,
                    url TEXT,
                    description TEXT,
                    embedding VECTOR(1024)
                );
            """
            )

        # Fetch source data from S3 using the prefix defined in s3.tf
        print(f"Fetching source data from S3: mock/{target_file}")
        s3_response = s3.get_object(
            Bucket=os.environ["BUCKET_NAME"], Key=f"mock/{target_file}"
        )

        # 'utf-8-sig' handles potential Byte Order Marks (BOM) from Excel exports
        lines = s3_response["Body"].read().decode("utf-8-sig").splitlines()
        reader = csv.DictReader(lines)

        print(f"Starting ingestion for {target_table}...")
        for i, row in enumerate(reader):
            if target_table == "dept_contacts":
                # Construct a descriptive context string for high-quality vector embeddings
                text_to_embed = (
                    f"Department: {row['department']}. "
                    f"Service: {row['service_name']}. "
                    f"Topic: {row['topic']}. "
                    f"Keywords: {row['tags']}. "
                    f"Detailed Description: {row['description']}"
                )

                # Attempt to get vectors from Bedrock
                vector = get_embedding(bedrock, text_to_embed)

                # --- MOCK FALLBACK (Commented out for future testing) ---
                # Use a dummy vector of 1024 zeros until Bedrock connection issue resolved
                # vector = [0.0] * 1024

                # Format for pgvector [val1, val2, ...]
                vector_string = "[" + ",".join(map(str, vector)) + "]"

                # Insert processed record and vector into the database
                conn.run(
                    f"""INSERT INTO {target_table}
                    (uid, service_name, department, phone_number, topic, user_type, tags, url, description, embedding)
                    VALUES (:u, :s, :d, :p, :t, :ut, :tg, :url, :desc, :emb)""",
                    u=row["uid"],
                    s=row["service_name"],
                    d=row["department"],
                    p=row["phone_number"],
                    t=row["topic"],
                    ut=row["user_type"],
                    tg=row["tags"],
                    url=row["url"],
                    desc=row["description"],
                    emb=vector_string,
                )

                # Log progress every 5 rows to monitor Lambda health in CloudWatch
                if i % 5 == 0:
                    print(f"Successfully processed {i} rows...")

        # Commit only if everything succeeded
        conn.run("COMMIT")
        print(f"Ingestion complete. Total rows processed: {i + 1}")
        return {"status": "success", "table": target_table}

    except Exception as e:
        # Roll back if any row fails
        conn.run("ROLLBACK")
        print(f"ERROR: {str(e)}")
        raise e
    finally:
        conn.close()


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
