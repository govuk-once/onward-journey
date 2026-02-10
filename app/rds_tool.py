"""
RDS Tool Lambda (MCP Server).
Handles semantic search requests forwarded by the AgentCore Gateway.
"""

import os
import json
import boto3
import pg8000.native

BEDROCK_URL = os.environ.get("BEDROCK_RUNTIME_ENDPOINT")
SECRETS_URL = os.environ.get("SECRETS_ENDPOINT_URL")

bedrock = boto3.client(
    "bedrock-runtime", region_name="eu-west-2", endpoint_url=f"https://{BEDROCK_URL}"
)

secrets_client = boto3.client(
    "secretsmanager", region_name="eu-west-2", endpoint_url=f"https://{SECRETS_URL}"
)


def get_db_password() -> str:
    """Retrieves the DB password by explicitly checking the format."""
    response = secrets_client.get_secret_value(SecretId=os.environ["DB_SECRET_ARN"])
    raw_value = response["SecretString"].strip()

    # 1. If it looks like JSON, parse it and get the key
    if raw_value.startswith("{"):
        db_password = json.loads(raw_value)
        return str(db_password.get("password"))

    # 2. Otherwise, it's the raw password string
    return str(raw_value)


def lambda_handler(event, context):
    """AgentCore Gateway sends the tool arguments inside the 'arguments' key."""

    print(f"📥 GATEWAY CALL: {json.dumps(event)}")

    # 1. Extract Query and Request ID
    query = (
        event.get("arguments", {}).get("query")
        or event.get("query")
        or "No query provided"
    )
    request_id = event.get("id")

    # 2. Generate Vector Embedding using Amazon Titan v2
    # We use 1024 dimensions to match our RDS pgvector settings
    embed_body = json.dumps({"inputText": query, "dimensions": 1024, "normalize": True})
    embed_resp = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0", body=embed_body
    )
    embedding = json.loads(embed_resp["body"].read())["embedding"]

    # 3. Connect and Query RDS using pgvector similarity
    db_password = get_db_password()
    conn = pg8000.native.Connection(
        user=os.environ["DB_USER"],
        password=db_password,
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        port=5432,
    )

    results = conn.run(
        """
        SELECT service_name, phone_number, live_chat_identifier, description
        FROM dept_contacts_v2
        ORDER BY embedding <=> :embed::vector LIMIT 3
        """,
        embed=str(embedding),
    )
    conn.close()

    # 4. Format results for the MCP-compatible response
    formatted_results = [
        {"service": r[0], "phone": r[1], "live_chat_identifier": r[2], "info": r[3]}
        for r in results
    ]

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps(formatted_results)}]
        },
    }
