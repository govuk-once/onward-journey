# """
# RDS Tool Lambda (MCP Server).
# Handles semantic search requests forwarded by the AgentCore Gateway.
# """

# import os
# import json
# import boto3
# import psycopg2


# def get_db_password():
#     """Fetches the password from Secrets Manager."""
#     client = boto3.client("secretsmanager")
#     response = client.get_secret_value(SecretId=os.environ["DB_SECRET_ARN"])
#     return response["SecretString"]


# def lambda_handler(event, context):
#     """AgentCore Gateway sends the tool arguments inside the 'arguments' key."""
#     # 1. Extract the query argument defined in your Terraform property block
#     query = event.get("arguments", {}).get("query", "")

#     if not query:
#         return {"content": [{"type": "text", "text": "No search query provided."}]}

#     # 2. Convert query to vector embedding using Amazon Titan v2
#     bedrock = boto3.client("bedrock-runtime")
#     embed_body = json.dumps({"inputText": query, "dimensions": 1024, "normalize": True})
#     embed_resp = bedrock.invoke_model(
#         modelId="amazon.titan-embed-text-v2:0", body=embed_body
#     )
#     embedding = json.loads(embed_resp["body"].read())["embedding"]

#     # 3. Connect and Query RDS using pgvector similarity
#     conn = psycopg2.connect(
#         host=os.environ["DB_HOST"],
#         database=os.environ["DB_NAME"],
#         user=os.environ["DB_USER"],
#         password=get_db_password(),
#     )

#     with conn.cursor() as cur:
#         # Use <=> for cosine distance search
#         cur.execute(
#             """
#             SELECT service_name, phone_number, genesys_queue_id, description
#             FROM dept_contacts_v2  -- FIX: Query the V2 table
#             ORDER BY embedding <=> %s::vector LIMIT 3
#             """,
#             (embedding,),
#         )
#         rows = cur.fetchall()
#     conn.close()

#     # 4. Return results in the MCP-compatible format the Gateway expects
#     results = [
#         {"service": r[0], "phone": r[1], "queue_id": r[2], "info": r[3]} for r in rows
#     ]

#     return {"content": [{"type": "text", "text": json.dumps(results)}]}


# TODO: Revert back to code above following debug
# Code below for testing agent connectivity to RDS

import json


def lambda_handler(event, context):
    """
    RDS Tool - Minimal MCP Implementation.
    No external dependencies (to avoid ImportModuleErrors).
    """
    print(f"📥 GATEWAY CALL: {json.dumps(event)}")

    # 1. Flexible Extraction
    # The Gateway might send the query at the top level OR inside arguments
    query = (
        event.get("query")
        or event.get("arguments", {}).get("query")
        or event.get("params", {})
        .get("arguments", {})
        .get("query", "No query provided")
    )

    # 2. Echo the ID
    # If the orchestrator sent an ID, we MUST return it.
    request_id = event.get("id")

    # 3. Simulate Search
    if "DWP" in query.upper() or "WORK AND PENSIONS" in query.upper():
        results = "DWP Contact: 0800 328 5644. Address: Caxton House, London."
    else:
        results = f"Search result for '{query}': No specific contact found."

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": results}]},
    }
