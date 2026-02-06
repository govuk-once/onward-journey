import os
import json
import uuid
from typing import List, Dict, Any

async def initiate_live_handoff(reason: str, deployment_id_env: str, history: List[Dict[str, Any]]) -> str:
    """Generic logic for all live chat transfers to reduce code duplication."""
    # Extract recent user queries for context summary
    user_queries = [
        c['text'] for m in history 
        for c in m['content'] 
        if m['role'] == 'user' and c['type'] == 'text'
    ]
    summary = f"User is asking about: {reason}. Context: {' | '.join(user_queries[-3:])}"

    handoff_config = {
        "action": "initiate_live_handoff",
        "deploymentId": os.getenv(deployment_id_env),
        "region": os.getenv('GENESYS_REGION', 'euw2.pure.cloud'),
        "token": str(uuid.uuid4()),
        "reason": reason,
        "summary": summary
    }
    return f"SIGNAL: initiate_live_handoff {json.dumps(handoff_config)}"

# --- Specialized Tool Wrappers ---
async def connect_to_moj(reason: str, history: List[Dict[str, Any]]):
    return await initiate_live_handoff(reason, 'GENESYS_DEPLOYMENT_ID_MOJ', history)

async def connect_to_immigration(reason: str, history: List[Dict[str, Any]]):
    return await initiate_live_handoff(reason, 'GENESYS_DEPLOYMENT_ID_IMMIGRATION', history)

async def connect_to_hmrc(reason: str, history: List[Dict[str, Any]]):
    return await initiate_live_handoff(reason, 'GENESYS_DEPLOYMENT_ID_PENSIONS_FORMS_AND_RETURNS', history)

def get_tool_definitions(strategy: int) -> List[Dict[str, Any]]:
    """Returns the JSON declarations for Bedrock based on the selected strategy."""
    
    oj_tool = {
        "name": "query_internal_kb",
        "description": "Search specialized internal Onward Journey data for status and private guidance.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }

    govuk_tool = {
        "name": "query_govuk_kb",
        "description": "Search public GOV.UK policy, legislation, and public-facing services.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }

    live_chat_tools = [
        {
            "name": "connect_to_live_chat_MOJ",
            "description": "Call for MOJ human assistance or phone transfers.",
            "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}
        },
        {
            "name": "connect_to_live_chat_immigration",
            "description": "Call for immigration human assistance or phone transfers.",
            "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}
        },
        {
            "name": "connect_to_live_chat_HMRC_pensions_forms_and_returns",
            "description": "Call for HMRC pensions/forms human assistance or phone transfers.",
            "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}
        }
    ]

    if strategy == 1: return [oj_tool]
    if strategy == 2: return [govuk_tool]
    if strategy == 4: return [oj_tool] + live_chat_tools
    return [oj_tool, govuk_tool] # Default strategy 3