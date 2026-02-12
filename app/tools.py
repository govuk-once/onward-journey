import os
import json
import uuid
from typing import List, Dict, Any

async def initiate_live_handoff(reason: str, deployment_id_env: str, history: List[Dict[str, Any]], triage_data: Dict[str, Any] = {}) -> str:
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
async def connect_to_moj(reason: str, history: List[Dict[str, Any]], triage_data: Dict[str, Any] = {}, **kwargs):
    return await initiate_live_handoff(reason, 'GENESYS_DEPLOYMENT_ID_MOJ', history, triage_data)

async def connect_to_immigration(reason: str, history: List[Dict[str, Any]], triage_data: Dict[str, Any] = {}, **kwargs):
    return await initiate_live_handoff(reason, 'GENESYS_DEPLOYMENT_ID_IMMIGRATION', history, triage_data)

async def connect_to_hmrc(reason: str, history: List[Dict[str, Any]], triage_data: Dict[str, Any] = {}, **kwargs):
    return await initiate_live_handoff(reason, 'GENESYS_DEPLOYMENT_ID_PENSIONS_FORMS_AND_RETURNS', history, triage_data)

def get_internal_kb_definition() -> List[Dict[str, Any]]:
    return [{
        "name": "query_internal_kb",
        "description": "Search specialized internal Onward Journey data for guidance.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    }]

def get_govuk_definitions() -> List[Dict[str, Any]]:
    return [{
        "name": "query_govuk_kb",
        "description": "Search public GOV.UK policy and legislation.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    }]

def get_live_chat_definitions() -> List[Dict[str, Any]]:
    """Returns schemas for all live chat endpoints including triage data requirements."""
    tools_list = []
    names = [
        ("MOJ", "Ministry of Justice"),
        ("immigration", "Immigration and visas"),
        ("HMRC_pensions_forms_and_returns", "HMRC pensions and returns")
    ]

    for short, full in names:
        tools_list.append({
            "name": f"connect_to_live_chat_{short}",
            "description": f"Connect to a live agent for {full} queries. Requires pre-collected triage data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "A concise summary of the user's situation and the reason for handoff."
                    },
                    "triage_data": {
                        "type": "object",
                        "description": "Structured data collected during the triage process (e.g., visa type)."
                    }
                },
                "required": ["reason", "triage_data"],
            }
        })
    return tools_list
