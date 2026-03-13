from typing import List, Dict, Any

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

def get_genesys_kb_definition() -> List[Dict[str, Any]]:
    return [{
        "name": "query_genesys_kb",
        "description": (
            "Search the official Genesys Cloud Knowledge Base for live "
            "department policies, legal guidance, and updated procedural rules. "
            "Use this for questions about specific rules like passport photos or tax eligibility."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The specific policy question to search for."}
            },
            "required": ["query"],
        }
    }]