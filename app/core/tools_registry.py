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