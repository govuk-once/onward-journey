import os
import json
import re
import uuid
import PureCloudPlatformClientV2

from typing import List, Dict, Any
from PureCloudPlatformClientV2.rest import ApiException

class GenesysServiceDiscovery:
    def __init__(self):
        self._setup_genesys_sdk()

        self.api_client = PureCloudPlatformClientV2.api_client.ApiClient().get_client_credentials_token(
                os.getenv("GENESYS_CLOUD_CLIENT_ID"),
                os.getenv("GENESYS_CLOUD_CLIENT_SECRET")
            )
        self.web_api = PureCloudPlatformClientV2.WebDeploymentsApi(self.api_client)
        self.arch_api = PureCloudPlatformClientV2.ArchitectApi(self.api_client)
        self.conversations_api = PureCloudPlatformClientV2.ConversationsApi(self.api_client)

    def _setup_genesys_sdk(self, region = "eu_west_2"):
        """Authenticates with Genesys using the Region Enum key."""

        try:
            region = PureCloudPlatformClientV2.PureCloudRegionHosts[region]
            PureCloudPlatformClientV2.configuration.host = region.get_api_host()

        except Exception as e:
            print(f"Auth Failure: {e}")

    def get_config_from_deployment(self, deployment_id: str) -> dict:
        """
        Traverses: Deployment -> Config -> Messaging Flow -> Published Version Schema
        to identify required triage fields for the AI Agent.
        """
        if not deployment_id:
            return {}

        try:
            # 1. Get deployment to find the configuration reference
            deployment = self.web_api.get_webdeployments_deployment(deployment_id)
            flow_id = deployment.flow.id
            flow_data = self.arch_api.get_flow(flow_id)
            version_id = flow_data.published_version.id
            config = self.arch_api.get_flow_version_configuration(flow_id, version_id)

            return config

        except ApiException as e:
            print(f"Genesys API Error ({e.status}): {e.body}")
            return {}
        except Exception as e:
            print(f"Discovery Traversal Error: {e}")
            return {}

    def extract_triage_data(self, config: dict):
        """
        Dynamically extracts ALL triage fields, their specific options,
        and the relevant prompts from the Genesys config.
        """
        actions = config.get('flowSequenceItemList', [{}])[0].get('actionList', [])

        # We now store fields as a dictionary to map each field to its specific options
        triage_fields = {}
        global_prompt = ""

        # 1. Iterate through all actions to find all Switches (Decision points)
        for action in actions:
            if action.get('__type') == 'SwitchAction':
                # Extract the variable name being checked
                cases = action.get('cases', [])
                if cases:
                    ref = cases[0]['value']['metaData']['references'][0]
                    field_name = ref.get('name')

                    # Extract all possible options for THIS specific field
                    options = [
                        c['value']['config']['==']['operands'][1]['lit']['text']
                        for c in cases if 'lit' in str(c)
                    ]
                    triage_fields[field_name] = options

            # 2. Collect the human-readable prompt(s)
            elif action.get('__type') == 'SendResponseAction':
                text = action.get('messageBody', {}).get('text', "")
                if text:
                    global_prompt += f" {text}"

        return {
            "missing": list(triage_fields.keys()),
            "field_options": triage_fields, # Specific options per field
            "prompt": global_prompt.strip()
        }

async def initiate_live_handoff(reason: str, deployment_id_env: str, history: List[Dict[str, Any]], triage_data: Dict[str, Any] = {}) -> str:
    """Generic logic for all live chat transfers to reduce code duplication."""
    # Extract recent user queries for context summary
    user_queries = [
        c['text'] for m in history
        for c in m['content']
        if m['role'] == 'user' and c['type'] == 'text'
    ]
    summary = f"User is asking about: {reason}. Context: {' | '.join(user_queries[-3:])}"



    print('DEBUG: Triage Data for Handoff:', triage_data)

    handoff_config = {
        "action": "initiate_live_handoff",
        "deploymentId": os.getenv(deployment_id_env),
        "region": os.getenv('GENESYS_REGION', 'euw2.pure.cloud'),
        "token": str(uuid.uuid4()),
        "reason": reason,
        "summary": summary,
        "customAttributes": triage_data
    }
    return f"SIGNAL: initiate_live_handoff {json.dumps(handoff_config)}"

async def connect_to_moj(reason: str, history: List[Dict[str, Any]], triage_data: Dict[str, Any] = {}, **kwargs):
    return await initiate_live_handoff(reason, 'GENESYS_DEPLOYMENT_ID_MOJ', history, triage_data)

async def connect_to_immigration_and_visas(reason: str, history: List[Dict[str, Any]], triage_data: Dict[str, Any] = {}, **kwargs):
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

def get_triage_data():

    department_data = {0: {"name" : "moj", "description": "Ministry of Justice", "deployment_id": os.getenv("GENESYS_DEPLOYMENT_ID_MOJ")},
                   1: {"name": "immigration_and_visas", "description": "Immigration and visas", "deployment_id": os.getenv("GENESYS_DEPLOYMENT_ID_IMMIGRATION")},
                   2: {"name": "hmrc", "description": "Pensions, forms and returns", "deployment_id": os.getenv("GENESYS_DEPLOYMENT_ID_PENSIONS_FORMS_AND_RETURNS")}}

    # if the crendentials are not present return empty metadata and skip APi call.
    client_id = os.getenv("GENESYS_CLOUD_CLIENT_ID")
    client_secret = os.getenv("GENESYS_CLOUD_CLIENT_SECRET")
    if not client_id or not client_secret:
        for key in department_data:
                department_data[key]["triage_data"] = {"missing": [], "field_options": {}, "prompt": ""}
        return department_data
    discovery = GenesysServiceDiscovery()

    for key, dept in department_data.items():
        config = discovery.get_config_from_deployment(dept['deployment_id'])
        triage_data = discovery.extract_triage_data(config)
        department_data[key]['triage_data'] = triage_data

    return department_data

def get_live_chat_definitions() -> List[Dict[str, Any]]:
    # without gensys creds disable live chat tools to avoid unusable tool calls
    if not  os.get_env ("GENESYS_CLOUD_CLIENT_ID") or not os.getenv("GENESYS_CLOUD_CLIENT_SECRET"):
        return []

    tools_list = []

    department_data = get_triage_data()

    for _, dept in department_data.items():

        clean_name = dept['name'].lower().replace(" ", "_")
        short_name = re.sub(r'[^a-z0-9_]', '', clean_name)
        description = dept['description']
        triage_data = dept['triage_data']
        required_fields = triage_data.get('missing', [])

        tools_list.append({
            "name": f"connect_to_live_chat_{short_name}",
            "description": f"Connect to a live agent for {description}. Requires: {', '.join(required_fields)} to be collected from the user first with options: {triage_data.get('field_options', {})}. Use the following prompt to ask the user for missing information: {triage_data.get('prompt', 'No specific prompt available.')}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "triage_data": {
                        "type": "object",
                        "properties": {field: {"type": "string"} for field in required_fields},
                        "required": required_fields
                    }
                },
                "required": ["reason", "triage_data"],
            }
        })
    return tools_list
