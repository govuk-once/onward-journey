import os 
import re 
import json 
import uuid 
import PureCloudPlatformClientV2
from typing import Dict, Any, List 
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
        """Safe traversal: Deployment -> Flow -> Version -> Configuration"""
        if not deployment_id:
            return {}

        try:
            # Get deployment and check for flow existence
            deployment = self.web_api.get_webdeployments_deployment(deployment_id)
            if not deployment or not getattr(deployment, 'flow', None):
                return {}

            flow_id = deployment.flow.id
            
            # Get flow and check for a published version
            flow_data = self.arch_api.get_flow(flow_id)
            if not flow_data or not getattr(flow_data, 'published_version', None):
                return {}

            version_id = flow_data.published_version.id
            
            #  Get configuration - if this fails, catch it in the except block
            return self.arch_api.get_flow_version_configuration(flow_id, version_id)

        except ApiException as e:
            # Log the 404/403 but return empty so the app continues
            print(f"Genesys API Error ({e.status}) for ID: {deployment_id}")
            return {}
        except Exception as e:
            print(f"Discovery Traversal Error: {e}")
            return {}

    def extract_triage_data(self, config: dict):
        """Safely extracts triage fields, returning defaults if config is empty."""
        # If config is empty, return a 'no triage required' structure
        if not config:
            return {"missing": [], "field_options": {}, "prompt": ""}

        # Safely navigate the JSON structure using .get()
        flow_items = config.get('flowSequenceItemList', [])
        actions = flow_items[0].get('actionList', []) if flow_items else []

        triage_fields = {}
        global_prompt = ""

        for action in actions:
            if action.get('__type') == 'SwitchAction':
                cases = action.get('cases', [])
                if cases:
                    # Use get() and check types to avoid crashes
                    ref = cases[0].get('value', {}).get('metaData', {}).get('references', [{}])[0]
                    field_name = ref.get('name')
                    
                    if field_name:
                        options = [
                            c['value']['config']['==']['operands'][1]['lit']['text']
                            for c in cases if 'lit' in str(c)
                        ]
                        triage_fields[field_name] = options

            elif action.get('__type') == 'SendResponseAction':
                text = action.get('messageBody', {}).get('text', "")
                if text:
                    global_prompt += f" {text}"

        return {
            "missing": list(triage_fields.keys()),
            "field_options": triage_fields,
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

def get_triage_data():

    department_data = {0: {"name" : "moj", "description": "Ministry of Justice", "deployment_id": os.getenv("GENESYS_DEPLOYMENT_ID_MOJ")},
                   1: {"name": "immigration_and_visas", "description": "Immigration and visas", "deployment_id": os.getenv("GENESYS_DEPLOYMENT_ID_IMMIGRATION")},
                   2: {"name": "hmrc", "description": "Pensions, forms and returns", "deployment_id": os.getenv("GENESYS_DEPLOYMENT_ID_PENSIONS_FORMS_AND_RETURNS")}}

    discovery = GenesysServiceDiscovery()

    for key, dept in department_data.items():
        config = discovery.get_config_from_deployment(dept['deployment_id'])
        triage_data = discovery.extract_triage_data(config)
        department_data[key]['triage_data'] = triage_data

    return department_data

def get_live_chat_definitions() -> List[Dict[str, Any]]:

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