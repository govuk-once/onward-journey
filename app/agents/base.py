import re 
import json
import boto3
import asyncio
import numpy as np
from copy import deepcopy
from app.core.engine import PromptGuidance

import app.integrations as live_registry

from sklearn.metrics.pairwise import cosine_similarity
from app.core.data                  import SearchResult
from typing                   import List, Dict, Any, Optional

class QueryEmbeddingMixin:
    def _get_embedding(self, text: str, dimensions: int = 1024) -> List[float]:
        """Standardized embedding for all KBs."""
        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": True
        })
        response = self.client.invoke_model(
            modelId=self.embedding_model,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        return json.loads(response.get('body').read()).get('embedding', [])

class OJSearchMixin:
    " Capability to search local data. This expects the host class to have .embeddings and .chunk_data"
    def query_internal_kb(self, query: str) -> str:
        """Local RAG search."""
        query_vec = np.array(self._get_embedding(query)).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = sims.argsort()[-self.top_K_OJ:][::-1]
        return "Internal Context:\n" + "\n".join([self.chunk_data[i] for i in top_idx])

class GovUKSearchMixin:
    " Capability to search public GOV.UK OpenSearch."
    def query_govuk_kb(self, query: str) -> str:
        """OpenSearch RAG search."""
        search_body = {
            "size": self.top_K_govuk,
            "query": {"knn": {"titan_embedding": {"vector": self._get_embedding(query), "k": self.top_K_govuk}}}
        }
        resp = self.os_client.search(index=self.os_index, body=search_body)

        results = []
        for hit in resp["hits"]["hits"]:
            result = hit["_source"]
            result["url"] = f"https://www.gov.uk{result['exact_path']}"
            result["score"] = hit["_score"]
            results.append(SearchResult(**result))

        return 'Retrieved GOV.UK Context:\n' + "\n".join(
            [f"Title: {res.title}\nURL: {res.url}\nDescription: {res.description or 'N/A'}\nScore: {res.score}\n"
             for res in results]
        ) if results else "No GOV.UK info found."

class LiveChatMixin:
    " Capability to initiate live handoff. Expects tools library to have named functions."
    def _get_live_chat_registry(self):
            return {
                "connect_to_live_chat_moj": live_registry.connect_to_moj,
                "connect_to_live_chat_immigration_and_visas": live_registry.connect_to_immigration_and_visas,
                "connect_to_live_chat_hmrc": live_registry.connect_to_hmrc
            }

class HandOffMixin:
    async def process_handoff(self)-> Optional[str]:
        """
        Processes handoff context
        """
        history = self.handoff_package.get('final_conversation_history', [])

        if not history:
            if self.verbose:
                print("Handoff history is empty. Treating as a standard chat.")
            return None # avoid LLM hallucinating an empty string

        history_str = json.dumps(history)

        initial_prompt = (
            f"Previous conversation history: {history_str}. "
            f"INSTRUCTION: Based on the history above, provide the next response to the user. "
        "Please analyze the history and fulfill the user's request, using your specialized tools if necessary."
        "If more than one phone number is available semantically, ask a clarifying question."
        )
        return await self._send_message_and_tools(initial_prompt)

class ServiceTriageQMixin:
    SERVICE_SCHEMAS = live_registry.get_triage_data()

    async def extract_triage_data(self, service: str, history: List[Dict]) -> Dict[str, Any]:
        """
        Performs a semantic gap analysis by calling the LLM to extract data from history C.
        """
        found_key = None
        for key, value in self.SERVICE_SCHEMAS.items():
            if value.get('name') == service:
                found_key = key

        # Identify the last question asked to anchor short answers
            last_assistant_question = "N/A"
            for msg in reversed(history):
                if msg['role'] == 'assistant' and msg.get('content'):
                    # Grab the text content of the last assistant turn
                    last_assistant_question = next((c['text'] for c in msg['content'] if c['type'] == 'text'), "N/A")
                    break

        required = self.SERVICE_SCHEMAS[found_key]['triage_data'].get('missing', []) if found_key is not None else []
        # We only send the text content to save tokens and focus the extraction
        history_str = json.dumps(history)


        # crafting a prompt to extract required triage data from the conversation history
        extraction_prompt = (
                    f"ACT AS A DATA EXTRACTOR. Analyze this conversation history: {history_str}\n"
                    f"Target Fields: {required}.\n"
                    f"Field Options/Constraints: {self.SERVICE_SCHEMAS[found_key]['triage_data'].get('field_options', {})}\n\n"
                    f"CONTEXTUAL ANCHOR: \n"
                    f"The last question asked was: '{last_assistant_question}'\n\n"
                    
                    "### RULES FOR SCALABLE EXTRACTION: \n"
                    "1. SEMANTIC MAPPING: Do not look for exact words. If the user describes a state, extract the corresponding field. "
                    "Examples: ('speed up' or 'no news' -> Update: Yes); ('already sent' or 'last month' -> Applied: Yes).\n"
                    
                    "2. DEPENDENCY LOGIC (COMMON SENSE): Use the relationship between fields to fill 'ghost' answers:\n"
                    "   - PRE-REQUISITES: If a user is at a 'Late Stage' (asking for updates or status), automatically set 'Early Stage' fields (like Help Applying) to 'No'.\n"
                    "   - MUTUAL EXCLUSIVITY: If a user confirms one path, logically infer 'No' for the alternative paths within the same service schema.\n"
                    "   - NEGATIVE INFERENCE: If a user explicitly complains about one specific hurdle, and the schema asks about other hurdles (e.g., 'Trouble Proving Status'), infer 'No' for the others to avoid redundant questioning.\n"
                    
                    "3. CONFIDENCE & NULLS: Only fill a field if it is explicitly stated or logically necessitated by Rule 2. "
                    "If the information is completely absent and cannot be logically inferred, mark as null.\n"
                    
                    "4. OUTPUT FORMAT: Return the a JSON object AND reasons for each individual extraction. Do not include any preamble, conversational filler, or markdown formatting other than the JSON itself. If you cannot extract data, return empty extracted object."                    " - 'extracted' : {{field: value}} for identified or logically inferred data.\n"
                    " - 'missing' : [list] of fields that truly remain unknown.\n"
                    " - 'follow_up': A natural question for the FIRST missing field only.\n\n"
                    
                    "Maintain the logic established by long-form narrative; do not let a 'Yes/No' answer to a specific question overwrite a complex situation described earlier."
                )
        
        # llm call, use a low temperature for deterministic extraction and client assumed to be part of obj
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": extraction_prompt}],
            "max_tokens": 1000,
            "temperature": 0.0 # Strict extraction, no creativity
        })

        response = self.client.invoke_model(
            modelId=self.model_name,
            body=body
        )

        # Parse the response back into a dictionary
        response_body = json.loads(response['body'].read())
        raw_text = response_body['content'][0]['text']
        try:
            # Matches everything between the first '{' and the last '}'
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return {"extracted": {}, "missing": [], "follow_up": None}
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            return {"extracted": {}, "missing": [], "follow_up": None}
class GenesysKBSearchMixin:
    def query_genesys_kb(self, query: str, top_k: int = 3) -> str:
        query_vec = np.array(self._get_embedding(query)).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.genesys_embeddings)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        return "\n\n".join([self.genesys_chunks[i] for i in top_idx])

class BaseAgent:
    def __init__(self, model_name: str, aws_region: str, temperature: float = 0.0, **kwargs):

            self.client = boto3.client(service_name="bedrock-runtime", region_name=aws_region)
            self.aws_region = aws_region
            self.model_name = model_name
            self.temperature = temperature
            self.history: List[Dict[str, Any]] = []
            self.available_tools = {}
            self.bedrock_tools = []
            self.system_instruction = ""
            self.prompt_guidance = PromptGuidance()
            self.triage_state = {}

    def _add_to_history(self, role: str, text: str = '', tool_calls: list = [], tool_results: list = []):
        """Standardized history management for all subclasses."""
        message = {"role": role, "content": []}
        if text: message["content"].append({"type": "text", "text": text})
        if tool_calls: message["content"].extend(tool_calls)
        if tool_results: message["content"].extend(tool_results)
        self.history.append(message)

    async def _execute_tool(self, tool_use_block: Dict[str, Any]) -> str:
        tool_name = tool_use_block['name']
        args = tool_use_block['input']
        
        # Handle specialized triage/handoff logic
        if tool_name.startswith("connect_to_live_chat_"):
            return await self._handle_handoff_gate(tool_name, args)

        # Standard tool execution
        func = self.available_tools.get(tool_name)
        if not func:
            return f"Error: Tool {tool_name} not found."

        # Inject history if it's an integration tool
        if 'app.integrations' in func.__module__:
            args['history'] = deepcopy(self.history)
            
        return await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)

    def _call_llm(self, system_prompt: str):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": self.history,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "tools": self.bedrock_tools
        }
        resp = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body))
        return json.loads(resp['body'].read())
    #def _is_relevant_service(self, svc_id: str, extracted_data: Dict[str, Any], sim_score=0.75) -> bool:
        """
        Scalable Vector Similarity check to prevent cross-contamination.
        """
        schema = self.SERVICE_SCHEMAS.get(svc_id)
        if not schema or not extracted_data:
            return False

        # Hard Constraint: Field Name Check
        # Still useful to ensure the LLM didn't hallucinate keys from other depts
        valid_fields = schema.get('triage_data', {}).get('missing', [])
        if not any(key in valid_fields for key in extracted_data.keys()):
            return False

        # Semantic Constraint: Vector Similarity
        # Compare the extracted data content to the Service's defined purpose
        try:
            # Flatten the extracted data into a string for embedding
            data_str = " ".join([f"{k}: {v}" for k, v in extracted_data.items()])
            data_vector = np.array(self._get_embedding(data_str)).reshape(1, -1) #

            # Get the pre-cached embedding for the service description
            # (You should generate this once at startup and store it in svc_info)
            service_description = schema.get('description', "")
            service_vector = np.array(self._get_embedding(service_description)).reshape(1, -1) #

            similarity = cosine_similarity(data_vector, service_vector)[0][0] #
            
            # Debugging is key for fine-tuning your threshold
            print(f"[DEBUG] Similarity for {svc_id}: {similarity:.4f}")

            
            return similarity > sim_score
            
        except Exception as e:
            print(f"[ERROR] Vector relevance check failed: {e}")
            return True # Fallback to True to avoid blocking the flow on API errors
    def _is_relevant_service(self, svc_id: str, extracted_data: Dict[str, Any]) -> bool:
        """
        Determines if the extracted data truly belongs to the given service.
        """
        # 1. Get the list of field names specific to this department from the registry
        # In your case, Immigration fields often start with 'Task.'
        schema = self.SERVICE_SCHEMAS.get(svc_id, {})
        valid_fields = schema.get('triage_data', {}).get('missing', [])
        
        # 2. Check if any of the keys found by the LLM match this specific schema
        # This prevents cross-contamination (e.g., MOJ fields being added during Immigration)
        has_matching_fields = any(key in valid_fields for key in extracted_data.keys())
        
        # 3. Use semantic keywords to verify department relevance
        # This acts as a secondary safety gate
        content_str = str(extracted_data).lower()
        keywords = {
            "immigration_and_visas": ["visa", "permit", "settlement", "immigrat"],
            "moj": ["court", "justice", "legal aid", "prison"],
            "hmrc": ["tax", "pension", "vat", "revenue"]
        }
        
        dept_keywords = keywords.get(svc_id, [])
        keyword_match = any(word in content_str for word in dept_keywords) if dept_keywords else True

        return has_matching_fields and keyword_match

    async def _finalize_handoff(self, system_instruction: str, handoff_signal: str) -> str:
        """
        Triggers the final LLM response before the technical handoff occurs.
        """
        # Use the current history (which now includes the tool result)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_instruction,
            "messages": self.history,
            "max_tokens": 512, 
            "temperature": self.temperature
        }
        
        resp = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body))
        resp_body = json.loads(resp['body'].read())
        
        # Extract the assistant's parting words
        final_text = next(
            (c['text'] for c in resp_body.get('content', []) if c['type'] == 'text'), 
            "Transferring you to an agent..."
        )
        self._add_to_history("user", "[SYSTEM]: User has been moved to a live chat queue. Awaiting hand-back.")
        return f"{final_text}\n\n{handoff_signal}"

    async def _run_triage_extraction(self) -> str:
        """
        """
        
        # check if agent has the mixin injected, if not triage extraction is an identity function
        if not hasattr(self, 'extract_triage_data') or not hasattr(self, 'SERVICE_SCHEMAS'):
            return ""
        
        # pull service id from servicetriageqmixin 
        active_svc_id = getattr(self, 'active_service_id', None)
        services_to_check = {active_svc_id: self.SERVICE_SCHEMAS[active_svc_id]} if active_svc_id else self.SERVICE_SCHEMAS

        injection_parts = []

        for svc_id, svc_info in services_to_check.items():

            report = await self.extract_triage_data(svc_info['name'], self.history)
            extracted = report.get("extracted", {})
            #print(f"\n Running extraction for: {svc_info['name']}")
            if extracted and self._is_relevant_service(svc_id, extracted):

                self.active_service_id = svc_id
                self.triage_state.update(extracted)
                required_fields = svc_info['triage_data'].get('missing', [])
                still_missing = [f for f in required_fields if f not in self.triage_state]
                print(f"COLLECTED TRIAGE INFO: {self.triage_state}")
                print(f"MISSING TRIAGE INFO: {still_missing}")
                injection_parts.append(
                    f"\n\n### MANDATORY TRIAGE PROTOCOL ###\n"
                    f"The following fields are already VERIFIED in the session database: {json.dumps(self.triage_state)}\n"
                    f"1. DO NOT mention, confirm, or ask the user about these verified fields.\n"
                    f"2. YOUR NEXT QUESTION MUST ONLY BE ABOUT: {still_missing}.\n"
                    f"3. IF 'Still Missing' is empty, YOU MUST IMMEDIATELY CALL the appropriate 'connect_to_live_chat' tool. "
                    f"DO NOT ask the user for permission to connect if you have already gathered the data; just execute the tool."
                )
                if not still_missing:
                    injection_parts.append(
                        f"\n\n### TRIAGE COMPLETE ###\n"
                        f"Verified Data: {json.dumps(self.triage_state)}\n"
                        f"INSTRUCTIONS:\n"
                        f"1. Inform the user you have all the necessary details.\n"                        
                        f"2. ASK the user if they would like you to connect them to a live specialist now.\n"                        
                        f"3. STOP. Do not call the 'connect_to_live_chat' tool yet. Wait for their agreement."
                    )
                
        return "\n\n".join(injection_parts)

    async def _handle_handoff_gate(self, tool_name: str, args: dict) -> str:
        
        service_id = tool_name.replace("connect_to_live_chat_", "")

        triage_report = await self.extract_triage_data(service_id, self.history)
        final_triage = {**self.triage_state, **triage_report.get("extracted", {})}
        required = self.SERVICE_SCHEMAS.get(service_id, {}).get('triage_data', {}).get('missing', [])
        missing_now = [f for f in required if f not in final_triage]

        if missing_now:

            return (f"HANDOFF_BLOCKED: Information missing for {service_id}: {missing_now}."
                    f"INSTRUCTION: Ask the user for this missing information using {triage_report.get('follow up')}")

        args['triage_data'] = final_triage 
        func = self.available_tools[tool_name]
        if 'app.integrations' in func.__module__:
            args['history'] = deepcopy(self.history)
        return await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)        

    async def _send_message_and_tools(self, prompt: str) -> str:
        """The core orchestration loop shared by all agents, now with dynamic triage gating."""
        
        self._add_to_history("user", prompt)

        while True:
            # Refresh triage injection every loop so the LLM sees what it just collected
            triage_injection = await self._run_triage_extraction()

            effective_system_instruction = self.prompt_guidance.compose_system_instruction(
                self.system_instruction + triage_injection,
                prompt,
                self.history
            )
        
            resp_body = self._call_llm(effective_system_instruction)
            content = resp_body.get('content', [])

            text = next((c['text'] for c in content if c['type'] == 'text'), None)
            tool_use = [c for c in content if c['type'] == 'tool_use']

            self._add_to_history("assistant", text, tool_calls=tool_use)

            if not tool_use:
                return text or "I encountered an error."

            results = []

            for call in tool_use:

                out = await self._execute_tool(call)

                results.append({
                    "type": "tool_result",
                    "tool_use_id": call['id'],
                    "content": [{"type": "text", "text": str(out)}]
                })

            self._add_to_history("user", tool_results=results)

            if "SIGNAL: initiate_live_handoff" in str(out):
                return await self._finalize_handoff(effective_system_instruction, out)

            self._add_to_history("user", tool_results=results)
