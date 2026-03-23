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
    SERVICE_SCHEMAS = live_registry.get_triage_fields()

    triage_state: Dict[str, Any] = {}
    active_service_id: Optional[str] = None 

    def _build_extraction_system_prompt(self, schema: Dict, history: List[Dict], last_q: str) -> str:
        """Constructs the structured prompt for the LLM"""
        required = schema['triage_fields'].get('missing', [])
        options = schema['triage_fields'].get('field_options', {})

        return (
                    f"ACT AS A DATA EXTRACTOR. Analyze this conversation history: {history}\n"
                    f"Target Fields: {required}.\n"
                    f"Field Options/Constraints: {options}\n\n"
                    f"CONTEXTUAL ANCHOR: \n"
                    f"The last question asked was: '{last_q}'\n\n"
                    
                    "1. RULES FOR SCALABLE EXTRACTION: \n"
                    "1. SEMANTIC MAPPING: Do not look for exact words. If the user describes a state, extract the corresponding field. "
                    "Examples: ('speed up' or 'no news' -> Update: Yes); ('already sent' or 'last month' -> Applied: Yes).\n"
                    
                    "2. DEPENDENCY LOGIC (COMMON SENSE): Use the relationship between fields to fill 'ghost' answers:\n"
                    "   - PRE-REQUISITES: If a user is at a 'Late Stage' (asking for updates or status), automatically set 'Early Stage' fields (like Help Applying) to 'No'.\n"
                    "   - MUTUAL EXCLUSIVITY: If a user confirms one path, logically infer 'No' for the alternative paths within the same service schema.\n"
                    
                    "3. CONFIDENCE & NULLS: Only fill a field if it is explicitly stated or logically necessitated by COMMON SENSE / DEPENDENCY LOGIC. "
                    "If the information is completely absent and cannot be logically inferred, mark as null.\n"
                    
                    "4. OUTPUT FORMAT: Return the a JSON object for each individual extraction. Do not include any preamble, conversational filler, or markdown formatting other than the JSON itself. If you cannot extract data, return empty extracted object."                    " - 'extracted' : {{field: value}} for identified or logically inferred data.\n"
                    " - 'missing' : [list] of fields that truly remain unknown.\n"
                    " - 'follow_up': A natural question for the FIRST missing field only.\n\n"
                    " - 'reasons' : The reasons for each extraction per variable."
                    
                    "Maintain the logic established by long-form narrative; do not let a 'Yes/No' answer to a specific question overwrite a complex situation described earlier."
                )

    def _is_relevant_service(self, svc_id: str, extracted_data: Dict[str, Any]) -> bool:
        """
        Determines if the extracted data truly belongs to the given service.
        """
        # 1. Get the list of field names specific to this department from the registry
        # In your case, Immigration fields often start with 'Task.'
        schema = self.SERVICE_SCHEMAS.get(svc_id, {})
        valid_fields = schema.get('triage_fields', {}).get('missing', [])
        
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
    
    def _get_schema_key_by_service(self, service: str) -> Optional[str]:
        """Finds key for a given service name"""
        for key, value in self.SERVICE_SCHEMAS.items():
            if value.get('name') == service:
                return key
        return None 

    def _get_last_assistant_question(self, history: List[Dict]) -> str:
        """Extracts text of the most recent assistant message to anchor context"""
        for msg in reversed(history):

            if msg['role'] == 'assistant' and msg.get('content'):
                return next((c['text'] for c in msg['content'] if c['type'] == 'text'), "N/A")

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Safely extract JSON from LLM response"""

        try:
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")

    async def _call_llm_for_extraction(self, prompt: str) -> str:
        """Handles Bedrock API call"""

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.0 # Strict extraction, no creativity
        })
        
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=body
        )
        response_body = json.loads(response['body'].read())

        return response_body['content'][0]['text']

    async def coordinate_service_triage(self, history: List[Dict]) -> str:
        """This is the top level function that conducts slot filling for each service. """

        # determine which services to triage (an actively assigned one or all to begin with)
        services_to_check = {self.active_service_id: self.SERVICE_SCHEMAS[self.active_service_id]} if self.active_service_id else self.SERVICE_SCHEMAS

        # skip if active service is already complete (e.g. no more fields to find)
        if self.active_service_id:
            required = self.SERVICE_SCHEMAS[self.active_service_id]['triage_fields']
            if all(f in self.triage_state for f in required):
                return ""
        
        injection_parts = []

        for svc_id, svc_info in services_to_check.items():

            slot_report = await self.slot_extraction(svc_info['name'], history)
            extracted = slot_report.get("extracted", {})

            print(f"[DEBUG - SLOT REPORT]: Extracted: {extracted}")
            print(f"[DEBUG - CURRENT STATE]: {self.triage_state}")

            if extracted and self._is_relevant_service(svc_id, extracted):

                self.active_service_id = svc_id 
                self.triage_state.update(extracted)

                required_fields = svc_info['triage_fields'].get('missing', [])
                still_missing = [f for f in required_fields if f not in self.triage_state]
                print(f"[DEBUG - TRIAGE PROGRESS]: Service: {svc_id} | Still Missing: {still_missing}")
                if still_missing:
                    injection_parts.append(
                    f"\n\n### MANDATORY TRIAGE IN PROGRESS ###\n"
                    f"Verified Data: {json.dumps(self.triage_state)}\n"
                    f"STILL MISSING: {still_missing}\n"
                    f"CRITICAL INSTRUCTION: You are FORBIDDEN from calling any 'connect_to_live_chat' tools yet. "
                    f"The triage is incomplete. You must ask the user: '{slot_report.get('follow_up')}'"
                )
                else:
                    injection_parts.append(
                        f"\n\n### TRIAGE COMPLETE ###\n"
                        f"Verified Data: {json.dumps(self.triage_state)}\n"
                        f"INSTRUCTIONS:\n"
                        f"1. Inform the user you have all the necessary details.\n"                        
                        f"2. ASK the user if they would like you to connect them to a live specialist now.\n"                        
                        f"3. STOP. Do not call the 'connect_to_live_chat' tool yet. Wait for their agreement."
                    )
        return "\n\n".join(injection_parts)
 
    async def slot_extraction(self, service: str, history: List[Dict]) -> Dict[str, Any]:
        """
        Performs a semantic gap analysis by calling the LLM to extract data from history C.
        """

        # resolve schema details 
        schema_key = self._get_schema_key_by_service(service)
        if not schema_key:
            return {"extracted": {}, "missing": [], "follow_up": None}
        
        # prepare contextual info 
        last_q = self._get_last_assistant_question(history)
        schema_data = self.SERVICE_SCHEMAS[schema_key]

        # build and execute request
        prompt = self._build_extraction_system_prompt(schema_data, history, last_q)
        raw_response = await self._call_llm_for_extraction(prompt)

        print(f"\n[DEBUG - RAW EXTRACTION]: Service: {service}")
        print(f"LLM Response: {raw_response}") # See the raw JSON/Reasoning from the extractor

        parsed = self._parse_llm_response(raw_response)
        # parse and return
        return parsed 
    
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
            self.logs = []
    def _add_to_history(self, role: str, text: str = '', tool_calls: list = [], tool_results: list = []):
        """Standardized history management for all subclasses."""
        message = {"role": role, "content": []}
        if text: message["content"].append({"type": "text", "text": text})
        if tool_calls: message["content"].extend(tool_calls)
        if tool_results: message["content"].extend(tool_results)
        self.history.append(message)

    def _log_thought(self, message: str):
            """Helper to print to console AND save for the frontend."""
            print(f"[AGENT_THOUGHT]: {message}")
            self.logs.append(message)

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

    def _is_triage_complete(self, svc_id: str) -> bool:
        required = self.SERVICE_SCHEMAS.get(svc_id, {}).get('triage_fields', {})
        return all(f in self.triage_state for f in required)
    
    def _get_missing_fields(self, svc_id: str) -> List[str]:
        required = self.SERVICE_SCHEMAS.get(svc_id, {}).get('triage_fields', {}).get('missing', [])
        return [f for f in required if f not in self.triage_state]

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

    async def _handle_handoff_gate(self, tool_name: str, args: dict) -> str:
        
        service_id = tool_name.replace("connect_to_live_chat_", "")
        print(f"[DEBUG - HANDOFF GATE]: Checking {service_id}")
        # check: is local state already complete?
        if not self._is_triage_complete(service_id):
            self._log_thought('Triage not fully completed, I\'m attempting a final extraction.') 
            print(f"[DEBUG - HANDOFF GATE]: Triage incomplete. Attempting final extraction.")
            slot_report  = await self.slot_extraction(service_id, self.history)
            final_report = {**self.triage_state, **slot_report.get("extracted", {})}

            required = self.SERVICE_SCHEMAS.get(service_id, {}).get('triage_fields', {}).get('missing', [])
            missing_now = [f for f in required if f not in final_report]

            if missing_now:
                self._log_thought(f"HANDOFF_BLOCKED: Information missing for {service_id}: {missing_now}."
                        f"INSTRUCTION: Ask the user for this missing information using {slot_report.get('follow up')}")
                print(f"[DEBUG - HANDOFF GATE]: BLOCKED! Missing: {missing_now}")
                return (f"HANDOFF_BLOCKED: Information missing for {service_id}: {missing_now}."
                        f"INSTRUCTION: Ask the user for this missing information using {slot_report.get('follow up')}")

            # sync state 
            self.triage_state.update(slot_report.get("extracted"), {})
        print(f"[DEBUG - HANDOFF GATE]: PASSED. Executing {tool_name}")
        self._log_thought(f"HANDOFF GATE PASSED. Executing {tool_name}")
        args['triage_fields'] = self.triage_state

        func = self.available_tools[tool_name]
        
        if 'app.integrations' in func.__module__:
            args['history'] = deepcopy(self.history)
        return await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)        

    async def _send_message_and_tools(self, prompt: str) -> str:
        """The core orchestration loop shared by all agents, now with dynamic triage gating."""
        
        print('[PROMPT]:', prompt)

        self._add_to_history("user", prompt)

        loop_count = 0
        triage_injected = False 


        last_triage_log = ""

        while True:
            loop_count += 1
            print(f"\n[DEBUG - LOOP ITERATION]: {loop_count}")
            # Refresh triage injection every loop so the LLM sees what it just collected
            triage_injection = ""

            if hasattr(self, 'coordinate_service_triage') and not triage_injected:
                triage_injection = await self.coordinate_service_triage(self.history)
                if triage_injection and triage_injection != last_triage_log:
                        self._log_thought(f"TRIAGE UPDATED: {triage_injection}")
                        last_triage_log = triage_injection
                
                if triage_injection:
                    print('[TRIAGE INJECTION]:', triage_injection)
                triage_injected = True 
            effective_system_instruction = self.prompt_guidance.compose_system_instruction(
                self.system_instruction + triage_injection,
                prompt,
                self.history
            )
        
            resp_body = self._call_llm(effective_system_instruction)
            content = resp_body.get('content', [])

            text = next((c['text'] for c in content if c['type'] == 'text'), None)
            tool_use = [c for c in content if c['type'] == 'tool_use']

            if text: 
                print(f"[DEBUG - ASSISTANT TEXT]: {text}")
            if tool_use: 
                print(f"[DEBUG - TOOL CALLS]: {[t['name'] for t in tool_use]}")
                self._log_thought(f"LLM decided to use tools: {[t['name'] for t in tool_use]}")
            self._add_to_history("assistant", text, tool_calls=tool_use)

            if not tool_use:
                return text or "I encountered an error."

            results = []

            for call in tool_use:

                out = await self._execute_tool(call)
                self._log_thought(f"Here is the TOOL RESULT: {call['name']} -> {str(out)[:100]}...")
                print(f"[DEBUG - TOOL RESULT]: {call['name']} -> {str(out)[:100]}...") # Truncated for readability
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call['id'],
                    "content": [{"type": "text", "text": str(out)}]
                })

            self._add_to_history("user", tool_results=results)

            if "SIGNAL: initiate_live_handoff" in str(out):
                self._log_thought(f"Finalizing handoff, as the handoff signal is present")
                return await self._finalize_handoff(effective_system_instruction, out)
