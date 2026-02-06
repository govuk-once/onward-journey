import json
import numpy as np
import boto3
import os

from typing                   import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from opensearchpy             import OpenSearch
from dotenv                   import load_dotenv

from helpers                  import SearchResult


import asyncio
import uuid
import tools 

load_dotenv()

def default_handoff():
    return {'handoff_agent_id': 'GOV.UK Chat', 'final_conversation_history': []}

class OnwardJourneyAgent:
    def __init__(self,
                 handoff_package: dict,
                 vector_store_embeddings: np.ndarray,
                 vector_store_chunks: list[str],
                 embedding_model:str = "amazon.titan-embed-text-v2:0",
                 model_name: str = "anthropic.claude-3-7-sonnet-20250219-v1:0",
                 aws_region: str = 'eu-west-2',
                 temperature: float = 0.0,
                 strategy: int = 4,
                 top_K_OJ: int = 3,
                 top_K_govuk: int = 1,
                 verbose: bool = False):

        self.verbose = verbose
        self.client = boto3.client(service_name="bedrock-runtime", region_name=aws_region)

        # Local KB (Onward Journey)
        self.embeddings = vector_store_embeddings
        self.chunk_data = vector_store_chunks

        self.model_name      = model_name
        self.embedding_model = embedding_model
        self.embedding_model = embedding_model
        self.temperature     = temperature
        self.top_K_OJ        = top_K_OJ
        self.top_K_govuk     = top_K_govuk

        # Remote KB (GOV.UK OpenSearch)
        self.os_client = OpenSearch(
            hosts=[os.getenv("OPENSEARCH_URL")],
            http_auth=(os.getenv("OPENSEARCH_USERNAME"), os.getenv("OPENSEARCH_PASSWORD"))
        )
        self.os_index = 'govuk_chat_chunked_content'

        # State & History
        self.handoff_package = handoff_package
        self.history: List[Dict[str, Any]] = []

        self.strategy = strategy

        self._tool_declarations()

        self.system_instruction = (
                    "You are the **Onward Journey Agent**. Your sole purpose is to process "
                    "and help with the user's request. **Your priority is aiding and clarifying until you have all the information needed to provide a final answer.** "
                    "This includes:"
                    "1. **Ambiguity Check:** If the user's request is ambiguous or requires a specific detail (e.g., 'Tax Credits'), your first turn **MUST BE A TEXT RESPONSE** asking a single, specific clarifying question. **DO NOT CALL THE TOOL YET.** "
                    "This can include when there are MULTIPLE phone numbers so can you clarify until only one phone number remains following user clarification."
                    "2. **Tool Use:** If the request is clear, OR if the user has just provided the clarification, you must call the `query_internal_kb` and/or `query_govuk_kb` tools to find answers to the user query. "
                    "3. **Final Answer:** After the tool call(s) is/are complete, provide the final, grounded answer unless clarification is needed." \
                    "You have access to two knowledge bases which you can query using the tools provided. "
                    "Make sure your responses are formatted well for the user to read." \
                    "Always be looking to clarify if there is any ambiguity in the user's request."
                    "You can use both tools if the query requires a cross-referenced answer."
                    "If a phone number is provided for a MOJ-related query, you must call the `connect_to_live_chat_MOJ` tool" 
                    "to transfer the user to a live agent IF they want a human agent. If a phone number is provided for an "
                    "immigration-related query, you must call the `connect_to_live_chat_immigration` tool to transfer the user" 
                    "to a live agent IF they want a human agent. All other live chats are currently not available."
                    "If a phone number is provided for a HMRC pensions, forms and returns related query, you must call the `connect_to_live_chat_HMRC_pensions_forms_and_returns` tool" \
                    "to transfer the user to a live agent IF they want a human agent."
                    "CRITICAL FORMATTING RULES:\n"
                    "1. Use **Markdown** for all responses.\n"
                    "2. Use ### Headers for distinct sections.\n"
                    "3. Use **bold** for emphasis, phone numbers, and key terms.\n"
                    "4. Use bullet points or numbered lists for steps or multiple contact details.\n"
                    "5. Use > blockquotes for important notes or warnings.\n\n"
                    "Example structure:\n"
                    "### Section Title\n"
                    "* **Phone:** `0300...`\n"
                    "* **Hours:** 9am - 5pm\n\n"
                    "Always clarify ambiguity before calling tools."
                                  )

    def _add_to_history(self, role: str, text: str = '', tool_calls: list = [], tool_results: list = []):
        """Ensures content is always a list of valid dictionaries."""
        message = {"role": role, "content": []}

        # Text must be wrapped in a dictionary with a 'type' key
        if text:
            message["content"].append({"type": "text", "text": text})

        # Tool calls from the model are already dictionaries
        if tool_calls:
            message["content"].extend(tool_calls)

        # Tool results must be wrapped correctly
        if tool_results:
            message["content"].extend(tool_results)

        self.history.append(message)

    def _get_embedding(self, text: str) -> List[float]:
        """Standardized embedding for all KBs."""
        body = json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        })
        response = self.client.invoke_model(
            modelId=self.embedding_model,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        return json.loads(response.get('body').read()).get('embedding', [])

    # orchestration of messaging and tool calling 
    async def _send_message_and_tools(self, prompt: str) -> str:
        self._add_to_history("user", prompt) #

        while True:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "system": self.system_instruction, #
                "messages": self.history,
                "messages": self.history,
                "max_tokens": 4096,
                "temperature": self.temperature,
                "tools": self.bedrock_tools
            }

            resp = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body))
            resp_body = json.loads(resp['body'].read())

            content = resp_body.get('content', [])
            text = next((c['text'] for c in content if c['type'] == 'text'), None)
            tool_use = [c for c in content if c['type'] == 'tool_use']

            self._add_to_history("assistant", text, tool_calls=tool_use) #

            if not tool_use:
                return text or "I encountered an error."

            results = []
            handoff_signal = None 

            for call in tool_use:
                func = self.available_tools[call['name']]
                args = call['input']
                
                if func.__module__ == 'tools':
                    from copy import deepcopy
                    args['history'] = deepcopy(self.history)

                out = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)

                if "SIGNAL: initiate_live_handoff" in str(out):
                    handoff_signal = out 

                results.append({
                    "type": "tool_result",
                    "tool_use_id": call['id'],
                    "content": [{"type": "text", "text": out}]
                })


            self._add_to_history("user", tool_results=results)

            # If we have a handoff signal, we stop the loop here and return
            # This prevents a second invoke_model call that might fail or lose the signal
            if handoff_signal:

                final_body = body.copy()
                final_body["messages"] = self.history
                
                final_resp = self.client.invoke_model(modelId=self.model_name, body=json.dumps(final_body))
                final_resp_body = json.loads(final_resp['body'].read())
                final_text = next((c['text'] for c in final_resp_body.get('content', []) if c['type'] == 'text'), "Transferring...")
                
                return f"{final_text}\n\n{handoff_signal}"

    def _tool_declarations(self):
        """Maps Bedrock tool names to Python functions based on strategy."""

        # Internal RAG logic stays in Agent to access local embeddings
        self.available_tools = {
            "query_internal_kb": self.query_internal_kb,
            "query_govuk_kb": self.query_govuk_kb
        }

        # External logic mapped from tools.py
        external_mapping = {
            "connect_to_live_chat_MOJ": tools.connect_to_moj,
            "connect_to_live_chat_immigration": tools.connect_to_immigration,
            "connect_to_live_chat_HMRC_pensions_forms_and_returns": tools.connect_to_hmrc
        }

        if self.strategy == 4:
            self.available_tools = self.available_tools | external_mapping
            self.available_tools.pop("query_govuk_kb", None)
        elif self.strategy == 1:
            self.available_tools = {"query_internal_kb": self.query_internal_kb}
        elif self.strategy == 2:
            self.available_tools = {"query_govuk_kb": self.query_govuk_kb}

        self.bedrock_tools = tools.get_tool_definitions(self.strategy)




        if self.verbose:
            active_tool_names = [t['name'] for t in self.bedrock_tools]
            print(f"DEBUG: Strategy {self.strategy} active. Tools available: {active_tool_names}")

    async def process_handoff(self)-> Optional[str]:
        """
        Processes handoff context with three specific tool-use strategies:
        1: Use OJ KB only (Ignore GOVUK)
        2: Use GOVUK KB only (Ignore OJ)
        3: Use both (Standard autonomous mode)
        """
        history = self.handoff_package.get('final_conversation_history', [])

        if not history:
            if self.verbose:
                print("Handoff history is empty. Treating as a standard chat.")
            return None # avoid LLM hallucinating an empty string

        history_str = json.dumps(history)

        # Strategy-specific constraints
        constraints = {
            1: "CRITICAL: You must ONLY use the 'query_internal_kb' tool for this initial turn.",
            2: "CRITICAL: You must ONLY use the 'query_govuk_kb' tool for this initial turn.",
            3: "CRITICAL: Use both 'query_internal_kb', 'query_govuk_kb' to answer."
        }

        selected_constraint = constraints.get(self.strategy, constraints[3])
        initial_prompt = (
            f"Previous conversation history: {history_str}. "
            f"INSTRUCTION: Based on the history above, provide the next response to the user. "
            f"{selected_constraint}\n"
        "Please analyze the history and fulfill the user's request, using your specialized tools if necessary."
        "If more than one phone number is available semantically, ask a clarifying question."
        )
        return await self._send_message_and_tools(initial_prompt)

    async def _initiate_handoff(self, reason: str, deployment_id_env: str):
            """Generic handoff logic to reduce duplication."""
            user_queries = [
                c['text'] for m in self.history 
                for c in m['content'] 
                if m['role'] == 'user' and c['type'] == 'text'
            ]
            summary = f"User asking about: {reason}. History: {' | '.join(user_queries[-3:])}"

            handoff_config = {
                "action": "initiate_live_handoff",
                "deploymentId": os.getenv(deployment_id_env),
                "region": os.getenv('GENESYS_REGION', 'euw2.pure.cloud'),
                "token": str(uuid.uuid4()),
                "reason": reason,
                "summary": summary
            }
            return f"SIGNAL: initiate_live_handoff {json.dumps(handoff_config)}"

    async def connect_to_live_chat_MOJ(self, reason: str):
        """
        Returns handoff configuration for the frontend. 
        """
        return await self._initiate_handoff(reason, 'GENESYS_DEPLOYMENT_ID_MOJ')

    async def connect_to_live_chat_immigration(self, reason: str):
        """
        Returns handoff configuration for the frontend. 
        """
        return await self._initiate_handoff(reason, 'GENESYS_DEPLOYMENT_ID_IMMIGRATION')

    async def connect_to_live_chat_HMRC_pensions_forms_and_returns(self, reason: str):
        """
        Returns handoff configuration for the frontend. 
        """
        return await self._initiate_handoff(reason, 'GENESYS_DEPLOYMENT_ID_PENSIONS_FORMS_AND_RETURNS') 

    def query_internal_kb(self, query: str) -> str:
        """Local RAG search."""
        query_vec = np.array(self._get_embedding(query)).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = sims.argsort()[-self.top_K_OJ:][::-1]
        return "Internal Context:\n" + "\n".join([self.chunk_data[i] for i in top_idx])
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
    
    def run_conversation(self) -> None:
            """
            Interactive terminal loop that mirrors the original functionality
            but uses the new unified multi-tool logic.
            """
            # Display the specialized agent's first response
            print("\n" + "-" * 100)
            print("You are now speaking with the Onward Journey Agent.")
            #print(f"Onward Journey Agent: {first_response}")
            print("-" * 100 + "\n")

            # Handle handoff if history exists
            if self.handoff_package.get('final_conversation_history'):
                print("Processing context from previous agent...")
                initial_response = self.process_handoff()
                print(f"\nAgent: {initial_response}\n")

            # Standard interactive loop
            while True:
                try:
                    user_input = input("You: ").strip()

                    if user_input.lower() in ["exit", "quit", "end"]:
                        print("\n👋 Conversation with Onward Journey Agent ended.")
                        break

                    if not user_input:
                        continue

                    response = self._send_message_and_tools(user_input)
                    print(f"\n Onward Journey Agent: {response}\n")

                except KeyboardInterrupt:
                    break
