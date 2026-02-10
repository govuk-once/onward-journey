import json
import numpy as np
import boto3
import os

from typing                   import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from opensearchpy             import OpenSearch
from dotenv                   import load_dotenv
from copy import deepcopy

from helpers                  import SearchResult

import asyncio
import tools

load_dotenv()

def default_handoff():
    return {'handoff_agent_id': 'GOV.UK Chat', 'final_conversation_history': []}

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
                "connect_to_live_chat_MOJ": tools.connect_to_moj,
                "connect_to_live_chat_immigration": tools.connect_to_immigration,
                "connect_to_live_chat_HMRC_pensions_forms_and_returns": tools.connect_to_hmrc
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

class RunConversationMixin:
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

class BaseAgent:
    def __init__(self, model_name: str, aws_region: str, temperature: float = 0.0):

            self.client = boto3.client(service_name="bedrock-runtime", region_name=aws_region)
            self.aws_region = aws_region
            self.model_name = model_name
            self.temperature = temperature
            self.history: List[Dict[str, Any]] = []
            self.available_tools = {}
            self.bedrock_tools = []
            self.system_instruction = ""

    def _add_to_history(self, role: str, text: str = '', tool_calls: list = [], tool_results: list = []):
        """Standardized history management for all subclasses."""
        message = {"role": role, "content": []}
        if text: message["content"].append({"type": "text", "text": text})
        if tool_calls: message["content"].extend(tool_calls)
        if tool_results: message["content"].extend(tool_results)
        self.history.append(message)

    async def _send_message_and_tools(self, prompt: str) -> str:
            """The core orchestration loop shared by all agents."""
            self._add_to_history("user", prompt)

            while True:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "system": self.system_instruction,
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

                self._add_to_history("assistant", text, tool_calls=tool_use)

                if not tool_use:
                    return text or "I encountered an error."

                results = []
                handoff_signal = None

                for call in tool_use:
                    func = self.available_tools[call['name']]
                    args = call['input']
                    if func.__module__ == 'tools':
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

class GovUKAgent(BaseAgent, HandOffMixin, QueryEmbeddingMixin, GovUKSearchMixin, RunConversationMixin):
    def __init__(self, handoff_package: dict,
                 embedding_model:str = "amazon.titan-embed-text-v2:0",
                 model_name: str = "anthropic.claude-3-7-sonnet-20250219-v1:0",
                 aws_region: str = 'eu-west-2',
                 temperature: float = 0.0, top_K_govuk: int = 3):
        super().__init__(model_name, aws_region, temperature)

        self.handoff_package = handoff_package

        self.embedding_model = embedding_model
        self.top_K_govuk     = top_K_govuk

        self.os_client = OpenSearch(
            hosts=[os.getenv("OPENSEARCH_URL")],
            http_auth=(os.getenv("OPENSEARCH_USERNAME"), os.getenv("OPENSEARCH_PASSWORD"))
        )
        self.os_index = 'govuk_chat_chunked_content'

        self._tool_declarations()

    def _tool_declarations(self):
        self.available_tools = {
            "query_govuk_kb": self.query_govuk_kb,
        }
        self.bedrock_tools = tools.get_govuk_definitions()

class OnwardJourneyAgent(BaseAgent, HandOffMixin, QueryEmbeddingMixin, OJSearchMixin, LiveChatMixin, RunConversationMixin):
    def __init__(self,
                 handoff_package: dict,
                 vector_store_embeddings: np.ndarray,
                 vector_store_chunks: list[str],
                 embedding_model:str = "amazon.titan-embed-text-v2:0",
                 model_name: str = "anthropic.claude-3-7-sonnet-20250219-v1:0",
                 aws_region: str = 'eu-west-2',
                 temperature: float = 0.0,
                 top_K_OJ: int = 3,
                 verbose: bool = False):

        super().__init__(model_name=model_name, aws_region=aws_region, temperature=temperature)

        self.verbose = verbose
        self.client = boto3.client(service_name="bedrock-runtime", region_name=aws_region)

        # Onward Journey related
        self.handoff_package = handoff_package
        self.embeddings = vector_store_embeddings
        self.chunk_data = vector_store_chunks


        self.embedding_model = embedding_model
        self.temperature     = temperature
        self.top_K_OJ        = top_K_OJ


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

    def _tool_declarations(self):
        """Maps Bedrock tool names to Python functions based on strategy."""

        # Internal RAG logic stays in Agent to access local embeddings
        self.available_tools = {
            "query_internal_kb": self.query_internal_kb,
        }

        self.available_tools = self.available_tools | self._get_live_chat_registry()

        self.bedrock_tools = (tools.get_internal_kb_definition() + tools.get_live_chat_definitions())

class hybridAgent(OnwardJourneyAgent, HandOffMixin, QueryEmbeddingMixin, OJSearchMixin, GovUKSearchMixin, LiveChatMixin, RunConversationMixin):
    def __init__(self,
                 handoff_package: dict,
                 vector_store_embeddings: np.ndarray,
                 vector_store_chunks: list[str],
                 embedding_model:str = "amazon.titan-embed-text-v2:0",
                 model_name: str = "anthropic.claude-3-7-sonnet-20250219-v1:0",
                 aws_region: str = 'eu-west-2',
                 temperature: float = 0.0,
                 top_K_OJ: int = 3,
                 top_K_govuk: int = 3,
                 verbose: bool = False):

        super().__init__(handoff_package, vector_store_embeddings, vector_store_chunks, embedding_model, model_name, aws_region, temperature, top_K_OJ)

        self.handoff_package = handoff_package

        self.embedding_model =embedding_model
        self.top_K_OJ        = top_K_OJ
        self.top_K_govuk     = top_K_govuk

        self._tool_declarations()
    def _tool_declarations(self):
        """Maps Bedrock tool names to Python functions based on strategy."""

        # Internal RAG logic stays in Agent to access local embeddings
        self.available_tools = {
            "query_internal_kb": self.query_internal_kb,
            "query_govuk_kb": self.query_govuk_kb,
        }

        self.available_tools = self.available_tools | self._get_live_chat_registry()

        self.bedrock_tools = tools.get_internal_kb_definition() + tools.get_govuk_definitions() + tools.get_live_chat_definitions()
