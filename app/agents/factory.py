import os
import boto3
import numpy as np
from opensearchpy             import OpenSearch
import app.integrations as integrations_registry
import app.core.tools_registry as tools_registry
from app.agents.base import BaseAgent, HandOffMixin, QueryEmbeddingMixin, GovUKSearchMixin, OJSearchMixin, LiveChatMixin, ServiceTriageQMixin

class GovUKAgent(BaseAgent, HandOffMixin, QueryEmbeddingMixin, GovUKSearchMixin):
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
        self.bedrock_tools = tools_registry.get_govuk_definitions()

class OnwardJourneyAgent(BaseAgent, HandOffMixin, QueryEmbeddingMixin, OJSearchMixin, LiveChatMixin, ServiceTriageQMixin):
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
            "and help with the user's request. **Your priority is aiding and clarifying until you have "
            "all the information needed to provide a final answer.**\n\n"

            "### 1. MANDATORY CLARIFICATION RULE\n"
                "* **CRITICAL:** If your tools (like query_internal_kb) return more than one relevant service or phone number, you **MUST NOT** display any phone numbers or specific contact details yet.\n"
                "* **Zero-Disclosure Policy:** If multiple options exist, your response must ONLY consist of a clarifying question asking the user to specify which service they need (e.g., 'Are you looking for personal tax or business tax?').\n"
                "* **Wait for Input:** You are forbidden from providing contact details until the user's ambiguity is resolved to a single specific service.\n\n"
            "### 2. AMBIGUITY AND TRIAGE\n"
            "* **Ambiguity Check:** your first turn **MUST BE A TEXT RESPONSE** asking a single, specific clarifying question. **DO NOT CALL THE TOOL YET** if the initial user prompt is vague.\n"            "your first turn **MUST BE A TEXT RESPONSE** asking a single, specific clarifying question. **DO NOT CALL THE TOOL YET.**\n"
            "* **Post-Tool Ambiguity:** If tools return multiple semantically similar results, treat this as ambiguity and revert to a text response asking for clarification.\n"
            "* **Data Collection:** Before connecting to a live agent, you must ensure all triage questions associated "
            "with that service (e.g., visa type) have been answered. CRITICAL: Only ask a single question for each triage requirement.\n"
            "* **Handling Blocks:** If a tool call returns a 'HANDOFF_BLOCKED' message, do not apologize for a system error. "
            "Instead, naturally ask the user for the missing information specified in that message.\n\n"

            "### 3. TOOL USE AND ROUTING\n"
            "* **Internal Search:** If the request is clear, use your tools to find answers to the user query. \n"
            "CRITICAL: If more than one piece of contact information exists, clarify with the user before providing any contact details. \n"
            "* **Live Chat Tool:** If a live chat is available for a service, ask the user if they want to be connected. "
            "Only call the handoff tool if they explicitly say 'yes'.\n"
            "* **Handoff Summary:** When calling a handoff tool, provide a concise 'reason' summary of the user's "
            "primary concern and the key details collected.\n\n"

            "### 4. FINAL RESPONSES AND TRANSITIONS\n"
            "* **Post-Handoff Transition:** Once a 'SIGNAL: initiate_live_handoff' is returned, your final response "
            "must ONLY be a simple transition message (e.g., 'I am now connecting you to a specialist...').\n"
            "* **No Hallucinations:** DO NOT summarize the outcome of the specialist's work or invent 'next steps' after "
            "a handoff signal, as the specialist will handle the resolution.\n"
            "* **Grounded Answers:** For non-handoff queries, provide a final, grounded answer once tool calls are complete.\n\n"

            "### 5. FORMATTING RULES\n"
            "1. Use **Markdown** for all responses.\n"
            "2. Use ### Headers for distinct sections.\n"
            "3. Use **bold** for emphasis, phone numbers, and key terms.\n"
            "4. Use bullet points or numbered lists for steps or multiple contact details.\n"
            "5. Use > blockquotes for important notes or warnings.\n\n"

            "Always clarify ambiguity before calling tools.")
    def _tool_declarations(self):
        """Maps Bedrock tool names to Python functions based on strategy."""

        # Internal RAG logic stays in Agent to access local embeddings
        self.available_tools = {
            "query_internal_kb": self.query_internal_kb,
        }

        self.available_tools = self.available_tools | self._get_live_chat_registry()

        self.bedrock_tools = (tools_registry.get_internal_kb_definition() + integrations_registry.get_live_chat_definitions())

class hybridAgent(OnwardJourneyAgent, HandOffMixin, QueryEmbeddingMixin, OJSearchMixin, GovUKSearchMixin, LiveChatMixin):
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

        self.bedrock_tools = tools_registry.get_internal_kb_definition() + tools_registry.get_govuk_definitions() + integrations_registry.get_live_chat_definitions()
