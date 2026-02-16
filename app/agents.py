import json
import numpy as np
import boto3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from typing import List, Dict, Any, Optional

from memory_store import MemoryStore, MemoryItem

class OnwardJourneyAgent:
    """
    Agent designed to take a handoff package, initialize with specialized tools/data,
    and immediately continue the user's journey based on the previous context.
    """
    def __init__(
        self,
        handoff_package: dict,
        vector_store_embeddings: np.ndarray,
        vector_store_embeddings_text_chunks: list[str],
        embedding_model: SentenceTransformer,
        aws_role_arn: Optional[str] = None,
        model_name: str = "anthropic.claude-3-7-sonnet-20250219-v1:0",
        aws_region: str = "eu-west-2",
        aws_role_session_name: str = "onward-journey-inference",
        temperature: float = 0.0,
        verbose: bool = False,
        seed: int = 1,
        top_K: int = 3,
        memory_store: Optional[MemoryStore] = None,
        session_id: str = "default-session",
        memory_k: int = 5,
        best_practice_store: Optional[MemoryStore] = None,
        best_practice_k: int = 3,
        best_practice_outcome: str = "good",
        fast_answer_threshold: float = 0.95,
        fast_answer_exclude_outcome: str = "bad",
        guardrail_tags: Optional[list[str]] = None,
    ):

        # declare tools for bedrock
        self._tool_declarations()

        # initial aws bedrock client and role assumption
        self._initialise_aws(aws_region, role_arn=aws_role_arn, role_session_name=aws_role_session_name)

        # Clarification finite state
        self.awaiting_clarification: bool = False

        # Model configuration
        self.model_name      = model_name
        self.temperature     = temperature

        # Store the handoff package for processing
        self.handoff_package = handoff_package

        # Define available tools and their names
        self.specialized_tools = [self.query_csv_rag]
        self.available_tools   = {f.__name__: f for f in self.specialized_tools}
        self.top_K             = top_K


        # Accessibility to vector store embeddings, vector store text chunk equivalents and vector store embedding model to embed user queries
        self.embeddings      = vector_store_embeddings
        self.chunk_data      = vector_store_embeddings_text_chunks
        self.embedding_model = embedding_model

        # Seed for reproducibility
        self.seed            = seed

        # Memory configuration
        self.memory_store    = memory_store
        self.session_id      = session_id
        self.memory_k        = memory_k
        self.best_practice_store = best_practice_store
        self.best_practice_k = best_practice_k
        self.best_practice_outcome = best_practice_outcome
        self.fast_answer_threshold = fast_answer_threshold
        self.fast_answer_exclude_outcome = fast_answer_exclude_outcome
        self.guardrail_tags = guardrail_tags

        # Specialized System Instruction for onward journey agent
        self.system_instruction = (
            "You are the **Onward Journey Agent**. Your sole purpose is to process "
            "and complete the user's request. **Your priority is correctness.** "
            "1. **Ambiguity Check:** If the user's request is ambiguous or requires a specific detail (e.g., 'Tax Credits'), your first turn **MUST BE A TEXT RESPONSE** asking a single, specific clarifying question. **DO NOT CALL THE TOOL YET.** "
            "2. **Tool Use:** If the request is clear, OR if the user has just provided the clarification, you must call the `query_csv_rag` tool to find the answer. "
            "3. **Final Answer:** After the tool call is complete, provide the final, grounded answer."
            "Make sure your responses are formatted well for the user to read."
        )
        # Verbosity for debugging
        self.verbose = verbose

        # Initialize conversation history
        self.history: List[Dict[str, Any]] = [  ]

        # Track last applied tags from :bp for per-turn updates
        self._session_tags: Optional[list[str]] = None

    def _initialise_aws(self, aws_region: str, role_arn: Optional[str], role_session_name: str):
        if role_arn != None:
            # Assume given role
            try:
                self.client = boto3.client(service_name="sts", region_name=aws_region)
                assume_role_response = self.client.assume_role(
                    RoleArn = role_arn,
                    RoleSessionName = role_session_name
                )
                role_credentials = assume_role_response["Credentials"]
            except:
                raise ValueError(f"Failed to assume role %s using session name %s" % (role_arn, role_session_name))

            try:
                self.client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=aws_region,
                    aws_access_key_id = role_credentials["AccessKeyId"],
                    aws_secret_access_key = role_credentials["SecretAccessKey"],
                    aws_session_token = role_credentials["SessionToken"]
                )
            except:
                raise ValueError("Failed to initialize Bedrock client. Check your AWS configuration.")

        else:
            # Use credentials from environment and config files without assuming a role
            try:
                self.client = boto3.client(service_name="bedrock-runtime", region_name=aws_region)
            except:
                raise ValueError("Failed to initialize Bedrock client. Check your AWS configuration.")
        return
    def _tool_declarations(self):

        self.query_csv_declaration = {
            "name": "query_csv_rag",
            "description": "Performs a RAG search on internal data...",
            "input_schema": {
                            "type": "object",
                            "properties": {
                                            "user_query": {
                                                            "type": "string",
                                            "description": "The user's specific natural language request..."
                                                          }
                                          },
                            "required": ["user_query"],
                          },
        }

        self.bedrock_tools = [
            self.query_csv_declaration
        ]

        return

    def _add_to_history(self, role: str, content: Optional[str] = None, tool_calls: Optional[List[Dict]] = None, tool_results: Optional[List[Dict]] = None):
            """Adds a message to the internal history list in Bedrock format."""
            message: Dict[str, Any] = {"role": role, "content": []}

            if content:
                message['content'].append({"type": "text", "text": content})

            if tool_calls:
                for call in tool_calls:
                    message['content'].append(call)

            if tool_results:
                for result in tool_results:
                    message['content'].append(result)

            # Only append messages that have actual content
            if message['content']:
                self.history.append(message)

    def _build_system_instruction(self, memory_context: Optional[str]) -> str:
        """Combine base instruction with optional memory context."""
        if not memory_context:
            return self.system_instruction
        return (
            f"{self.system_instruction}\n\n"
            f"Prior notes from this user/session:\n{memory_context}\n"
            "Use them if relevant; ignore if not."
        )

    def _get_memory_context(self, query: str) -> Optional[str]:
        """Retrieve formatted memory snippets for the current session."""
        if not self.memory_store:
            return None
        memories: List[MemoryItem] = self.memory_store.search(
            session_id=self.session_id, query=query, k=self.memory_k
        )

        formatted_sections = []
        if memories:
            formatted_sections.append(
                "Recent session notes:\n" + "\n".join(m.format_for_prompt() for m in memories)
            )
        guardrails = self._build_guardrail_block(query=query)
        if guardrails:
            formatted_sections.append(guardrails)

        if not formatted_sections:
            return None
        return "\n\n".join(formatted_sections)

    def _build_guardrail_block(self, query: str) -> Optional[str]:
        """
        Fetch best-practice patterns and render as concise guardrail bullets.
        """
        if not self.best_practice_store:
            return None
        results = self.best_practice_store.search_best_practice(
            query=query,
            outcome=self.best_practice_outcome,
            tags=self.guardrail_tags,
            k=self.best_practice_k,
        )
        if not results:
            return None

        lines = []
        for item, score in results:
            tag_str = f"[{', '.join(item.tags)}] " if item.tags else ""
            lines.append(f"- {tag_str}{item.summary}")
        return "Proven helpful patterns (guardrails):\n" + "\n".join(lines)

    def _get_fast_answer(self, query: str) -> Optional[str]:
        """
        Check prior session memories for a highly similar question and reuse the stored answer
        when the cosine similarity exceeds the configured threshold.
        """
        if not self.memory_store or self.fast_answer_threshold <= 0:
            return None
        results = self.memory_store.search_with_scores(
            session_id=self.session_id, query=query, k=self.memory_k
        )
        if not results:
            return None
        # filter out low-quality outcomes if tagged
        filtered = [
            (item, score)
            for item, score in results
            if item.outcome != self.fast_answer_exclude_outcome
        ]
        if not filtered:
            return None
        best_item, score = filtered[0]
        if score >= self.fast_answer_threshold:
            return best_item.text
        return None

    def _record_memory(self, user_text: str, assistant_text: Optional[str], tags: Optional[list[str]] = None) -> None:
        """Persist a concise memory of the turn."""
        if not self.memory_store:
            return
        if assistant_text is None:
            assistant_text = ""
        # Summary is just the user question to anchor similarity search on queries
        summary = user_text.strip()
        self.memory_store.add(
            session_id=self.session_id,
            role="assistant",
            text=assistant_text,
            summary=summary,
            outcome=None,
            tags=tags,
        )

    def _record_best_practice(
        self,
        user_text: str,
        assistant_text: str,
        outcome: str,
        tags: Optional[list[str]] = None,
    ) -> None:
        """Persist a best-practice memory entry when marked helpful."""
        if not self.best_practice_store:
            return
        summary = f"Helpful pattern: User: {user_text.strip()} | Assistant: {assistant_text.strip()}"
        self.best_practice_store.add(
            session_id="best_practice",
            role="assistant",
            text=assistant_text,
            summary=summary,
            outcome=outcome,
            tags=tags,
        )

    def _flush_session_memory(self) -> None:
        """
        Persist any deferred memory entries (for JSON store) and clear deferral.
        """
        if not self.memory_store:
            return
        try:
            from memory_store import JsonMemoryStore
            if isinstance(self.memory_store, JsonMemoryStore):
                self.memory_store.set_defer_persist(False)
                self.memory_store.persist()
        except Exception:
            pass

    def _send_message_and_handle_tools(self, prompt: str) -> str:
        """
        Sends a message to the Bedrock model and handles tool calls in a loop.
        """
        # Fast path: reuse a previous answer if the question is very similar
        fast_answer = self._get_fast_answer(prompt)
        if fast_answer:
            self._add_to_history(role="user", content=prompt)
            self._add_to_history(role="assistant", content=fast_answer)
            if self.verbose:
                print("Fast answer served from session memory (no model call).")
            return fast_answer

        memory_context = self._get_memory_context(prompt)

        # Add the new user prompt to history
        self._add_to_history(role="user", content=prompt)

        # Handle clarification state
        if self.awaiting_clarification:
            self.awaiting_clarification = False
            if self.verbose:
                print("Clarification received from user. Proceeding to tool call.")

        response_text = ""
        # The loop condition is based on whether the last model response contained tool use
        while True:
            # Prepare the request body for the Bedrock API
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": self.history,
                "system": self._build_system_instruction(memory_context),
                "max_tokens": 4096,
                "temperature": self.temperature,
                "tools": self.bedrock_tools,
            }

            # Send the request to Bedrock
            bedrock_response = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )

            # Parse the response
            response_body = json.loads(bedrock_response.get('body').read())

            model_content = response_body.get('content', [])

            # Extract Text or Tool Calls
            tool_calls   = [c for c in model_content if c.get('type') == 'tool_use']
            text_content = next((c.get('text') for c in model_content if c.get('type') == 'text'), None)

            # Add the model's response to history
            self._add_to_history(role="assistant", content=text_content, tool_calls=tool_calls)

            # if no tool calls and text content is present and we were not awaiting clarification
            if not tool_calls and text_content and not self.awaiting_clarification:
                # Model responds with text and we were not awaiting clarification
                if self.verbose:
                    print("Onward Journey Agent detects ambiguity. Returning clarification question.")
                self.awaiting_clarification = True
                return text_content
            if not tool_calls:
                response_text = text_content if text_content else "I couldn't generate a response."
                break

            # Execute Tools and prepare results for history
            tool_results_for_history = []

            for call in tool_calls:
                tool_use_id   = call['id'] # Unique ID for this tool use
                function_name = call['name'] # The tool to call
                args          = call['input'] # 'input' holds the arguments for the tool

                if self.verbose:
                    print(f"Onward Journey requests tool call: {function_name}({args})")

                # Execute the tool if it's available
                if function_name in self.available_tools:
                    tool_function = self.available_tools[function_name]
                    tool_result = tool_function(**args)

                    if self.verbose:
                        print(f"Tool execution result: {tool_result}")

                    # Prepare the result in the Bedrock 'tool_result' format
                    tool_results_for_history.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [
                            {"type": "text", "text": str(tool_result)}
                        ]
                    })
                else:
                    if self.verbose:
                        print(f"Warning: Model attempted to call unauthorized tool: {function_name}")
                    tool_results_for_history.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [
                            {"type": "text", "text": f"Tool '{function_name}' is not allowed."}
                        ]
                    })

            # Add the tool results to history and continue the loop for the next LLM turn
            self._add_to_history(role="user", tool_results=tool_results_for_history)

            # Loop continues: The next iteration of the while loop sends the history
            # including the tool results back to the model.

        return response_text

    def process_handoff(self) -> str:
        """
        Processes the initial handoff data to generate the first specialized response.
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print(f"Agent Processing Handoff from: {self.handoff_package['handoff_agent_id']}")

        # Bedrock's history starts fresh with this complex handoff message
        context_prompt = (
                f"Previous conversation history: {json.dumps(self.handoff_package['final_conversation_history'])}. "
                f"The user's final request is: {self.handoff_package['next_agent_prompt']}. "
                "Please analyze the history and fulfill the user's request, using your specialized tools if necessary."
            )
        print('User: ', self.handoff_package['next_agent_prompt'])

        # Use the internal send method to process the handoff, which manages history
        first_response = self._send_message_and_handle_tools(context_prompt)

        return first_response

    def run_conversation(self)-> None:
        """
        Runs the full conversation: handles the handoff first, then starts the
        interactive loop with the user.
        """
        # TODO: self.process_handoff() not yet implemented

        # Display the specialized agent's first response
        print("\n" + "-" * 100)
        print("You are now speaking with the Onward Journey Agent.")
        #print(f"Onward Journey Agent: {first_response}")
        print("-" * 100 + "\n")

        # 2. Start the interactive loop
        last_user: Optional[str] = None
        last_answer: Optional[str] = None
        self._session_tags = None

        # Defer disk writes for JSON memory store; enable at session end
        try:
            from memory_store import JsonMemoryStore  # local import to avoid cycle
            if isinstance(self.memory_store, JsonMemoryStore):
                self.memory_store.set_defer_persist(True)
        except Exception:
            pass

        while True:
            user_input = input("You: ")

            # Allow user to end the conversation
            if user_input.strip().lower() in ["quit", "exit", "end"]:
                self._flush_session_memory()
                print("\n👋 Conversation with Onward Journey Agent ended.")
                break

            # Admin command: promote last turn to best-practice with tags
            if user_input.strip().startswith(":bp"):
                if not last_user or not last_answer:
                    print("No previous turn to tag yet.")
                    continue
                # syntax: :bp outcome tag1,tag2
                parts = user_input.strip().split(" ", 2)
                outcome = parts[1] if len(parts) > 1 else "good"
                tags = []
                if len(parts) > 2:
                    tags = [t.strip() for t in parts[2].split(",") if t.strip()]
                self._record_best_practice(
                    user_text=last_user,
                    assistant_text=last_answer,
                    outcome=outcome,
                    tags=tags,
                )
                if tags:
                    # Apply tags to latest turn memory entry immediately
                    self._session_tags = tags
                    if self.memory_store:
                        self.memory_store.update_last_tags(session_id=self.session_id, tags=tags)
                print(f"Saved last turn as best-practice (outcome={outcome}, tags={tags}).")
                continue

            if not user_input.strip():
                continue

            # Send the new user message and handle any tool calls it triggers
            llm_response = self._send_message_and_handle_tools(user_input)
            print(f"\n Onward Journey Agent: {llm_response}\n")
            last_user = user_input
            last_answer = llm_response
            # Record turn immediately for fast-answer reuse (disk write deferred)
            self._record_memory(user_text=user_input, assistant_text=llm_response, tags=None)

    def query_csv_rag(self, user_query: str) -> str:
        """
        Performs Retrieval Augmented Generation (RAG) on internal CSV data.
        Use this tool to answer user queries on available data.

        Args:
            user_query (str): The user's specific request (e.g., "Tell me about tax").

        Returns:
            str: A string containing the top K most relevant text chunks (context).
        """
        if self.embeddings is None:
            return "RAG system is not initialized. Cannot access data."

        # 1. Embed the user query
        query_embedding = self.embedding_model.encode(user_query)

        # 2. Perform Similarity Search (Retrieval)
        # Compute cosine similarity between the query and all chunk embeddings
        similarity_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]

        # Get the indices of the top K relevant chunks
        top_indices = similarity_scores.argsort()[-self.top_K:][::-1]

        # 3. Augment Context
        retrieved_chunks = [self.chunk_data[i] for i in top_indices]

        # 4. Return Context for LLM Generation
        context_string = "\n".join(retrieved_chunks)

        # The model receives this context and uses it to answer the user_query
        return f"Retrieved Context:\n{context_string}"

    def get_forced_response(self, user_query: str) -> str:
            """
            Processes a single user query by forcing the LLM to call the RAG tool
            immediately, bypassing the Clarification Logic. Used for quantitative testing.
            """

            # force system instruction to call tool immediately
            forced_system_instruction = (
                "You are the **Onward Journey Agent**. Your sole goal is to answer the user's query. "
                "**CRITICAL RULE: YOU MUST NOT ASK CLARIFYING QUESTIONS.** "
                "Analyze the user query. You must call the 'query_csv_rag' tool immediately with the "
                "best possible query argument to retrieve context. ONLY output the final, grounded "
                "answer after the tool call is complete."
            )

            initial_messages = [{"role": "user", "content": [{"type": "text", "text": user_query}]}]

            body1 = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": initial_messages,
                "system": forced_system_instruction,
                "max_tokens": 4096,
                "temperature": self.temperature,
                "tools": self.bedrock_tools,
            }

            # First LLM Call to get Tool Call
            bedrock_response1 = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body1))
            response_body1 = json.loads(bedrock_response1.get('body').read())

            model_content1 = response_body1.get('content', [])
            tool_calls = [c for c in model_content1 if c.get('type') == 'tool_use']

            if not tool_calls:
                return "ERROR: LLM failed to call tool despite explicit instruction."

            # Execute Tool and Get Final Answer
            function_call = tool_calls[0]
            tool_output = self.query_csv_rag(function_call['input']['user_query'])

            tool_result_part = {
                "type": "tool_result",
                "tool_use_id": function_call['id'],
                "content": [{"type": "text", "text": tool_output}]
            }

            history_with_tool_output = [
                {"role": "user", "content": [{"type": "text", "text": user_query}]},
                {"role": "assistant", "content": [function_call]},
                {"role": "user", "content": [tool_result_part]}
            ]

            body2 = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": history_with_tool_output,
                "system": forced_system_instruction,
                "max_tokens": 4096,
                "temperature": self.temperature,
            }
            # Second LLM Call to get Final Response
            bedrock_response2 = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body2))
            response_body2 = json.loads(bedrock_response2.get('body').read())
            # Extract final text content
            final_text_content = next((c.get('text') for c in response_body2.get('content', []) if c.get('type') == 'text'), None)
            return final_text_content if final_text_content else "Error generating final response after forced tool call."
