# Onward Journey Agent System

This project demonstrates a specialized RAG (Retrieval-Augmented Generation) agent built using Amazon Bedrock and the boto3 SDK. The agent is designed to take over a conversation (handoff) from a general chatbot to provide a focused, data-driven response using custom data and specialized tools (function calling).

The system uses local sentence transformers for generating embeddings and cosine similarity for data retrieval, enhancing the LLM's knowledge base before delegating the final generation to a Bedrock model (Claude Sonnet 3.7).

---

## Project Structure

The project is organized into the following Python files:

| File Name          | Description                                                                                                                                    |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`          | Sets up the environment, loads data, initializes the agent, and runs a sample conversation loop.                                               |
| `agents.py`        | Contains the `OnwardJourneyAgent` class, which handles LLM configuration, tool declaration, RAG implementation, and the interactive chat loop. |
| `data.py`          | Contains the `container` class and utility functions (`df_to_text_chunks`) for loading CSV data, chunking it, and generating embeddings.       |
| `preprocessing.py` | Contains utility functions for data preparation, specifically for transforming raw conversation logs into structured JSON formats.             |

---

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Make sure you have the repository pre-requisites from [the root README](../README.MD) installed.
- **AWS Account** with configured **IAM credentials** (via CLI or environment variables).
- Model Access Granted for the desired Claude model (e.g., Claude 3.5 Sonnet) in your target AWS region.

### 2. Data Preparation

You will need a mock CSV file to simulate your internal data source for the RAG tool.

Mock data source files should be added to `../mock_data`. This will ensure that the file is added as an object to the datasets S3 bucket following a Terraform build (for future prototyping use).

`mock_rag_data.csv` has already been added to the `mock_data` folder.

It contains the columns expected by the df_to_text_chunks function in data.py: `uid`, `service_name`, `department`, `phone_number`, `topic`, `user_type`, `tags`, `url`, `last_update`, and `description`.

Example `mock` Structure:

```bash
uid,service_name,department,phone_number,topic,user_type,tags,url,last_update,description
1001,Childcare Tax Credit,HMRC,0300 123 4567,childcare,Individual,"tax, benefit",/childcare-tax,2024-01-15,"Information about claiming tax credits for childcare costs."
1002,Self Assessment Help,HMRC,0300 987 6543,self assessment,Individual,"tax, self employed",/self-assessment-guide,2024-02-01,"Guide to filing your annual Self Assessment tax return."
# Add more rows of relevant data...
```

### 3. Run the Agent

You can view all configuration options and a description of each by passing the `--help` flag:

```shell
uv run main.py --help
```

#### A. Interactive Mode (Conversation Demo)

Use this to see the agent handle the initial handoff and subsequent chat turns.
(Run from ../onward-journey/app)

```shell
uv run main.py interactive
```

#### B. Testing Mode (Performance Analysis)

Use this to run the agent against a suite of pre-defined queries and generate the performance report and confusion matrix plot.

```shell
uv run main.py test \
    --kb_path ./data/processed/mock/mock.csv \
    --test_data ./data/test/prototype1/user_test_data/user_prompts.csv \
    --region eu-west-2
```

#### Key Components and AWS Integration

##### 1. Bedrock Integration (`agents.py`)

- **Client Initialization**: The agent uses `boto3.client('bedrock-runtime', region_name=...)` for secure authentication and connection to the Bedrock service. You can pass the ARN of an IAM role to assume for calls to Bedrock via the `--role_arn` command line argument

- **Tool Declaration**: Functions are declared using the JSON Schema format required by Anthropic's models on Bedrock.

- **Inference Pipeline**: The agent uses `client.invoke_model()` to send requests. The tool-use logic involves a multi-step loop where the agent sends the prompt, receives the tool call, executes the local Python `query_csv_rag` function, and sends the results back to Bedrock as a subsequent user message for final answer generation.

##### 2. RAG Implementation (`agents.py` and `data.py`)

The RAG tool (query_csv_rag) remains the core component that operates locally to:

- Encode the user query using `SentenceTransformer`.

- Perform Cosine Similarity retrieval against pre-computed embeddings.

- Augment the LLM's prompt with the top 3 relevant text chunks.

##### 3. Expected Output Flow

- **Agent Initialization**: Prints confirmation of successful boto3 client connection and tool declaration.

- **Handoff Processing**: The agent sends the initial conversation context to the Bedrock model.

- **First Response**: The Bedrock model calls the query_csv_rag tool, receives the RAG context, and generates a specialized, grounded response.

- **Interactive Loop**: The console enters an interactive chat where each user turn triggers a new invoke_model call, potentially engaging the RAG tool.

### Memory configuration

- Conversation memory is enabled by default via a JSON file-backed vector store. Control it with CLI flags:
  - `--memory_store json|in_memory|none` (`json` persists to disk by default)
  - `--memory_k <int>` (top-k memory snippets to retrieve per turn)
  - `--memory_max_items <int>` (per-session cap; evicts oldest)
  - `--session_id <id>` (scope memories to a session/user)
  - `--memory_path <path>` (JSON file location; default `app/output/memory.json`)
- Best-practice memories (helpful-only) can be stored separately and retrieved alongside session notes:
  - `--best_practice_store json|in_memory|none` (default `json`)
  - `--best_practice_path <path>` (default `app/output/best_practice.json`)
  - `--best_practice_k <int>` (top-k helpful snippets to include)
  - `--prompt_for_feedback` to ask “Was this helpful?” after each response (saves helpful replies to best-practice store)

### Fast answer reuse

- The agent can reuse a previous answer from the same session without an LLM call when the closest prior question is very similar.
- Control via `--fast_answer_threshold` (cosine similarity 0–1, default `0.95`; set to `0` to disable).
- Memories tagged with outcome `bad` are automatically excluded from reuse.

### Best-practice guardrails

- Helpful interactions are stored in the best-practice store and injected into the system prompt as concise rules.
- Use `--guardrail_tags billing,identity` to limit which tagged guardrails are included (optional).
- Outcome filtering: only entries with outcome `good` are included; outcome `bad` is ignored.

### Interactive admin command

- Inside interactive mode, you can promote the last turn to a best-practice rule with tags:
  - `:bp <outcome> <tag1,tag2>` (example: `:bp good billing,refund`)
  - This is handy to curate guardrails without relying on the feedback prompt.
