# Onward Journey Agent System

This project implements a specialized Multi-Tool RAG Agent built with Amazon Bedrock (Claude 3.7 Sonnet). It is designed to handle conversation handoffs from a general chatbot, providing grounded answers using internal data, public GOV.UK records, or escalating to a live human agent via Genesys Cloud.

---

## Project Structure

The project is organized into the following Python files:

| File Name          | Description                                                                                                                                    |
| :----------------- | :-------------------------------------------------
| `oj_toggle_server.py` | **API Layer**: FastAPI server managing agent state, chat endpoints, and the hand-back logic from live chat. |
| `base.py` | **Core Logic**: Contains the `BaseAgent` and capability **Mixins** (`LiveChatMixin`, `HandOffMixin`, `ServiceTriageQMixin`) that handle orchestration and slot extraction. |
| `factory.py` | **Agent Factory**: Defines specialized agents (`OnwardJourneyAgent`, `GovUKAgent`, `hybridAgent`) by combining Mixins. |
| `genesys.py` | **Integrations**: Handles communication with Genesys Cloud SDK for Knowledge Base discovery and Live Chat configuration. |
| `data.py` | **Data Layer**: Implements Vector Stores for Local CSVs and Genesys Cloud content using Amazon Titan embeddings. |
| `engine.py` | **Prompt Engine**: Manages politeness of system instructions and `Cache Augmented Generation (CAG)` to reduce LLM costs. |
| `tools_registry.py` | **Tool Definitions**: Centralized JSON schemas for Bedrock tool-calling. |

| `main.py`          | CLI entry point: the primary script for running the agent in interactive mode or executing batch test runs                                               |

| `frontend/src/routes/+page.svelte`          | Svelte 5 reactive frontend managing real-time WebSocket connections to Genesys.                                              |

| `evaluation/`          | contains `test.py` and `benchmarking.py` for performance scoring and topic mapping                                              |

---

## Advanced Features

### 1. Multi-Stage Triage (Slot Filling)
The system uses the `ServiceTriageQMixin` to perform "Semantic Mapping." It doesn't just look for keywords; it uses the LLM as a data extractor to determine if a user has logically provided information.

### 2. The Handoff "Gate"
To protect live agent capacity, the `_handle_handoff_gate` function in `base.py` acts as a hard validator. Even if the LLM *wants* to connect the user, the backend will block the tool execution and force a follow-up question if mandatory fields are missing. 

### 3. Hybrid RAG
The system can simultaneously search:
* **Local CSVs**: For internal contact routing data.
* **Genesys KB**: For live-synced department policies.
* **OpenSearch**: For public GOV.UK documentation.
### 4. Cache Augmented Generation (CAG)
The `CAGQueryCache` reduces latency by performing a TF-IDF similarity check on incoming queries against a local cache of previously answered questions. It is not part of the main code, but was implemented as a technical spike to reduce latency. 

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Make sure you have the repository pre-requisites from [the root README](../README.MD) installed.
- **AWS Account** with configured **IAM credentials** (via CLI or environment variables).
- Model Access Granted for the desired Claude model (e.g., Claude 3.7 Sonnet) in your target AWS region.
- Environment: A .env file required in the app directory to store API credentials and service URLs.

### 2. .env Configuration
Ensure your .env file contains the following variables for OpenSearch and Genesys Cloud integration:
```bash
# OpenSearch (GOV.UK Knowledge Base)
OPENSEARCH_URL=your_opensearch_url
OPENSEARCH_USERNAME=your_username
OPENSEARCH_PASSWORD=your_password

# Genesys Knowledge Base ID
GENESYS_KB_ID=kb_id

# Genesys Cloud (Live Agent Handoff)
GENESYS_CLOUD_CLIENT_ID=your_id
GENESYS_CLOUD_CLIENT_SECRET=your_secret
GENESYS_DEPLOYMENT_ID_MOJ=your_deployment_id
GENESYS_DEPLOYMENT_ID_PENSIONS_FORMS_AND_RETURNS=your_deployment_id
GENESYS_DEPLOYMENT_ID_IMMIGRATION=your_deployment_id
GENESYS_REGION=euw2.pure.cloud

# Local Knowledge Base
KB_PATH=../app/resources/data/oj_rag_data.csv
```
### 3. Data Preparation

You will need a CSV file to simulate your internal data source for the RAG tool.

Mock data source files should be added to `app/resources/data/oj_rag_data.csv`. This will ensure that the file is added as an object to the datasets S3 bucket following a Terraform build (for future prototyping use).

`oj_rag_data.csv` has already been added to the `app/resources/data` folder.

It contains the columns expected by the df_to_text_chunks function in data.py: `uid`, `service_name`, `department`, `phone_number`, `topic`, `user_type`, `tags`, `url`, `last_update`, and `description`.

Example `mock` Structure:

```bash
uid,service_name,department,phone_number,topic,user_type,tags,url,last_update,description
1001,Childcare Tax Credit,HMRC,0300 123 4567,childcare,Individual,"tax, benefit",/childcare-tax,2024-01-15,"Information about claiming tax credits for childcare costs."
1002,Self Assessment Help,HMRC,0300 987 6543,self assessment,Individual,"tax, self employed",/self-assessment-guide,2024-02-01,"Guide to filing your annual Self Assessment tax return."
# Add more rows of relevant data...
```
The `LocalCSVVectorStore` in data.py expects the CSV at the path defined by the KB_PATH environment variable. 


### 4. Usage

#### A. Interactive Mode (Conversation Demo)

Use this to see the agent handle the initial handoff and subsequent chat turns.
(Run from ../onward-journey)

```shell
gds-cli aws once-onwardjourney-development-admin -- uv run main.py interactive --agent_type 0
```
Note: --agent_type 0 uses the Onward Journey Agent, 1 uses GovUK, and 2 is Hybrid.
#### B. Testing Mode (Performance Analysis)

Use this to run the agent against a suite of pre-defined queries and generate the performance report and confusion matrix plot.

```shell
gds-cli aws once-onwardjourney-development-admin -- uv run main.py test \
    --output_dir path/to/output \
    --test_data ./ \
uv run main.py test \
    --kb_path app/resources/data/oj_rag_data.csv \
    --test_data app/resources/data/test_queries_large_80.json \
```

#### C. Frontend Chat Interaction (Demo-ing)

You will need two terminal windows; one to run the backend and the other for the frontend.

In the first, navigate to the top level directory onward-journey and run: 
```shell
gds-cli aws once-onwardjourney-development-admin -- uv run uvicorn app.api.oj_toggle_server:app --reload
```
In the second, navigate to the "frontend" folder and run:
```shell
npm run dev
```
Once these have been run and are hosted, go to a browser and go to http://localhost:5173/ . There you can interact with the Onward Journey Agent as a user.

#### Key Components and AWS Integration

##### 1. Bedrock Integration (`base.py`)

- **Client Initialization**: The agent uses `boto3.client('bedrock-runtime', region_name=...)` for secure authentication and connection to the Bedrock service.

- **Tool Declaration**: Functions are declared using the JSON Schema format required by Anthropic's models on Bedrock.

- **Inference Pipeline**: The agent uses `client.invoke_model()` to send requests. The tool-use logic involves a multi-step loop where the agent sends the prompt, receives the tool call, executes the local Python `query_internal_kb` function, and sends the results back to Bedrock as a subsequent user message for final answer generation.

##### 2. RAG Implementation (`base.py, factory.py` and `data.py`)

The RAG tools are the core component that operates locally to:

- Encode the user query using `Amazon Titan Text Embeddings v2` (specifically `amazon.titan-embed-text-v2:0`) model.

- Performs Cosine Similarity against pre-computed embeddings (transformed oj_rag_data.csv data).

- Augment the LLM's prompt with the top k relevant text chunks for a given Agent.

##### 3. Mixin-Based Agents

Instead of one file, capabilities are injected into agents using Mixins. For example, `OnwardJourneyAgent`
inherits from `OJSearchMixin` to gain local RAG powers and `LiveChatMixin` for Genesys integration.    

##### 4. Tool-Use Loop (Bedrock)
The system follows a reactive loop:

1. User Prompt sent to Bedrock.

2. Bedrock requests Tool Call (e.g., `query_internal_kb`).
 
3. Local Execution: The Python code queries the CSV/OpenSearch.

4. Final Synthesis: Results are fed back to Bedrock to produce a natural language answer.

##### 5. Cache Augmented Generation (CAG)
The `CAGQueryCache` object found in `engine.py` uses TF-IDF vectorization to check if a similar question has been answered before, reducing LLM costs and latency for common queries. 

##### 6. The Handoff Gate
The BaseAgent includes a programmatic fail-safe (_handle_handoff_gate) that intercepts tool calls to ensure triage is 100% complete before a WebSocket connection is permitted.

##### 7. Svelte 5 Frontend
The frontend (+page.svelte) uses reactive logic to intercept SIGNAL: strings from the backend to trigger connection to Genesys Cloud.
