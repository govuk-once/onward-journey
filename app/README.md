# Onward Journey Agent System

This project implements a specialized Multi-Tool RAG Agent built with Amazon Bedrock (Claude 3.7 Sonnet). It is designed to handle conversation handoffs from a general chatbot, providing grounded answers using internal data, public GOV.UK records, or escalating to a live human agent via Genesys Cloud.

---

## Project Structure

The project is organized into the following Python files:

| File Name          | Description                                                                                                                                    |
| :----------------- | :-------------------------------------------------
| `main.py`          | Sets up the environment, loads data, initializes the agent, and runs a sample conversation loop.                                               |
| `agents.py`        | Contains the `OnwardJourneyAgent` class, which handles LLM configuration, tool declaration, RAG implementation, and the interactive chat loop. |
| `data.py`          | Contains the `container` class and utility functions (`df_to_text_chunks`) for loading CSV data, chunking it, and generating embeddings.       |
| `preprocessing.py` | Contains utility functions for data preparation, specifically for transforming raw conversation logs into structured JSON formats.             |
| `test.py`          | *Evaluator*: Logic for running batch test cases and mapping results to topics for analysis. |
| `metrics.py`       | *Analytics*: Calculates the Clarification Success Gain (CSG) score |
| `helpers.py`       | *Utilities*: Standardizes UK phone numbers and maps labels for confusion matrices. |
| `plotting.py`      | *Visuals*: Generates Seaborn-based heatmaps for performance reporting.|

---

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

# Genesys Cloud (Live Agent Handoff)
GENESYS_CLOUD_CLIENT_ID=your_id
GENESYS_CLOUD_CLIENT_SECRET=your_secret
GENESYS_DEPLOYMENT_ID_MOJ=your_deployment_id
GENESYS_DEPLOYMENT_ID_HMRC=your_deployment_id
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



### 4. Usage

#### A. Interactive Mode (Conversation Demo)

Use this to see the agent handle the initial handoff and subsequent chat turns.
(Run from ../onward-journey/app)

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
    --kb_path ../mock_data/oj_rag_data.csv \
    --test_data ../test_data/prototype2/test_queries_large_80.json \
```

#### C. Frontend Chat Interaction (Demo-ing)

You will need two terminal windows; one to run the backend and the other for the frontend.

In the first, navigate to the "app" folder and run:
```shell
gds-cli aws once-onwardjourney-development-admin -- uv run uvicorn app.api.server:app --reload
```
In the second, navigate to the "frontend" folder and run:
```shell
npm run dev
```
Once these have been run and are hosted, go to a browser and go to http://localhost:6173/ . There you can interact with the Onward Journey Agent as a user.

#### Key Components and AWS Integration

##### 1. Bedrock Integration (`base.py`)

- **Client Initialization**: The agent uses `boto3.client('bedrock-runtime', region_name=...)` for secure authentication and connection to the Bedrock service. You can pass the ARN of an IAM role to assume for calls to Bedrock via the `--role_arn` command line argument

- **Tool Declaration**: Functions are declared using the JSON Schema format required by Anthropic's models on Bedrock.

- **Inference Pipeline**: The agent uses `client.invoke_model()` to send requests. The tool-use logic involves a multi-step loop where the agent sends the prompt, receives the tool call, executes the local Python `query_csv_rag` function, and sends the results back to Bedrock as a subsequent user message for final answer generation.

##### 2. RAG Implementation (`base.py, factory.py` and `data.py`)

The RAG tools are the core component that operates locally to:

- Encode the user query using `Amazon Titan Text Embeddings v2` model.

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