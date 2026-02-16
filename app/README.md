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
GENESYS_DEPLOYMENT_ID=your_deployment_id
GENESYS_REGION=euw2.pure.cloud

# Local Knowledge Base
KB_PATH=../mock_data/mock_rag_data.csv
```
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



### 3. Usage

#### A. Interactive Mode (Conversation Demo)

Use this to see the agent handle the initial handoff and subsequent chat turns.
(Run from ../onward-journey/app)

```shell
gds-cli aws once-onwardjourney-development-admin -- uv run main.py interactive
```

#### B. Testing Mode (Performance Analysis)

Use this to run the agent against a suite of pre-defined queries and generate the performance report and confusion matrix plot.

```shell
gds-cli aws once-onwardjourney-development-admin -- uv run main.py test \
    --output_dir path/to/output \
    --test_data ./ \
uv run main.py test \
    --kb_path ../mock_data/mock_rag_data.csv \
    --test_data ../test_data/prototype2/test_queries_large_80.json \
```

#### C. Frontend Chat Interaction (Demo-ing)

You will need two terminal windows; one to run the backend and the other for the frontend.

In the first, navigate to the "app" folder and run:
```shell
gds-cli aws once-onwardjourney-development-admin -- uv run uvicorn chat_server:app --reload
```
In the second, navigate to the "frontend" folder and run:
```shell
npm run dev
```
Once these have been run and are hosted, go to a browser and go to http://localhost:6173/ . There you can interact with the Onward Journey Agent as a user.

#### Key Components and AWS Integration

##### 1. Bedrock Integration (`agents.py`)

- **Client Initialization**: The agent uses `boto3.client('bedrock-runtime', region_name=...)` for secure authentication and connection to the Bedrock service. You can pass the ARN of an IAM role to assume for calls to Bedrock via the `--role_arn` command line argument

- **Tool Declaration**: Functions are declared using the JSON Schema format required by Anthropic's models on Bedrock.

- **Inference Pipeline**: The agent uses `client.invoke_model()` to send requests. The tool-use logic involves a multi-step loop where the agent sends the prompt, receives the tool call, executes the local Python `query_csv_rag` function, and sends the results back to Bedrock as a subsequent user message for final answer generation.

##### 2. RAG Implementation (`agents.py` and `data.py`)

The RAG tool (query_internal_kb) remains the core component that operates locally to:

- Encode the user query using `Amazon Titan Text Embeddings v2` model.

- Performs Cosine Similarity against pre-computed embeddings.

- Augment the LLM's prompt with the top 3 relevant text chunks.

##### 3. Expected Output Flow

- **Agent Initialization**: Prints confirmation of successful boto3 client connection and tool declaration.

- **Handoff Processing**: The agent sends the initial conversation context to the Bedrock model.

- **First Response**: The Bedrock model calls its tools (depending on strategy - e.g. govuk kb, oj kb or both), receives the RAG context, and generates a specialized, grounded response.

- **Interactive Loop**: The console enters an interactive chat where each user turn triggers a new invoke_model call, potentially engaging the RAG tool.
