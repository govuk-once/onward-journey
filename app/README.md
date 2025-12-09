# Onward Journey Agent System

This project demonstrates a specialized RAG (Retrieval-Augmented Generation) agent built using the **Gemini API** that takes over a conversation (**handoff**) from a general chatbot to provide a more focused and data-driven response using custom data and specialized tools.

The system utilizes **Sentence Transformers** for generating embeddings and **Cosine Similarity** for data retrieval, enhancing the LLM's knowledge base.

---

## Project Structure

The project is organized into the following Python files:

| File Name | Description |
| :--- | :--- |
| `main.py` | Sets up the environment, loads data, initializes the agent, and runs a sample conversation loop. |
| `agents.py` | Contains the `OnwardJourneyAgent` class, which handles LLM configuration, tool declaration, RAG implementation, and the interactive chat loop. |
| `data.py` | Contains the `container` class and utility functions (`df_to_text_chunks`) for loading CSV data, chunking it, and generating embeddings. |
| `preprocessing.py` | Contains utility functions for data preparation, specifically for transforming raw conversation logs into structured JSON formats. |

---

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

Make sure you have the repository pre-requisites from (the root README)[../README.MD] installed.

You will need a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/api-keys) to run this code

### 2. Data Preparation

You will need a mock CSV file to simulate your internal data source for the RAG tool.

Create a file named mock_rag_data.csv (or similar) in your project directory.

Ensure it contains the columns expected by the df_to_text_chunks function in data.py: `uid`, `service_name`, `department`, `phone_number`, `topic`, `user_type`, `tags`, `url`, `last_update`, and `description`.

Example `mock_rag_data.csv` Structure:

```bash
uid,service_name,department,phone_number,topic,user_type,tags,url,last_update,description
1001,Childcare Tax Credit,HMRC,0300 123 4567,childcare,Individual,"tax, benefit",/childcare-tax,2024-01-15,"Information about claiming tax credits for childcare costs."
1002,Self Assessment Help,HMRC,0300 987 6543,self assessment,Individual,"tax, self employed",/self-assessment-guide,2024-02-01,"Guide to filing your annual Self Assessment tax return."
# Add more rows of relevant data...
```

### 3. Configure `main.py`
Open main.py and update the placeholder paths and API key in the if __name__ == "__main__": block:

```bash
if __name__ == "__main__":
    main(
        RUN_MODE='test',  # Change to 'interactive' for interactive mode
        PATH_TO_KB='knowledge_base.csv',
        TEST_DATA_PATH='test_queries.csv',
        API_KEY='your_api_key_here')
```

### 5. How to Run the Agent
Execute the `main.py` script from your terminal:
```bash
uv run main.py
```
**Expected Output Flow**

1. **Agent Initialization**: You will see a confirmation that the agent has been initialized with its specialized instruction and RAG tool.

2. **Handoff Processing**: The agent will immediately process the simulated handoff_package data, which contains the user's final query (e.g., "Can you help me with childcare options?").

3. **First Response**: The Onward Journey Agent will use its internal RAG tool (query_csv_rag) to retrieve context related to the user's query from the loaded CSV data and generate a specialized response.

4. **Interactive Loop**: The console will enter an interactive loop, allowing you to ask follow-up questions to the specialized agent.
```
... (Initialization messages) ...
User: Can you help me with childcare options?

----------------------------------------------------------------------------------------------------
🤖 You are now speaking with the Onward Journey Agent.
🤖 Onward Journey Agent: Of course! I see you're looking for information on childcare. Based on our specialized resources, I can certainly help you with details on claiming tax credits for childcare costs. What specific details are you interested in?
----------------------------------------------------------------------------------------------------

You:
```

To exit the conversation, type `quit`, `exit`, or `end`.

### Key Components in `agents.py`

#### RAG Implementation (`query_csv_rag`)
The RAG tool connects the LLM to your custom data. It handles the following steps:

1. **Embedding**: It encodes the user_query using the SentenceTransformer('all-MiniLM-L6-v2').

2. **Retrieval**: It calculates the cosine similarity between the query embedding and all pre-computed data embeddings (self.embeddings).

3. **Augmentation**: It retrieves the top 3 most relevant chunks of text (K=3) from your CSV data.

4. **Generation**: It sends the retrieved context back to the Gemini model for generating the final, informed response.

#### Conversation Handoff (`process_handoff`)
The agent initiates the conversation by sending a complex initial prompt containing the previous final_conversation_history and the user's critical question (next_agent_prompt). This ensures the agent retains full context and immediately focuses on solving the user's need.
