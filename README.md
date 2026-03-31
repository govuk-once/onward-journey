# Onward Journey Codebase

This repository contains the **Onward Journey Codebase** for local prototyping.
The software is object oriented, utilising the core BaseAgent class for all agents. It facilitates orchestration - e.g. message and tool handling. 

The Onward Journey Agent inherits from the BaseAgent along with capabilities for live chat connections, knowledge based searches and handoffs. It has a "LLM brain" for its decision making/routing on tool use. In particular, the **Amazon Bedrock (Claude 3.7 Sonnet)** model is used for the agent brain. The system is designed to provide grounded, data-backed answers and intelligently manage transitions between automated support and live human agents via **Genesys Cloud**.

---

## 🏗️ Project Architecture

The codebase is split into a robust Python backend and a reactive Svelte 5 frontend, integrated through a FastAPI orchestration layer.

### 1. Backend (Python/FastAPI)
The backend manages the "brain" of the operation, utilizing a Mixin-based architecture to swap agent capabilities dynamically.
* **Core Orchestration**: `base.py` and `factory.py` define the logic for specialized agents like the `OnwardJourneyAgent` and `GovUKAgent`.
* **Data & RAG**: `data.py` implements vector stores using **Amazon Titan Text Embeddings v2** to search local CSVs and Genesys Knowledge Bases.
* **The Handoff Gate**: A programmatic validator in `base.py` that ensures all mandatory user information is collected before allowing an escalation to a live agent.
* **Performance**: `engine.py` includes **Cache Augmented Generation (CAG)** to reduce latency and LLM costs via TF-IDF similarity checks.

### 2. Frontend (Svelte 5)
A modern, reactive interface located in the `/frontend` directory.
* **Real-time Interaction**: Manages WebSocket connections to Genesys Cloud.
* **Signal Handling**: Intercepts specific backend signals to trigger UI changes and live chat handoffs.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Hybrid RAG** | Simultaneously searches internal CSVs, Genesys KBs, and GOV.UK (OpenSearch). |
| **Multi-Stage Triage** | Uses the LLM as a semantic data extractor to fill required "slots" before escalation. |
| **Fail-Safe Handoff** | Prevents premature human agent connection if mandatory fields are missing. |
| **Evaluation Suite** | Includes tools for performance scoring, benchmarking, and topic mapping. |

---

## 🛠️ Getting Started

### Prerequisites
* AWS Account with access to **Claude 3.7 Sonnet** and **Titan Embeddings**.
* Python 3.x (with `uv` package manager) and Node.js.
* A `.env` file configured with OpenSearch and Genesys Cloud credentials (see app README.md for more details)

### Running the System
1.  **Backend**: Navigate to the root and run the FastAPI server:
    ```bash
    gds-cli aws [your-profile] -- uv run uvicorn app.api.oj_toggle_server:app --reload
    ```
2.  **Frontend**: Navigate to `/frontend` and start the development server:
    ```bash
    npm install
    npm run dev
    ```

---
The application will be available at http://localhost:5173/.  
## 🧪 Testing & Evaluation
The system supports multiple modes of execution via `main.py`:
* **Interactive Mode**: For real-time conversation demos in terminal.
* **Testing Mode**: For batch processing pre-defined queries to generate performance reports and confusion matrices.

---