import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Core imports from your factory
from app.core.data import vectorStore, GenesysVectorStore
from app.integrations.genesys import GenesysServiceDiscovery

# environment setup
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from app.agents.base import BaseAgent
from app.agents.factory import OnwardJourneyAgent

app = FastAPI()

# global state
AGENT_CONFIG = {"oja_enabled": True}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# data initialization (onward journey knowledge base and live chat service discovery)
KB_PATH = os.getenv("KB_PATH", "./your_kb_file.csv")
vs = vectorStore(file_path=KB_PATH)
y = GenesysServiceDiscovery()
raw_gen_data = y.get_all_kb_content(os.getenv("GENESYS_KB_ID"))
vs_genesys = GenesysVectorStore(raw_gen_data)

# Initialize the specialized OnwardJourneyAgent
# This agent keeps its unique system_instructions intact (defined within object)
oja_internal = OnwardJourneyAgent(
    handoff_package={'final_conversation_history': []},
    vector_store_embeddings=vs.get_embeddings(),
    vector_store_chunks=vs.get_chunks(),
    genesys_embeddings=vs_genesys.get_embeddings(),
    genesys_chunks=vs_genesys.get_chunks(),
    model_name="anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region='eu-west-2'
)

# Initialize the top level agent 
base_agent = BaseAgent(
    model_name="anthropic.claude-3-7-sonnet-20250219-v1:0",
    aws_region='eu-west-2'
)

# Capability management endpoints

@app.post("/agent/toggle")
async def toggle_oja(enabled: bool):
    """Simple toggle for the routing logic."""
    AGENT_CONFIG["oja_enabled"] = enabled
    return {"status": "success", "oja_enabled": enabled}

@app.get("/agent/status")
async def get_status():
    return AGENT_CONFIG

# Chat Endpoints

class ChatRequest(BaseModel):
    message: str

class HandBackRequest(BaseModel):
    transcript: list[dict]

@app.get("/handoff/package")
async def get_handoff_package():
    return base_agent.handoff_package

@app.post("/handoff/process")
async def process_handoff_endpoint():
    response_text = await base_agent.process_handoff()
    return {"response": response_text or "Context processed."}

@app.post("/handoff/back")
async def hand_back_to_agent(request: HandBackRequest):
    try:
        # 1. Check if the transcript actually has content
        if not request.transcript or len(request.transcript) == 0:
            base_agent._add_to_history(role="user", text="[SYSTEM NOTIFICATION]: User returned from the live chat queue without starting a conversation.")
            ai_response = await base_agent._send_message_and_tools(
                "I'm back, but I didn't end up speaking with a live agent. Please acknowledge this and ask how you can help me further."
            )
            return {"status": "success", "summary": ai_response}

        # 2. Process actual transcript if it exists
        for entry in request.transcript:
            speaker = "Live Agent" if entry['role'] == 'assistant' else "User"
            base_agent._add_to_history(role=entry['role'], text=f"[{speaker}]: {entry['text']}")

        summary_prompt = (
            "The user has finished their live chat session. Provide a concise Markdown summary "
            "of what was discussed in the live chat only. If the transcript provided no new information, "
            "simply acknowledge the return and offer further GOV.UK help."
        )
        ai_response = await base_agent._send_message_and_tools(summary_prompt)
        return {"status": "success", "summary": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process handback: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Delegation Logic:
    If OJA is enabled, we pass the user intent to the specialized agent.
    If disabled, we use the standard BaseAgent.
    """
    if AGENT_CONFIG["oja_enabled"]:
            oja_internal.history = base_agent.history
            response_text = await oja_internal._send_message_and_tools(request.message)
            base_agent.history = oja_internal.history
            
            # Capture the current internal state to send to the frontend
            active_svc = getattr(oja_internal, 'active_service_id', 'Unknown')
            return {
                "response": response_text,
                "debug": {
                    "active_service": active_svc,
                    "triage_state": oja_internal.triage_state,
                    "missing_fields": oja_internal.SERVICE_SCHEMAS.get(active_svc, {})
                                    .get('triage_data', {}).get('missing', [])
                                    if active_svc in oja_internal.SERVICE_SCHEMAS else []
                }
            }
    else:
        # Standard GOV.UK flow for general queries
        response_text = await base_agent._send_message_and_tools(request.message)
        
    return {"response": response_text}

@app.post("/chat/reset")
async def reset_chat():
    """Resets history for both agents to ensure a clean slate."""
    base_agent.history = []
    oja_internal.history = []
    return {"status": "success"}

@app.on_event("startup")
async def startup_event():
    """ Runs automatically when Uvicorn starts the app """
    print(f"OJA Initialized: {AGENT_CONFIG['oja_enabled']}")