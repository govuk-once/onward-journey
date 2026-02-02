import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import OnwardJourneyAgent
from handoff_examples import example_handoff_pension_schemes_nohelp
from data import vectorStore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Knowledge Base
KB_PATH = os.getenv("KB_PATH", "./your_kb_file.csv")
vs = vectorStore(file_path=KB_PATH)

# Initialize Agent with Strategy 4
agent = OnwardJourneyAgent(
    handoff_package=example_handoff_pension_schemes_nohelp(),
    vector_store_embeddings=vs.get_embeddings(),
    vector_store_chunks=vs.get_chunks(),
    strategy=4 
)

class ChatRequest(BaseModel):
    message: str

class HandBackRequest(BaseModel):
    transcript: list[dict]

@app.get("/handoff/package")
async def get_handoff_package():
    try:
        return agent.handoff_package
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/handoff/process")
async def process_handoff_endpoint():
    try:
        response_text = await agent.process_handoff()
        return {"response": response_text or "Context processed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/handoff/back")
async def hand_back_to_agent(request: HandBackRequest):
    try:
        # 1. Sync the transcript into the agent's memory
        for entry in request.transcript:
            speaker = "Live Agent" if entry['role'] == 'assistant' else "User"
            agent._add_to_history(role=entry['role'], text=f"[{speaker}]: {entry['text']}")
        
        # 2. Make an LLM call to summarize or check for remaining needs
        summary_prompt = (
            "I have just returned from a live chat session. "
            "Based on the transcript above, provide a very brief summary of what was resolved "
            "and ask the user if there is anything else related to GOV.UK services I can help with."
        )
        ai_response = await agent._send_message_and_tools(summary_prompt) 
        
        return {
            "status": "History updated",
            "summary": ai_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = await agent._send_message_and_tools(request.message)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))