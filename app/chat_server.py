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

# Initialize Agent
agent = OnwardJourneyAgent(
    handoff_package=example_handoff_pension_schemes_nohelp(),
    vector_store_embeddings=vs.get_embeddings(),
    vector_store_chunks=vs.get_chunks())

# Overriding system instruction for better visual structure
agent.system_instruction += (
    "\n\nCRITICAL: Use Markdown formatting. Use ### for headers, **bold** for phone numbers, "
    "and bullet points for lists of contact details to ensure readability."
)

# Overriding system instruction for better visual structure
agent.system_instruction += (
    "\n\nCRITICAL: Use Markdown formatting. Use ### for headers, **bold** for phone numbers, "
    "and bullet points for lists of contact details to ensure readability."
)

class ChatRequest(BaseModel):
    message: str

class HandBackRequest(BaseModel):
    transcript: list[dict]

@app.get("/handoff/package")
async def get_handoff_package():
    return agent.handoff_package

@app.post("/handoff/process")
async def process_handoff_endpoint():
    response_text = await agent.process_handoff()
    return {"response": response_text or "Context processed."}

@app.post("/handoff/back")
async def hand_back_to_agent(request: HandBackRequest):
    try:
        for entry in request.transcript:
            speaker = "Live Agent" if entry['role'] == 'assistant' else "User"
            agent._add_to_history(role=entry['role'], text=f"[{speaker}]: {entry['text']}")

        # Explicitly asking for a structured summary in Markdown
        summary_prompt = (
            "I am back. Please provide a structured Markdown summary of the live chat "
            "using bullet points, and ask if I can help with GOV.UK services."
        )
        ai_response = await agent._send_message_and_tools(summary_prompt)
        return {"status": "success", "summary": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process handback")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response_text = await agent._send_message_and_tools(request.message)
    return {"response": response_text}

@app.post("/chat/reset")
async def reset_chat():
    agent.history = []
    agent.handoff_package = {'final_conversation_history': []}
    return {"status": "success"}
