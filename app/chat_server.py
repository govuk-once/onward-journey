import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import OnwardJourneyAgent, default_handoff
from data   import vectorStore

app = FastAPI()

# Enable CORS for the Svelte dev server
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

# Initialize Agent with Strategy 4 (Internal KB + Live Chat)
agent = OnwardJourneyAgent(
    handoff_package=default_handoff(),
    vector_store_embeddings=vs.get_embeddings(),
    vector_store_chunks=vs.get_chunks(),
    strategy=4 
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # The agent processes the message and potentially calls 'connect_to_live_chat
        # If called, 'connect_to_live_chat' returns the JSON handoff string
        response_text = await agent._send_message_and_tools(request.message)
        
        # Return the text containing the signal to the frontend
        return {"response": response_text}
        
    except Exception as e:
        print(f"Chat Logic Error: {e}")
        # Return a 500 error if the Bedrock call or tool fails
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")