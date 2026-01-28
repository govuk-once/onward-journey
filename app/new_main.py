import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import OnwardJourneyAgent, default_handoff
from data import vectorStore

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Specify the origins allowed to make requests to this backend
origins = [
    "http://localhost:5173",  # Svelte dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your Agent & Vector Store once on startup
# store KB_PATH in .env file e.g. ../mock_data/mock_rag_data.csv 
KB_PATH = os.getenv("KB_PATH", "./your_kb_file.csv")
vs = vectorStore(file_path=KB_PATH)

agent = OnwardJourneyAgent(
    handoff_package=default_handoff(),
    vector_store_embeddings=vs.get_embeddings(),
    vector_store_chunks=vs.get_chunks(),
    strategy=4 
)

# 4. Define the Data Model for incoming messages
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Await the coroutine to get the string response
        response_text = await agent._send_message_and_tools(request.message)
        print(f"Response: {response_text}")
        # Ensure we are returning a serializable dictionary
        return {"response": response_text}
    except Exception as e:
        print(f"Handoff Logic Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))