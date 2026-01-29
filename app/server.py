from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import OnwardJourneyAgent
from app.main import default_handoff
import numpy as np

app = FastAPI()

# Allow Svelte to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock/Load your Vector Store (Replace with your actual loader from main.py)
# For this example, we assume you have your embeddings and chunks ready
mock_embeddings = np.random.rand(10, 1024).astype('float32')
mock_chunks = ["Chunk 1", "Chunk 2"]

# Initialize the Agent
agent = OnwardJourneyAgent(
    handoff_package=default_handoff(),
    vector_store_embeddings=mock_embeddings,
    vector_store_chunks=mock_chunks,
    strategy=3 # Use both KBs
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # We use the internal _send_message_and_tools logic
        response = agent._send_message_and_tools(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)