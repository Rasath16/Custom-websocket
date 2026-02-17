import json
import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import AsyncGroq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# precise imports
from functools import lru_cache

# Initialize Groq client
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Cache system prompt to avoid disk I/O on every turn
@lru_cache(maxsize=1)
def get_system_prompt():
    system_prompt_file = "system_prompt.txt"
    if os.path.exists(system_prompt_file):
        with open(system_prompt_file, "r") as f:
            return f.read().strip()
    return "You are a helpful assistant."

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Retell LLM Server is running"}

@app.websocket("/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    print(f"Accepted connection for call_id: {call_id}")
    
    # Track the current generation task so we can cancel it if needed
    current_generation_task = None

    try:
        # Wait for user input (no initial greeting)
        async for data in websocket.iter_json():
            interaction_type = data.get("interaction_type")
            
            # --- CRITICAL FIX: CONCURRENCY HANDLING ---
            # If a new response is required, we MUST cancel any ongoing generation.
            # This prevents the "Stutter Effect" where the AI replies to old audio fragments.
            if interaction_type == "response_required":
                if current_generation_task and not current_generation_task.done():
                    print("Cancelling previous generation task...")
                    current_generation_task.cancel()
                
                # Create a new task for this response and track it
                current_generation_task = asyncio.create_task(
                    handle_response(websocket, data)
                )

            elif interaction_type == "ping_pong":
                await websocket.send_json({
                    "response_type": "ping_pong",
                    "timestamp": data.get("timestamp")
                })
                
    except WebSocketDisconnect:
        print(f"Client disconnected for call {call_id}")
    except Exception as e:
        print(f"Error processing call {call_id}: {e}")

async def handle_response(websocket: WebSocket, data: dict):
    """
    Handles the LLM generation in a separate task.
    This allows the main loop to listen for new interruptions constantly.
    """
    response_id = data.get("response_id")
    transcript = data.get("transcript", [])
    
    # Load system prompt
    system_prompt = get_system_prompt()
    
    # Construct messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Use larger context window (20 turns)
    recent_transcript = transcript[-20:] if len(transcript) > 20 else transcript
    
    for turn in recent_transcript:
        role = "assistant" if turn["role"] == "agent" else "user"
        messages.append({"role": role, "content": turn["content"]})
    
    print(f"Generating response for ID {response_id}...")

    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=messages,
            stream=True,
            temperature=0.2, 
            max_tokens=50,
        )
        
        async for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                await websocket.send_json({
                    "response_type": "response",
                    "response_id": response_id,
                    "content": content,
                    "content_complete": False,
                    "end_call": False
                })
        
        # Signal completion
        await websocket.send_json({
            "response_type": "response",
            "response_id": response_id,
            "content": "",
            "content_complete": True,
            "end_call": False
        })
        print(f"Response complete for ID {response_id}")

    except asyncio.CancelledError:
        print(f"Generation for ID {response_id} was cancelled by new input.")
        # Important: Do not try to send anything to websocket here, just exit.
        raise
    except Exception as e:
        print(f"Error calling Groq API: {e}")