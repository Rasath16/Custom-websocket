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

# Add CORS to allow connections from Retell servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
# Ensure GROQ_API_KEY is set in your Railway Variables
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Function to load system prompt safely
def get_system_prompt():
    system_prompt_file = "system_prompt.txt"
    if os.path.exists(system_prompt_file):
        with open(system_prompt_file, "r") as f:
            return f.read().strip()
    return "You are a helpful assistant." # Fallback

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Retell LLM Server is running"}

@app.websocket("/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    print(f"Accepted connection for call_id: {call_id}")

    try:
        # 1. Send Initial Greeting
        # The initial greeting is hardcoded here to ensure low latency on the first turn
        initial_greeting = "Hey there, Am I speaking with Marcus?"
        
        await websocket.send_json({
            "response_type": "response",
            "response_id": 0,
            "content": initial_greeting,
            "content_complete": True,
            "end_call": False
        })

        async for data in websocket.iter_json():
            interaction_type = data.get("interaction_type")
            
            if interaction_type == "ping_pong":
                timestamp = data.get("timestamp")
                await websocket.send_json({
                    "response_type": "ping_pong",
                    "timestamp": timestamp
                })
                continue
            
            if interaction_type == "update_only":
                continue
            
            if interaction_type == "response_required":
                response_id = data.get("response_id")
                transcript = data.get("transcript", [])
                
                # Load the latest system prompt
                current_system_prompt = get_system_prompt()
                
                # Construct messages for Groq
                messages = [{"role": "system", "content": current_system_prompt}]
                
                # --- CRITICAL FIX: Increased Context Window ---
                # We now keep the last 20 turns instead of 6. 
                # Llama 3.1 8B is fast enough to handle this, and it prevents the agent
                # from "forgetting" that the user already answered "Homeowner".
                recent_transcript = transcript[-20:] if len(transcript) > 20 else transcript
                
                # Map Retell transcript to Groq message format
                for turn in recent_transcript:
                    role = "assistant" if turn["role"] == "agent" else "user"
                    messages.append({"role": role, "content": turn["content"]})
                
                print(f"Generating response for ID {response_id}...")
                
                try:
                    # Call Groq API with streaming
                    completion = await groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant", 
                        messages=messages,
                        stream=True,
                        # --- OPTIMIZATION: Lower Temperature ---
                        # 0.4 keeps it natural but prevents it from drifting off-script
                        temperature=0.4, 
                        max_tokens=150,
                        stop=None 
                    )
                    
                    # Stream chunks back to Retell
                    full_response = ""
                    async for chunk in completion:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
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
                    print(f"Response complete: {full_response}")
                except Exception as e:
                    print(f"Error calling Groq API: {e}")
                    continue

    except WebSocketDisconnect:
        print(f"Client disconnected for call {call_id}")
    except Exception as e:
        print(f"Error processing call {call_id}: {e}")