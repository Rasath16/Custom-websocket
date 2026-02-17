import json
import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import AsyncGroq
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()


# Initialize Groq client
# Ensure GROQ_API_KEY is set in .env
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Load the system prompt from file
SYSTEM_PROMPT_FILE = "system_prompt.txt"
if os.path.exists(SYSTEM_PROMPT_FILE):
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        SYSTEM_PROMPT = f.read().strip()
else:
    SYSTEM_PROMPT = "You are a helpful assistant." # Fallback

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Retell LLM Server is running"}

@app.websocket("/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    print(f"Accepted connection for call_id: {call_id}")

    try:
        # 1. Send Initial Greeting
        # The prompt specifies the opening line.
        initial_greeting = "Hey there, Am I speaking with Marcus?"
        
        # Send the initial response to Retell
        # Retell Custom LLM expects the first message to start the conversation
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
                # Handle updates - usually just logging or keeping track of state
                continue
            
            if interaction_type == "response_required":
                response_id = data.get("response_id")
                transcript = data.get("transcript", [])
                
                # Construct messages for Groq
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                
                # Optimization: Truncate history to keep context small (last 6 turns)
                # This significantly speeds up processing
                recent_transcript = transcript[-6:] if len(transcript) > 6 else transcript
                
                # Map Retell transcript to Groq message format
                for turn in recent_transcript:
                    role = "assistant" if turn["role"] == "agent" else "user"
                    messages.append({"role": role, "content": turn["content"]})
                
                print(f"Generating response for ID {response_id}...")
                # print(f"Messages sent to Groq: {json.dumps(messages, indent=2)}") # Comment out explicitly to save I/O time
                
                try:
                    # Call Groq API with streaming
                    # utilizing Llama 3.1 8B Instant for speed
                    completion = await groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant", 
                        messages=messages,
                        stream=True,
                        temperature=0.6, # Slightly lower temperature for faster deterministic tokens
                        max_tokens=150, # Reduced max tokens
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
                    # Isolate error so we don't kill the connection, though Retell might timeout
                    continue

    except WebSocketDisconnect:
        print(f"Client disconnected for call {call_id}")
    except Exception as e:
        print(f"Error processing call {call_id}: {e}")
        # maintain connection or close depending on severity
