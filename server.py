import json
import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Pre-initialize client at module level — no per-request overhead
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Cache system prompt once at startup, not per-request
SYSTEM_PROMPT = ""
SYSTEM_MESSAGE = {}

@app.on_event("startup")
async def startup():
    global SYSTEM_PROMPT, SYSTEM_MESSAGE
    try:
        with open("system_prompt.txt", "r") as f:
            SYSTEM_PROMPT = f.read().strip()
    except FileNotFoundError:
        SYSTEM_PROMPT = "You are a helpful assistant."
    # Pre-build the system message dict once
    SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}


@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.websocket("/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()

    current_task = None

    try:
        async for data in websocket.iter_json():
            interaction_type = data.get("interaction_type")

            if interaction_type == "response_required":
                # Cancel stale generation immediately
                if current_task and not current_task.done():
                    current_task.cancel()

                current_task = asyncio.create_task(
                    handle_response(websocket, data)
                )

            elif interaction_type == "ping_pong":
                await websocket.send_json({
                    "response_type": "ping_pong",
                    "timestamp": data.get("timestamp")
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error call {call_id}: {e}")
    finally:
        # Clean up on disconnect
        if current_task and not current_task.done():
            current_task.cancel()


async def handle_response(websocket: WebSocket, data: dict):
    response_id = data.get("response_id")
    transcript = data.get("transcript", [])

    # Build messages — pre-cached system message + trimmed transcript
    # 12 turns is plenty for a qualification call, fewer tokens = faster TTFT
    recent = transcript[-12:] if len(transcript) > 12 else transcript

    messages = [SYSTEM_MESSAGE]
    for turn in recent:
        messages.append({
            "role": "assistant" if turn["role"] == "agent" else "user",
            "content": turn["content"]
        })

    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True,
            temperature=0.2,
            max_tokens=120,
            # Stop sequences cut generation early at natural turn boundaries
            # Saves 50-150ms by not generating tokens you'll throw away
            stop=["\nUser:", "\nuser:", "\nMarcus:", "\n\n"],
        )

        async for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                try:
                    await websocket.send_json({
                        "response_type": "response",
                        "response_id": response_id,
                        "content": delta.content,
                        "content_complete": False,
                        "end_call": False,
                    })
                except Exception:
                    return  # websocket dead, bail

        # Signal done
        try:
            await websocket.send_json({
                "response_type": "response",
                "response_id": response_id,
                "content": "",
                "content_complete": True,
                "end_call": False,
            })
        except Exception:
            return

    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"Groq error: {e}")