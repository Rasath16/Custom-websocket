# Retell Custom LLM Server with Groq

This project sets up a FastAPI server to act as a Custom LLM backend for Retell AI, using Groq's Llama 3 models for ultra-low latency responses.

## Prerequisites

- Python 3.8+
- [Groq API Key](https://console.groq.com/)
- [ngrok](https://ngrok.com/) (for exposing local server to public internet)

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    -   Rename `.env.example` to `.env`.
    -   Edit `.env` and add your `GROQ_API_KEY`.

3.  **Run Server**:
    -   Double-click `start_server.bat` or run:
        ```bash
        uvicorn server:app --host 0.0.0.0 --port 8000 --reload
        ```

4.  **Expose to Internet**:
    -   In a new terminal window, run:
        ```bash
        ngrok http 8000
        ```
    -   Copy the **Forwarding URL** (e.g., `https://<id>.ngrok-free.app`).
    -   Note: Retell uses WebSockets, so replace `https://` with `wss://`.

## Connect to Retell

1.  Go to your [Retell Dashboard](https://console.retellai.com/).
2.  Create a new Agent or edit an existing one.
3.  Select **Custom LLM**.
4.  In the **Custom LLM URL** field, enter your WebSocket URL:
    ```
    wss://<your-ngrok-url>/llm-websocket
    ```
    (Retell will automatically append the `call_id` to the end of this URL).

## Customization

-   **System Prompt**: Edit `system_prompt.txt` to change the agent's behavior.
-   **Model**: The server uses `llama3-8b-8192` by default. You can change this in `server.py` to `llama-3.1-70b-versatile` if needed.
