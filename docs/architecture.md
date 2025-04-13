## Architecture

The FabricAI Inference Server is a modular, hackable platform for running large language models (LLMs) locally or in a hybrid environment. It leverages FastAPI for REST/SSE endpoints, Socket.IO for real-time events, and Redis for ephemeral memory or conversation states. This design aims to be lightweight, extensible, and easy to modify or swap components.

### 1. Data & Request Flow

![Data & Request Flow](/docs/fabricai-inference-server-design-1.0.png)

1. Client sends either an HTTP request (SSE) or a Socket.IO event (inference_prompt).

2. FastAPI or Socket.IO checks concurrency (Queue + Semaphore).

3. Engine uses local quantized LLM to generate tokens.

4. Tokens stream back as SSE or Socket.IO events.

### 2. High-Level Components

FastAPI Application

- Manages HTTP routes, SSE endpoints (/v1/inference_sse), and status checks (/v1/status).
- Exposes a minimal REST interface for server-sent events and concurrency management.

Socket.IO (python-socketio)

- Handles real-time bidirectional communication.
- Receives events and streams tokens back with events like inference_token or inference_complete.

Llama Engine (or other LLM engine)

- Responsible for loading a quantized local model.
- Streams tokens via a generate_stream() generator function.
- Swappable for other model backends if needed.

Redis Session Manager

- Maintains ephemeral conversation memory or sessions.
- Could be replaced with any store (SQL, file, or memory) with minimal code changes.

Concurrency Control

- A queue plus a semaphore limit how many inferences run in parallel (MAX_CONCURRENT_REQUESTS).
- Additional requests are queued until a slot is free, or return HTTP 429 if the queue is at capacity.

Settings / Env

- A settings.py uses pydantic.BaseSettings to load environment variables (via .env or Docker).
- Includes model paths, concurrency limits, and other tunable parameters.

### 3. Deployment & Configuration

#### Docker

- A lightweight Dockerfile building from Python 3.11-slim.
- Installs model dependencies (llama-cpp-python) and your code.

#### docker-compose

- Orchestrates `fabricai-inference-server` container and redis container.
- Reads .env for model paths (LLM_MODEL), concurrency settings, etc.

### 4. Extending / Hacking

- **Engine Swap:** Replace LlamaEngine with a different LLM or language model backend.

- **Memory Store:** Switch Redis for an in-process dictionary or a SQL database.

- **Socket.IO:** Add new events or rooms if building multi-user chat apps.

- **Concurrency:** Adjust or replace the queue logic for advanced scheduling or batch processing.

- **Auth / Security:** Insert JWT checks or custom middlewares in FastAPI if needed.
