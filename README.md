# FabricAI Inference Server

A hackable, modular, containerized inference server for deploying large language models in local or hybrid environments.

---

## Architecture

See docs page.

## Prerequisites

- **Python 3.10+** (if running locally)
- **Docker & Docker Compose with Docker Desktop** (recommended for containerized usage)
- **Poetry** (if installing locally)

---

## Getting Started

1. Clone the Repository

```bash
git clone https://github.com/tmcarmichael/fabricai-inference-server.git
cd fabricai-inference-server
```

2. Download the Model

Suggested: TheBloke/Llama-2-13B-Ensemble-v5-GGUF (https://huggingface.co/TheBloke/Llama-2-13B-Ensemble-v5-GGUF)

Check hardware compatibility (Huggingface supports a check for this), if needed use 4bit or 3bit quantization.

3. Configure Model Path

Create a `.env` file at the project root:

```bash
cp .env.example .env
```

Edit `.env` to set:

```env
LOCAL_MODEL_DIR=/absolute/path/to/your/large-model
LLM_MODEL=/models/llama-2-13b-ensemble-v5.Q4_K_M.gguf
```

4. Run with Docker

Build & start:

```bash
docker-compose up --build
```

This spins up:

- fabricai-inference-server (FastAPI, uvicorn)
- Redis (for session/conversation memory)

5. Test the server

SSE Endpoint:

```bash
curl -N -X POST http://localhost:8000/v1/inference_sse \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello from Docker!"}'
```

Status:

```bash
curl http://localhost:8000/v1/status
```

6. [Optional] Local Development Environment without Docker

Install Poetry:

```bash
pip install --upgrade poetry
```

Install Dependencies:

```bash
poetry install
```

Start the Server:

```bash
poetry run uvicorn fabricai_inference_server.server:app --host 0.0.0.0 --port 8000
```

7. [Optional] Event-based Streaming

Socket.IO Support: Connect via Socket.IO at ws://localhost:8000 and emit the "inference_prompt" event.
