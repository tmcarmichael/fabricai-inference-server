# FabricAI Inference Server

[![Status](https://img.shields.io/badge/status-active_development-blue)](#)
[![Python](https://img.shields.io/badge/python-3.14%2B_free--threaded-blue)](https://docs.python.org/3/howto/free-threading-python.html)
[![License](https://img.shields.io/github/license/tmcarmichael/fabricai-inference-server)](https://github.com/tmcarmichael/fabricai-inference-server/blob/main/LICENSE)
![](https://img.shields.io/github/last-commit/tmcarmichael/fabricai-inference-server)

A local/cloud hybrid LLM routing gateway. Exposes an OpenAI-compatible API that analyzes incoming requests and routes them between local inference via [Ollama](https://ollama.com) and cloud providers (Anthropic Claude, OpenAI GPT, Google Gemini).

Any tool that speaks the OpenAI API, including SDKs, agents, and CLI tools, can point at `localhost:8000` and the gateway handles backend selection.

## How Routing Works

Requests pass through a layered routing stack:

1. **Explicit routing**: `model: "anthropic/claude-sonnet-4-20250514"` goes directly to that backend
2. **Session pinning**: conversations stay on the same backend via atomic Redis HSETNX
3. **Heuristic rules**: 9 rules analyze token count, code presence, instruction complexity, domain (math/legal/medical), keyword signals, conversation depth, and back-reference density. Each rule sets a confidence score.
4. **Cascade fallback**: low-confidence decisions run on local first, score output quality (length, repetition, coherence, uncertainty), and escalate to cloud if quality is below threshold

Rule confidence self-adjusts over time. Cascade outcomes feed back into per-rule thresholds via exponential moving average. Rules that produce good local results trend higher, and rules that frequently escalate trend lower.

## Quick Start

```bash
git clone https://github.com/tmcarmichael/fabricai-inference-server.git
cd fabricai-inference-server
uv sync --dev
ollama pull llama3.2
cp .env.example .env
./run.sh
```

Or with Docker: `docker compose up --build` (starts gateway + Redis + Ollama)

## Usage

```bash
# Routes automatically based on request analysis
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'

# Force a specific backend
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "anthropic/claude-sonnet-4-20250514", "messages": [...]}'

# Preview routing decision without running inference
curl http://localhost:8000/v1/route \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Summarize this"}]}'
```

Cloud backends activate when API keys are set in `.env`. Without them, all requests route to Ollama.

## API

| Endpoint               | Method | Description                                        |
| ---------------------- | ------ | -------------------------------------------------- |
| `/v1/chat/completions` | POST   | OpenAI-compatible chat (streaming + non-streaming) |
| `/v1/route`            | POST   | Dry-run routing decision                           |
| `/v1/models`           | GET    | List models across all backends                    |
| `/v1/status`           | GET    | Capacity and backend health                        |
| `/v1/usage`            | GET    | Cost tracking, routing stats, adaptive thresholds  |
| `/health`              | GET    | Backend + Redis connectivity (no auth)             |

Response headers include `X-FabricAI-Backend`, `X-FabricAI-Route-Layer`, and `X-FabricAI-Route-Rule`.

## Configuration

All via environment variables (`.env`):

| Variable               | Default                  | Description                  |
| ---------------------- | ------------------------ | ---------------------------- |
| `OLLAMA_BASE_URL`      | `http://localhost:11434` | Ollama endpoint              |
| `OLLAMA_DEFAULT_MODEL` | `llama3.2`               | Default local model          |
| `ANTHROPIC_API_KEY`    | none                     | Enables Claude backend       |
| `OPENAI_API_KEY`       | none                     | Enables GPT backend          |
| `GOOGLE_API_KEY`       | none                     | Enables Gemini backend       |
| `AUTH_ENABLED`         | `false`                  | API key auth                 |
| `FABRICAI_API_KEYS`    | none                     | Comma-separated allowed keys |
| `RATE_LIMIT_RPM`       | `60`                     | Requests per minute per key  |
| `LOG_FORMAT`           | `text`                   | `text` or `json`             |

Routing rules, thresholds, and keyword lists are configurable in [`config/routing.yaml`](config/routing.yaml).

## Architecture

- **Python 3.14 free-threaded**: GIL disabled (`PYTHON_GIL=0`) for true parallelism across I/O and CPU-bound prompt analysis
- **Backend protocol**: all providers implement a shared `Protocol` with streaming, health checks, and auto-registration
- **Single capacity semaphore**: unified backpressure with fast-fail before routing work
- **Atomic session pinning**: Redis `HSETNX` prevents race conditions on concurrent requests
- **Redis connection pool + RDB snapshots**: bounded connections, LRU eviction, no per-command fsync
- **Middleware stack**: request ID tracing, sliding-window rate limiting, Bearer token auth

See [docs/architecture.md](docs/architecture.md) for detailed design decisions and tradeoffs.

## Development

### Prerequisites

- [Python 3.14+](https://www.python.org/downloads/) (free-threaded build)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com)
- Redis

### Setup

```bash
uv sync --dev
cp .env.example .env
```

### Testing

```bash
uv run pytest tests/ -v        # 90 tests
uv run ruff check .            # lint
uv run ruff format .           # format
```

### Project Structure

```
fabricai_inference_server/
  app.py              # Application factory + lifespan
  server.py           # OpenAI-compatible route handlers
  settings.py         # Environment configuration
  schemas/            # OpenAI request/response models
  backends/           # Ollama, Anthropic, OpenAI, Google
  router/             # Heuristic rules, cascade, adaptive thresholds
  middleware/         # Auth, rate limiting, request ID
  telemetry/          # Cost tracking, event collection
  session/            # Redis session manager
  utils/              # Token estimation, prompt analysis
```

## Contributing

Contributions are welcome. Please open an issue to discuss before submitting a PR.

## License

This project is licensed under the [MIT License](LICENSE).
