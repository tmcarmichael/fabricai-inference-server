## Architecture

FabricAI is a local/cloud hybrid LLM routing gateway. It exposes an OpenAI-compatible API endpoint and routes requests between local inference (Ollama) and cloud providers (Anthropic, OpenAI, Google) based on request characteristics.

### 1. Request Flow

```
Client (any OpenAI SDK)
    │
    │  POST /v1/chat/completions
    │  Authorization: Bearer <key>
    ▼
┌─────────────────────────────────────────────────────┐
│  Middleware Stack                                    │
│  Request ID → Rate Limiter → API Key Auth           │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Capacity Check (fast-fail if semaphore exhausted)  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Routing Dispatch                                   │
│                                                     │
│  1. Explicit prefix? (anthropic/model) → direct     │
│  2. Session pinned? (HSETNX atomic) → pinned backend│
│  3. Heuristic router (9 rules, <1ms)                │
│     ├── High confidence → execute directly          │
│     └── Low confidence → Cascade fallback           │
│           ├── Try local → score quality             │
│           ├── Quality OK → return local             │
│           └── Quality bad → re-run on cloud         │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Backend (implements Backend protocol)              │
│  Ollama │ Anthropic │ OpenAI │ Google               │
│                                                     │
│  → Streaming SSE or non-streaming JSON response     │
│  → Cost tracked per request                         │
│  → Routing decision in X-FabricAI-* headers         │
└─────────────────────────────────────────────────────┘
```

### 2. Key Architectural Decisions

#### Python 3.14 Free-Threaded (no GIL)

The server targets Python 3.14+ and runs with `PYTHON_GIL=0` to disable the Global Interpreter Lock. This enables true thread parallelism for a workload that mixes async I/O (proxying to backends, streaming SSE) with CPU-bound work (regex-based prompt analysis, quality scoring, adaptive threshold computation). Under the GIL, CPU-bound routing analysis blocks the event loop. Other requests stall while one request's prompt is being classified. With free-threading, prompt analysis runs in parallel with request streaming. The single-threaded penalty is ~5-10%, but concurrent throughput scales near-linearly with CPU cores. This matters most on the hardware this project targets: multi-core Macs with unified memory and multi-GPU PCs, where parallelism is available but the GIL would leave cores idle.

#### Ollama as Local Backend (not raw llama-cpp-python)

The server proxies to Ollama's OpenAI-compatible API via httpx rather than loading models directly with llama-cpp-python. This delegates model management, chat template handling, Metal/CUDA acceleration, and quantization to Ollama, a mature runtime that handles these concerns better than inline model loading. The tradeoff is an extra network hop (localhost), but it eliminates C extension complexity, simplifies the Docker build, and lets users manage models with `ollama pull` independently.

#### Single Capacity Semaphore (not queue + semaphore)

An earlier design used a separate queue and semaphore for concurrency control. This caused misleading 429 errors: requests could fill the queue while waiting for slow backends, making new requests think the server was overloaded when the real bottleneck was inference latency. A single `asyncio.Semaphore` with `total_capacity = max_concurrent + queue_size` provides unified, accurate backpressure.

#### Atomic Session Pinning via Redis HSETNX

Once a conversation is routed to a backend, switching mid-conversation degrades coherence. Session pinning uses Redis `HSETNX` (set-if-not-exists) so the first request to pin wins. Concurrent requests for the same session atomically resolve to the same backend without a read-then-write race.

#### Redis Connection Pool + RDB Snapshots

Session state and routing metadata are ephemeral. Losing them on a Redis restart is acceptable. A bounded `ConnectionPool(max_connections=20)` prevents connection exhaustion under load. RDB snapshots (`save 60 1000`) with LRU eviction replace AOF, providing 10-100x faster writes since there's no per-command fsync.

#### Heuristic Router with Deep Prompt Analysis

The routing stack has 9 rules across two tiers:

**Fast-path rules** (cheapest checks first): trivial query detection, long context threshold, code + action keyword matching, simple task keywords, code without action keywords.

**Deep analysis rules** (sub-ms, pre-compiled regex): instruction complexity scoring (multi-step markers, imperative verb density, multiple questions), weak domain detection (math, legal, medical, where local models underperform), and back-reference density (heavy context dependency favors cloud).

All keyword matching and prompt analysis uses pre-compiled regex alternation patterns cached via `lru_cache`. One regex engine pass replaces k sequential substring searches.

#### Adaptive Confidence Thresholds

Each heuristic rule sets a base confidence (0.50-0.95). An adaptive layer adjusts this based on observed cascade outcomes using an exponential moving average:
- Rules where local consistently passes quality → confidence trends up → fewer cascades
- Rules where local frequently fails quality → confidence trends down → more cascades

This creates a system that self-tunes from production traffic. No ML model, no training pipeline, no cold start, just statistics. The EMA adapts to shifting usage patterns: if users start asking harder "summarize" tasks, the success rate for that rule drops and the system automatically routes more of them through the cascade safety net.

#### Cascade Fallback

For low-confidence decisions (confidence < 0.7, non-streaming, cloud backend available), the cascade runs the request on local first, scores output quality (length, repetition, degeneration, completeness, uncertainty), and escalates to cloud if quality is below threshold. Each cascade outcome feeds back to the adaptive thresholds, closing the feedback loop.

#### Heuristics + Cascade, not ML Classifier

An earlier design planned a trained ML classifier (sentence-transformers + sklearn) between the heuristic rules and the cascade. This approach was rejected because the cascade already evaluates quality in real-time with the actual output. A classifier would predict what the cascade decides, adding complexity for marginal latency savings on ~9% of requests. Local inference is free (user's hardware), so the "wasted" local attempt in cascade is only a latency cost, not a monetary one. The adaptive thresholds achieve the same outcome (routing improvement over time) without the cold start, training pipeline, model weights, or maintenance burden of an ML classifier.

#### Fast-Fail Capacity Check

The server checks the capacity semaphore before performing any routing work (session lookups, heuristic analysis, backend resolution). Under load, this avoids wasting CPU cycles on requests that will be rejected anyway. A request at capacity gets a 429 in microseconds instead of after milliseconds of routing computation.

#### Health Check Timeouts (3s)

All backend health checks are wrapped in `asyncio.wait_for(timeout=3.0)`. A flaky cloud API that hangs during startup cannot block the server. It's marked unhealthy and skipped. The server starts with whatever backends are reachable, and the remaining backends can be retried later.

### 3. Components

#### Backends (`backends/`)

All backends implement the `Backend` protocol:
```python
class Backend(Protocol):
    async def chat_completion(request) -> ChatCompletionResponse
    async def chat_completion_stream(request) -> AsyncIterator[ChatCompletionChunk]
    async def health() -> BackendHealth
    async def startup() -> None
    async def shutdown() -> None
```

| Backend | Transport | Notes |
|---------|-----------|-------|
| Ollama | httpx → localhost:11434 | Connection pool (50 max, 20 keepalive) |
| Anthropic | `anthropic` SDK | System message extracted to top-level param |
| OpenAI | `openai` SDK | Near-passthrough, response normalization |
| Google | `google-genai` SDK | Role translation (assistant→model), parts format |

Backends auto-activate when their API key is set in `.env`. Health checks have a 3-second timeout, so a hanging cloud API cannot block server startup.

#### Router (`router/`)

- `heuristic.py`: 9-rule engine with configurable thresholds, keyword matching, instruction complexity scoring, domain detection, and reference density analysis. All rules pass through adaptive confidence adjustment.
- `adaptive.py`: Self-tuning confidence thresholds using exponential moving average of cascade success rates per rule. Rules that consistently produce good local results trend higher; rules that often escalate trend lower.
- `quality.py`: Output quality scorer (5 heuristic checks: length, repetition, degeneration, completeness, uncertainty)
- `cascade.py`: Try-local-then-escalate layer. Feeds outcomes back to adaptive thresholds.

#### Middleware (`middleware/`)

Applied in order: Request ID → Rate Limit → Auth.

- `request_id.py`: Generates/propagates `X-Request-ID` for log tracing
- `rate_limit.py`: Sliding window deque per API key/IP, O(1) amortized
- `auth.py`: Bearer token validation, `/health` always public

#### Telemetry (`telemetry/`)

- `collector.py`: Bounded deque of request events with routing rule distribution and cascade stats
- `cost.py`: Per-model USD cost estimation for Anthropic, OpenAI, and Google models

### 4. API Surface

```
POST /v1/chat/completions    # OpenAI-compatible (streaming + non-streaming)
POST /v1/route               # Dry-run routing decision (no inference)
GET  /v1/models              # List models across all backends
GET  /v1/status              # Capacity + backend health
GET  /v1/usage               # Cost, tokens, routing distribution, cascade stats
GET  /health                 # Backend + Redis health (public, no auth)
```

### 5. Deployment

```yaml
# docker-compose.yaml, 3 services
services:
  fabricai-inference-server:  # FastAPI app, Python 3.14 free-threaded
  redis:                      # RDB snapshots, LRU eviction, 256MB cap
  ollama:                     # Local model runtime
```

### 6. Routing Rules

9 heuristic rules applied in priority order (first match wins):

| Rule | Signal | Routes to | Base Confidence |
|------|--------|-----------|-----------------|
| Trivial query | < 50 tokens, no code | Local | 0.95 |
| Long context | > 4000 tokens | Cloud | 0.90 |
| Code + action keyword | Code block + refactor/debug/review/... | Cloud | 0.90 |
| Simple task keyword | summarize/extract/translate/... | Local | 0.85 |
| Code without action | Code present, no action keyword | Local | 0.65 |
| High complexity | Multi-step instructions, multiple tasks | Cloud | 0.85 |
| Weak domain | Math, legal, or medical content | Cloud | 0.85 |
| Heavy back-references | References to prior conversation context | Cloud | 0.80 |
| Deep conversation | > 10 message turns | Cloud | 0.85 |

Base confidence is adjusted at runtime by adaptive thresholds based on observed cascade outcomes.

### 7. Configuration

All via environment variables (`.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_DEFAULT_MODEL` | `llama3.2` | Default local model |
| `ANTHROPIC_API_KEY` | none | Enables Claude backend |
| `OPENAI_API_KEY` | none | Enables GPT backend |
| `GOOGLE_API_KEY` | none | Enables Gemini backend |
| `AUTH_ENABLED` | `false` | API key auth |
| `FABRICAI_API_KEYS` | none | Comma-separated allowed keys |
| `RATE_LIMIT_RPM` | `60` | Requests per minute per key |
| `LOG_FORMAT` | `text` | `text` or `json` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ROUTING_CONFIG_PATH` | `config/routing.yaml` | Routing rules config |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `QUEUE_MAX_SIZE` | `10` | Queued request slots |
| `MAX_CONCURRENT_REQUESTS` | `4` | Active inference slots |

Routing thresholds and keyword lists are in [`config/routing.yaml`](../config/routing.yaml).
