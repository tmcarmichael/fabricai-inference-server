"""
Application factory with lifespan management.

Creates the FastAPI app, wires up backends, session manager, router,
middleware, and logging. Cloud backends activate when API keys are set.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from fabricai_inference_server.backends.ollama import OllamaBackend
from fabricai_inference_server.exceptions import QueueFullError
from fabricai_inference_server.logging_config import setup_logging
from fabricai_inference_server.middleware.auth import AuthMiddleware
from fabricai_inference_server.middleware.rate_limit import RateLimitMiddleware
from fabricai_inference_server.middleware.request_id import RequestIDMiddleware
from fabricai_inference_server.router.adaptive import AdaptiveThresholds
from fabricai_inference_server.router.cascade import CascadeLayer
from fabricai_inference_server.router.heuristic import (
    HeuristicRouter,
    RoutingConfig,
)
from fabricai_inference_server.router.quality import QualityScorer
from fabricai_inference_server.session.manager import SessionManager
from fabricai_inference_server.settings import settings
from fabricai_inference_server.telemetry.collector import TelemetryCollector

logger = logging.getLogger(__name__)


async def _register_backend(backends, name, backend):
    """Start a backend and register it if healthy."""
    await backend.startup()
    health = await backend.health()
    if health.healthy:
        backends[name] = backend
        logger.info(
            "%s backend connected. Models: %s", name, health.models
        )
    else:
        logger.warning(
            "%s backend not reachable (%s). Skipping.",
            name,
            health.error,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    backends: dict = {}
    all_backends = []

    # Ollama (always attempted)
    ollama = OllamaBackend(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_default_model,
    )
    all_backends.append(("ollama", ollama))
    await _register_backend(backends, "ollama", ollama)

    # Cloud backends (opt-in via API keys)
    if settings.anthropic_api_key:
        from fabricai_inference_server.backends.anthropic import (
            AnthropicBackend,
        )

        ab = AnthropicBackend(api_key=settings.anthropic_api_key)
        all_backends.append(("anthropic", ab))
        await _register_backend(backends, "anthropic", ab)

    if settings.openai_api_key:
        from fabricai_inference_server.backends.openai_backend import (
            OpenAIBackend,
        )

        ob = OpenAIBackend(api_key=settings.openai_api_key)
        all_backends.append(("openai", ob))
        await _register_backend(backends, "openai", ob)

    if settings.google_api_key:
        from fabricai_inference_server.backends.google import GoogleBackend

        gb = GoogleBackend(api_key=settings.google_api_key)
        all_backends.append(("google", gb))
        await _register_backend(backends, "google", gb)

    if not backends:
        logger.error(
            "No backends available. "
            "Set OLLAMA_BASE_URL or cloud API keys."
        )

    # Session manager
    session_mgr = SessionManager(
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
    )
    await session_mgr.startup()

    # Telemetry
    telemetry = TelemetryCollector()

    # Router with adaptive thresholds
    routing_config = RoutingConfig.from_yaml(
        settings.routing_config_path
    )
    adaptive = AdaptiveThresholds()
    heuristic_router = HeuristicRouter(
        routing_config, backends, adaptive=adaptive
    )

    # Cascade layer
    cascade = None
    if routing_config.cascade_enabled:
        cascade = CascadeLayer(
            quality_scorer=QualityScorer(
                quality_threshold=routing_config.cascade_quality_threshold,
            ),
            confidence_threshold=routing_config.cascade_confidence_threshold,
        )

    # Wire state
    app.state.backends = backends
    app.state.session_manager = session_mgr
    app.state.telemetry = telemetry
    app.state.router = heuristic_router
    app.state.adaptive = adaptive
    app.state.cascade = cascade
    # Single capacity semaphore replaces the old queue+semaphore pair.
    # Total slots = active inference slots + queued waiting slots.
    total_capacity = (
        settings.max_concurrent_requests + settings.queue_max_size
    )
    app.state.capacity = asyncio.Semaphore(total_capacity)

    logger.info(
        "Server ready — backends: %s, auth: %s, rate_limit: %d rpm",
        list(backends.keys()),
        "enabled" if settings.auth_enabled else "disabled",
        settings.rate_limit_rpm,
    )

    yield

    # --- Shutdown ---
    for _, backend in all_backends:
        await backend.shutdown()
    await session_mgr.shutdown()
    logger.info("Server shut down cleanly.")


def create_app() -> FastAPI:
    # Configure logging before anything else
    setup_logging(fmt=settings.log_format, level=settings.log_level)

    app = FastAPI(
        title="FabricAI Inference Server",
        description="Intelligent local/cloud hybrid LLM routing gateway",
        version="0.3.0",
        lifespan=lifespan,
    )

    # --- Middleware (applied in reverse order) ---

    # Request ID (outermost — runs first)
    app.add_middleware(RequestIDMiddleware)

    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_rpm,
    )

    # Auth (innermost — runs after rate limiting)
    if settings.auth_enabled:
        api_keys = settings.api_key_set
        if not api_keys:
            logger.warning(
                "AUTH_ENABLED=true but no FABRICAI_API_KEYS set. "
                "All requests will be rejected."
            )
        app.add_middleware(AuthMiddleware, api_keys=api_keys)

    # --- Exception handlers ---

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request, exc: QueueFullError):
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": exc.message,
                    "type": "rate_limit_error",
                }
            },
        )

    # --- Routes ---

    from fabricai_inference_server.server import router

    app.include_router(router)

    # Health endpoint (always public, no auth required)
    @app.get("/health")
    async def health(request):
        backends = request.app.state.backends
        checks = {}
        all_healthy = True

        for name, backend in backends.items():
            health_info = await backend.health()
            checks[name] = {
                "healthy": health_info.healthy,
                "latency_ms": health_info.latency_ms,
            }
            if not health_info.healthy:
                checks[name]["error"] = health_info.error
                all_healthy = False

        # Redis check
        try:
            await request.app.state.session_manager.client.ping()
            checks["redis"] = {"healthy": True}
        except Exception as e:
            checks["redis"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        return JSONResponse(
            status_code=200 if all_healthy else 503,
            content={
                "status": "healthy" if all_healthy else "degraded",
                "checks": checks,
            },
        )

    return app


app = create_app()
