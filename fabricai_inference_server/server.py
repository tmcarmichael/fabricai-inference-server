"""
API routes — OpenAI-compatible chat completions endpoint.

Thin route handlers that delegate to backends. The server owns
concurrency control, routing dispatch, cost tracking, and
response formatting.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from fabricai_inference_server.exceptions import (
    BackendError,
    BackendUnavailableError,
    QueueFullError,
)
from fabricai_inference_server.router import RoutingDecision
from fabricai_inference_server.schemas.openai import ChatCompletionRequest
from fabricai_inference_server.settings import settings
from fabricai_inference_server.telemetry.collector import RequestEvent
from fabricai_inference_server.telemetry.cost import estimate_cost

logger = logging.getLogger(__name__)

router = APIRouter()

# Recognized backend prefixes for explicit model routing
_BACKEND_PREFIXES = ("anthropic/", "openai/", "google/", "ollama/")


def _resolve_cloud_backend(request: Request):
    """Find the first available cloud backend for cascade escalation."""
    backends = request.app.state.backends
    for name in ("anthropic", "openai", "google"):
        if name in backends:
            return backends[name]
    return None


async def _resolve_routing(
    request: Request, chat_request: ChatCompletionRequest
) -> tuple[object, RoutingDecision]:
    """Determine which backend handles this request.

    Priority:
      1. Explicit prefix (anthropic/model, openai/model, etc.)
      2. Session pinning (if X-Session-ID + pinned model)
      3. Heuristic router (model="auto" or unprefixed)
    """
    backends = request.app.state.backends
    model = chat_request.model

    # 1. Explicit prefix routing
    for prefix in _BACKEND_PREFIXES:
        if model.startswith(prefix):
            backend_name = prefix.rstrip("/")
            if backend_name not in backends:
                raise HTTPException(
                    404,
                    detail=(
                        f"Backend '{backend_name}' not configured. "
                        f"Set the API key in .env to enable it."
                    ),
                )
            return backends[backend_name], RoutingDecision(
                backend_name=backend_name,
                model=model,
                layer="explicit",
                rule_matched="prefix",
                reason=f"Explicit prefix: {prefix}",
            )

    # 2. Session pinning
    session_id = request.headers.get("x-session-id")
    if session_id:
        session_mgr = request.app.state.session_manager
        pinned = await session_mgr.get_pinned_model(session_id)
        if pinned and pinned in backends:
            return backends[pinned], RoutingDecision(
                backend_name=pinned,
                model="auto",
                layer="session_pin",
                rule_matched="pinned_model",
                reason=f"Session {session_id[:8]}... pinned to {pinned}",
            )

    # 3. Heuristic router
    app_router = request.app.state.router
    decision = await app_router.route(chat_request, backends)

    if decision.backend_name not in backends:
        # Fallback if routed backend isn't available
        if backends:
            fallback = next(iter(backends))
            decision.backend_name = fallback
            decision.reason += f" (fallback: {fallback})"
        else:
            raise HTTPException(503, detail="No backends available")

    # Atomically pin session — first request wins the race
    if session_id:
        session_mgr = request.app.state.session_manager
        pinned_to = await session_mgr.pin_model_if_unset(
            session_id, decision.backend_name
        )
        # Another concurrent request may have pinned a different backend
        if pinned_to != decision.backend_name and pinned_to in backends:
            decision.backend_name = pinned_to
            decision.layer = "session_pin"
            decision.reason = (
                f"Concurrent pin resolved to {pinned_to}"
            )

    return backends[decision.backend_name], decision


def _routing_headers(decision: RoutingDecision) -> dict[str, str]:
    """Build response headers with routing metadata."""
    return {
        "X-FabricAI-Backend": decision.backend_name,
        "X-FabricAI-Route-Layer": decision.layer,
        "X-FabricAI-Route-Rule": decision.rule_matched or "",
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    chat_request: ChatCompletionRequest, request: Request
):
    capacity = request.app.state.capacity
    telemetry = request.app.state.telemetry

    # Fast-fail: reject immediately if at total capacity
    if capacity._value <= 0:
        raise QueueFullError()

    backend, decision = await _resolve_routing(request, chat_request)

    if chat_request.stream:

        async def event_stream():
            start = time.monotonic()
            try:
                async with capacity:
                    async for chunk in backend.chat_completion_stream(
                        chat_request
                    ):
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                    latency = (time.monotonic() - start) * 1000
                    telemetry.record(
                        RequestEvent(
                            request_id="stream",
                            timestamp=datetime.now(timezone.utc),
                            model=chat_request.model,
                            backend=backend.name,
                            input_tokens=0,
                            output_tokens=0,
                            latency_ms=latency,
                            estimated_cost_usd=0.0,
                            routing_layer=decision.layer,
                            routing_rule=decision.rule_matched,
                            routing_confidence=decision.confidence,
                        )
                    )
            except (BackendUnavailableError, BackendError) as e:
                error_payload = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                    }
                }
                yield f"data: {json.dumps(error_payload)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                **_routing_headers(decision),
            },
        )
    else:
        start = time.monotonic()
        try:
            async with capacity:
                # Check cascade eligibility
                cascade = request.app.state.cascade
                cloud_backend = _resolve_cloud_backend(request)
                cascade_result = None

                if (
                    cascade
                    and cloud_backend
                    and cascade.should_cascade(
                        decision.confidence,
                        has_cloud=True,
                        is_streaming=False,
                    )
                ):
                    response, cascade_result = (
                        await cascade.execute(
                            chat_request, backend, cloud_backend
                        )
                    )

                    # Feed outcome to adaptive thresholds
                    adaptive = request.app.state.adaptive
                    if adaptive and decision.rule_matched:
                        adaptive.record_outcome(
                            rule=decision.rule_matched,
                            success=not cascade_result.escalated,
                            quality_score=cascade_result.quality_score,
                        )

                    if cascade_result.escalated:
                        decision.layer = "cascade"
                        decision.reason += (
                            f" → escalated (quality="
                            f"{cascade_result.quality_score:.2f})"
                        )
                else:
                    response = await backend.chat_completion(
                        chat_request
                    )

                latency = (time.monotonic() - start) * 1000

                cost = 0.0
                input_t = output_t = 0
                if response.usage:
                    input_t = response.usage.prompt_tokens
                    output_t = response.usage.completion_tokens
                    cost = estimate_cost(
                        response.model, input_t, output_t
                    )

                telemetry.record(
                    RequestEvent(
                        request_id=response.id,
                        timestamp=datetime.now(timezone.utc),
                        model=response.model,
                        backend=(
                            "cascade"
                            if cascade_result
                            and cascade_result.escalated
                            else backend.name
                        ),
                        input_tokens=input_t,
                        output_tokens=output_t,
                        latency_ms=latency,
                        estimated_cost_usd=cost,
                        routing_layer=decision.layer,
                        routing_rule=decision.rule_matched,
                        routing_confidence=decision.confidence,
                        cascade_triggered=(
                            cascade_result.escalated
                            if cascade_result
                            else None
                        ),
                        cascade_quality_score=(
                            cascade_result.quality_score
                            if cascade_result
                            else None
                        ),
                    )
                )

                return response.model_dump()
        except BackendUnavailableError as e:
            raise HTTPException(502, detail=str(e)) from e
        except BackendError as e:
            raise HTTPException(500, detail=str(e)) from e


@router.post("/v1/route")
async def route_preview(
    chat_request: ChatCompletionRequest, request: Request
):
    """Dry-run routing — returns the routing decision without running inference."""
    _, decision = await _resolve_routing(request, chat_request)
    return decision.to_dict()


@router.get("/v1/models")
async def list_models(request: Request):
    """List available models across all backends."""
    all_models = []
    for backend_name, backend in request.app.state.backends.items():
        health_info = await backend.health()
        if health_info.healthy:
            for model_id in health_info.models:
                all_models.append(
                    {
                        "id": f"{backend_name}/{model_id}",
                        "object": "model",
                        "owned_by": backend_name,
                    }
                )
    return {"object": "list", "data": all_models}


@router.get("/v1/status")
async def status(request: Request):
    """Server health, capacity, and backend status."""
    capacity = request.app.state.capacity
    total = settings.max_concurrent_requests + settings.queue_max_size

    backend_health = {}
    for backend_name, backend in request.app.state.backends.items():
        health_info = await backend.health()
        backend_health[backend_name] = {
            "healthy": health_info.healthy,
            "models": health_info.models,
            "latency_ms": health_info.latency_ms,
            "error": health_info.error,
        }

    return {
        "active_requests": total - capacity._value,
        "total_capacity": total,
        "available_slots": capacity._value,
        "backends": backend_health,
    }


@router.get("/v1/usage")
async def usage(request: Request):
    """Cost tracking, usage summary, routing distribution, and adaptive state."""
    summary = request.app.state.telemetry.get_summary()
    adaptive = request.app.state.adaptive
    if adaptive:
        summary["adaptive_thresholds"] = adaptive.get_stats()
    return summary
