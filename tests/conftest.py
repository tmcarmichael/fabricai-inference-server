"""
Shared test fixtures.

Creates a test app with a mock backend and heuristic router
so tests don't need a running Ollama instance or Redis server.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from fabricai_inference_server.backends import BackendCapabilities, BackendHealth
from fabricai_inference_server.exceptions import QueueFullError
from fabricai_inference_server.router.adaptive import AdaptiveThresholds
from fabricai_inference_server.router.cascade import CascadeLayer
from fabricai_inference_server.router.heuristic import (
    HeuristicRouter,
    RoutingConfig,
)
from fabricai_inference_server.router.quality import QualityScorer
from fabricai_inference_server.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    Usage,
)
from fabricai_inference_server.server import router
from fabricai_inference_server.settings import settings
from fabricai_inference_server.telemetry.collector import TelemetryCollector


class MockBackend:
    """In-memory backend that returns canned responses."""

    _default_model = "mock-model"

    @property
    def name(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-test123",
            model="mock-model",
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant", content="Hello from mock!"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ):
        chunks = [
            ChatCompletionChunk(
                id="chatcmpl-test123",
                model="mock-model",
                choices=[
                    ChunkChoice(delta=ChoiceDelta(role="assistant"))
                ],
            ),
            ChatCompletionChunk(
                id="chatcmpl-test123",
                model="mock-model",
                choices=[
                    ChunkChoice(delta=ChoiceDelta(content="Hello"))
                ],
            ),
            ChatCompletionChunk(
                id="chatcmpl-test123",
                model="mock-model",
                choices=[
                    ChunkChoice(delta=ChoiceDelta(content=" from mock!"))
                ],
            ),
            ChatCompletionChunk(
                id="chatcmpl-test123",
                model="mock-model",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(), finish_reason="stop"
                    )
                ],
            ),
        ]
        for chunk in chunks:
            yield chunk

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, models=["mock-model"])

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def mock_session_manager():
    mgr = MagicMock()
    mgr.get_pinned_model = AsyncMock(return_value=None)
    mgr.pin_model_if_unset = AsyncMock(side_effect=lambda sid, m: m)
    return mgr


@pytest.fixture
def app(mock_backend, mock_session_manager):
    """Create a test app with mock backend and router."""
    test_app = FastAPI()

    @test_app.exception_handler(QueueFullError)
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

    test_app.include_router(router)
    backends = {"ollama": mock_backend}
    test_app.state.backends = backends
    test_app.state.telemetry = TelemetryCollector()
    test_app.state.session_manager = mock_session_manager
    config = RoutingConfig()
    adaptive = AdaptiveThresholds()
    test_app.state.router = HeuristicRouter(
        config, backends, adaptive=adaptive
    )
    test_app.state.adaptive = adaptive
    test_app.state.cascade = CascadeLayer(
        quality_scorer=QualityScorer(
            quality_threshold=config.cascade_quality_threshold,
        ),
        confidence_threshold=config.cascade_confidence_threshold,
    )
    total_capacity = (
        settings.max_concurrent_requests + settings.queue_max_size
    )
    test_app.state.capacity = asyncio.Semaphore(total_capacity)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)
