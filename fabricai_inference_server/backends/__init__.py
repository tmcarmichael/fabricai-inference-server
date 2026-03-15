"""
Backend protocol and registry for inference providers.

Every backend (Ollama, Anthropic, OpenAI, Google) implements the Backend
protocol so the server and router can treat them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

from fabricai_inference_server.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


@dataclass
class BackendHealth:
    healthy: bool
    latency_ms: float | None = None
    error: str | None = None
    models: list[str] = field(default_factory=list)


@dataclass
class BackendCapabilities:
    streaming: bool = True
    tool_calling: bool = False
    vision: bool = False
    max_context_window: int = 8192
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@runtime_checkable
class Backend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse: ...

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def health(self) -> BackendHealth: ...

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...
