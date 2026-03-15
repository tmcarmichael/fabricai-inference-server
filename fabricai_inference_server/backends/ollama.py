"""
Ollama backend: proxies to Ollama's OpenAI-compatible API.

Ollama runs locally and handles model management, chat templates,
Metal/CUDA acceleration. We talk to it via its /v1/ endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

import httpx

from fabricai_inference_server.backends import BackendCapabilities, BackendHealth
from fabricai_inference_server.exceptions import (
    BackendError,
    BackendUnavailableError,
)
from fabricai_inference_server.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)


class OllamaBackend:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2",
    ):
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            tool_calling=True,
            vision=True,
            max_context_window=131072,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=5.0, read=120.0, write=5.0, pool=5.0
            ),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
            ),
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "OllamaBackend not started. Call startup() first."
            )
        return self._client

    def _resolve_model(self, model: str) -> str:
        """Resolve model name, stripping 'ollama/' prefix if present."""
        if model.startswith("ollama/"):
            return model[7:]
        if model in ("auto", ""):
            return self._default_model
        return model

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = self._resolve_model(payload.get("model", ""))
        payload["stream"] = False

        try:
            response = await self.client.post(
                "/v1/chat/completions", json=payload
            )
            response.raise_for_status()
            return ChatCompletionResponse(**response.json())
        except httpx.ConnectError as e:
            raise BackendUnavailableError("ollama", str(e)) from e
        except httpx.HTTPStatusError as e:
            raise BackendError(
                "ollama",
                f"HTTP {e.response.status_code}: {e.response.text}",
            ) from e

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = self._resolve_model(payload.get("model", ""))
        payload["stream"] = True

        try:
            async with self.client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        yield ChatCompletionChunk(**json.loads(data))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse SSE chunk: %s", data
                        )
        except httpx.ConnectError as e:
            raise BackendUnavailableError("ollama", str(e)) from e
        except httpx.HTTPStatusError as e:
            raise BackendError(
                "ollama", f"HTTP {e.response.status_code}"
            ) from e

    async def health(self) -> BackendHealth:
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self.client.get("/v1/models"), timeout=3.0
            )
            response.raise_for_status()
            latency = (time.monotonic() - start) * 1000
            data = response.json()
            models = [
                m.get("id", m.get("model", "unknown"))
                for m in data.get("data", [])
            ]
            return BackendHealth(
                healthy=True, latency_ms=round(latency, 1), models=models
            )
        except TimeoutError:
            latency = (time.monotonic() - start) * 1000
            return BackendHealth(
                healthy=False,
                latency_ms=round(latency, 1),
                error="Health check timed out (3s)",
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return BackendHealth(
                healthy=False,
                latency_ms=round(latency, 1),
                error=str(e),
            )
