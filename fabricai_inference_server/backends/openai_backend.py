"""
OpenAI backend — GPT models via the Chat Completions API.

Near-passthrough since our schema matches OpenAI's format.
We still normalize SDK response objects to our Pydantic models
for consistency across all backends.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

import openai

from fabricai_inference_server.backends import BackendCapabilities, BackendHealth
from fabricai_inference_server.exceptions import (
    BackendError,
    BackendUnavailableError,
)
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

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
]


class OpenAIBackend:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: openai.AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            tool_calling=True,
            vision=True,
            max_context_window=128_000,
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.010,
        )

    async def startup(self) -> None:
        self._client = openai.AsyncOpenAI(api_key=self._api_key)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            raise RuntimeError(
                "OpenAIBackend not started. Call startup() first."
            )
        return self._client

    def _resolve_model(self, model: str) -> str:
        if model.startswith("openai/"):
            return model[7:]
        return model

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        kwargs: dict = {
            "model": self._resolve_model(request.model),
            "messages": [
                m.model_dump(exclude_none=True) for m in request.messages
            ],
            "stream": False,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop:
            kwargs["stop"] = request.stop
        if request.tools:
            kwargs["tools"] = [
                t.model_dump(exclude_none=True) for t in request.tools
            ]
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice
        if request.response_format:
            kwargs["response_format"] = request.response_format

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except openai.APIConnectionError as e:
            raise BackendUnavailableError("openai", str(e)) from e
        except openai.APIStatusError as e:
            raise BackendError(
                "openai", f"HTTP {e.status_code}: {e.message}"
            ) from e

        return ChatCompletionResponse(
            id=response.id,
            model=response.model,
            created=response.created,
            choices=[
                Choice(
                    index=c.index,
                    message=ChatMessage(
                        role=c.message.role,
                        content=c.message.content,
                    ),
                    finish_reason=c.finish_reason,
                )
                for c in response.choices
            ],
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            if response.usage
            else None,
        )

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        kwargs: dict = {
            "model": self._resolve_model(request.model),
            "messages": [
                m.model_dump(exclude_none=True) for m in request.messages
            ],
            "stream": True,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop:
            kwargs["stop"] = request.stop

        try:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                yield ChatCompletionChunk(
                    id=chunk.id,
                    model=chunk.model,
                    created=chunk.created,
                    choices=[
                        ChunkChoice(
                            index=c.index,
                            delta=ChoiceDelta(
                                role=c.delta.role if c.delta else None,
                                content=(
                                    c.delta.content if c.delta else None
                                ),
                            ),
                            finish_reason=c.finish_reason,
                        )
                        for c in chunk.choices
                    ],
                )
        except openai.APIConnectionError as e:
            raise BackendUnavailableError("openai", str(e)) from e
        except openai.APIStatusError as e:
            raise BackendError(
                "openai", f"HTTP {e.status_code}: {e.message}"
            ) from e

    async def health(self) -> BackendHealth:
        start = time.monotonic()
        try:
            models = await asyncio.wait_for(
                self.client.models.list(), timeout=3.0
            )
            latency = (time.monotonic() - start) * 1000
            model_ids = [
                m.id
                for m in models.data
                if m.id in DEFAULT_MODELS
            ]
            return BackendHealth(
                healthy=True,
                latency_ms=round(latency, 1),
                models=model_ids or DEFAULT_MODELS,
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
