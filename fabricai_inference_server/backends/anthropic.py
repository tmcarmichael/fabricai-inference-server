"""
Anthropic backend — Claude models via the Messages API.

Translates between OpenAI chat format and Anthropic's Messages API.
Key differences: system message is a top-level param, content uses
blocks, stop reasons differ.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator

import anthropic

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

STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}

DEFAULT_MODELS = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
]


class AnthropicBackend:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            tool_calling=True,
            vision=True,
            max_context_window=200_000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )

    async def startup(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            raise RuntimeError(
                "AnthropicBackend not started. Call startup() first."
            )
        return self._client

    def _resolve_model(self, model: str) -> str:
        if model.startswith("anthropic/"):
            return model[10:]
        return model

    def _extract_system(
        self, messages: list[ChatMessage]
    ) -> tuple[list[dict], str | None]:
        """Split system message out (Anthropic takes it as a top-level param)."""
        system = None
        translated = []
        for msg in messages:
            if msg.role == "system":
                system = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )
            else:
                translated.append(
                    {"role": msg.role, "content": msg.content}
                )
        return translated, system

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        messages, system = self._extract_system(request.messages)
        kwargs: dict = {
            "model": self._resolve_model(request.model),
            "max_tokens": request.max_tokens or 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop_sequences"] = (
                request.stop
                if isinstance(request.stop, list)
                else [request.stop]
            )

        try:
            response = await self.client.messages.create(**kwargs)
        except anthropic.APIConnectionError as e:
            raise BackendUnavailableError("anthropic", str(e)) from e
        except anthropic.APIStatusError as e:
            raise BackendError(
                "anthropic", f"HTTP {e.status_code}: {e.message}"
            ) from e

        content = "".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        return ChatCompletionResponse(
            model=response.model,
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant", content=content
                    ),
                    finish_reason=STOP_REASON_MAP.get(
                        response.stop_reason, response.stop_reason
                    ),
                )
            ],
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=(
                    response.usage.input_tokens
                    + response.usage.output_tokens
                ),
            ),
        )

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        messages, system = self._extract_system(request.messages)
        kwargs: dict = {
            "model": self._resolve_model(request.model),
            "max_tokens": request.max_tokens or 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model = self._resolve_model(request.model)

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                # Role chunk
                yield ChatCompletionChunk(
                    id=chunk_id,
                    model=model,
                    created=created,
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(role="assistant")
                        )
                    ],
                )
                # Content chunks
                async for text in stream.text_stream:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        model=model,
                        created=created,
                        choices=[
                            ChunkChoice(
                                delta=ChoiceDelta(content=text)
                            )
                        ],
                    )
                # Final chunk
                yield ChatCompletionChunk(
                    id=chunk_id,
                    model=model,
                    created=created,
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(), finish_reason="stop"
                        )
                    ],
                )
        except anthropic.APIConnectionError as e:
            raise BackendUnavailableError("anthropic", str(e)) from e
        except anthropic.APIStatusError as e:
            raise BackendError(
                "anthropic", f"HTTP {e.status_code}: {e.message}"
            ) from e

    async def health(self) -> BackendHealth:
        start = time.monotonic()
        try:
            await asyncio.wait_for(
                self.client.messages.count_tokens(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "ping"}],
                ),
                timeout=3.0,
            )
            latency = (time.monotonic() - start) * 1000
            return BackendHealth(
                healthy=True,
                latency_ms=round(latency, 1),
                models=DEFAULT_MODELS,
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
