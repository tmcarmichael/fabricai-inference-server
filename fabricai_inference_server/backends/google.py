"""
Google backend: Gemini models via the GenAI SDK.

Translates between OpenAI chat format and Google's content/parts
structure. Key differences: "assistant" role → "model",
content is wrapped in parts, system instruction is separate config.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator

from google import genai
from google.genai.types import GenerateContentConfig

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
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


class GoogleBackend:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: genai.Client | None = None

    @property
    def name(self) -> str:
        return "google"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            tool_calling=True,
            vision=True,
            max_context_window=1_000_000,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
        )

    async def startup(self) -> None:
        self._client = genai.Client(api_key=self._api_key)

    async def shutdown(self) -> None:
        pass  # genai Client has no close method

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            raise RuntimeError(
                "GoogleBackend not started. Call startup() first."
            )
        return self._client

    def _resolve_model(self, model: str) -> str:
        if model.startswith("google/"):
            return model[7:]
        return model

    def _translate_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[dict], str | None]:
        """Convert OpenAI messages to Google contents format."""
        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )
            else:
                role = "model" if msg.role == "assistant" else "user"
                text = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content or "")
                )
                contents.append(
                    {"role": role, "parts": [{"text": text}]}
                )
        return contents, system_instruction

    def _build_config(
        self,
        request: ChatCompletionRequest,
        system_instruction: str | None,
    ) -> GenerateContentConfig:
        kwargs: dict = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop_sequences"] = (
                request.stop
                if isinstance(request.stop, list)
                else [request.stop]
            )
        return GenerateContentConfig(**kwargs)

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        contents, system_instruction = self._translate_messages(
            request.messages
        )
        config = self._build_config(request, system_instruction)

        try:
            response = await self.client.aio.models.generate_content(
                model=self._resolve_model(request.model),
                contents=contents,
                config=config,
            )
        except Exception as e:
            error_str = str(e)
            if "connect" in error_str.lower():
                raise BackendUnavailableError(
                    "google", error_str
                ) from e
            raise BackendError("google", error_str) from e

        content = response.text or ""
        usage_meta = response.usage_metadata

        return ChatCompletionResponse(
            model=self._resolve_model(request.model),
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant", content=content
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=(
                    usage_meta.prompt_token_count
                    if usage_meta
                    else 0
                ),
                completion_tokens=(
                    usage_meta.candidates_token_count
                    if usage_meta
                    else 0
                ),
                total_tokens=(
                    usage_meta.total_token_count if usage_meta else 0
                ),
            ),
        )

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        contents, system_instruction = self._translate_messages(
            request.messages
        )
        config = self._build_config(request, system_instruction)

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model = self._resolve_model(request.model)

        try:
            # Role chunk
            yield ChatCompletionChunk(
                id=chunk_id,
                model=model,
                created=created,
                choices=[
                    ChunkChoice(delta=ChoiceDelta(role="assistant"))
                ],
            )

            async for chunk in await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        model=model,
                        created=created,
                        choices=[
                            ChunkChoice(
                                delta=ChoiceDelta(content=chunk.text)
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
        except Exception as e:
            error_str = str(e)
            if "connect" in error_str.lower():
                raise BackendUnavailableError(
                    "google", error_str
                ) from e
            raise BackendError("google", error_str) from e

    async def health(self) -> BackendHealth:
        start = time.monotonic()
        try:
            await asyncio.wait_for(
                self.client.aio.models.list(
                    config={"page_size": 5}
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
