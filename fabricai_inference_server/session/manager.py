"""
Redis-backed session manager for conversation persistence.

Stores messages in OpenAI chat message format. Supports session pinning
(keeping a conversation on the same model) via session metadata.
Uses a connection pool to prevent connection exhaustion under load.
"""

from __future__ import annotations

import json
import uuid

import redis.asyncio as aioredis

from fabricai_inference_server.schemas.openai import ChatMessage


class SessionManager:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        max_connections: int = 20,
    ):
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._max_connections = max_connections
        self._pool: aioredis.ConnectionPool | None = None
        self._redis: aioredis.Redis | None = None

    async def startup(self) -> None:
        self._pool = aioredis.ConnectionPool(
            host=self._redis_host,
            port=self._redis_port,
            max_connections=self._max_connections,
            decode_responses=True,
        )
        self._redis = aioredis.Redis(connection_pool=self._pool)

    async def shutdown(self) -> None:
        if self._redis:
            await self._redis.aclose()
        if self._pool:
            await self._pool.aclose()

    @property
    def client(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError(
                "SessionManager not started. Call startup() first."
            )
        return self._redis

    async def get_or_create_session(
        self, session_id: str | None = None
    ) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        key = f"session:{session_id}"
        if not await self.client.exists(key):
            await self.client.set(key, json.dumps([]))
        return session_id

    async def add_message(
        self, session_id: str, message: ChatMessage
    ) -> None:
        key = f"session:{session_id}"
        data = await self.client.get(key)
        messages = json.loads(data) if data else []
        messages.append(message.model_dump(exclude_none=True))
        await self.client.set(key, json.dumps(messages))

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        key = f"session:{session_id}"
        data = await self.client.get(key)
        if not data:
            return []
        return [ChatMessage(**m) for m in json.loads(data)]

    async def pin_model_if_unset(
        self, session_id: str, model: str
    ) -> str:
        """Atomically pin a session to a backend. Returns the pinned value.

        Uses HSETNX so the first request to pin wins — concurrent
        requests for the same session all see the same backend.
        """
        meta_key = f"session_meta:{session_id}"
        was_set = await self.client.hsetnx(
            meta_key, "pinned_model", model
        )
        if was_set:
            return model
        # Another request pinned first — return their value
        return await self.client.hget(meta_key, "pinned_model") or model

    async def get_pinned_model(self, session_id: str) -> str | None:
        meta_key = f"session_meta:{session_id}"
        return await self.client.hget(meta_key, "pinned_model")
