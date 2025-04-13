"""
redis_session_manager.py

Session manager using Redis to handle sessions. Provide scaffolding for building on short-term and long-term chat memory.
"""

import os
import json
import uuid
import redis.asyncio as redis


class RedisSessionManager:
    def __init__(self):
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    async def get_or_create_session(self, session_id: str = "") -> str:
        """Return a valid session_id; create an empty conversation if not existing."""
        if not session_id:
            session_id = str(uuid.uuid4())
        key = f"session:{session_id}"
        if not await self.redis.exists(key):
            await self.redis.set(key, json.dumps([]))
        return session_id

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a new (role, content) to the conversation in Redis."""
        key = f"session:{session_id}"
        conversation_json = await self.redis.get(key)
        conversation = json.loads(conversation_json)
        conversation.append((role, content))
        await self.redis.set(key, json.dumps(conversation))

    async def build_prompt(self, session_id: str) -> str:
        """
        Concatenate messages into a prompt. Pattern: User, Role, System
        TODO: Improve building on messages.
        """
        key = f"session:{session_id}"
        conversation_json = await self.redis.get(key)
        if not conversation_json:
            return "Assistant:"

        conversation = json.loads(conversation_json)
        compiled = []
        for role, text in conversation:
            if role == "user":
                compiled.append(f"User: {text}\n")
            elif role == "assistant":
                compiled.append(f"Assistant: {text}\n")
            else:
                compiled.append(f"{role}: {text}\n")
        return "".join(compiled) + "Assistant:"
