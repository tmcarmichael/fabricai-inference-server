"""
Rate limiting middleware.

Sliding window counter per API key (or per IP if no auth).
Uses a deque with left-pruning — O(1) amortized per request
instead of O(n) list comprehension rebuild.

In-memory — suitable for single-worker deployments.
"""

from __future__ import annotations

import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._windows: dict[str, deque[float]] = {}

    def _get_key(self, request: Request) -> str:
        """Identify the client — API key if present, else IP."""
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return f"key:{auth[7:].strip()}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        key = self._get_key(request)
        now = time.monotonic()
        cutoff = now - 60

        window = self._windows.get(key)
        if window is None:
            window = deque()
            self._windows[key] = window

        # Prune expired entries from the left — O(expired) amortized
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.rpm:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"Rate limit exceeded ({self.rpm} req/min).",
                        "type": "rate_limit_error",
                    }
                },
                headers={"Retry-After": "60"},
            )

        window.append(now)
        return await call_next(request)
