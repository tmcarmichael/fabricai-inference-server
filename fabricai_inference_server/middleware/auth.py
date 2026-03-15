"""
API key authentication middleware.

Validates Bearer tokens against configured API keys.
Skips auth for health checks and OpenAPI docs.
"""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths that never require auth
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_keys: set[str]):
        super().__init__(app)
        self.api_keys = api_keys

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip auth for public paths
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Missing API key. Use Authorization: Bearer <key>",
                        "type": "auth_error",
                    }
                },
            )

        key = auth_header[7:].strip()
        if key not in self.api_keys:
            logger.warning("Invalid API key attempt: %s...", key[:8])
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid API key.",
                        "type": "auth_error",
                    }
                },
            )

        return await call_next(request)
