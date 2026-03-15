"""
Request ID middleware.

Ensures every request has a unique ID for tracing through logs.
Checks for an existing X-Request-ID header, generates one if absent.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(
            "x-request-id", uuid.uuid4().hex[:16]
        )
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
