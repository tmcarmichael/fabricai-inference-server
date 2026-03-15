"""
Tests for rate limiting middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fabricai_inference_server.middleware.rate_limit import (
    RateLimitMiddleware,
)


@pytest.fixture
def rate_limited_app():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=3)

    @app.get("/v1/test")
    async def test_route():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


@pytest.fixture
def rate_limited_client(rate_limited_app):
    return TestClient(rate_limited_app)


def test_under_limit_passes(rate_limited_client):
    for _ in range(3):
        response = rate_limited_client.get("/v1/test")
        assert response.status_code == 200


def test_over_limit_rejected(rate_limited_client):
    for _ in range(3):
        rate_limited_client.get("/v1/test")

    response = rate_limited_client.get("/v1/test")
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]["message"]
    assert response.headers.get("retry-after") == "60"


def test_health_bypasses_rate_limit(rate_limited_client):
    # Exhaust the limit
    for _ in range(3):
        rate_limited_client.get("/v1/test")

    # Health should still work
    response = rate_limited_client.get("/health")
    assert response.status_code == 200


def test_different_keys_have_separate_limits(rate_limited_app):
    client = TestClient(rate_limited_app)

    # Key A: 3 requests
    for _ in range(3):
        client.get(
            "/v1/test",
            headers={"Authorization": "Bearer key-a"},
        )

    # Key A: should be limited
    resp_a = client.get(
        "/v1/test",
        headers={"Authorization": "Bearer key-a"},
    )
    assert resp_a.status_code == 429

    # Key B: should still work
    resp_b = client.get(
        "/v1/test",
        headers={"Authorization": "Bearer key-b"},
    )
    assert resp_b.status_code == 200
