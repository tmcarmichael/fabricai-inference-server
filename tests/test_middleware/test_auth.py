"""
Tests for API key authentication middleware.
"""


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fabricai_inference_server.middleware.auth import AuthMiddleware


@pytest.fixture
def authed_app():
    app = FastAPI()
    app.add_middleware(AuthMiddleware, api_keys={"sk-valid-key"})

    @app.get("/v1/test")
    async def test_route():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


@pytest.fixture
def authed_client(authed_app):
    return TestClient(authed_app)


def test_valid_key_passes(authed_client):
    response = authed_client.get(
        "/v1/test",
        headers={"Authorization": "Bearer sk-valid-key"},
    )
    assert response.status_code == 200


def test_missing_key_rejected(authed_client):
    response = authed_client.get("/v1/test")
    assert response.status_code == 401
    assert "Missing API key" in response.json()["error"]["message"]


def test_invalid_key_rejected(authed_client):
    response = authed_client.get(
        "/v1/test",
        headers={"Authorization": "Bearer sk-wrong-key"},
    )
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["error"]["message"]


def test_health_endpoint_bypasses_auth(authed_client):
    response = authed_client.get("/health")
    assert response.status_code == 200


def test_malformed_auth_header(authed_client):
    response = authed_client.get(
        "/v1/test",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert response.status_code == 401
