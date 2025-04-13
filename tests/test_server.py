import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from fabricai_inference_server.server import fastapi_app, request_queue


@pytest.fixture
def client():
    """Pytest fixture returning a TestClient for server tests."""
    return TestClient(fastapi_app)


def test_status_endpoint(client):
    """Basic test for /v1/status - checks JSON structure."""
    # Act
    resp = client.get("/v1/status")

    # Assert
    assert resp.status_code == 200
    data = resp.json()
    assert "queue_size" in data
    assert "queue_max_size" in data


@patch("fabricai_inference_server.server.get_engine")
def test_inference_sse_happy_path(mock_get_engine, client):
    """
    Mocks get_engine so we don't load a real file, then ensures SSE streams tokens.
    """
    # Arrange
    mock_engine = MagicMock()
    mock_engine.generate_stream.return_value = ["token1 ", "token2"]
    mock_get_engine.return_value = mock_engine
    payload = {"prompt": "Hello SSE test", "max_tokens": 10}

    # Act
    resp = client.post("/v1/inference_sse", json=payload, stream=True)

    # Assert
    assert resp.status_code == 200
    body = b"".join(resp.iter_content()).decode("utf-8")
    assert "token1" in body
    assert "token2" in body


def test_queue_full(client):
    """
    If the request_queue is already full, we expect 429 from SSE endpoint.
    """
    # Arrange
    with pytest.raises(RuntimeError):
        for _ in range(fastapi_app.QUEUE_MAX_SIZE + 1):
            request_queue.put_nowait(None)

    # Act
    resp = client.post("/v1/inference_sse", json={"prompt": "Queue test"})

    # Assert
    assert resp.status_code == 429
    while not request_queue.empty():
        request_queue.get_nowait()
        request_queue.task_done()
