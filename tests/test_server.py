"""
Tests for OpenAI-compatible API endpoints.
"""

import json


def test_chat_completion_non_streaming(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Hello from mock!"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == 15


def test_chat_completion_streaming(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    assert data_lines[-1] == "data: [DONE]"

    content_parts = []
    for line in data_lines[:-1]:
        chunk = json.loads(line[6:])
        assert chunk["object"] == "chat.completion.chunk"
        delta_content = chunk["choices"][0]["delta"].get("content")
        if delta_content:
            content_parts.append(delta_content)

    assert "".join(content_parts) == "Hello from mock!"


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    # Models now include backend prefix
    assert data["data"][0]["id"] == "ollama/mock-model"
    assert data["data"][0]["owned_by"] == "ollama"


def test_status(client):
    response = client.get("/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert "active_requests" in data
    assert "total_capacity" in data
    assert "backends" in data
    assert data["backends"]["ollama"]["healthy"] is True


def test_capacity_full(client):
    """Exhaust capacity semaphore, then verify we get 429."""
    cap = client.app.state.capacity
    # Drain all slots
    acquired = []
    while cap._value > 0:
        cap._value -= 1
        acquired.append(True)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 429

    # Restore slots
    cap._value += len(acquired)


def test_chat_completion_validates_messages(client):
    """Request without messages should fail validation."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test"},
    )
    assert response.status_code == 422


# --- Phase 2: Routing & Telemetry ---


def test_prefix_routing_unknown_backend(client):
    """Requesting an unconfigured backend returns 404."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "anthropic/claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 404
    assert "not configured" in response.json()["detail"]


def test_prefix_routing_ollama(client):
    """ollama/ prefix routes to the ollama backend."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ollama/llama3.3",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 200


def test_usage_endpoint(client):
    """Usage endpoint returns telemetry summary."""
    # Make a request first to generate telemetry
    client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    response = client.get("/v1/usage")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] >= 1
    assert "recent_events" in data
    assert "total_cost_usd" in data


def test_streaming_includes_backend_header(client):
    """Streaming responses include X-FabricAI-Backend header."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.headers.get("x-fabricai-backend") == "ollama"


# --- Phase 3a: Router integration ---


def test_auto_routing_returns_routing_headers(client):
    """model=auto triggers the heuristic router and returns routing headers."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert response.headers.get("x-fabricai-route-layer") == "heuristic"
    assert response.headers.get("x-fabricai-route-rule") is not None


def test_dry_run_route_endpoint(client):
    """POST /v1/route returns routing decision without running inference."""
    response = client.post(
        "/v1/route",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["layer"] == "heuristic"
    assert data["backend"] == "ollama"
    assert data["rule_matched"] == "trivial_query"
    assert "reason" in data


def test_dry_run_route_code_analysis(client):
    """Code + action keyword routes to cloud (ollama fallback in test)."""
    response = client.post(
        "/v1/route",
        json={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "```python\ndef foo(): pass\n```\nRefactor this",
                }
            ],
        },
    )
    data = response.json()
    assert data["rule_matched"] == "code_analysis"


def test_explicit_prefix_routing_header(client):
    """Explicit prefix sets layer=explicit in routing headers."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ollama/test",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.headers.get("x-fabricai-route-layer") == "explicit"


def test_usage_includes_routing_distribution(client):
    """Usage endpoint includes routing rule distribution."""
    client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    response = client.get("/v1/usage")
    data = response.json()
    assert "routing_distribution" in data
