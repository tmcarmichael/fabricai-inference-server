"""
Tests for Anthropic backend message translation.
"""

from fabricai_inference_server.backends.anthropic import AnthropicBackend
from fabricai_inference_server.schemas.openai import ChatMessage


def test_extract_system_message():
    backend = AnthropicBackend(api_key="test")
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello"),
    ]
    translated, system = backend._extract_system(messages)
    assert system == "You are helpful."
    assert len(translated) == 1
    assert translated[0]["role"] == "user"
    assert translated[0]["content"] == "Hello"


def test_extract_no_system_message():
    backend = AnthropicBackend(api_key="test")
    messages = [
        ChatMessage(role="user", content="Hello"),
    ]
    translated, system = backend._extract_system(messages)
    assert system is None
    assert len(translated) == 1


def test_resolve_model_strips_prefix():
    backend = AnthropicBackend(api_key="test")
    assert (
        backend._resolve_model("anthropic/claude-sonnet-4-20250514")
        == "claude-sonnet-4-20250514"
    )
    assert (
        backend._resolve_model("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"
    )
