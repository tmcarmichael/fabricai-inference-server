"""
Tests for Google backend message translation.
"""

from fabricai_inference_server.backends.google import GoogleBackend
from fabricai_inference_server.schemas.openai import ChatMessage


def test_translate_messages():
    backend = GoogleBackend(api_key="test")
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi!"),
        ChatMessage(role="user", content="How are you?"),
    ]
    contents, system = backend._translate_messages(messages)
    assert system == "You are helpful."
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Hello"}]
    assert contents[1]["role"] == "model"  # assistant → model
    assert contents[2]["role"] == "user"


def test_resolve_model_strips_prefix():
    backend = GoogleBackend(api_key="test")
    assert (
        backend._resolve_model("google/gemini-2.0-flash")
        == "gemini-2.0-flash"
    )
