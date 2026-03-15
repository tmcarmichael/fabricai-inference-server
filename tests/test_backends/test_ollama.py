"""
Tests for Ollama backend model resolution.
"""

from fabricai_inference_server.backends.ollama import OllamaBackend


def test_resolve_model_strips_prefix():
    backend = OllamaBackend(default_model="llama3.2")
    assert backend._resolve_model("ollama/llama3.3:8b") == "llama3.3:8b"


def test_resolve_model_auto_returns_default():
    backend = OllamaBackend(default_model="llama3.2")
    assert backend._resolve_model("auto") == "llama3.2"


def test_resolve_model_empty_returns_default():
    backend = OllamaBackend(default_model="llama3.2")
    assert backend._resolve_model("") == "llama3.2"


def test_resolve_model_passthrough():
    backend = OllamaBackend(default_model="llama3.2")
    assert backend._resolve_model("mistral") == "mistral"
