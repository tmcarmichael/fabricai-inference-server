"""
Tests for the heuristic routing engine.
"""

import pytest

from fabricai_inference_server.router.heuristic import (
    HeuristicRouter,
    RoutingConfig,
)
from fabricai_inference_server.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
)
from fabricai_inference_server.utils.tokens import (
    estimate_tokens,
    extract_last_user_content,
    has_any_keyword,
    has_code_block,
)

# --- Token utilities ---


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 2  # 2 * 1.3 = 2.6 → 2
    assert estimate_tokens("a " * 100) > 100


def test_has_code_block():
    assert has_code_block("```python\nprint('hi')\n```")
    assert has_code_block("def foo(): pass")
    assert has_code_block("class Foo: pass")
    assert not has_code_block("just regular text here")


def test_has_any_keyword():
    assert has_any_keyword("please summarize this", ["summarize"])
    assert has_any_keyword("REFACTOR the code", ["refactor"])
    assert not has_any_keyword("hello world", ["refactor", "debug"])


def test_extract_last_user_content():
    messages = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="second"),
    ]
    assert extract_last_user_content(messages) == "second"


def test_extract_last_user_content_empty():
    messages = [ChatMessage(role="assistant", content="reply")]
    assert extract_last_user_content(messages) == ""


# --- Heuristic routing rules ---


class MockOllamaBackend:
    _default_model = "llama3.3:8b"

    @property
    def name(self):
        return "ollama"


class MockCloudBackend:
    @property
    def name(self):
        return "anthropic"


def _make_request(content: str, msg_count: int = 1):
    messages = [ChatMessage(role="user", content=content)]
    # Pad with dummy turns to test conversation depth
    for i in range(msg_count - 1):
        messages.insert(
            0, ChatMessage(role="assistant", content=f"reply {i}")
        )
        messages.insert(
            0, ChatMessage(role="user", content=f"msg {i}")
        )
    return ChatCompletionRequest(model="auto", messages=messages)


@pytest.fixture
def heuristic_router():
    backends = {
        "ollama": MockOllamaBackend(),
        "anthropic": MockCloudBackend(),
    }
    return HeuristicRouter(RoutingConfig(), backends)


@pytest.mark.asyncio
async def test_trivial_query(heuristic_router):
    request = _make_request("Hi")
    decision = await heuristic_router.route(request, None)
    assert decision.backend_name == "ollama"
    assert decision.rule_matched == "trivial_query"
    assert decision.layer == "heuristic"


@pytest.mark.asyncio
async def test_long_context(heuristic_router):
    long_text = "word " * 4000
    request = _make_request(long_text)
    decision = await heuristic_router.route(request, None)
    assert decision.backend_name == "anthropic"
    assert decision.rule_matched == "long_context"


@pytest.mark.asyncio
async def test_code_analysis(heuristic_router):
    request = _make_request(
        "```python\ndef foo(): pass\n```\nPlease refactor this"
    )
    decision = await heuristic_router.route(request, None)
    assert decision.backend_name == "anthropic"
    assert decision.rule_matched == "code_analysis"


@pytest.mark.asyncio
async def test_simple_task_keyword(heuristic_router):
    # Must exceed trivial_max_tokens (50) to reach the keyword rules
    padding = "the document covers many topics and details "
    request = _make_request(
        f"Please summarize the following: {padding * 5}"
    )
    decision = await heuristic_router.route(request, None)
    assert decision.backend_name == "ollama"
    assert decision.rule_matched == "simple_task"


@pytest.mark.asyncio
async def test_code_no_action(heuristic_router):
    request = _make_request("```python\nprint('hello')\n```")
    decision = await heuristic_router.route(request, None)
    assert decision.backend_name == "ollama"
    assert decision.rule_matched == "code_no_action"


@pytest.mark.asyncio
async def test_deep_conversation(heuristic_router):
    # Last message must exceed trivial threshold (50 tokens) and
    # not match any keyword rules, so only conversation depth triggers
    filler = "here is a moderately long message with enough words "
    request = _make_request(filler * 6, msg_count=12)
    decision = await heuristic_router.route(request, None)
    assert decision.backend_name == "anthropic"
    assert decision.rule_matched == "deep_conversation"


@pytest.mark.asyncio
async def test_default_route(heuristic_router):
    # Must exceed trivial threshold, no code, no matching keywords
    filler = "I have a question about general knowledge and trivia "
    request = _make_request(filler * 6)
    decision = await heuristic_router.route(request, None)
    assert decision.backend_name == "ollama"
    assert decision.rule_matched == "default"


@pytest.mark.asyncio
async def test_no_cloud_fallback_to_local():
    """When no cloud backend is configured, cloud rules fall back to local."""
    backends = {"ollama": MockOllamaBackend()}
    router = HeuristicRouter(RoutingConfig(), backends)
    long_text = "word " * 4000  # Would normally route to cloud
    request = _make_request(long_text)
    decision = await router.route(request, None)
    # Should fall back to ollama since no cloud available
    assert decision.backend_name == "ollama"


@pytest.mark.asyncio
async def test_config_from_yaml(tmp_path):
    """Config loads from YAML file."""
    yaml_content = """
default_backend: ollama
cloud_backend: openai
trivial_max_tokens: 100
cloud_keywords:
  - custom_keyword
local_keywords:
  - custom_local
"""
    config_file = tmp_path / "routing.yaml"
    config_file.write_text(yaml_content)
    config = RoutingConfig.from_yaml(config_file)
    assert config.cloud_backend == "openai"
    assert config.trivial_max_tokens == 100
    assert "custom_keyword" in config.cloud_keywords


@pytest.mark.asyncio
async def test_config_defaults_when_file_missing(tmp_path):
    config = RoutingConfig.from_yaml(tmp_path / "nonexistent.yaml")
    assert config.default_backend == "ollama"
    assert config.trivial_max_tokens == 50
