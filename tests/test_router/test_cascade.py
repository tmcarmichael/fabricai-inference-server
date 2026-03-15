"""
Tests for the cascade fallback layer.
"""

import pytest

from fabricai_inference_server.router.cascade import CascadeLayer
from fabricai_inference_server.router.quality import QualityScorer
from fabricai_inference_server.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)


class FakeBackend:
    """Backend that returns a configurable response."""

    def __init__(self, content: str, name: str = "fake"):
        self._content = content
        self._name = name

    @property
    def name(self):
        return self._name

    async def chat_completion(self, request):
        return ChatCompletionResponse(
            id="test-123",
            model=f"{self._name}-model",
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant", content=self._content
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )


@pytest.fixture
def cascade():
    return CascadeLayer(
        quality_scorer=QualityScorer(quality_threshold=0.6),
        confidence_threshold=0.7,
    )


def test_should_cascade_low_confidence(cascade):
    assert cascade.should_cascade(0.5, has_cloud=True, is_streaming=False)


def test_should_not_cascade_high_confidence(cascade):
    assert not cascade.should_cascade(
        0.9, has_cloud=True, is_streaming=False
    )


def test_should_not_cascade_streaming(cascade):
    assert not cascade.should_cascade(
        0.5, has_cloud=True, is_streaming=True
    )


def test_should_not_cascade_no_cloud(cascade):
    assert not cascade.should_cascade(
        0.5, has_cloud=False, is_streaming=False
    )


@pytest.mark.asyncio
async def test_cascade_local_passes(cascade):
    """Good local response → no escalation."""
    local = FakeBackend(
        "A complete and well-formed response.", name="local"
    )
    cloud = FakeBackend("Cloud response.", name="cloud")
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    response, result = await cascade.execute(request, local, cloud)
    assert not result.escalated
    assert result.quality_score >= 0.6
    assert response.model == "local-model"


@pytest.mark.asyncio
async def test_cascade_escalates_on_bad_quality(cascade):
    """Bad local response → escalate to cloud."""
    local = FakeBackend("", name="local")  # Empty response
    cloud = FakeBackend(
        "A proper cloud response.", name="cloud"
    )
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    response, result = await cascade.execute(request, local, cloud)
    assert result.escalated
    assert result.quality_score < 0.6
    assert response.model == "cloud-model"
    assert result.local_content == ""


@pytest.mark.asyncio
async def test_cascade_saves_local_content_for_training():
    """Escalated cascade saves local content for future training."""
    # Use a strict threshold so "I don't know" fails
    strict_cascade = CascadeLayer(
        quality_scorer=QualityScorer(quality_threshold=0.8),
        confidence_threshold=0.7,
    )
    local = FakeBackend("I don't know", name="local")
    cloud = FakeBackend("The answer is 42.", name="cloud")
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Question")],
    )

    response, result = await strict_cascade.execute(
        request, local, cloud
    )
    assert result.escalated
    assert result.local_content == "I don't know"
