"""
Cascade fallback layer — try local, score quality, escalate if needed.

This is the most expensive routing path (pays for both local and cloud
inference when escalating) but provides a safety net for ambiguous
requests. Only triggers for low-confidence heuristic decisions on
non-streaming requests.

Cascade events generate training data for the future local classifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fabricai_inference_server.router.quality import QualityScorer
from fabricai_inference_server.schemas.openai import ChatCompletionRequest

logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Outcome of a cascade attempt."""

    escalated: bool
    quality_score: float
    quality_reasons: list[str] = field(default_factory=list)
    # Saved for classifier training data
    local_content: str | None = None


class CascadeLayer:
    """Try local model first. If output quality is low, escalate to cloud."""

    def __init__(
        self,
        quality_scorer: QualityScorer,
        confidence_threshold: float = 0.7,
    ):
        self.scorer = quality_scorer
        self.confidence_threshold = confidence_threshold

    def should_cascade(
        self, routing_confidence: float, has_cloud: bool, is_streaming: bool
    ) -> bool:
        """Determine if this request should go through cascade."""
        if is_streaming:
            return False
        if not has_cloud:
            return False
        return routing_confidence < self.confidence_threshold

    async def execute(
        self,
        request: ChatCompletionRequest,
        local_backend,
        cloud_backend,
    ) -> tuple[object, CascadeResult]:
        """Run request on local, score quality, escalate if needed.

        Returns (response, cascade_result).
        """
        # Step 1: Try local
        local_response = await local_backend.chat_completion(request)
        content = ""
        if local_response.choices:
            content = local_response.choices[0].message.content or ""

        # Step 2: Score quality
        quality = self.scorer.score(content)

        if quality.passed:
            logger.debug(
                "Cascade: local passed (score=%.2f)", quality.score
            )
            return local_response, CascadeResult(
                escalated=False,
                quality_score=quality.score,
                quality_reasons=quality.reasons,
            )

        # Step 3: Escalate to cloud
        logger.info(
            "Cascade: escalating to cloud (score=%.2f, reasons=%s)",
            quality.score,
            quality.reasons,
        )
        cloud_response = await cloud_backend.chat_completion(request)
        return cloud_response, CascadeResult(
            escalated=True,
            quality_score=quality.score,
            quality_reasons=quality.reasons,
            local_content=content,
        )
