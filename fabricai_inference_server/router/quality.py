"""
Quality scorer for local model output validation.

Lightweight heuristics that detect low-quality responses without
needing a separate model. Runs in < 1ms on the generated text.
When quality is below threshold, the cascade layer escalates to cloud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_UNCERTAINTY_PHRASES = (
    "i'm not sure",
    "i don't know",
    "i cannot",
    "i can't",
    "as an ai",
    "i'm unable",
    "i apologize, but",
)

_SENTENCE_ENDINGS = frozenset(".!?\"')]\n")

# Matches 3+ consecutive repeated words
_REPEAT_PATTERN = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)


@dataclass
class QualityScore:
    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


class QualityScorer:
    """Score response quality using fast heuristics."""

    def __init__(
        self,
        quality_threshold: float = 0.6,
        min_response_length: int = 10,
        max_repetition_ratio: float = 0.5,
    ):
        self.quality_threshold = quality_threshold
        self.min_response_length = min_response_length
        self.max_repetition_ratio = max_repetition_ratio

    def score(self, text: str) -> QualityScore:
        """Evaluate response quality. Returns score 0.0-1.0."""
        if not text:
            return QualityScore(
                score=0.0, passed=False, reasons=["empty_response"]
            )

        penalties: list[str] = []
        value = 1.0
        stripped = text.strip()

        # 1. Very short response
        if len(stripped) < self.min_response_length:
            value -= 0.5
            penalties.append("very_short_response")

        # 2. Repetition: low unique word ratio
        words = stripped.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < (1.0 - self.max_repetition_ratio):
                value -= 0.3
                penalties.append("high_repetition")

        # 3. Repeated word sequences (degeneration)
        if _REPEAT_PATTERN.search(stripped):
            value -= 0.3
            penalties.append("word_degeneration")

        # 4. Incomplete sentence
        if stripped and stripped[-1] not in _SENTENCE_ENDINGS:
            value -= 0.15
            penalties.append("incomplete_sentence")

        # 5. Self-reported uncertainty
        text_lower = stripped.lower()
        for phrase in _UNCERTAINTY_PHRASES:
            if phrase in text_lower:
                value -= 0.2
                penalties.append(f"uncertainty:{phrase}")
                break

        value = max(0.0, min(1.0, value))
        return QualityScore(
            score=round(value, 3),
            passed=value >= self.quality_threshold,
            reasons=penalties,
        )
