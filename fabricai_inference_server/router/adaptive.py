"""
Adaptive confidence thresholds.

Learns from cascade outcomes to self-tune routing confidence per
heuristic rule. Uses an exponential moving average so the system
adapts to changing usage patterns without explicit retraining.

No ML, no training pipeline, no cold start — just statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum observations before we trust the adaptive signal
_MIN_SAMPLES = 10

# EMA smoothing factor — higher = adapts faster, noisier
_ALPHA = 0.1


@dataclass
class RuleStats:
    """Running statistics for a single routing rule."""

    total: int = 0
    successes: int = 0
    # Exponential moving average of success rate
    ema_success: float = 0.5
    # EMA of quality scores from cascade
    ema_quality: float = 0.7


class AdaptiveThresholds:
    """Adjusts routing confidence based on observed cascade outcomes.

    When the cascade processes a request, it reports back:
    - Which rule triggered
    - Whether local passed quality check (success)
    - The quality score

    This data tunes the confidence of each heuristic rule:
    - Rules that consistently produce good local results get higher
      confidence → fewer cascade attempts → lower latency.
    - Rules that often fail quality get lower confidence → more
      cascade attempts → better output quality.
    """

    def __init__(self, blend_weight: float = 0.6):
        self._stats: dict[str, RuleStats] = {}
        # How much to weight observed data vs base confidence
        # 0.0 = ignore observations, 1.0 = fully adaptive
        self._blend = blend_weight

    def record_outcome(
        self,
        rule: str,
        success: bool,
        quality_score: float,
    ) -> None:
        """Record a cascade outcome for adaptive tuning."""
        stats = self._stats.setdefault(rule, RuleStats())
        stats.total += 1
        if success:
            stats.successes += 1
        stats.ema_success = (
            _ALPHA * (1.0 if success else 0.0)
            + (1 - _ALPHA) * stats.ema_success
        )
        stats.ema_quality = (
            _ALPHA * quality_score + (1 - _ALPHA) * stats.ema_quality
        )

    def adjust_confidence(
        self, rule: str, base_confidence: float
    ) -> float:
        """Blend base confidence with observed success rate.

        Returns base_confidence unchanged until we have enough data.
        """
        stats = self._stats.get(rule)
        if not stats or stats.total < _MIN_SAMPLES:
            return base_confidence

        # Blend: base confidence weighted by (1-blend), observed by blend
        adjusted = (
            base_confidence * (1 - self._blend)
            + stats.ema_success * self._blend
        )
        return round(max(0.1, min(0.99, adjusted)), 3)

    def get_stats(self) -> dict:
        """Export adaptive state for telemetry/dashboard."""
        return {
            rule: {
                "total": s.total,
                "successes": s.successes,
                "success_rate": (
                    round(s.successes / s.total, 3) if s.total else 0
                ),
                "ema_success": round(s.ema_success, 3),
                "ema_quality": round(s.ema_quality, 3),
            }
            for rule, s in self._stats.items()
        }
