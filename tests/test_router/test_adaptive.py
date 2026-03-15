"""
Tests for adaptive confidence thresholds.
"""

from fabricai_inference_server.router.adaptive import AdaptiveThresholds


def test_no_data_returns_base_confidence():
    adaptive = AdaptiveThresholds()
    assert adaptive.adjust_confidence("unknown_rule", 0.85) == 0.85


def test_insufficient_data_returns_base():
    adaptive = AdaptiveThresholds()
    # Record only 5 outcomes (below _MIN_SAMPLES=10)
    for _ in range(5):
        adaptive.record_outcome("my_rule", success=True, quality_score=0.9)
    assert adaptive.adjust_confidence("my_rule", 0.85) == 0.85


def test_high_success_rate_increases_confidence():
    adaptive = AdaptiveThresholds(blend_weight=0.6)
    for _ in range(20):
        adaptive.record_outcome("good_rule", success=True, quality_score=0.9)
    adjusted = adaptive.adjust_confidence("good_rule", 0.50)
    # Should be higher than base 0.50
    assert adjusted > 0.60


def test_low_success_rate_decreases_confidence():
    adaptive = AdaptiveThresholds(blend_weight=0.6)
    for _ in range(20):
        adaptive.record_outcome("bad_rule", success=False, quality_score=0.3)
    adjusted = adaptive.adjust_confidence("bad_rule", 0.85)
    # Should be lower than base 0.85
    assert adjusted < 0.60


def test_mixed_outcomes():
    adaptive = AdaptiveThresholds(blend_weight=0.6)
    for _ in range(10):
        adaptive.record_outcome("mixed", success=True, quality_score=0.8)
    for _ in range(10):
        adaptive.record_outcome("mixed", success=False, quality_score=0.4)
    adjusted = adaptive.adjust_confidence("mixed", 0.85)
    # Should be somewhere between high success and low success
    assert 0.3 < adjusted < 0.85


def test_get_stats():
    adaptive = AdaptiveThresholds()
    adaptive.record_outcome("rule_a", success=True, quality_score=0.9)
    adaptive.record_outcome("rule_a", success=False, quality_score=0.4)
    stats = adaptive.get_stats()
    assert "rule_a" in stats
    assert stats["rule_a"]["total"] == 2
    assert stats["rule_a"]["successes"] == 1
    assert stats["rule_a"]["success_rate"] == 0.5


def test_ema_adapts_to_recent_pattern():
    adaptive = AdaptiveThresholds(blend_weight=0.6)
    # Start with successes
    for _ in range(15):
        adaptive.record_outcome("shifting", success=True, quality_score=0.9)
    high = adaptive.adjust_confidence("shifting", 0.5)

    # Then failures
    for _ in range(30):
        adaptive.record_outcome("shifting", success=False, quality_score=0.3)
    low = adaptive.adjust_confidence("shifting", 0.5)

    # EMA should have shifted down
    assert low < high


def test_confidence_clamped():
    adaptive = AdaptiveThresholds(blend_weight=1.0)
    for _ in range(20):
        adaptive.record_outcome("extreme", success=True, quality_score=1.0)
    adjusted = adaptive.adjust_confidence("extreme", 0.99)
    assert adjusted <= 0.99

    for _ in range(100):
        adaptive.record_outcome("extreme2", success=False, quality_score=0.0)
    adjusted = adaptive.adjust_confidence("extreme2", 0.1)
    assert adjusted >= 0.1
