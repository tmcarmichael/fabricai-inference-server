"""
Tests for cost estimation and telemetry.
"""

from datetime import datetime, timezone

from fabricai_inference_server.telemetry.collector import (
    RequestEvent,
    TelemetryCollector,
)
from fabricai_inference_server.telemetry.cost import estimate_cost


def test_estimate_cost_known_model():
    cost = estimate_cost("claude-sonnet-4-20250514", 1000, 500)
    # $3/1M input = $0.003/1K, $15/1M output = $0.015/1K
    expected = (1000 / 1000 * 0.003) + (500 / 1000 * 0.015)
    assert abs(cost - expected) < 0.0001


def test_estimate_cost_local_model():
    cost = estimate_cost("llama3.3:8b", 1000, 500)
    assert cost == 0.0


def test_estimate_cost_unknown_model():
    cost = estimate_cost("some-unknown-model", 1000, 500)
    assert cost == 0.0


def test_telemetry_collector():
    collector = TelemetryCollector(max_events=5)

    for i in range(3):
        collector.record(
            RequestEvent(
                request_id=f"req-{i}",
                timestamp=datetime.now(timezone.utc),
                model="gpt-4o",
                backend="openai",
                input_tokens=100,
                output_tokens=50,
                latency_ms=200.0,
                estimated_cost_usd=0.001,
            )
        )

    summary = collector.get_summary()
    assert summary["total_requests"] == 3
    assert summary["total_cost_usd"] == 0.003
    assert len(summary["recent_events"]) == 3


def test_telemetry_collector_bounded():
    collector = TelemetryCollector(max_events=2)

    for i in range(5):
        collector.record(
            RequestEvent(
                request_id=f"req-{i}",
                timestamp=datetime.now(timezone.utc),
                model="gpt-4o",
                backend="openai",
                input_tokens=100,
                output_tokens=50,
                latency_ms=200.0,
                estimated_cost_usd=0.001,
            )
        )

    summary = collector.get_summary()
    assert summary["total_requests"] == 5
    # Only last 2 events retained in deque
    assert len(summary["recent_events"]) == 2
    assert summary["recent_events"][0]["request_id"] == "req-3"
