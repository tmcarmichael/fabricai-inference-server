"""
Lightweight telemetry collector.

Stores recent request events in a bounded deque for the dashboard
and summary endpoint. Non-blocking — callers fire-and-forget.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RequestEvent:
    request_id: str
    timestamp: datetime
    model: str
    backend: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    # Routing telemetry
    routing_layer: str | None = None
    routing_rule: str | None = None
    routing_confidence: float | None = None
    # Cascade telemetry
    cascade_triggered: bool | None = None
    cascade_quality_score: float | None = None


class TelemetryCollector:
    def __init__(self, max_events: int = 1000):
        self._events: deque[RequestEvent] = deque(maxlen=max_events)
        self._total_cost: float = 0.0
        self._total_requests: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._routing_counts: dict[str, int] = {}
        self._cascade_total: int = 0
        self._cascade_escalated: int = 0

    def record(self, event: RequestEvent) -> None:
        self._events.append(event)
        self._total_cost += event.estimated_cost_usd
        self._total_requests += 1
        self._total_input_tokens += event.input_tokens
        self._total_output_tokens += event.output_tokens
        if event.routing_rule:
            self._routing_counts[event.routing_rule] = (
                self._routing_counts.get(event.routing_rule, 0) + 1
            )
        if event.cascade_triggered is not None:
            self._cascade_total += 1
            if event.cascade_triggered:
                self._cascade_escalated += 1

    def get_summary(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "total_cost_usd": round(self._total_cost, 6),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "routing_distribution": dict(self._routing_counts),
            "cascade": {
                "total": self._cascade_total,
                "escalated": self._cascade_escalated,
            },
            "recent_events": [
                {
                    "request_id": e.request_id,
                    "timestamp": e.timestamp.isoformat(),
                    "model": e.model,
                    "backend": e.backend,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cost_usd": round(e.estimated_cost_usd, 6),
                    "latency_ms": round(e.latency_ms, 1),
                    "routing_layer": e.routing_layer,
                    "routing_rule": e.routing_rule,
                }
                for e in list(self._events)[-20:]
            ],
        }
