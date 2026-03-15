"""
Router protocol and routing decision model.

The router examines an incoming request and decides which backend
and model should handle it. Routing decisions are logged for
telemetry and future classifier training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fabricai_inference_server.schemas.openai import ChatCompletionRequest


@dataclass
class RoutingDecision:
    backend_name: str
    model: str
    layer: str  # "explicit", "session_pin", "heuristic", "classifier", "cascade"
    rule_matched: str | None = None
    confidence: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "backend": self.backend_name,
            "model": self.model,
            "layer": self.layer,
            "rule_matched": self.rule_matched,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class Router(Protocol):
    async def route(
        self,
        request: ChatCompletionRequest,
        available_backends: dict,
    ) -> RoutingDecision: ...
