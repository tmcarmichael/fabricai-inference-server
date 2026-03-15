"""
Heuristic router: rule-based routing with deep prompt analysis.

Applies rules in priority order. First match wins. Rules use token
counting, code detection, keyword matching, instruction complexity,
domain detection, and context reference density. Confidence is
optionally adjusted by adaptive thresholds learned from cascade data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fabricai_inference_server.router import RoutingDecision
from fabricai_inference_server.router.adaptive import AdaptiveThresholds
from fabricai_inference_server.schemas.openai import ChatCompletionRequest
from fabricai_inference_server.utils.prompt_analysis import (
    detect_weak_domain,
    estimate_instruction_complexity,
    measure_reference_density,
)
from fabricai_inference_server.utils.tokens import (
    estimate_tokens,
    extract_last_user_content,
    has_any_keyword,
    has_code_block,
)

logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    default_backend: str = "ollama"
    cloud_backend: str = "anthropic"
    trivial_max_tokens: int = 50
    long_context_min_tokens: int = 4000
    deep_conversation_turns: int = 10
    cloud_keywords: list[str] = field(
        default_factory=lambda: [
            "refactor",
            "review",
            "debug",
            "analyze",
            "explain",
            "compare",
            "evaluate",
            "architect",
            "optimize",
            "critique",
        ]
    )
    local_keywords: list[str] = field(
        default_factory=lambda: [
            "summarize",
            "extract",
            "reformat",
            "translate",
            "classify",
            "list",
            "convert",
            "format",
            "count",
            "sort",
        ]
    )

    # Prompt analysis thresholds
    complexity_cloud_threshold: float = 0.5
    reference_density_cloud_threshold: float = 0.5

    # Cascade settings
    cascade_enabled: bool = True
    cascade_confidence_threshold: float = 0.7
    cascade_quality_threshold: float = 0.6

    @classmethod
    def from_yaml(cls, path: str | Path) -> RoutingConfig:
        path = Path(path)
        if not path.exists():
            logger.info(
                "Routing config %s not found, using defaults.", path
            )
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {
            k: v
            for k, v in data.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**valid)


class HeuristicRouter:
    """Rule-based router with deep prompt analysis and adaptive tuning."""

    def __init__(
        self,
        config: RoutingConfig,
        available_backends: dict,
        adaptive: AdaptiveThresholds | None = None,
    ):
        self.config = config
        self.backends = available_backends
        self.adaptive = adaptive

    def _resolve_local(self) -> tuple[str, str]:
        if "ollama" in self.backends:
            backend = self.backends["ollama"]
            model = getattr(backend, "_default_model", "auto")
            return "ollama", model
        name = next(iter(self.backends))
        return name, "auto"

    def _resolve_cloud(self) -> tuple[str, str]:
        preferred = self.config.cloud_backend
        if preferred in self.backends:
            return preferred, "auto"
        for name in ("anthropic", "openai", "google"):
            if name in self.backends:
                return name, "auto"
        return self._resolve_local()

    def _decide(
        self,
        target: str,
        rule: str,
        reason: str,
        confidence: float,
    ) -> RoutingDecision:
        if target == "cloud":
            backend, model = self._resolve_cloud()
        else:
            backend, model = self._resolve_local()

        # Adaptive adjustment if we have enough data
        if self.adaptive:
            confidence = self.adaptive.adjust_confidence(
                rule, confidence
            )

        return RoutingDecision(
            backend_name=backend,
            model=model,
            layer="heuristic",
            rule_matched=rule,
            confidence=confidence,
            reason=reason,
        )

    async def route(
        self,
        request: ChatCompletionRequest,
        available_backends: dict | None = None,
    ) -> RoutingDecision:
        """Apply heuristic rules in priority order. First match wins."""
        if available_backends:
            self.backends = available_backends

        user_content = extract_last_user_content(request.messages)
        token_count = estimate_tokens(user_content)
        code_present = has_code_block(user_content)
        message_count = len(request.messages)
        cfg = self.config

        # --- Fast-path rules (cheapest checks first) ---

        # Rule 1: Trivial query, short, no code
        if token_count < cfg.trivial_max_tokens and not code_present:
            return self._decide(
                "local",
                "trivial_query",
                f"Short query ({token_count} tokens, no code)",
                0.95,
            )

        # Rule 2: Long context. Needs a strong model
        if token_count > cfg.long_context_min_tokens:
            return self._decide(
                "cloud",
                "long_context",
                f"Long input ({token_count} tokens)",
                0.90,
            )

        # Rule 3: Code + action keyword. Needs reasoning
        if code_present and has_any_keyword(
            user_content, cfg.cloud_keywords
        ):
            return self._decide(
                "cloud",
                "code_analysis",
                "Code with cloud action keyword",
                0.90,
            )

        # Rule 4: Simple task keyword, local handles fine
        if has_any_keyword(user_content, cfg.local_keywords):
            return self._decide(
                "local",
                "simple_task",
                "Simple task keyword detected",
                0.85,
            )

        # Rule 5: Code without action keyword, local can try
        if code_present:
            return self._decide(
                "local",
                "code_no_action",
                "Code present but no complex action keyword",
                0.65,
            )

        # --- Deep analysis rules (more expensive, fewer hits) ---

        # Rule 6: High instruction complexity
        complexity = estimate_instruction_complexity(user_content)
        if complexity >= cfg.complexity_cloud_threshold:
            return self._decide(
                "cloud",
                "high_complexity",
                f"Instruction complexity {complexity:.2f}",
                0.85,
            )

        # Rule 7: Weak local domain (math, legal, medical)
        domain = detect_weak_domain(user_content)
        if domain:
            return self._decide(
                "cloud",
                "weak_domain",
                f"Domain '{domain}', local models underperform",
                0.85,
            )

        # Rule 8: Heavy back-references to prior context
        ref_density = measure_reference_density(user_content)
        if ref_density >= cfg.reference_density_cloud_threshold:
            return self._decide(
                "cloud",
                "high_reference_density",
                f"Reference density {ref_density:.2f}",
                0.80,
            )

        # Rule 9: Deep conversation, cloud for coherence
        if message_count > cfg.deep_conversation_turns:
            return self._decide(
                "cloud",
                "deep_conversation",
                f"Long conversation ({message_count} messages)",
                0.85,
            )

        # Default: low confidence, cascade eligible
        return self._decide(
            "local",
            "default",
            "No heuristic rule matched, defaulting to local",
            0.50,
        )
