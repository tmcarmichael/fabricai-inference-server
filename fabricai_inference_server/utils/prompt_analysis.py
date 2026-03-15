"""
Deep prompt analysis for routing decisions.

Goes beyond keyword matching to assess instruction complexity,
domain classification, and context dependency. These signals
feed the heuristic router to distinguish tasks that genuinely
need a frontier model from those a local model can handle.
"""

from __future__ import annotations

import re
from functools import lru_cache

# --- Instruction complexity ---

_MULTI_STEP_MARKERS = (
    "then",
    "after that",
    "next",
    "finally",
    "first",
    "second",
    "third",
    "step 1",
    "step 2",
    "additionally",
    "furthermore",
    "followed by",
    "once you",
    "before that",
)

_IMPERATIVE_VERBS = (
    "compare",
    "evaluate",
    "analyze",
    "design",
    "implement",
    "create",
    "build",
    "optimize",
    "migrate",
    "refactor",
    "explain",
    "describe",
    "calculate",
    "derive",
    "prove",
    "plan",
    "propose",
    "assess",
    "critique",
    "synthesize",
)


@lru_cache(maxsize=2)
def _compile_markers(markers: tuple[str, ...]) -> re.Pattern:
    escaped = "|".join(re.escape(m) for m in markers)
    return re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)


def estimate_instruction_complexity(text: str) -> float:
    """Score 0.0-1.0 based on multi-step, multi-task signals.

    High complexity → cloud model likely needed.
    """
    if not text:
        return 0.0

    score = 0.0

    # Count imperative verbs (multiple tasks)
    verb_pattern = _compile_markers(_IMPERATIVE_VERBS)
    verb_hits = len(verb_pattern.findall(text))
    score += min(verb_hits * 0.12, 0.45)

    # Multi-step indicators
    step_pattern = _compile_markers(_MULTI_STEP_MARKERS)
    step_hits = len(step_pattern.findall(text))
    score += min(step_hits * 0.1, 0.3)

    # Multiple questions
    q_count = text.count("?")
    if q_count > 1:
        score += min((q_count - 1) * 0.1, 0.25)

    return min(round(score, 3), 1.0)


# --- Domain detection ---

# Domains where local models are known to underperform
_WEAK_LOCAL_DOMAINS: dict[str, tuple[str, ...]] = {
    "math": (
        "equation",
        "integral",
        "derivative",
        "theorem",
        "proof",
        "calculate",
        "matrix",
        "eigenvalue",
        "polynomial",
    ),
    "legal": (
        "pursuant to",
        "whereas",
        "jurisdiction",
        "statute",
        "liability",
        "plaintiff",
        "defendant",
        "arbitration",
    ),
    "medical": (
        "diagnosis",
        "symptoms",
        "treatment",
        "prognosis",
        "contraindication",
        "pathology",
        "dosage",
        "etiology",
    ),
}

_DOMAIN_PATTERNS: dict[str, re.Pattern] = {}


def _get_domain_pattern(domain: str) -> re.Pattern:
    if domain not in _DOMAIN_PATTERNS:
        markers = _WEAK_LOCAL_DOMAINS[domain]
        escaped = "|".join(re.escape(m) for m in markers)
        _DOMAIN_PATTERNS[domain] = re.compile(
            rf"\b(?:{escaped})\b", re.IGNORECASE
        )
    return _DOMAIN_PATTERNS[domain]


def detect_weak_domain(text: str) -> str | None:
    """Detect if text falls in a domain where local models struggle.

    Returns domain name or None.
    Requires 2+ marker matches to avoid false positives.
    """
    if not text:
        return None
    for domain in _WEAK_LOCAL_DOMAINS:
        pattern = _get_domain_pattern(domain)
        if len(pattern.findall(text)) >= 2:
            return domain
    return None


# --- Reference density ---

_BACK_REFERENCES = (
    "above",
    "previously",
    "earlier",
    "as i mentioned",
    "as mentioned",
    "given the",
    "based on what",
    "considering what",
    "the previous",
    "my earlier",
    "you said",
    "you mentioned",
    "we discussed",
    "from before",
)


@lru_cache(maxsize=1)
def _back_ref_pattern() -> re.Pattern:
    escaped = "|".join(re.escape(r) for r in _BACK_REFERENCES)
    return re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)


def measure_reference_density(text: str) -> float:
    """Score 0.0-1.0. How heavily does text reference prior context?

    High density → needs strong context window, favor cloud.
    """
    if not text:
        return 0.0
    hits = len(_back_ref_pattern().findall(text))
    return min(round(hits * 0.25, 3), 1.0)
