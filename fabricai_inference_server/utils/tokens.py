"""
Fast token estimation and text analysis for routing decisions.

These are heuristics, not exact counts. Good enough for routing
where we need sub-millisecond decisions, not billing accuracy.
"""

from __future__ import annotations

import re
from functools import lru_cache

_CODE_MARKERS = ("```", "def ", "class ", "function ", "import ", "const ")


def estimate_tokens(text: str) -> int:
    """Estimate token count. ~1.3 tokens per word for English."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def has_code_block(text: str) -> bool:
    """Detect code blocks or code-like patterns in text."""
    return any(marker in text for marker in _CODE_MARKERS)


@lru_cache(maxsize=4)
def _keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern:
    """Compile a regex alternation from keywords. Cached per unique set."""
    escaped = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(escaped, re.IGNORECASE)


def has_any_keyword(text: str, keywords: list[str]) -> bool:
    """Check if text contains any keyword. Uses compiled regex for speed."""
    if not keywords:
        return False
    pattern = _keyword_pattern(tuple(keywords))
    return pattern.search(text) is not None


def extract_last_user_content(messages: list) -> str:
    """Get the content of the last user message."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return (
                msg.content
                if isinstance(msg.content, str)
                else str(msg.content)
            )
    return ""
