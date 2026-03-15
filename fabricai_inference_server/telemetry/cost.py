"""
Per-model cost estimation.

Prices in USD per 1K tokens. Local models are free.
These are approximate. Update as pricing changes.
"""

# (input_cost_per_1k, output_cost_per_1k)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-20250514": (0.015, 0.075),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-haiku-4-5-20251001": (0.0008, 0.004),
    # OpenAI
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
    # Google
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-2.5-pro": (0.00125, 0.010),
    "gemini-2.5-flash": (0.00015, 0.0006),
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float:
    """Estimate USD cost for a request. Returns 0.0 for local/unknown models."""
    costs = MODEL_COSTS.get(model, (0.0, 0.0))
    return (input_tokens / 1000 * costs[0]) + (
        output_tokens / 1000 * costs[1]
    )
