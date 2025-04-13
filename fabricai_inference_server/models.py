"""
models.py

Validation on inference.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class InferenceRequest(BaseModel):
    """
    Pydantic model to validate the fields needed for inference.
    """

    prompt: str = Field(..., min_length=1, description="Prompt text to generate from.")
    max_tokens: int = Field(128, gt=0, le=2048, description="Maximum tokens to generate.")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature for generation."
    )
    top_p: float = Field(
        0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter."
    )
    repeat_penalty: float = Field(
        1.1,
        ge=1.0,
        le=2.0,
        description="Penalty for repeated tokens (1.0 means no penalty).",
    )
    stop: Optional[List[str]] = Field(
        default=None, description="Optional list of stop sequences."
    )
    session_id: Optional[str] = Field(
        None, description="Optional session ID for conversation tracking."
    )
