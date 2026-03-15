"""
Tests for deep prompt analysis utilities.
"""

from fabricai_inference_server.utils.prompt_analysis import (
    detect_weak_domain,
    estimate_instruction_complexity,
    measure_reference_density,
)

# --- Instruction complexity ---


def test_single_step_low_complexity():
    score = estimate_instruction_complexity("What is the capital of France?")
    assert score < 0.2


def test_multi_step_high_complexity():
    score = estimate_instruction_complexity(
        "First compare the two approaches, then evaluate their "
        "tradeoffs, and finally propose a migration plan."
    )
    assert score >= 0.5


def test_multiple_questions():
    score = estimate_instruction_complexity(
        "What is X? How does it relate to Y? Why is Z important?"
    )
    assert score >= 0.2


def test_empty_text_complexity():
    assert estimate_instruction_complexity("") == 0.0


def test_imperative_verbs_detected():
    score = estimate_instruction_complexity(
        "Analyze this data, compare with historical trends, "
        "and synthesize a recommendation."
    )
    assert score >= 0.3


# --- Domain detection ---


def test_math_domain_detected():
    domain = detect_weak_domain(
        "Solve the integral of x^2 and prove the theorem."
    )
    assert domain == "math"


def test_legal_domain_detected():
    domain = detect_weak_domain(
        "The plaintiff filed pursuant to the jurisdiction statute."
    )
    assert domain == "legal"


def test_medical_domain_detected():
    domain = detect_weak_domain(
        "The diagnosis suggests treatment for these symptoms."
    )
    assert domain == "medical"


def test_no_domain_for_general_text():
    domain = detect_weak_domain(
        "Tell me about the history of programming languages."
    )
    assert domain is None


def test_single_marker_not_enough():
    # Needs 2+ markers to trigger
    domain = detect_weak_domain("Solve this equation.")
    assert domain is None


# --- Reference density ---


def test_no_references():
    density = measure_reference_density("What is Python?")
    assert density == 0.0


def test_heavy_references():
    density = measure_reference_density(
        "As I mentioned earlier, given the previous discussion, "
        "based on what you said, can you expand on that?"
    )
    assert density >= 0.5


def test_moderate_references():
    density = measure_reference_density(
        "Given the above, what do you think?"
    )
    assert 0.0 < density < 1.0
