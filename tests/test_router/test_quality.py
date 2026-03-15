"""
Tests for the quality scorer.
"""

from fabricai_inference_server.router.quality import QualityScorer


def test_good_response():
    scorer = QualityScorer()
    result = scorer.score(
        "The quick brown fox jumps over the lazy dog. "
        "This is a complete and well-formed response."
    )
    assert result.passed is True
    assert result.score > 0.8
    assert len(result.reasons) == 0


def test_empty_response():
    scorer = QualityScorer()
    result = scorer.score("")
    assert result.passed is False
    assert result.score == 0.0
    assert "empty_response" in result.reasons


def test_very_short_response():
    scorer = QualityScorer(min_response_length=20)
    result = scorer.score("Yes.")
    assert result.passed is False
    assert "very_short_response" in result.reasons


def test_high_repetition():
    scorer = QualityScorer()
    # More than 50% repeated words
    result = scorer.score("the the the the the the other word.")
    assert "high_repetition" in result.reasons


def test_word_degeneration():
    scorer = QualityScorer()
    result = scorer.score(
        "This is a normal start but then loop loop loop loop."
    )
    assert "word_degeneration" in result.reasons


def test_incomplete_sentence():
    scorer = QualityScorer()
    result = scorer.score(
        "This response starts well but ends abruptly and the"
    )
    assert "incomplete_sentence" in result.reasons


def test_uncertainty_detected():
    scorer = QualityScorer()
    result = scorer.score(
        "I'm not sure about this, but I think the answer is 42."
    )
    assert any("uncertainty" in r for r in result.reasons)


def test_multiple_penalties_stack():
    scorer = QualityScorer(min_response_length=20)
    # Short + incomplete + uncertain → 1.0 - 0.5 - 0.15 - 0.2 = 0.15
    result = scorer.score("I don't know")
    assert result.score < 0.5
    assert len(result.reasons) >= 3


def test_custom_threshold():
    scorer = QualityScorer(quality_threshold=0.9)
    # Slightly imperfect response (incomplete sentence)
    result = scorer.score(
        "A reasonable response with enough words but no ending punctuation"
    )
    # 1.0 - 0.15 (incomplete) = 0.85 < 0.9 threshold
    assert result.passed is False
