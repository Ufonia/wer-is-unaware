"""Tests for Tier 3 metrics: SeMaScore, Intelligibility, Heval."""

from __future__ import annotations

import pytest

from metrics.model_cache import models


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Clear model cache after each test to prevent cross-test state leakage."""
    yield
    models.clear()


# ---------------------------------------------------------------------------
# Heval
# ---------------------------------------------------------------------------

class TestHeval:
    def test_identity(self):
        from metrics import calculate_metric
        score = calculate_metric("heval", "the cat sat on the mat", "the cat sat on the mat")
        assert score < 0.05  # identical → near 0

    def test_different(self):
        from metrics import calculate_metric
        score = calculate_metric("heval", "the cat sat on the mat", "quantum physics is fascinating")
        assert score > 0.0

    def test_empty_string(self):
        from metrics import calculate_metric
        score = calculate_metric("heval", "", "the cat sat on the mat")
        assert score == 1.0  # fallback

    def test_model_cache_loaded(self):
        from metrics import calculate_metric
        calculate_metric("heval", "hello world", "hello world")
        assert "heval_roberta" in models.loaded()

    def test_lower_is_better(self):
        from metrics import calculate_metric
        similar = calculate_metric("heval", "the cat sat on the mat", "the cat sat on a mat")
        different = calculate_metric("heval", "the cat sat on the mat", "quantum physics is fascinating")
        assert similar < different


# ---------------------------------------------------------------------------
# Intelligibility
# ---------------------------------------------------------------------------

class TestIntelligibility:
    def test_identity(self):
        from metrics import calculate_metric
        score = calculate_metric("intelligibility", "the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_similar(self):
        from metrics import calculate_metric
        score = calculate_metric("intelligibility", "the cat sat on the mat", "the cat sat on a mat")
        assert score > 0.7

    def test_different(self):
        from metrics import calculate_metric
        score = calculate_metric("intelligibility", "the cat sat on the mat", "quantum physics is fascinating")
        assert score < 0.7

    def test_empty_string(self):
        from metrics import calculate_metric
        score = calculate_metric("intelligibility", "", "the cat sat on the mat")
        assert score == 0.0  # fallback

    def test_model_cache_loaded(self):
        from metrics import calculate_metric
        calculate_metric("intelligibility", "hello world", "hello world")
        assert "intelligibility_nli" in models.loaded()


# ---------------------------------------------------------------------------
# SeMaScore
# ---------------------------------------------------------------------------

class TestSeMaScore:
    def test_identity(self):
        from metrics import calculate_metric
        score = calculate_metric("semascore", "the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_similar(self):
        from metrics import calculate_metric
        score = calculate_metric("semascore", "the cat sat on the mat", "the cat sat on a mat")
        assert score > 0.5

    def test_different(self):
        from metrics import calculate_metric
        score = calculate_metric("semascore", "the cat sat on the mat", "quantum physics is fascinating")
        assert score < 0.5

    def test_empty_string(self):
        from metrics import calculate_metric
        score = calculate_metric("semascore", "", "the cat sat on the mat")
        assert score == 0.0  # fallback

    def test_model_cache_loaded(self):
        from metrics import calculate_metric
        calculate_metric("semascore", "hello world", "hello world")
        assert "semascore_deberta" in models.loaded()
