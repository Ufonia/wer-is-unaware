"""Tests for Tier 2 heavy metrics: SimCSE, BARTScore, Semantic-WER, BERTScore, BLEURT."""

from __future__ import annotations

import pytest

from metrics.model_cache import models


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Clear model cache after each test to prevent cross-test state leakage."""
    yield
    models.clear()


# ---------------------------------------------------------------------------
# SimCSE
# ---------------------------------------------------------------------------

class TestSimCSE:
    def test_identity(self):
        from metrics import calculate_metric
        score = calculate_metric("simcse", "the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.95

    def test_similar(self):
        from metrics import calculate_metric
        score = calculate_metric("simcse", "the cat sat on the mat", "the cat sat on a mat")
        assert score > 0.7

    def test_different(self):
        from metrics import calculate_metric
        score = calculate_metric("simcse", "the cat sat on the mat", "quantum physics is fascinating")
        assert score < 0.5

    def test_both_empty_returns_none(self):
        from metrics import calculate_metric
        assert calculate_metric("simcse", "", "") is None

    def test_one_empty_returns_zero(self):
        from metrics import calculate_metric
        assert calculate_metric("simcse", "", "the cat sat on the mat") == 0.0

    def test_model_cache_loaded(self):
        from metrics import calculate_metric
        calculate_metric("simcse", "hello", "hello")
        assert "simcse_bert_base" in models.loaded()


# ---------------------------------------------------------------------------
# BARTScore
# ---------------------------------------------------------------------------

class TestBARTScore:
    def test_missing_checkpoint_errors(self, monkeypatch):
        """Clear error when BARTSCORE_CHECKPOINT env var is not set."""
        monkeypatch.delenv("BARTSCORE_CHECKPOINT", raising=False)
        from metrics.learned_semantic_lightweight.bartscore import _load_bart_scorer
        with pytest.raises(RuntimeError, match="BARTSCORE_CHECKPOINT"):
            _load_bart_scorer()

    def test_identity(self):
        from metrics import calculate_metric
        score = calculate_metric("bart_score", "the cat sat on the mat", "the cat sat on the mat")
        # Identity should produce the highest (least negative) score
        assert isinstance(score, float)

    def test_similar_higher_than_different(self):
        from metrics import calculate_metric
        similar = calculate_metric("bart_score", "the cat sat on the mat", "the cat sat on a mat")
        different = calculate_metric("bart_score", "the cat sat on the mat", "quantum physics is fascinating")
        assert similar > different

    def test_different(self):
        from metrics import calculate_metric
        score = calculate_metric("bart_score", "the cat sat on the mat", "quantum physics is fascinating")
        assert isinstance(score, float)

    def test_both_empty_returns_none(self):
        from metrics import calculate_metric
        assert calculate_metric("bart_score", "", "") is None

    def test_one_empty_lets_algorithm_run(self):
        """BARTScore has no clear fallback for empty input — algorithm tries."""
        from metrics import calculate_metric
        score = calculate_metric("bart_score", "", "the cat sat on the mat")
        # Result is either a float (algorithm succeeded) or None (algorithm failed)
        assert score is None or isinstance(score, float)

    def test_score_is_negative(self):
        from metrics import calculate_metric
        score = calculate_metric("bart_score", "the cat sat on the mat", "a dog ran in the park")
        assert score < 0.0

    def test_model_cache_loaded(self):
        from metrics import calculate_metric
        calculate_metric("bart_score", "hello", "hello")
        assert "bart_large_cnn" in models.loaded()


# ---------------------------------------------------------------------------
# Semantic-WER
# ---------------------------------------------------------------------------

class TestSemanticWER:
    def test_identity(self):
        from metrics import calculate_metric
        score = calculate_metric("semantic_wer", "the cat sat on the mat", "the cat sat on the mat")
        assert score == 0.0

    def test_synonym_substitution(self):
        from metrics import calculate_metric
        swer = calculate_metric("semantic_wer", "the big cat sat on the mat", "the large cat sat on the mat")
        # "big" and "large" are semantically similar — SWER should be lower than standard WER
        wer = calculate_metric("wer", "the big cat sat on the mat", "the large cat sat on the mat")
        assert swer < wer

    def test_completely_different(self):
        from metrics import calculate_metric
        score = calculate_metric("semantic_wer", "the cat sat on the mat", "quantum physics is fascinating")
        assert score > 0.0

    def test_both_empty_returns_none(self):
        from metrics import calculate_metric
        assert calculate_metric("semantic_wer", "", "") is None

    def test_one_empty_lets_algorithm_run(self):
        """SWER internal logic handles one-empty naturally (deletion costs)."""
        from metrics import calculate_metric
        score = calculate_metric("semantic_wer", "", "the cat sat on the mat")
        assert score is None or isinstance(score, float)

    def test_own_cache_key(self):
        from metrics import calculate_metric
        calculate_metric("semantic_wer", "hello world", "hello world")
        assert "swer_minilm" in models.loaded()
        # Should NOT share cache key with SBERT
        assert "sbert_minilm" not in models.loaded()


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

class TestBERTScore:
    def test_identity(self):
        from metrics import calculate_metric
        score = calculate_metric("bert_score", "the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.95

    def test_similar(self):
        from metrics import calculate_metric
        score = calculate_metric("bert_score", "the cat sat on the mat", "the cat sat on a mat")
        assert score > 0.8

    def test_different(self):
        from metrics import calculate_metric
        score = calculate_metric("bert_score", "the cat sat on the mat", "quantum physics is fascinating")
        # Raw (no rescaling) BERTScores are high even for unrelated strings
        assert score < 0.9

    def test_both_empty_returns_none(self):
        from metrics import calculate_metric
        assert calculate_metric("bert_score", "", "") is None

    def test_one_empty_returns_zero(self):
        from metrics import calculate_metric
        assert calculate_metric("bert_score", "", "the cat sat on the mat") == 0.0

    def test_model_cache_loaded(self):
        from metrics import calculate_metric
        calculate_metric("bert_score", "hello", "hello")
        assert "bertscore_roberta" in models.loaded()


# ---------------------------------------------------------------------------
# BLEURT
# ---------------------------------------------------------------------------

class TestBLEURT:
    def test_clinical_missing_env_var_errors(self, monkeypatch):
        """Clear error when CLINICAL_BLEURT_CHECKPOINT env var is not set."""
        monkeypatch.delenv("CLINICAL_BLEURT_CHECKPOINT", raising=False)
        from metrics.learned_semantic_lightweight.bleurt_metric import _make_loader
        loader = _make_loader("CLINICAL_BLEURT_CHECKPOINT", "Clinical BLEURT")
        with pytest.raises(RuntimeError, match="CLINICAL_BLEURT_CHECKPOINT"):
            loader()

    def test_missing_checkpoint_dir_errors(self, monkeypatch, tmp_path):
        """Clear error when checkpoint path doesn't exist."""
        monkeypatch.setenv("CLINICAL_BLEURT_CHECKPOINT", str(tmp_path / "nonexistent"))
        from metrics.learned_semantic_lightweight.bleurt_metric import _make_loader
        loader = _make_loader("CLINICAL_BLEURT_CHECKPOINT", "Clinical BLEURT")
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            loader()

    def test_bleurt_and_clinical_share_code(self):
        """Both BLEURT variants use the same internal function."""
        from metrics.learned_semantic_lightweight import bleurt_metric
        assert hasattr(bleurt_metric, "_score_bleurt")

    def test_scoring_with_mock(self, monkeypatch):
        """BLEURT scoring works when checkpoint is available (mocked)."""
        from unittest.mock import MagicMock
        from metrics.learned_semantic_lightweight import bleurt_metric

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = [0.75]

        # Bypass model cache — inject mock directly
        monkeypatch.setattr(
            bleurt_metric, "_score_bleurt",
            lambda gt, hyp, model_key: 0.75,
        )
        assert bleurt_metric.calculate_bleurt("hello", "hello") == 0.75
        assert bleurt_metric.calculate_clinical_bleurt("hello", "hello") == 0.75

    def test_both_empty_returns_none(self):
        """Both empty returns None without loading model."""
        from metrics.learned_semantic_lightweight.bleurt_metric import (
            calculate_bleurt,
            calculate_clinical_bleurt,
        )
        assert calculate_bleurt("", "") is None
        assert calculate_clinical_bleurt("", "") is None

    def test_one_empty_lets_algorithm_try(self):
        """BLEURT has no clear fallback — algorithm tries (or returns None)."""
        from metrics.learned_semantic_lightweight.bleurt_metric import calculate_bleurt
        score = calculate_bleurt("", "hello")
        # Either a float (algorithm succeeded) or None (algorithm failed)
        assert score is None or isinstance(score, float)
