"""Tests for Tier 2 lightweight metrics: SBERT similarity and NLI mutual entailment."""

import pytest

from metrics import calculate_metric, list_metrics, get_metric_info
from metrics.model_cache import models


@pytest.fixture(autouse=True)
def _clear_model_cache():
    yield
    models.clear()


class TestSBERTSimilarity:
    def test_identical_strings(self):
        score = calculate_metric("sbert_similarity", "the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.95

    def test_similar_strings(self):
        score = calculate_metric("sbert_similarity", "the cat sat on the mat", "the cat sat on a mat")
        assert score > 0.8

    def test_different_strings(self):
        score = calculate_metric("sbert_similarity", "the cat sat on the mat", "bananas are yellow fruit")
        assert score < 0.5

    def test_both_empty_returns_none(self):
        score = calculate_metric("sbert_similarity", "", "")
        assert score is None

    def test_empty_gt_returns_zero(self):
        score = calculate_metric("sbert_similarity", "", "some text")
        assert score == 0.0

    def test_empty_hyp_returns_zero(self):
        score = calculate_metric("sbert_similarity", "some text", "")
        assert score == 0.0

    def test_returns_float(self):
        score = calculate_metric("sbert_similarity", "hello world", "hello there")
        assert isinstance(score, float)


class TestNLIXSmall:
    def test_paraphrase_high_entailment(self):
        score = calculate_metric(
            "nli_xsmall",
            "the patient has a severe headache",
            "the patient is suffering from a bad headache",
        )
        assert score > 0.5

    def test_contradiction_low_entailment(self):
        score = calculate_metric(
            "nli_xsmall",
            "the patient is healthy",
            "the patient is critically ill",
        )
        assert score < 0.3

    def test_both_empty_returns_none(self):
        score = calculate_metric("nli_xsmall", "", "")
        assert score is None

    def test_empty_returns_zero(self):
        score = calculate_metric("nli_xsmall", "", "some text")
        assert score == 0.0

    def test_returns_float(self):
        score = calculate_metric("nli_xsmall", "hello", "hello")
        assert isinstance(score, float)


class TestAllNLISizes:
    @pytest.mark.parametrize("metric_name", ["nli_xsmall", "nli_base", "nli_large"])
    def test_returns_float(self, metric_name):
        score = calculate_metric(metric_name, "the cat sat on the mat", "the cat sat on a mat")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestModelCache:
    def test_loaded_after_sbert(self):
        calculate_metric("sbert_similarity", "hello", "hello")
        assert "sbert_minilm" in models.loaded()

    def test_loaded_after_nli(self):
        calculate_metric("nli_xsmall", "hello", "hello")
        assert "nli_xsmall" in models.loaded()

    def test_clear_empties_cache(self):
        calculate_metric("sbert_similarity", "hello", "hello")
        assert len(models.loaded()) > 0
        models.clear()
        assert len(models.loaded()) == 0

    def test_unload_single(self):
        calculate_metric("sbert_similarity", "hello", "hello")
        assert "sbert_minilm" in models.loaded()
        models.unload("sbert_minilm")
        assert "sbert_minilm" not in models.loaded()


class TestRegistryIntegration:
    def test_tier2_metrics_in_list(self):
        all_metrics = list_metrics()
        assert "learned_semantic_lightweight" in all_metrics
        tier2 = all_metrics["learned_semantic_lightweight"]
        for name in ["sbert_similarity", "nli_xsmall", "nli_base", "nli_large"]:
            assert name in tier2

    def test_metric_info_sbert(self):
        info = get_metric_info("sbert_similarity")
        assert info.tier == "learned_semantic_lightweight"
        assert info.higher_is_better is True
        assert info.extra == "learned-semantic"

    def test_metric_info_nli(self):
        info = get_metric_info("nli_xsmall")
        assert info.tier == "learned_semantic_lightweight"
        assert info.higher_is_better is True
        assert info.extra == "learned-semantic"
