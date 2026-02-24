"""Tests for all 15 Tier 1 metrics."""

import pytest
from metrics import calculate_metric, calculate_all_metrics, list_metrics, get_metric_info

GT = "the cat sat on the mat"
HYP = "the cat sat on a mat"

TIER_1_METRICS = [
    "wer", "cer", "mer", "wil",
    "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "rouge_1", "rouge_2", "rouge_l", "rouge_w",
    "chrf", "chrf_plus_plus", "meteor",
]


class TestListMetrics:
    def test_list_metrics_has_tier1(self):
        tiers = list_metrics()
        assert "edit_distance_and_ngram" in tiers
        assert len(tiers["edit_distance_and_ngram"]) == 15

    def test_all_tier1_names_present(self):
        tiers = list_metrics()
        tier1_names = tiers["edit_distance_and_ngram"]
        for name in TIER_1_METRICS:
            assert name in tier1_names, f"{name} missing from tier1"


class TestMetricInfo:
    def test_wer_info(self):
        info = get_metric_info("wer")
        assert info.name == "wer"
        assert info.tier == "edit_distance_and_ngram"
        assert info.higher_is_better is False

    def test_bleu_1_info(self):
        info = get_metric_info("bleu_1")
        assert info.higher_is_better is True

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            get_metric_info("nonexistent_metric")


class TestKnownPair:
    """One substitution: 'the' → 'a'."""

    def test_wer(self):
        score = calculate_metric("wer", GT, HYP)
        assert isinstance(score, float)
        assert 0.15 < score < 0.20

    def test_cer(self):
        score = calculate_metric("cer", GT, HYP)
        assert isinstance(score, float)
        assert 0.0 < score < 0.2

    def test_mer(self):
        score = calculate_metric("mer", GT, HYP)
        assert isinstance(score, float)
        assert 0.0 < score < 0.3

    def test_wil(self):
        score = calculate_metric("wil", GT, HYP)
        assert isinstance(score, float)
        assert 0.0 < score < 0.5

    def test_bleu_1(self):
        score = calculate_metric("bleu_1", GT, HYP)
        assert isinstance(score, float)
        assert 0.7 < score <= 1.0

    def test_bleu_2(self):
        score = calculate_metric("bleu_2", GT, HYP)
        assert isinstance(score, float)
        assert 0.5 < score <= 1.0

    def test_bleu_3(self):
        score = calculate_metric("bleu_3", GT, HYP)
        assert isinstance(score, float)
        assert 0.3 < score <= 1.0

    def test_bleu_4(self):
        score = calculate_metric("bleu_4", GT, HYP)
        assert isinstance(score, float)
        assert 0.2 < score <= 1.0

    def test_rouge_1(self):
        score = calculate_metric("rouge_1", GT, HYP)
        assert isinstance(score, float)
        assert 0.7 < score <= 1.0

    def test_rouge_2(self):
        score = calculate_metric("rouge_2", GT, HYP)
        assert isinstance(score, float)
        assert 0.5 < score <= 1.0

    def test_rouge_l(self):
        score = calculate_metric("rouge_l", GT, HYP)
        assert isinstance(score, float)
        assert 0.7 < score <= 1.0

    def test_rouge_w(self):
        score = calculate_metric("rouge_w", GT, HYP)
        assert isinstance(score, float)
        assert 0.7 < score <= 1.0

    def test_chrf(self):
        score = calculate_metric("chrf", GT, HYP)
        assert isinstance(score, float)
        assert 0.5 < score <= 1.0

    def test_chrf_plus_plus(self):
        score = calculate_metric("chrf_plus_plus", GT, HYP)
        assert isinstance(score, float)
        assert 0.5 < score <= 1.0

    def test_meteor(self):
        score = calculate_metric("meteor", GT, HYP)
        assert isinstance(score, float)
        assert 0.5 < score <= 1.0


class TestPerfectMatch:

    def test_wer_zero(self):
        assert calculate_metric("wer", GT, GT) == 0.0

    def test_cer_zero(self):
        assert calculate_metric("cer", GT, GT) == 0.0

    def test_bleu_4_one(self):
        score = calculate_metric("bleu_4", GT, GT)
        assert score > 0.99

    def test_rouge_l_one(self):
        score = calculate_metric("rouge_l", GT, GT)
        assert score > 0.99

    def test_meteor_one(self):
        score = calculate_metric("meteor", GT, GT)
        assert score > 0.99


class TestCompleteMismatch:
    MISMATCH_GT = "the cat sat on the mat"
    MISMATCH_HYP = "yellow purple green blue orange red"

    def test_wer_high(self):
        score = calculate_metric("wer", self.MISMATCH_GT, self.MISMATCH_HYP)
        assert score >= 1.0

    def test_bleu_1_low(self):
        score = calculate_metric("bleu_1", self.MISMATCH_GT, self.MISMATCH_HYP)
        assert score < 0.1

    def test_rouge_l_low(self):
        score = calculate_metric("rouge_l", self.MISMATCH_GT, self.MISMATCH_HYP)
        assert score < 0.1


class TestCalculateAllMetrics:
    def test_tier_returns_all_15(self):
        scores = calculate_all_metrics(GT, HYP, tier="edit_distance_and_ngram")
        assert len(scores) == 15
        for name in TIER_1_METRICS:
            assert name in scores
            assert isinstance(scores[name], float)

    def test_specific_metrics(self):
        scores = calculate_all_metrics(GT, HYP, metrics=["wer", "bleu_1"])
        assert len(scores) == 2
        assert "wer" in scores
        assert "bleu_1" in scores

    def test_tier_and_metrics_mutually_exclusive(self):
        with pytest.raises(ValueError):
            calculate_all_metrics(GT, HYP, tier="edit_distance_and_ngram", metrics=["wer"])

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError):
            calculate_all_metrics(GT, HYP, tier="nonexistent_tier")

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            calculate_all_metrics(GT, HYP, metrics=["nonexistent_metric"])
