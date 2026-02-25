"""Intelligibility — composite metric: NLI + BERTScore + phonetic similarity.

Models:
  - NLI: ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli (~1.4GB)
  - BERTScore: roberta-large (~1.4GB, via bert_score library)
Weights: 0.40 * NLI + 0.28 * BERTScore_F1 + 0.32 * phonetic_similarity
Higher is better, clipped to [0, 1].

Source: factual-consistency/FER/run_metrics.py L427-588
"""

from __future__ import annotations


def calculate_intelligibility(gt: str, hyp: str, **kwargs) -> float:
    """Compute intelligibility score. Higher is better."""
    raise NotImplementedError("Intelligibility not yet implemented")
