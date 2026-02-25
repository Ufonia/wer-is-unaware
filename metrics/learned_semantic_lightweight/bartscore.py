"""BARTScore — average bidirectional log-likelihood. Model: facebook/bart-large-cnn (~1.6GB, auto-downloaded)."""

from __future__ import annotations


def calculate_bart_score(gt: str, hyp: str, **kwargs) -> float:
    raise NotImplementedError("BARTScore not yet implemented")
