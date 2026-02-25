"""SeMaScore — importance-weighted semantic similarity with edit alignment.

Model: microsoft/deberta-large-mnli (~1.7GB, auto-downloaded), pruned to 18 layers.
Algorithm: Character-level edit alignment -> segment mapping -> importance-weighted
semantic similarity * character-level MER.
Higher is better, range [0, 1].

Source: factual-consistency/FER/run_metrics.py L80-424
"""

from __future__ import annotations


def calculate_semascore(gt: str, hyp: str, **kwargs) -> float:
    """Compute SeMaScore. Higher is better."""
    raise NotImplementedError("SeMaScore not yet implemented")
