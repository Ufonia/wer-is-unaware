"""Semantic Word Error Rate — DP-based weighted edit distance with semantic similarity.

Model: all-MiniLM-L6-v2 (~80MB, auto-downloaded). Own cache key (decoupled from SBERT).
"""

from __future__ import annotations


def calculate_semantic_wer(gt: str, hyp: str, **kwargs) -> float:
    raise NotImplementedError("Semantic-WER not yet implemented")
