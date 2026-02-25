"""Heval — hybrid ASR evaluation metric with semantic distance weighting.

Model: roberta-base (~500MB, auto-downloaded from HuggingFace).
Algorithm: Extract keywords by semantic distance threshold, count word errors
by class (keyword vs non-keyword), combine with overall semantic distance.
Lower is better (like WER).

Source: factual-consistency/FER/run_metrics.py L590-759
"""

from __future__ import annotations


def calculate_heval(gt: str, hyp: str, **kwargs) -> float:
    """Compute Heval score. Lower is better."""
    raise NotImplementedError("Heval not yet implemented")
