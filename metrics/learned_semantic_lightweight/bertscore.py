"""BERTScore — contextual embedding F1. Model: microsoft/deberta-xlarge-mnli (~2.7GB, auto-downloaded)."""

from __future__ import annotations


def calculate_bert_score(gt: str, hyp: str, **kwargs) -> float:
    raise NotImplementedError("BERTScore not yet implemented")
