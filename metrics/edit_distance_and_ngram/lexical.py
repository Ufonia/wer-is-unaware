"""Lexical edit-distance metrics: WER, CER, MER, WIL.

Thin wrappers around jiwer. Each function expects **pre-cleaned** text
(the public API in metrics/__init__.py handles cleaning).
"""

import jiwer


def calculate_wer(gt: str, hyp: str) -> float:
    """Word Error Rate."""
    try:
        return jiwer.wer(gt, hyp)
    except (ValueError, TypeError):
        return 1.0


def calculate_cer(gt: str, hyp: str) -> float:
    """Character Error Rate."""
    try:
        return jiwer.cer(gt, hyp)
    except (ValueError, TypeError):
        return 1.0


def calculate_mer(gt: str, hyp: str) -> float:
    """Match Error Rate."""
    try:
        return jiwer.mer(gt, hyp)
    except (ValueError, TypeError):
        return 1.0


def calculate_wil(gt: str, hyp: str) -> float:
    """Word Information Lost."""
    try:
        return jiwer.wil(gt, hyp)
    except (ValueError, TypeError):
        return 1.0
