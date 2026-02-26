"""Lexical edit-distance metrics: WER, CER, MER, WIL.

Thin wrappers around jiwer. Each function expects **pre-cleaned** text
(the public API in metrics/__init__.py handles cleaning).
"""

from __future__ import annotations

from typing import Optional

import jiwer


def calculate_wer(gt: str, hyp: str) -> Optional[float]:
    """Word Error Rate."""
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 1.0
    return jiwer.wer(gt, hyp)


def calculate_cer(gt: str, hyp: str) -> Optional[float]:
    """Character Error Rate."""
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 1.0
    return jiwer.cer(gt, hyp)


def calculate_mer(gt: str, hyp: str) -> Optional[float]:
    """Match Error Rate."""
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 1.0
    return jiwer.mer(gt, hyp)


def calculate_wil(gt: str, hyp: str) -> Optional[float]:
    """Word Information Lost."""
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 1.0
    return jiwer.wil(gt, hyp)
