"""BLEURT and Clinical BLEURT — TensorFlow-based metrics with manual checkpoint download.

Checkpoint paths configured via environment variables:
- BLEURT_CHECKPOINT: path to standard BLEURT checkpoint directory
- CLINICAL_BLEURT_CHECKPOINT: path to Clinical BLEURT checkpoint directory
"""

from __future__ import annotations


def calculate_bleurt(gt: str, hyp: str, **kwargs) -> float:
    raise NotImplementedError("BLEURT not yet implemented")


def calculate_clinical_bleurt(gt: str, hyp: str, **kwargs) -> float:
    raise NotImplementedError("Clinical BLEURT not yet implemented")
