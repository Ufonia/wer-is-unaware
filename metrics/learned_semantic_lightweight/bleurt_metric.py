"""BLEURT and Clinical BLEURT — learned evaluation metric (TensorFlow-based).

Checkpoint paths configured via environment variables:
- BLEURT_CHECKPOINT: path to standard BLEURT checkpoint directory
- CLINICAL_BLEURT_CHECKPOINT: path to Clinical BLEURT checkpoint directory

Requires the ``bleurt`` optional dependency group: uv sync --extra bleurt
"""

from __future__ import annotations

import os

from metrics.model_cache import models

_BLEURT_KEY = "bleurt_scorer"
_CLINICAL_BLEURT_KEY = "clinical_bleurt_scorer"


def _make_loader(env_var: str, label: str):
    """Create a loader function that reads a checkpoint path from an env var."""

    def _load():
        checkpoint_path = os.environ.get(env_var)
        if not checkpoint_path:
            raise RuntimeError(
                f"{env_var} environment variable is not set. "
                f"Set it to the path of your {label} checkpoint directory. "
                f"See .env.example for details."
            )
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"{label} checkpoint directory not found: {checkpoint_path}"
            )

        from bleurt import score as bleurt_score

        return bleurt_score.BleurtScorer(checkpoint=checkpoint_path)

    return _load


models.register_loader(_BLEURT_KEY, _make_loader("BLEURT_CHECKPOINT", "BLEURT"))
models.register_loader(
    _CLINICAL_BLEURT_KEY,
    _make_loader("CLINICAL_BLEURT_CHECKPOINT", "Clinical BLEURT"),
)


def _score_bleurt(gt: str, hyp: str, model_key: str) -> float:
    """Score a single (gt, hyp) pair with a BLEURT checkpoint."""
    if not gt or not hyp:
        return 0.0
    scorer = models.get(model_key)
    scores = scorer.score(references=[gt], candidates=[hyp])
    return scores[0]


def calculate_bleurt(gt: str, hyp: str, **kwargs) -> float:
    """BLEURT score for a single (gt, hyp) pair."""
    return _score_bleurt(gt, hyp, _BLEURT_KEY)


def calculate_clinical_bleurt(gt: str, hyp: str, **kwargs) -> float:
    """Clinical BLEURT score for a single (gt, hyp) pair."""
    return _score_bleurt(gt, hyp, _CLINICAL_BLEURT_KEY)
