"""NLI mutual-entailment metrics (xsmall / base / large).

Models (auto-downloaded from HuggingFace Hub):
- cross-encoder/nli-deberta-v3-xsmall (~160MB)
- cross-encoder/nli-deberta-v3-base   (~740MB)
- cross-encoder/nli-deberta-v3-large  (~1.7GB)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import CrossEncoder

from metrics.model_cache import models

_MODELS = {
    "nli_xsmall": "cross-encoder/nli-deberta-v3-xsmall",
    "nli_base": "cross-encoder/nli-deberta-v3-base",
    "nli_large": "cross-encoder/nli-deberta-v3-large",
}

for _key, _hf_id in _MODELS.items():
    models.register_loader(_key, lambda hf_id=_hf_id: CrossEncoder(hf_id, max_length=512))


def _softmax_entailment(logits: np.ndarray) -> float:
    """Softmax over [contradiction, entailment, neutral], return entailment prob."""
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    return float(probs[0][1])


def _mutual_entailment(gt: str, hyp: str, model_key: str) -> Optional[float]:
    """min(P(gt entails hyp), P(hyp entails gt)) — both directions must be high."""
    if not gt and not hyp:
        return None
    if not gt or not hyp:
        return 0.0

    model = models.get(model_key)
    pe_fwd = _softmax_entailment(model.predict([(gt, hyp)], convert_to_numpy=True))
    pe_bwd = _softmax_entailment(model.predict([(hyp, gt)], convert_to_numpy=True))
    return min(pe_fwd, pe_bwd)


def calculate_nli_xsmall(gt: str, hyp: str, **kwargs) -> Optional[float]:
    return _mutual_entailment(gt, hyp, "nli_xsmall")


def calculate_nli_base(gt: str, hyp: str, **kwargs) -> Optional[float]:
    return _mutual_entailment(gt, hyp, "nli_base")


def calculate_nli_large(gt: str, hyp: str, **kwargs) -> Optional[float]:
    return _mutual_entailment(gt, hyp, "nli_large")
