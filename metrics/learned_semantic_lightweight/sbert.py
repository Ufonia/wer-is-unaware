"""SBERT cosine similarity. Model: all-MiniLM-L6-v2 (~80MB, auto-downloaded)."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer, util

from metrics.model_cache import models

_MODEL_KEY = "sbert_minilm"

models.register_loader(
    _MODEL_KEY, lambda: SentenceTransformer("all-MiniLM-L6-v2")
)


def calculate_sbert_similarity(gt: str, hyp: str, **kwargs) -> float:
    if not gt or not hyp:
        return 0.0

    model = models.get(_MODEL_KEY)
    gt_embedding = model.encode(gt, convert_to_tensor=True)
    hyp_embedding = model.encode(hyp, convert_to_tensor=True)
    return util.cos_sim(gt_embedding, hyp_embedding).item()
