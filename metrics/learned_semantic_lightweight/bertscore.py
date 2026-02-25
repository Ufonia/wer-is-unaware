"""BERTScore — contextual embedding F1. Model: microsoft/deberta-xlarge-mnli (~2.7GB, auto-downloaded).

Scores are baseline-rescaled: (raw - baseline) / (1 - baseline), matching the
official bert-score package defaults. Baselines computed on Common Crawl for
english, sourced from the bert-score 0.3.13 package data.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from metrics.learned_semantic_lightweight.bertscore_utils import (
    bert_cos_score_idf,
    get_model,
    get_tokenizer,
    model2layers,
)
from metrics.model_cache import get_device, models

_MODEL_KEY = "bertscore_deberta"
_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
_NUM_LAYERS = model2layers[_MODEL_TYPE]

# Baseline values for layer 40, english, from bert-score 0.3.13 package data.
# Used for rescaling: (raw - baseline) / (1 - baseline)
_BASELINE_P = 0.5169066
_BASELINE_R = 0.5170288
_BASELINE_F = 0.5150192


def _load_bertscore():
    device = get_device()
    tokenizer = get_tokenizer(_MODEL_TYPE)
    model = get_model(_MODEL_TYPE, _NUM_LAYERS)
    model.to(device)

    # Uniform IDF weights (no corpus-level IDF), CLS/SEP zeroed out
    idf_dict = defaultdict(lambda: 1.0)
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    return {
        "model": model,
        "tokenizer": tokenizer,
        "idf_dict": idf_dict,
        "device": device,
    }


models.register_loader(_MODEL_KEY, _load_bertscore)


def score(
    cands: list[str],
    refs: list[str],
    model,
    tokenizer,
    idf_dict: dict,
    device: str,
    batch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute BERTScore P, R, F1 for candidate/reference pairs (baseline-rescaled)."""
    assert len(cands) == len(refs), "Different number of candidates and references"

    all_preds = bert_cos_score_idf(
        model, refs, cands, tokenizer, idf_dict,
        device=device, batch_size=batch_size,
    ).cpu()

    # Baseline rescaling: (raw - baseline) / (1 - baseline)
    baselines = torch.tensor([_BASELINE_P, _BASELINE_R, _BASELINE_F])
    all_preds = (all_preds - baselines) / (1 - baselines)

    return all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]


def calculate_bert_score(gt: str, hyp: str, **kwargs) -> float:
    """BERTScore F1 for a single (gt, hyp) pair."""
    if not gt or not hyp:
        return 0.0

    bundle = models.get(_MODEL_KEY)
    P, R, F1 = score(
        cands=[hyp],
        refs=[gt],
        model=bundle["model"],
        tokenizer=bundle["tokenizer"],
        idf_dict=bundle["idf_dict"],
        device=bundle["device"],
    )
    return F1.item()
