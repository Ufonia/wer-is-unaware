"""SimCSE cosine similarity via CLS pooling.

Model: princeton-nlp/sup-simcse-bert-base-uncased (~440MB, auto-downloaded).
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from metrics.model_cache import get_device, models

_MODEL_KEY = "simcse_bert_base"
_MODEL_NAME = "princeton-nlp/sup-simcse-bert-base-uncased"
_MAX_LENGTH = 128


def _load_simcse():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(_MODEL_NAME).to(device).eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


models.register_loader(_MODEL_KEY, _load_simcse)


@torch.no_grad()
def _embed(text: str, tokenizer, model, device: str) -> torch.Tensor:
    """Encode a single string → L2-normalised CLS embedding."""
    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=_MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, return_dict=True)
    # CLS pooling: use pooler_output if available, else first token
    emb = getattr(out, "pooler_output", None)
    if emb is None:
        emb = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def calculate_simcse(gt: str, hyp: str, **kwargs) -> float:
    """Cosine similarity of SimCSE sentence embeddings."""
    if not gt or not hyp:
        return 0.0

    bundle = models.get(_MODEL_KEY)
    tokenizer, model, device = bundle["tokenizer"], bundle["model"], bundle["device"]

    gt_emb = _embed(gt, tokenizer, model, device)
    hyp_emb = _embed(hyp, tokenizer, model, device)
    return float((gt_emb * hyp_emb).sum().item())
