"""Heval — hybrid ASR evaluation metric with semantic distance weighting.

Model: roberta-base (~500MB, auto-downloaded from HuggingFace).
Algorithm: Extract keywords by semantic distance threshold (gamma=0.4),
count word errors by class (keyword vs non-keyword), combine with overall
semantic distance.  Lower is better (like WER).

Source: factual-consistency/FER/run_metrics.py L590-759
"""

from __future__ import annotations

import logging
import re
from typing import List, Set, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from metrics.model_cache import get_device, models

LOGGER = logging.getLogger(__name__)

_MODEL_KEY = "heval_roberta"
_DEFAULT_MODEL = "roberta-base"


def _load_heval_model():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
    model = AutoModel.from_pretrained(_DEFAULT_MODEL).to(device).eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


models.register_loader(_MODEL_KEY, _load_heval_model)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mean_pool_last_hidden(
    model, tokenizer, texts: List[str], device: str,
) -> torch.Tensor:
    """Mean-pool the last hidden state with attention mask weighting."""
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    last = out.last_hidden_state
    attn = enc["attention_mask"].unsqueeze(-1)
    summed = (last * attn).sum(dim=1)
    counts = attn.sum(dim=1).clamp(min=1)
    return summed / counts


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = torch.nn.functional.normalize(b, p=2, dim=-1)
    return float((a * b).sum().item())


def _semantic_distance(
    gt: str, hyp: str, tokenizer, model, device: str,
) -> float:
    """1.0 - clipped(cosine_similarity), in [0, 1]."""
    emb = _mean_pool_last_hidden(model, tokenizer, [gt, hyp], device)
    cos = _cosine(emb[0], emb[1])
    sd = 1.0 - max(-1.0, min(1.0, cos))
    return max(0.0, min(1.0, sd))


def _minmax_scale(vals: List[float]) -> List[float]:
    """Min-max normalisation of a list of floats."""
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


def _levenshtein_ops(
    gt_words: List[str], hyp_words: List[str],
) -> List[Tuple[str, int, int]]:
    """Word-level Levenshtein with operation tracking (match/sub/del/ins)."""
    n, m = len(gt_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    ops: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1]
            and gt_words[i - 1] == hyp_words[j - 1]
        ):
            ops.append(("match", i, j))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i, j))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i, j))
            i -= 1
        else:
            ops.append(("ins", i, j))
            j -= 1
    ops.reverse()
    return ops


def _extract_keywords_by_sd(
    gt_words: List[str],
    full_gt: str,
    gamma: float,
    tokenizer,
    model,
    device: str,
) -> Tuple[Set[str], Set[str]]:
    """Select keywords (low semantic distance to full GT) vs non-keywords."""
    if not gt_words:
        return set(), set()
    uniq = list(dict.fromkeys(gt_words))
    scores = [
        _semantic_distance(full_gt, w, tokenizer, model, device)
        for w in uniq
    ]
    scaled = _minmax_scale(scores)
    keywords = {w for w, s in zip(uniq, scaled) if s < gamma}
    non_keywords = set(uniq) - keywords
    return keywords, non_keywords


def _count_wrong_by_class(
    gt_words: List[str],
    hyp_words: List[str],
    keywords: Set[str],
    non_keywords: Set[str],
) -> Tuple[int, int]:
    """Count substitutions/deletions at keyword vs non-keyword positions."""
    Nwk = 0
    Nwnk = 0
    ops = _levenshtein_ops(gt_words, hyp_words)
    wrong_positions = {
        i - 1 for op, i, _ in ops if op in ("sub", "del") and i > 0
    }
    for idx in wrong_positions:
        w = gt_words[idx]
        if w in keywords:
            Nwk += 1
        else:
            Nwnk += 1
    return Nwk, Nwnk


def _heval(
    gt: str,
    hyp: str,
    gamma: float,
    tokenizer,
    model,
    device: str,
) -> float:
    """Core Heval computation."""
    gt_words = re.findall(r"[A-Za-z]+", gt.lower())
    hyp_words = re.findall(r"[A-Za-z]+", hyp.lower())

    N = len(gt_words)
    if N == 0:
        return 0.0

    K, NK = _extract_keywords_by_sd(
        gt_words, " ".join(gt_words), gamma, tokenizer, model, device,
    )
    Nk = max(1, len(K))
    Nnk = max(1, len(NK))

    Nwk, Nwnk = _count_wrong_by_class(gt_words, hyp_words, K, NK)

    SD = _semantic_distance(
        " ".join(gt_words), " ".join(hyp_words), tokenizer, model, device,
    )
    NKER = Nwnk / Nnk

    p = N / Nk
    alpha1 = (Nwk * p) / N
    alpha2 = Nwnk / N

    heval = alpha1 * SD + alpha2 * NKER
    return max(0.0, min(1.0, float(heval)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_heval(gt: str, hyp: str, **kwargs) -> float | None:
    """Compute Heval score. Lower is better.

    Args:
        gt: Ground-truth transcript.
        hyp: Hypothesis transcript.
        **kwargs: Optional overrides (gamma, etc.).
    """
    if not gt and not hyp:
        return None

    gamma = kwargs.get("gamma", 0.4)
    bundle = models.get(_MODEL_KEY)
    tokenizer, model, device = bundle["tokenizer"], bundle["model"], bundle["device"]

    if not gt or not hyp:
        try:
            return _heval(gt, hyp, gamma, tokenizer, model, device)
        except Exception:
            return None

    return _heval(gt, hyp, gamma, tokenizer, model, device)
