"""SeMaScore — importance-weighted semantic similarity with edit alignment.

Model: microsoft/deberta-large-mnli (~1.7GB, auto-downloaded), pruned to 18 layers.
Algorithm: Character-level edit alignment -> segment mapping -> importance-weighted
semantic similarity * character-level MER.
Higher is better, range [0, 1].
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from metrics.model_cache import get_device, models

_MODEL_KEY = "semascore_deberta"
_MODEL_NAME = "microsoft/deberta-large-mnli"
_NUM_LAYERS = 18


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _prune_layers(model: torch.nn.Module, num_layers: int) -> torch.nn.Module:
    """Prune a transformer model to *num_layers* encoder layers."""
    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    if hasattr(model, "n_layers"):
        assert 0 <= num_layers <= model.n_layers
        model.n_layers = num_layers
    elif hasattr(model, "layer"):
        assert 0 <= num_layers <= len(model.layer)
        model.layer = torch.nn.ModuleList(list(model.layer[:num_layers]))
    elif hasattr(model, "encoder"):
        enc = model.encoder
        if hasattr(enc, "albert_layer_groups"):
            assert 0 <= num_layers <= enc.config.num_hidden_layers
            enc.config.num_hidden_layers = num_layers
        elif hasattr(enc, "block"):
            assert 0 <= num_layers <= len(enc.block)
            enc.block = torch.nn.ModuleList(list(enc.block[:num_layers]))
        else:
            assert 0 <= num_layers <= len(enc.layer)
            enc.layer = torch.nn.ModuleList(list(enc.layer[:num_layers]))
    elif hasattr(model, "transformer"):
        assert 0 <= num_layers <= len(model.transformer.layer)
        model.transformer.layer = torch.nn.ModuleList(
            list(model.transformer.layer[:num_layers])
        )
    elif hasattr(model, "layers"):
        assert 0 <= num_layers <= len(model.layers)
        model.layers = torch.nn.ModuleList(list(model.layers[:num_layers]))
    else:
        raise ValueError("Unsupported architecture for layer pruning.")

    return model


def _load_semascore() -> dict:
    device = get_device()
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.sep_token

    mdl = AutoModel.from_pretrained(_MODEL_NAME)
    mdl.eval()
    mdl = _prune_layers(mdl, _NUM_LAYERS)
    mdl = mdl.to(device)

    idf_dict: defaultdict = defaultdict(lambda: 1.0)
    if tok.sep_token_id is not None:
        idf_dict[tok.sep_token_id] = 0.0
    if tok.cls_token_id is not None:
        idf_dict[tok.cls_token_id] = 0.0

    return {"tokenizer": tok, "model": mdl, "idf_dict": idf_dict, "device": device}


models.register_loader(_MODEL_KEY, _load_semascore)


# ---------------------------------------------------------------------------
# Tokenisation & embedding
# ---------------------------------------------------------------------------

def _sent_encode(tokenizer: AutoTokenizer, sent: str) -> List[int]:
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    max_len = min(tokenizer.model_max_length, 512)
    return tokenizer.encode(
        sent, add_special_tokens=True, max_length=max_len, truncation=True,
    )


def _padding(arr: Sequence[Sequence[int]], pad_token: int, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = int(lens.max().item()) if len(lens) else 0
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        if len(a) == 0:
            continue
        padded[i, : len(a)] = torch.tensor(a, dtype=dtype)
        mask[i, : len(a)] = 1
    return padded, lens, mask


def _collate_idf(texts, tokenizer, idf_dict, device):
    token_ids = [_sent_encode(tokenizer, t) for t in texts]
    idf_w = [[idf_dict[i] for i in ids] for ids in token_ids]
    pad_id = tokenizer.pad_token_id
    padded, lens, mask = _padding(token_ids, pad_id, dtype=torch.long)
    padded_idf, _, _ = _padding(idf_w, 0, dtype=torch.float)
    return padded.to(device), mask.to(device), lens.to(device), padded_idf.to(device)


@torch.no_grad()
def _bert_encode(model, x, attention_mask):
    out = model(x, attention_mask=attention_mask, output_hidden_states=False)
    return out[0]


@torch.no_grad()
def _get_bert_embedding(texts, model, tokenizer, idf_dict, device):
    x, mask, lens, _ = _collate_idf(texts, tokenizer, idf_dict, device)
    emb = _bert_encode(model, x, attention_mask=mask)
    return emb, mask


# ---------------------------------------------------------------------------
# Character-level edit alignment
# ---------------------------------------------------------------------------

def _backtrace_changes(s1, s2, dp):
    i, j = len(s1), len(s2)
    ops: List[str] = []
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            ops.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append("$")
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            ops.append("-")
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            ops.append("+")
            j -= 1
    while i > 0:
        ops.append("-")
        i -= 1
    while j > 0:
        ops.append("+")
        j -= 1
    return ops[::-1]


def _edit_ops(hyp: str, ref: str) -> str:
    n, m = len(hyp), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j - 1], dp[i - 1][j])
    return "".join(_backtrace_changes(hyp, ref, dp))


def _split_by_spaces_with_ops(s: str, aligned: str, consume_marker: str):
    parts_s: List[str] = []
    parts_op: List[str] = []
    i = j = 0
    buf_s: List[str] = []
    buf_op: List[str] = []
    while i < len(s) and j < len(aligned):
        if s[i] == " " and aligned[j] == " ":
            parts_s.append("".join(buf_s))
            parts_op.append("".join(buf_op))
            buf_s, buf_op = [], []
            i += 1
            j += 1
        else:
            if aligned[j] == consume_marker:
                buf_op.append(aligned[j])
                j += 1
            else:
                buf_s.append(s[i])
                buf_op.append(aligned[j])
                i += 1
                j += 1
    if i < len(s):
        buf_s.append(s[i:])
    if j < len(aligned):
        buf_op.append(aligned[j:])
    parts_s.append("".join(buf_s))
    parts_op.append("".join(buf_op))
    return parts_s, len(parts_s) == len(parts_op)


def _mapped_sentence(ground_truth: str, hypothesis: str):
    aligned = _edit_ops(hypothesis, ground_truth)
    gt_chunks, gt_ok = _split_by_spaces_with_ops(ground_truth, aligned, consume_marker="-")
    hyp_chunks, hyp_ok = _split_by_spaces_with_ops(hypothesis, aligned, consume_marker="+")
    if gt_ok and hyp_ok:
        mismatches = sum(1 for c in aligned if c in ("+", "-", "$"))
        denom = max(len(ground_truth), len(hypothesis))
        mer = mismatches / denom if denom else 0.0
        return gt_chunks, hyp_chunks, aligned, mer
    return None, None, None, None


# ---------------------------------------------------------------------------
# Segment scoring
# ---------------------------------------------------------------------------

def _chunk_mer(s_gt: str, s_hyp: str) -> float:
    aligned = _edit_ops(s_hyp, s_gt)
    mismatches = sum(1 for c in aligned if c in ("+", "-", "$"))
    denom = max(len(s_gt), len(s_hyp))
    denom = denom if denom else 1
    return 1.0 - (mismatches / denom)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a.unsqueeze(0), p=2, dim=1)
    b = torch.nn.functional.normalize(b.unsqueeze(0), p=2, dim=1)
    return float(torch.mm(a, b.transpose(0, 1)).item())


def _token_strings(tokenizer, text: str) -> List[str]:
    ids = _sent_encode(tokenizer, text)
    return [tokenizer.decode([i]) for i in ids][1:-1]


def _segment_mean_embeddings(segments, full_embeddings, token_strs):
    k = 0
    out = []
    for seg in segments:
        seg = seg.strip()
        if seg == "":
            out.append(full_embeddings[0][0])
            continue
        word = ""
        start = k + 1
        while k < len(token_strs):
            word += token_strs[k]
            if word.strip() == seg:
                out.append(torch.mean(full_embeddings[0][start : k + 2], dim=0))
                k += 1
                break
            k += 1
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_sema_score(gt: str, hyp: str, **kwargs) -> float | None:
    """Compute SeMaScore. Higher is better."""
    if not gt and not hyp:
        return None

    bundle = models.get(_MODEL_KEY)
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    idf_dict = bundle["idf_dict"]
    device = bundle["device"]

    if not gt or not hyp:
        try:
            return _compute(gt, hyp, tokenizer, model, idf_dict, device)
        except Exception:
            return None

    return _compute(gt, hyp, tokenizer, model, idf_dict, device)


def _compute(
    gt: str, hyp: str, tokenizer, model, idf_dict, device: str,
) -> float:
    gt_clean = re.sub(r"[^\w\s]", "", gt.lower())
    hyp_clean = re.sub(r"[^\w\s]", "", hyp.lower())

    gt_emb, _ = _get_bert_embedding([gt_clean], model, tokenizer, idf_dict, device)
    hyp_emb, _ = _get_bert_embedding([hyp_clean], model, tokenizer, idf_dict, device)

    gt_chunks, hyp_chunks, aligned, _ = _mapped_sentence(gt_clean, hyp_clean)
    if gt_chunks is None:
        return 0.0

    gt_tok_strs = _token_strings(tokenizer, gt_clean)
    hyp_tok_strs = _token_strings(tokenizer, hyp_clean)

    gt_seg_embs = _segment_mean_embeddings(gt_chunks, gt_emb, gt_tok_strs)
    hyp_seg_embs = _segment_mean_embeddings(hyp_chunks, hyp_emb, hyp_tok_strs)

    total_gt_emb = (
        torch.mean(gt_emb[0][1:-1], dim=0) if gt_emb.size(1) > 2 else gt_emb[0][0]
    )

    weighted_sum = 0.0
    weight_total = 0.0

    for j in range(min(len(gt_seg_embs), len(hyp_seg_embs))):
        ss = (_cos_sim(gt_seg_embs[j], hyp_seg_embs[j]) + 1.0) / 2.0
        importance = (_cos_sim(gt_seg_embs[j], total_gt_emb) + 1.0) / 2.0
        mer_chunk = _chunk_mer(gt_chunks[j], hyp_chunks[j])

        weighted_sum += importance * ss * mer_chunk
        weight_total += importance

    return (weighted_sum / weight_total) if weight_total > 0 else 0.0
