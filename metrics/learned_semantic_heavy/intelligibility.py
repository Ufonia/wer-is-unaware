"""Intelligibility — composite metric: NLI + BERTScore + phonetic similarity.

Models:
  - NLI: ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli (~1.4GB)
  - BERTScore: roberta-large (~1.4GB, via bert_score library)
Weights: 0.40 * NLI + 0.28 * BERTScore_F1 + 0.32 * phonetic_similarity
Higher is better, clipped to [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from metrics.model_cache import get_device, models

_ALPHA = 0.40  # NLI weight
_BETA = 0.28   # BERTScore weight
_GAMMA = 0.32  # Phonetic weight

_NLI_MODEL_KEY = "intelligibility_nli"
_NLI_MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"


# ---------------------------------------------------------------------------
# NLI component
# ---------------------------------------------------------------------------

@dataclass
class _NLIPack:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    entail_id: int


def _load_nli() -> dict:
    device = get_device()
    tok = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_NAME)
    label2id = {v.lower(): k for k, v in mdl.config.id2label.items()}
    pack = _NLIPack(
        tokenizer=tok,
        model=mdl.eval().to(device),
        entail_id=label2id.get("entailment", 0),
    )
    return {"pack": pack, "device": device}


models.register_loader(_NLI_MODEL_KEY, _load_nli)


@torch.no_grad()
def _entail_prob(premise: str, hypothesis: str, pack: _NLIPack, device: str) -> float:
    inputs = pack.tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = pack.model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    return float(probs[pack.entail_id])


def _nli_mutual_entailment(gt: str, hyp: str, pack: _NLIPack, device: str) -> float:
    return min(
        _entail_prob(hyp, gt, pack, device),
        _entail_prob(gt, hyp, pack, device),
    )


# ---------------------------------------------------------------------------
# BERTScore component (via bert_score library)
# ---------------------------------------------------------------------------

def _semantic_similarity(gt: str, hyp: str, device: str) -> float:
    if not gt or not hyp:
        return 0.0
    from bert_score import score as bertscore
    _, _, F = bertscore(
        [hyp], [gt], lang="en", model_type="roberta-large",
        verbose=False, device=device,
    )
    return float(F.mean().item())


# ---------------------------------------------------------------------------
# Phonetic component (pure Python)
# ---------------------------------------------------------------------------

def _soundex_word(word: str) -> str:
    if not word:
        return "Z000"
    word = re.sub(r"[^A-Za-z]", "", word).upper()
    if not word:
        return "Z000"
    first = word[0]
    mapping = {
        **{c: "1" for c in "BFPV"},
        **{c: "2" for c in "CGJKQSXZ"},
        **{c: "3" for c in "DT"},
        "L": "4",
        **{c: "5" for c in "MN"},
        "R": "6",
    }
    encoded: List[str] = []
    prev = ""
    for ch in word[1:]:
        code = mapping.get(ch, "")
        if code != prev:
            encoded.append(code)
        if code:
            prev = code
        if ch in "AEIOUYHW":
            prev = ""
    digits = "".join(encoded)
    digits = re.sub(r"[^123456]", "", digits)
    return (first + digits + "000")[:4]


def _sentence_soundex(s: str) -> str:
    return " ".join(_soundex_word(w) for w in re.findall(r"[A-Za-z]+", s.lower()))


def _jaro_winkler(a: str, b: str, p: float = 0.1, max_l: int = 4) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    match_dist = max(0, max(la, lb) // 2 - 1)

    a_matches = [False] * la
    b_matches = [False] * lb
    matches = 0

    for i in range(la):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, lb)
        for j in range(start, end):
            if b_matches[j]:
                continue
            if a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    a_m = [a[i] for i in range(la) if a_matches[i]]
    b_m = [b[j] for j in range(lb) if b_matches[j]]
    transpositions = sum(aa != bb for aa, bb in zip(a_m, b_m)) // 2

    jaro = ((matches / la) + (matches / lb) + ((matches - transpositions) / matches)) / 3.0

    prefix = 0
    for i in range(min(max_l, la, lb)):
        if a[i] == b[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


def _phonetic_similarity(gt: str, hyp: str) -> float:
    return _jaro_winkler(_sentence_soundex(gt), _sentence_soundex(hyp))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_intelligibility(gt: str, hyp: str, **kwargs) -> float | None:
    """Compute intelligibility score. Higher is better."""
    if not gt and not hyp:
        return None

    bundle = models.get(_NLI_MODEL_KEY)
    pack, device = bundle["pack"], bundle["device"]

    if not gt or not hyp:
        try:
            s_nli = _nli_mutual_entailment(gt, hyp, pack, device)
            s_sem = _semantic_similarity(gt, hyp, device)
            s_pho = _phonetic_similarity(gt, hyp)
            score = _ALPHA * s_nli + _BETA * s_sem + _GAMMA * s_pho
            return max(0.0, min(1.0, float(score)))
        except Exception:
            return None

    s_nli = _nli_mutual_entailment(gt, hyp, pack, device)
    s_sem = _semantic_similarity(gt, hyp, device)
    s_pho = _phonetic_similarity(gt, hyp)
    score = _ALPHA * s_nli + _BETA * s_sem + _GAMMA * s_pho
    return max(0.0, min(1.0, float(score)))
