"""Semantic Word Error Rate — DP-based weighted edit distance with semantic similarity.

Model: all-MiniLM-L6-v2 (~80MB, auto-downloaded).

NE/sentiment word
sets default to empty (configurable via kwargs). Lower is better.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from metrics.model_cache import get_device, models

_MODEL_KEY = "swer_minilm"
_MODEL_NAME = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.6


def _load_swer_model():
    device = get_device()
    model = SentenceTransformer(_MODEL_NAME, device=device)
    return {"model": model, "device": device}


models.register_loader(_MODEL_KEY, _load_swer_model)


def _get_word_embeddings(words: list[str], model: SentenceTransformer,
                         device: str) -> torch.Tensor:
    return model.encode(words, convert_to_tensor=True, device=device)


def _calculate_swer(
    ref_words: list[str],
    hyp_words: list[str],
    model: SentenceTransformer,
    device: str,
    ne_and_sent_words: set[str],
    similarity_threshold: float,
) -> float:
    """DP-based weighted edit distance, normalised by hypothesis length."""
    if not ref_words:
        return len(hyp_words) * (1 / len(hyp_words)) if hyp_words else 0.0

    if not hyp_words:
        del_cost = sum(
            1.0 if word in ne_and_sent_words else 1 / len(ref_words)
            for word in ref_words
        )
        return del_cost / len(ref_words)

    # Pre-compute embeddings for all unique words
    unique_words = list(set(ref_words + hyp_words))
    embeddings = _get_word_embeddings(unique_words, model, device)
    word_to_embedding = {word: emb for word, emb in zip(unique_words, embeddings)}

    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    for i in range(len(ref_words) + 1):
        for j in range(len(hyp_words) + 1):
            if i == 0 and j == 0:
                continue

            if i > 0:
                ref_word = ref_words[i - 1]
                del_cost = 1.0 if ref_word in ne_and_sent_words else (1 / len(ref_words))

            if j > 0:
                ins_cost = 1 / len(hyp_words)

            deletion = dp[i - 1, j] + del_cost if i > 0 else float("inf")
            insertion = dp[i, j - 1] + ins_cost if j > 0 else float("inf")

            if i > 0 and j > 0:
                ref_word = ref_words[i - 1]
                hyp_word = hyp_words[j - 1]

                if ref_word == hyp_word:
                    sub_cost = 0.0
                elif ref_word in ne_and_sent_words:
                    sub_cost = 1.0
                else:
                    similarity = util.cos_sim(
                        word_to_embedding[ref_word], word_to_embedding[hyp_word]
                    ).item()
                    sub_cost = 0.0 if similarity >= similarity_threshold else 1.0

                substitution = dp[i - 1, j - 1] + sub_cost
            else:
                substitution = float("inf")

            dp[i, j] = min(deletion, insertion, substitution)

    return dp[len(ref_words), len(hyp_words)] / len(hyp_words)


def calculate_semantic_wer(gt: str, hyp: str, **kwargs) -> float:
    """Semantic Word Error Rate for a single (gt, hyp) pair.

    Kwargs:
        named_entities: set of NE strings (default: empty)
        sentiment_words: set of sentiment strings (default: empty)
        similarity_threshold: cosine sim threshold (default: 0.6)
    """
    if not gt or not hyp:
        return 1.0  # fallback

    bundle = models.get(_MODEL_KEY)
    model, device = bundle["model"], bundle["device"]

    named_entities: set[str] = kwargs.get("named_entities", set())
    sentiment_words: set[str] = kwargs.get("sentiment_words", set())
    ne_and_sent_words = named_entities | sentiment_words
    similarity_threshold = kwargs.get("similarity_threshold", _SIMILARITY_THRESHOLD)

    ref_words = gt.lower().split()
    hyp_words = hyp.lower().split()

    return _calculate_swer(
        ref_words, hyp_words, model, device, ne_and_sent_words, similarity_threshold
    )
