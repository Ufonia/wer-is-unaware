"""Tier 1: Edit-distance and n-gram overlap metrics (15 metrics).

Registers all metrics in the global registry on import.
"""

from metrics.registry import register

from metrics.edit_distance_and_ngram.lexical import (
    calculate_cer,
    calculate_mer,
    calculate_wer,
    calculate_wil,
)
from metrics.edit_distance_and_ngram.ngram import (
    calculate_bleu_1,
    calculate_bleu_2,
    calculate_bleu_3,
    calculate_bleu_4,
    calculate_chrf,
    calculate_chrf_plus_plus,
    calculate_meteor,
    calculate_rouge_1,
    calculate_rouge_2,
    calculate_rouge_l,
    calculate_rouge_w,
)

_TIER = "edit_distance_and_ngram"

# --- Lexical (error metrics: lower is better, fallback=1.0) ---
register("wer", _TIER, calculate_wer, fallback=1.0, higher_is_better=False, description="Word Error Rate")
register("cer", _TIER, calculate_cer, fallback=1.0, higher_is_better=False, description="Character Error Rate")
register("mer", _TIER, calculate_mer, fallback=1.0, higher_is_better=False, description="Match Error Rate")
register("wil", _TIER, calculate_wil, fallback=1.0, higher_is_better=False, description="Word Information Lost")

# --- N-gram (similarity metrics: higher is better, fallback=0.0) ---
register("bleu_1", _TIER, calculate_bleu_1, fallback=0.0, higher_is_better=True, description="BLEU-1 (unigram)")
register("bleu_2", _TIER, calculate_bleu_2, fallback=0.0, higher_is_better=True, description="BLEU-2 (uni+bigram)")
register("bleu_3", _TIER, calculate_bleu_3, fallback=0.0, higher_is_better=True, description="BLEU-3 (uni+bi+trigram)")
register("bleu_4", _TIER, calculate_bleu_4, fallback=0.0, higher_is_better=True, description="BLEU-4 (uni+bi+tri+4gram)")
register("rouge_1", _TIER, calculate_rouge_1, fallback=0.0, higher_is_better=True, description="ROUGE-1 F-measure (unigram)")
register("rouge_2", _TIER, calculate_rouge_2, fallback=0.0, higher_is_better=True, description="ROUGE-2 F-measure (bigram)")
register("rouge_l", _TIER, calculate_rouge_l, fallback=0.0, higher_is_better=True, description="ROUGE-L F-measure (LCS)")
register("rouge_w", _TIER, calculate_rouge_w, fallback=0.0, higher_is_better=True, description="ROUGE-W F-measure (weighted LCS)")
register("chrf", _TIER, calculate_chrf, fallback=0.0, higher_is_better=True, description="ChrF (character F-score)")
register("chrf_plus_plus", _TIER, calculate_chrf_plus_plus, fallback=0.0, higher_is_better=True, description="ChrF++ (ChrF with word bigrams)")
register("meteor", _TIER, calculate_meteor, fallback=0.0, higher_is_better=True, description="METEOR")
