"""Tier 2: Learned Semantic Lightweight metrics (SBERT, NLI, BERTScore, BARTScore, SimCSE, SWER).

Registers all metrics in the global registry on import.
Requires the ``learned-semantic`` optional dependency group.
BLEURT requires the separate ``bleurt`` optional dependency group.
"""

from metrics.registry import register

from metrics.learned_semantic_lightweight.sbert import calculate_sbert_similarity
from metrics.learned_semantic_lightweight.nli import (
    calculate_nli_base,
    calculate_nli_large,
    calculate_nli_xsmall,
)
from metrics.learned_semantic_lightweight.simcse import calculate_simcse
from metrics.learned_semantic_lightweight.bartscore import calculate_bart_score
from metrics.learned_semantic_lightweight.semantic_wer import calculate_semantic_wer
from metrics.learned_semantic_lightweight.bertscore import calculate_bert_score

_TIER = "learned_semantic_lightweight"
_EXTRA = "learned-semantic"

register(
    "sbert_similarity", _TIER, calculate_sbert_similarity,
    fallback=0.0, higher_is_better=True,
    description="SBERT cosine similarity (all-MiniLM-L6-v2)",
    extra=_EXTRA,
)
register(
    "nli_xsmall", _TIER, calculate_nli_xsmall,
    fallback=0.0, higher_is_better=True,
    description="NLI mutual entailment (DeBERTa-v3 xsmall)",
    extra=_EXTRA,
)
register(
    "nli_base", _TIER, calculate_nli_base,
    fallback=0.0, higher_is_better=True,
    description="NLI mutual entailment (DeBERTa-v3 base)",
    extra=_EXTRA,
)
register(
    "nli_large", _TIER, calculate_nli_large,
    fallback=0.0, higher_is_better=True,
    description="NLI mutual entailment (DeBERTa-v3 large)",
    extra=_EXTRA,
)
register(
    "simcse", _TIER, calculate_simcse,
    fallback=0.0, higher_is_better=True,
    description="SimCSE cosine similarity (sup-simcse-bert-base-uncased)",
    extra=_EXTRA,
)
register(
    "bart_score", _TIER, calculate_bart_score,
    fallback=0.0, higher_is_better=True,
    description="BARTScore average bidirectional log-likelihood (bart-large-cnn)",
    extra=_EXTRA,
)
register(
    "semantic_wer", _TIER, calculate_semantic_wer,
    fallback=1.0, higher_is_better=False,
    description="Semantic Word Error Rate — DP weighted edit distance (all-MiniLM-L6-v2)",
    extra=_EXTRA,
)
register(
    "bert_score", _TIER, calculate_bert_score,
    fallback=0.0, higher_is_better=True,
    description="BERTScore F1 (deberta-xlarge-mnli)",
    extra=_EXTRA,
)

# BLEURT metrics — separate optional dependency group
try:
    from metrics.learned_semantic_lightweight.bleurt_metric import (
        calculate_bleurt,
        calculate_clinical_bleurt,
    )

    _BLEURT_EXTRA = "bleurt"

    register(
        "bleurt", _TIER, calculate_bleurt,
        fallback=0.0, higher_is_better=True,
        description="BLEURT score (requires manual checkpoint download)",
        extra=_BLEURT_EXTRA,
    )
    register(
        "clinical_bleurt", _TIER, calculate_clinical_bleurt,
        fallback=0.0, higher_is_better=True,
        description="Clinical BLEURT score (requires manual checkpoint download)",
        extra=_BLEURT_EXTRA,
    )
except ImportError:
    pass
