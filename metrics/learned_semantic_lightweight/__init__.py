"""Tier 2: Learned Semantic Lightweight metrics (SBERT, NLI).

Registers all metrics in the global registry on import.
Requires the ``learned-semantic`` optional dependency group.
"""

from metrics.registry import register

from metrics.learned_semantic_lightweight.sbert import calculate_sbert_similarity
from metrics.learned_semantic_lightweight.nli import (
    calculate_nli_base,
    calculate_nli_large,
    calculate_nli_xsmall,
)

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
