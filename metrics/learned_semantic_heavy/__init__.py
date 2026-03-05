"""Tier 3: Learned Semantic Heavy metrics (SeMaScore, Intelligibility, Heval).

Registers all metrics in the global registry on import.
Requires the ``learned-semantic`` optional dependency group.
"""

from metrics.registry import register

from metrics.learned_semantic_heavy.semascore import calculate_sema_score
from metrics.learned_semantic_heavy.intelligibility import calculate_intelligibility
from metrics.learned_semantic_heavy.heval import calculate_heval

_TIER = "learned_semantic_heavy"
_EXTRA = "learned-semantic"

register(
    "sema_score", _TIER, calculate_sema_score,
    higher_is_better=True,
    description="SeMaScore — importance-weighted semantic similarity (DeBERTa-large-mnli, 18 layers)",
    extra=_EXTRA,
)
register(
    "intelligibility", _TIER, calculate_intelligibility,
    higher_is_better=True,
    description="Intelligibility — 0.40*NLI + 0.28*BERTScore + 0.32*Phonetic",
    extra=_EXTRA,
)
register(
    "heval", _TIER, calculate_heval,
    higher_is_better=False,
    description="Heval — hybrid ASR evaluation with semantic distance (roberta-base)",
    extra=_EXTRA,
)
