"""Public API for the metrics package.

Usage:
    from metrics import calculate_metric, calculate_all_metrics, list_metrics, get_metric_info

    score = calculate_metric("wer", gt="the cat sat on the mat", hyp="the cat sat on a mat")
    scores = calculate_all_metrics(gt, hyp, tier="edit_distance_and_ngram")
    available = list_metrics()
    info = get_metric_info("wer")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from metrics.cleaning import get_clean_transcript
from metrics.registry import REGISTRY, MetricUnavailableError

import metrics.edit_distance_and_ngram  # noqa: F401

try:
    import metrics.learned_semantic_lightweight  # noqa: F401
except ImportError:
    pass

try:
    import metrics.learned_semantic_heavy  # noqa: F401
except ImportError:
    pass


@dataclass(frozen=True)
class MetricInfo:
    """User-facing info about a registered metric."""

    name: str
    tier: str
    higher_is_better: bool
    description: str
    extra: Optional[str]


def list_metrics() -> Dict[str, List[str]]:
    """Return metric names grouped by tier.

    Returns:
        Dict mapping tier name → list of metric names.
    """
    tiers: Dict[str, List[str]] = {}
    for entry in REGISTRY.values():
        tiers.setdefault(entry.tier, []).append(entry.name)
    return tiers


def get_metric_info(name: str) -> MetricInfo:
    """Return metadata about a single metric."""
    entry = REGISTRY[name]
    return MetricInfo(
        name=entry.name,
        tier=entry.tier,
        higher_is_better=entry.higher_is_better,
        description=entry.description,
        extra=entry.extra,
    )


def calculate_metric(name: str, gt: str, hyp: str, **kwargs) -> float:
    """Calculate a single metric for a GT/HYP pair.

    Text is cleaned via ``get_clean_transcript`` before being passed to the
    metric function.

    Args:
        name: Registered metric name (e.g. ``"wer"``).
        gt: Ground-truth transcript.
        hyp: Hypothesis transcript.
        **kwargs: Forwarded to the metric function (e.g. model handles).

    Returns:
        Metric score as a float.

    Raises:
        KeyError: If *name* is not registered.
        MetricUnavailableError: If the metric's optional deps are missing.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown metric: {name!r}. Available: {sorted(REGISTRY)}")

    entry = REGISTRY[name]

    gt_clean = get_clean_transcript(gt)
    hyp_clean = get_clean_transcript(hyp)

    return entry.fn(gt_clean, hyp_clean, **kwargs)


def calculate_all_metrics(
    gt: str,
    hyp: str,
    *,
    tier: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, float]:
    """Calculate multiple metrics for a GT/HYP pair.

    Args:
        gt: Ground-truth transcript.
        hyp: Hypothesis transcript.
        tier: If given, run all metrics in this tier.
        metrics: If given, run these specific metrics (across any tier).
            Mutually exclusive with *tier*.
        **kwargs: Forwarded to each metric function.

    Returns:
        Dict mapping metric name → score.
    """
    if tier and metrics:
        raise ValueError("Specify 'tier' or 'metrics', not both.")

    if tier:
        names = [e.name for e in REGISTRY.values() if e.tier == tier]
        if not names:
            raise ValueError(
                f"Unknown tier: {tier!r}. Available: {sorted({e.tier for e in REGISTRY.values()})}"
            )
    elif metrics:
        names = metrics
    else:
        names = list(REGISTRY.keys())

    gt_clean = get_clean_transcript(gt)
    hyp_clean = get_clean_transcript(hyp)

    results: Dict[str, float] = {}
    for name in names:
        if name not in REGISTRY:
            raise KeyError(f"Unknown metric: {name!r}")
        entry = REGISTRY[name]
        results[name] = entry.fn(gt_clean, hyp_clean, **kwargs)

    return results
