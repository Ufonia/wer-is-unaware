"""Metric registry — central catalogue of all available metrics.

Empty-input handling is done inside each ``calculate_*`` function:

- Both empty → ``None`` (universal).
- One empty, clear answer → return it (e.g. WER → 1.0, BLEU → 0.0).
- One empty, ambiguous → let the algorithm try; ``try/except`` on the
  one-empty path only, returning ``None`` on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class MetricEntry:
    """A registered metric."""

    name: str
    tier: str
    fn: Callable[..., float]
    higher_is_better: bool
    description: str
    extra: Optional[str] = None  # optional-dependency group required (e.g. "learned-semantic")


class MetricUnavailableError(ImportError):
    """Raised when a metric's optional dependencies are not installed."""


# Global registry — populated by tier sub-packages on import.
REGISTRY: Dict[str, MetricEntry] = {}


def register(
    name: str,
    tier: str,
    fn: Callable[..., float],
    higher_is_better: bool,
    description: str,
    extra: Optional[str] = None,
) -> None:
    """Register a metric in the global registry."""
    REGISTRY[name] = MetricEntry(
        name=name,
        tier=tier,
        fn=fn,
        higher_is_better=higher_is_better,
        description=description,
        extra=extra,
    )


def get(name: str) -> MetricEntry:
    """Get a metric entry by name, raising if not found."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown metric: {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
