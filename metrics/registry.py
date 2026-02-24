"""Metric registry — central catalogue of all available metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class MetricEntry:
    """A registered metric."""

    name: str
    tier: str
    fn: Callable[..., float]
    fallback: float
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
    fallback: float,
    higher_is_better: bool,
    description: str,
    extra: Optional[str] = None,
) -> None:
    """Register a metric in the global registry."""
    REGISTRY[name] = MetricEntry(
        name=name,
        tier=tier,
        fn=fn,
        fallback=fallback,
        higher_is_better=higher_is_better,
        description=description,
        extra=extra,
    )


def get(name: str) -> MetricEntry:
    """Get a metric entry by name, raising if not found."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown metric: {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
