from typing import Any, Optional

from .gepa import build_gepa
from .mipro import build_mipro


def get_optimizer(name: str, metric, reflection_lm: Optional[Any] = None, **kwargs):
    """Return a configured optimizer by name."""
    name = name.lower()
    if name == "gepa":
        return build_gepa(metric=metric, reflection_lm=reflection_lm, **kwargs)
    if name in {"mipro", "miprov2"}:
        return build_mipro(metric=metric, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")
