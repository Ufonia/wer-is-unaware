"""Lazy-loading model cache and device helpers for learned-semantic metrics.

Usage:
    from metrics.model_cache import models, get_device

    models.register_loader("sbert", lambda: SentenceTransformer("all-MiniLM-L6-v2"))
    model = models.get("sbert")       # loads on first call, cached thereafter
    models.unload("sbert")            # free memory
    models.clear()                    # free all

    device = get_device()             # "cuda", "mps", or "cpu"
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


def get_device() -> str:
    """Return the best available torch device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelCache:

    def __init__(self) -> None:
        self._loaders: Dict[str, Callable[[], Any]] = {}
        self._models: Dict[str, Any] = {}

    def register_loader(self, key: str, loader: Callable[[], Any]) -> None:
        self._loaders[key] = loader

    def get(self, key: str) -> Any:
        if key in self._models:
            return self._models[key]
        if key not in self._loaders:
            raise KeyError(f"No loader registered for model key: {key!r}")
        self._models[key] = self._loaders[key]()
        return self._models[key]

    def loaded(self) -> List[str]:
        return list(self._models.keys())

    def unload(self, key: str) -> None:
        self._models.pop(key, None)

    def clear(self) -> None:
        self._models.clear()


models = ModelCache()
