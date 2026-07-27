"""Backend registry."""

from __future__ import annotations

from lorayaki.backends.anima import AnimaBackend
from lorayaki.backends.base import Backend, PresetConfig, TrainPaths
from lorayaki.backends.illustrious import IllustriousBackend

_BACKENDS = {
    "illustrious": IllustriousBackend,
    "anima": AnimaBackend,
}


def get_backend(name: str) -> Backend:
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise ValueError(
            f"不明なバックエンド: {name!r} (有効値: {', '.join(_BACKENDS)})"
        ) from None


__all__ = ["Backend", "PresetConfig", "TrainPaths", "get_backend"]
