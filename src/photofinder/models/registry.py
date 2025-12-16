from __future__ import annotations
from typing import Dict, Type

from .base import FaceEmbedder
from .dummy import DummyEmbedder

_REG: Dict[str, Type[FaceEmbedder]] = {
    DummyEmbedder.name: DummyEmbedder,
}

def register(name: str, cls: Type[FaceEmbedder]) -> None:
    _REG[name] = cls

def get_embedder(name: str) -> FaceEmbedder:
    if name == "dlib_resnet_v1":
        from .dlib_resnet import DlibResnetEmbedder  # local import
        register("dlib_resnet_v1", DlibResnetEmbedder)
    if name not in _REG:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REG.keys()) + ['dlib_resnet_v1']}")
    return _REG[name]()
