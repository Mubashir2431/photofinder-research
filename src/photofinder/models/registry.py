from __future__ import annotations
from typing import Dict, Type

from .base import FaceEmbedder
from .dummy import DummyEmbedder

_REG: Dict[str, Type[FaceEmbedder]] = {
    DummyEmbedder.name: DummyEmbedder,
}

_OPTIONAL = ["dlib_resnet_v1", "arcface_onnx"]

def register(name: str, cls: Type[FaceEmbedder]) -> None:
    _REG[name] = cls

def get_embedder(name: str) -> FaceEmbedder:
    if name == "dlib_resnet_v1":
        from .dlib_resnet import DlibResnetEmbedder
        register(DlibResnetEmbedder.name, DlibResnetEmbedder)

    if name == "arcface_onnx":
        from .arcface_onnx import ArcFaceOnnxEmbedder
        register(ArcFaceOnnxEmbedder.name, ArcFaceOnnxEmbedder)

    if name not in _REG:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REG.keys()) + _OPTIONAL}")

    return _REG[name]()
