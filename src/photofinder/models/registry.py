from __future__ import annotations

import inspect
from typing import Any, Dict, Type

from .base import FaceEmbedder
from .dummy import DummyEmbedder

_REG: Dict[str, Type[FaceEmbedder]] = {
    DummyEmbedder.name: DummyEmbedder,
}

# “Optional” means: they exist, but are lazy-imported so base install stays light.
_OPTIONAL = ["dlib_resnet_v1", "arcface_onnx", "opencv_sface", "mobilefacenet_onnx", "arcface_r100_onnx"]


def register(name: str, cls: Type[FaceEmbedder]) -> None:
    _REG[name] = cls


def _filter_kwargs_for_ctor(cls: Type[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs that the class constructor accepts.
    If it has **kwargs, keep everything.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}

    params = list(sig.parameters.values())

    # if **kwargs present, accept all
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs

    accepted = {p.name for p in params if p.name not in ("self",)}
    return {k: v for k, v in kwargs.items() if k in accepted}


def get_embedder(name: str, **kwargs) -> FaceEmbedder:
    """
    Lazy-import embedders so base install stays lightweight.
    kwargs are forwarded to the embedder's constructor when supported.
    """
    # ---- lazy imports ----
    if name == "dlib_resnet_v1":
        from .dlib_resnet import DlibResnetEmbedder
        register(DlibResnetEmbedder.name, DlibResnetEmbedder)

    elif name == "arcface_onnx":
        from .arcface_onnx import ArcFaceOnnxEmbedder
        register(ArcFaceOnnxEmbedder.name, ArcFaceOnnxEmbedder)

    elif name == "opencv_sface":
        from .opencv_sface import OpenCVSFaceEmbedder
        register(OpenCVSFaceEmbedder.name, OpenCVSFaceEmbedder)

    elif name == "mobilefacenet_onnx":
        from .mobilefacenet_onnx import MobileFaceNetOnnxEmbedder
        register(MobileFaceNetOnnxEmbedder.name, MobileFaceNetOnnxEmbedder)

    elif name == "arcface_r100_onnx":
        from .arcface_r100_onnx import ArcFaceR100OnnxEmbedder
        register(ArcFaceR100OnnxEmbedder.name, ArcFaceR100OnnxEmbedder)

    # ---- now validate ----
    if name not in _REG:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REG.keys()) + _OPTIONAL}")

    cls = _REG[name]
    kw = _filter_kwargs_for_ctor(cls, dict(kwargs))
    return cls(**kw)
