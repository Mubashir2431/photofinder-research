from __future__ import annotations

import os
from typing import List

import numpy as np

from .base import FaceEmbedder, FaceEmbedding


class ArcFaceOnnxEmbedder(FaceEmbedder):
    """
    ArcFace embedder via ONNXRuntime.

    Uses dlib for:
      - face detection
      - 68-landmark alignment (get_face_chip -> 112x112)

    Requires env vars:
      - DLIB_SHAPE_PREDICTOR_PATH  (68 landmarks .dat)
      - ARCFACE_ONNX_PATH          (ArcFace .onnx file)

    Knobs supported:
      - det_upsample (0/1/2)
      - arcface_padding (float)
      - arcface_preproc ("insightface" or "legacy")
    """

    name = "arcface_onnx"
    dim = 512

    def __init__(
        self,
        *,
        det_upsample: int = 1,
        arcface_padding: float = 0.25,
        arcface_preproc: str = "insightface",
    ):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime") from e

        try:
            import dlib  # type: ignore
        except Exception as e:
            raise RuntimeError("dlib not installed. Install dlib-bin or build dlib.") from e

        self._ort = ort
        self.dlib = dlib

        self.det_upsample = int(det_upsample)
        self.arcface_padding = float(arcface_padding)
        self.arcface_preproc = str(arcface_preproc).lower()

        shape_path = os.environ.get("DLIB_SHAPE_PREDICTOR_PATH")
        if not shape_path:
            raise RuntimeError("Set DLIB_SHAPE_PREDICTOR_PATH to your 68-landmarks .dat file.")
        if not os.path.exists(shape_path):
            raise RuntimeError(f"DLIB_SHAPE_PREDICTOR_PATH not found: {shape_path}")

        onnx_path = os.environ.get("ARCFACE_ONNX_PATH")
        if not onnx_path:
            raise RuntimeError("Set ARCFACE_ONNX_PATH to your ArcFace .onnx file.")
        if not os.path.exists(onnx_path):
            raise RuntimeError(f"ARCFACE_ONNX_PATH not found: {onnx_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_path)

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, rgb_112: np.ndarray) -> np.ndarray:
        """
        ArcFace preprocessing.

        - insightface: (x - 127.5) / 128.0
        - legacy:     (x - 127.5) / 127.5
        Output is NCHW float32.
        """
        x = rgb_112.astype(np.float32)
        if self.arcface_preproc == "legacy":
            x = (x - 127.5) / 127.5
        else:
            x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, 0)        # CHW -> NCHW
        return x

    def embed(self, bgr_image: np.ndarray) -> List[FaceEmbedding]:
        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1], dtype=np.uint8)
        faces = self.detector(rgb, self.det_upsample)

        out: List[FaceEmbedding] = []
        for f in faces:
            shape = self.shape_predictor(rgb, f)

            chip = self.dlib.get_face_chip(
                rgb, shape, size=112, padding=float(self.arcface_padding)
            )
            chip = np.ascontiguousarray(chip, dtype=np.uint8)

            inp = self._preprocess(chip)
            emb = self.session.run([self.output_name], {self.input_name: inp})[0]
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            out.append(
                FaceEmbedding(
                    embedding=emb,
                    bbox_xyxy=(int(f.left()), int(f.top()), int(f.right()), int(f.bottom())),
                )
            )
        return out
