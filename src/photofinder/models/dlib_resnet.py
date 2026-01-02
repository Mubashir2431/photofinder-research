from __future__ import annotations

import os
from typing import List

import numpy as np

from .base import FaceEmbedder, FaceEmbedding


class DlibResnetEmbedder(FaceEmbedder):
    """
    Dlib's face_recognition_model_v1 (128-D) baseline.

    Requires:
      - DLIB_SHAPE_PREDICTOR_PATH (68 landmarks .dat)
      - DLIB_FACE_REC_MODEL_PATH  (dlib_face_recognition_resnet_model_v1.dat)

    Knobs supported:
      - det_upsample (0/1/2)
      - chip_padding (float)
      - num_jitters (int)
    """

    name = "dlib_resnet_v1"
    dim = 128

    def __init__(
        self,
        *,
        det_upsample: int = 1,
        chip_padding: float = 0.25,
        num_jitters: int = 0,
    ):
        try:
            import dlib  # type: ignore
        except Exception as e:
            raise RuntimeError("dlib not installed. Install dlib-bin or build dlib.") from e

        self.dlib = dlib
        self.det_upsample = int(det_upsample)
        self.chip_padding = float(chip_padding)
        self.num_jitters = int(num_jitters)

        shape_path = os.environ.get("DLIB_SHAPE_PREDICTOR_PATH")
        if not shape_path:
            raise RuntimeError("Set DLIB_SHAPE_PREDICTOR_PATH to your 68-landmarks .dat file.")
        if not os.path.exists(shape_path):
            raise RuntimeError(f"DLIB_SHAPE_PREDICTOR_PATH not found: {shape_path}")

        rec_path = (
            os.environ.get("DLIB_FACE_REC_MODEL_PATH")
            or os.environ.get("DLIB_FACE_RECOGNITION_MODEL_PATH")
            or os.environ.get("DLIB_FACE_RECOGNITION_MODEL_V1_PATH")
        )
        if not rec_path:
            raise RuntimeError(
                "Set DLIB_FACE_REC_MODEL_PATH to your dlib_face_recognition_resnet_model_v1.dat"
            )
        if not os.path.exists(rec_path):
            raise RuntimeError(f"DLIB face recognition model not found: {rec_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_path)
        self.rec_model = dlib.face_recognition_model_v1(rec_path)

    def embed(self, bgr_image: np.ndarray) -> List[FaceEmbedding]:
        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1], dtype=np.uint8)
        rects = self.detector(rgb, self.det_upsample)

        out: List[FaceEmbedding] = []
        for r in rects:
            shape = self.shape_predictor(rgb, r)

            chip = self.dlib.get_face_chip(rgb, shape, size=150, padding=float(self.chip_padding))
            chip = np.ascontiguousarray(chip, dtype=np.uint8)

            try:
                vec = self.rec_model.compute_face_descriptor(chip, self.num_jitters)
            except TypeError:
                vec = self.rec_model.compute_face_descriptor(rgb, shape, self.num_jitters)

            emb = np.asarray(vec, dtype=np.float32).reshape(-1)

            out.append(
                FaceEmbedding(
                    embedding=emb,
                    bbox_xyxy=(int(r.left()), int(r.top()), int(r.right()), int(r.bottom())),
                )
            )
        return out
