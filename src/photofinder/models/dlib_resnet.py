from __future__ import annotations
import os
import numpy as np
from .base import FaceEmbedder, FaceEmbedding

class DlibResnetEmbedder(FaceEmbedder):
    """Dlib face_recognition_model_v1 (128-D). Requires local model files."""
    name = "dlib_resnet_v1"
    dim = 128

    def __init__(self):
        try:
            import dlib  # type: ignore
        except Exception as e:
            raise RuntimeError("dlib is not installed. `pip install -e '.[dlib]'`") from e

        shape_path = os.environ.get("DLIB_SHAPE_PREDICTOR_PATH")
        rec_path = os.environ.get("DLIB_FACE_REC_MODEL_PATH")
        if not shape_path or not rec_path:
            raise RuntimeError(
                "Set env vars DLIB_SHAPE_PREDICTOR_PATH and DLIB_FACE_REC_MODEL_PATH "
                "to your .dat model file paths."
            )

        self.dlib = dlib
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_path)
        self.face_rec = dlib.face_recognition_model_v1(rec_path)

    def embed(self, bgr_image: np.ndarray):
        # dlib expects RGB
        rgb = bgr_image[:, :, ::-1]
        faces = self.detector(rgb)
        out = []
        for f in faces:
            shape = self.shape_predictor(rgb, f)
            v = self.face_rec.compute_face_descriptor(rgb, shape)
            emb = np.asarray(v, dtype=np.float32)
            out.append(FaceEmbedding(embedding=emb, bbox_xyxy=(f.left(), f.top(), f.right(), f.bottom())))
        return out
