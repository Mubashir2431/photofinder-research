from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .base import FaceEmbedder, FaceEmbedding

# ArcFace 112x112 canonical 5-point template
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

def _shape68_to_5pts(shape) -> np.ndarray:
    pts = np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)],
        dtype=np.float32,
    )
    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    nose = pts[30]
    mouth_left = pts[48]
    mouth_right = pts[54]
    return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0)

def _align_112(bgr: np.ndarray, src5: np.ndarray) -> Optional[np.ndarray]:
    import cv2
    M, _ = cv2.estimateAffinePartial2D(src5, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)

class ArcFaceOnnxEmbedder(FaceEmbedder):
    """
    ArcFace (w600k_r50.onnx) via onnxruntime.
    Uses dlib for detection + 68 landmarks, then aligns using ArcFace 5-point template.

    Env vars:
      - DLIB_SHAPE_PREDICTOR_PATH (68 landmarks .dat)
      - ARCFACE_ONNX_PATH         (defaults to models/arcface/buffalo_l/w600k_r50.onnx)
    """
    name = "arcface_onnx"
    dim = 512

    def __init__(self):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime") from e

        try:
            import dlib  # type: ignore
        except Exception as e:
            raise RuntimeError("dlib is required for detection/landmarks.") from e

        # model path
        model_path = os.environ.get(
            "ARCFACE_ONNX_PATH",
            os.path.join("models", "arcface", "buffalo_l", "w600k_r50.onnx"),
        )
        if not os.path.exists(model_path):
            raise RuntimeError(f"ArcFace ONNX not found at: {model_path}")

        shape_path = os.environ.get("DLIB_SHAPE_PREDICTOR_PATH")
        if not shape_path or not os.path.exists(shape_path):
            raise RuntimeError("Set DLIB_SHAPE_PREDICTOR_PATH to your 68-landmarks .dat file.")

        self.dlib = dlib
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_path)

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, bgr_image: np.ndarray) -> List[FaceEmbedding]:
        import cv2

        bgr = np.ascontiguousarray(bgr_image, dtype=np.uint8)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)

        rects = self.detector(rgb, 1)  # upsample=1

        out: List[FaceEmbedding] = []
        for r in rects:
            shape = self.shape_predictor(rgb, r)
            src5 = _shape68_to_5pts(shape)

            aligned = _align_112(bgr, src5)
            if aligned is None:
                continue

            # Matches InsightFace default: (img - 127.5) / 127.5, swapRB=True :contentReference[oaicite:2]{index=2}
            blob = cv2.dnn.blobFromImage(
                aligned,
                scalefactor=1.0 / 127.5,
                size=(112, 112),
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
            ).astype(np.float32)

            vec = self.session.run([self.output_name], {self.input_name: blob})[0][0].astype(np.float32)

            # L2 normalize
            vec = vec / (np.linalg.norm(vec) + 1e-12)

            out.append(
                FaceEmbedding(
                    embedding=vec,
                    bbox_xyxy=(r.left(), r.top(), r.right(), r.bottom()),
                )
            )

        return out
