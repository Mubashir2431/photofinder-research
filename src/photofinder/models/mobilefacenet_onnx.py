from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import FaceEmbedder, FaceEmbedding

_ARCFACE_112_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _repo_root() -> Path:
    # .../src/photofinder/models/mobilefacenet_onnx.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _default_mbf_path() -> Path:
    return _repo_root() / "models" / "insightface" / "buffalo_sc" / "w600k_mbf.onnx"


def _default_yunet_path() -> Path:
    return _repo_root() / "models" / "opencv" / "yunet.onnx"


def _to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def _bbox_area_xyxy(b: List[int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _estimate_affine_5pt(src5: np.ndarray, dst5: np.ndarray) -> np.ndarray:
    """
    Estimate 2x3 affine transform from src5 -> dst5 using least squares.
    """
    n = src5.shape[0]
    X = np.zeros((2 * n, 6), dtype=np.float32)
    Y = np.zeros((2 * n,), dtype=np.float32)

    for i, ((x, y), (u, v)) in enumerate(zip(src5, dst5)):
        X[2 * i, 0:3] = [x, y, 1.0]
        Y[2 * i] = u

        X[2 * i + 1, 3:6] = [x, y, 1.0]
        Y[2 * i + 1] = v

    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coef.reshape(2, 3).astype(np.float32)


def _preprocess_insightface(bgr_112: np.ndarray) -> np.ndarray:
    """
    InsightFace-style preprocessing:
      BGR -> RGB
      float32
      (img - 127.5) / 128
      NCHW
    """
    rgb = bgr_112[:, :, ::-1].astype(np.float32)
    rgb = (rgb - 127.5) / 128.0
    chw = np.transpose(rgb, (2, 0, 1))
    return chw[np.newaxis, ...].astype(np.float32)


class MobileFaceNetOnnxEmbedder(FaceEmbedder):
    """
    YuNet (detection + 5pt landmarks) -> 112x112 align -> MobileFaceNet (ONNXRuntime)
    """
    name = "mobilefacenet_onnx"
    dim = 512

    def __init__(
        self,
        model_path: Optional[str] = None,
        yunet_path: Optional[str] = None,
        det_upsample: int = 1,
        yunet_score_thresh: float = 0.9,
        yunet_nms_thresh: float = 0.3,
        yunet_topk: int = 5000,
    ):
        self.det_upsample = int(det_upsample)

        mp = Path(model_path) if model_path else _default_mbf_path()
        if not mp.exists():
            raise FileNotFoundError(
                f"MobileFaceNet ONNX not found: {mp}\n"
                f"Expected: models/insightface/buffalo_sc/w600k_mbf.onnx"
            )
        self.model_path = str(mp)

        yp = Path(yunet_path) if yunet_path else _default_yunet_path()
        if not yp.exists():
            raise FileNotFoundError(
                f"YuNet ONNX not found: {yp}\n"
                f"Expected: models/opencv/yunet.onnx"
            )
        self.yunet_path = str(yp)

        import onnxruntime as ort  # type: ignore
        self.sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        import cv2  # type: ignore
        self.detector = cv2.FaceDetectorYN_create(
            self.yunet_path,
            "",
            (320, 320),
            float(yunet_score_thresh),
            float(yunet_nms_thresh),
            int(yunet_topk),
        )

    def embed(self, bgr_image: np.ndarray) -> List[FaceEmbedding]:
        import cv2  # type: ignore

        if bgr_image is None or bgr_image.size == 0:
            return []

        scale = float(self.det_upsample)
        if scale > 1:
            img_up = cv2.resize(bgr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            img_up = bgr_image

        h, w = img_up.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img_up)

        if faces is None or len(faces) == 0:
            return []

        out: List[FaceEmbedding] = []

        for f in faces:
            # YuNet: [x, y, w, h, l0x, l0y, ... l4x, l4y, score]
            x, y, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])

            if scale > 1:
                x /= scale
                y /= scale
                fw /= scale
                fh /= scale

            bbox_xyxy = list(_to_xyxy(x, y, fw, fh))

            lms = np.array(
                [
                    [float(f[4]), float(f[5])],
                    [float(f[6]), float(f[7])],
                    [float(f[8]), float(f[9])],
                    [float(f[10]), float(f[11])],
                    [float(f[12]), float(f[13])],
                ],
                dtype=np.float32,
            )
            if scale > 1:
                lms /= scale

            M = _estimate_affine_5pt(lms, _ARCFACE_112_TEMPLATE)
            aligned = cv2.warpAffine(bgr_image, M, (112, 112), flags=cv2.INTER_LINEAR)

            inp = _preprocess_insightface(aligned)
            feat = self.sess.run([self.out_name], {self.in_name: inp})[0]
            vec = np.asarray(feat, dtype=np.float32).reshape(-1)

            out.append(FaceEmbedding(embedding=vec, bbox_xyxy=bbox_xyxy))

        # sort largest first so face-policy=largest is stable
        out.sort(key=lambda fe: _bbox_area_xyxy(fe.bbox_xyxy), reverse=True)
        return out
