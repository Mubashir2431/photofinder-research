from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import FaceEmbedder, FaceEmbedding


def _repo_root() -> Path:
    # src/photofinder/models/opencv_sface.py -> repo root is 3 parents above "src"
    return Path(__file__).resolve().parents[3]


def _default_model_paths() -> Tuple[str, str]:
    root = _repo_root()
    yunet = root / "models" / "opencv" / "yunet.onnx"
    sface = root / "models" / "opencv" / "sface.onnx"
    return str(yunet), str(sface)


def _get_bbox_xyxy(fe: FaceEmbedding) -> Optional[List[int]]:
    """
    Your project uses bbox_xyxy, but keep this robust in case other models use bbox.
    """
    for attr in ("bbox_xyxy", "bbox", "xyxy", "box", "rect"):
        if hasattr(fe, attr):
            bb = getattr(fe, attr)
            if bb is None:
                return None
            return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
    return None


def _make_face_embedding(vec: np.ndarray, bbox_xyxy: List[int], score: float) -> FaceEmbedding:
    """
    Create FaceEmbedding robustly even if the dataclass has different field names.
    """
    sig = inspect.signature(FaceEmbedding)
    params = list(sig.parameters.values())

    # If it accepts 3 positionals, try (embedding, bbox, score)
    if len(params) >= 3:
        try:
            return FaceEmbedding(vec, bbox_xyxy, float(score))  # type: ignore[misc]
        except TypeError:
            pass

    # Try (embedding, bbox)
    try:
        return FaceEmbedding(vec, bbox_xyxy)  # type: ignore[misc]
    except TypeError:
        # Fall back to kwargs by common field names
        names = [p.name for p in params]
        kw = {}

        # embedding field
        if "embedding" in names:
            kw["embedding"] = vec
        elif "emb" in names:
            kw["emb"] = vec
        else:
            kw[names[0]] = vec

        # bbox field
        if "bbox_xyxy" in names:
            kw["bbox_xyxy"] = bbox_xyxy
        elif "bbox" in names:
            kw["bbox"] = bbox_xyxy
        else:
            if len(names) > 1:
                kw[names[1]] = bbox_xyxy

        # score field (optional)
        if "score" in names:
            kw["score"] = float(score)

        return FaceEmbedding(**kw)  # type: ignore[arg-type]


class OpenCVSFaceEmbedder(FaceEmbedder):
    """
    OpenCV YuNet detector + OpenCV SFace recognizer.

    Expects:
      models/opencv/yunet.onnx
      models/opencv/sface.onnx

    Optional env overrides:
      OPENCV_YUNET_PATH
      OPENCV_SFACE_PATH
    """
    name = "opencv_sface"

    def __init__(
        self,
        det_upsample: int = 1,
        min_face_area: int = 0,
        yunet_path: Optional[str] = None,
        sface_path: Optional[str] = None,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "OpenCV build missing FaceDetectorYN / FaceRecognizerSF. "
                "Install opencv-contrib-python and ensure you're importing that one."
            )

        env_yunet = os.environ.get("OPENCV_YUNET_PATH")
        env_sface = os.environ.get("OPENCV_SFACE_PATH")
        d_yunet, d_sface = _default_model_paths()

        self.yunet_path = yunet_path or env_yunet or d_yunet
        self.sface_path = sface_path or env_sface or d_sface

        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(f"YuNet model not found: {self.yunet_path}")
        if not os.path.exists(self.sface_path):
            raise FileNotFoundError(f"SFace model not found: {self.sface_path}")

        self.det_upsample = int(det_upsample)
        self.min_face_area = int(min_face_area)

        # Create once; we set input size per-image
        self._detector = cv2.FaceDetectorYN.create(
            self.yunet_path,
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )
        self._recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")

        # SFace is typically 128-D, but keep flexible
        self.dim = 128

    def _upsample_if_needed(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        det_upsample semantics:
          0 -> no scaling
          1 -> 2x
          2 -> 4x
        """
        if self.det_upsample <= 0:
            return img, 1.0
        scale = float(2 ** self.det_upsample)
        up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return up, scale

    def embed(self, bgr_image: np.ndarray) -> List[FaceEmbedding]:
        if bgr_image is None or bgr_image.size == 0:
            return []

        img_up, scale = self._upsample_if_needed(bgr_image)
        h, w = img_up.shape[:2]
        self._detector.setInputSize((w, h))

        _, faces = self._detector.detect(img_up)
        if faces is None or len(faces) == 0:
            return []

        out: List[FaceEmbedding] = []

        # IMPORTANT: In your OpenCV 4.10 output, score is the LAST value (face[-1]),
        # and landmarks are in the middle. The first 4 are still x,y,w,h.
        for face in faces:
            x = float(face[0])
            y = float(face[1])
            fw = float(face[2])
            fh = float(face[3])
            score = float(face[-1])  # ✅ correct for your output

            # bbox in upsampled coords
            x1_u = int(max(0, np.floor(x)))
            y1_u = int(max(0, np.floor(y)))
            x2_u = int(min(w - 1, np.ceil(x + fw)))
            y2_u = int(min(h - 1, np.ceil(y + fh)))
            if x2_u <= x1_u or y2_u <= y1_u:
                continue

            # min_face_area filter in ORIGINAL image coords
            bw_u, bh_u = (x2_u - x1_u), (y2_u - y1_u)
            bw = bw_u / scale
            bh = bh_u / scale
            if self.min_face_area > 0 and (bw * bh) < self.min_face_area:
                continue

            try:
                aligned = self._recognizer.alignCrop(img_up, face)
                feat = self._recognizer.feature(aligned)  # (1, D)
            except Exception:
                continue

            vec = np.asarray(feat, dtype=np.float32).reshape(-1)
            if vec.size != self.dim:
                self.dim = int(vec.size)

            # convert bbox back to ORIGINAL image coords
            x1 = int(round(x1_u / scale))
            y1 = int(round(y1_u / scale))
            x2 = int(round(x2_u / scale))
            y2 = int(round(y2_u / scale))
            bbox_xyxy = [x1, y1, x2, y2]

            out.append(_make_face_embedding(vec, bbox_xyxy, score))

        # Stable ordering: biggest face first (works with bbox_xyxy OR bbox)
        def _area(fe: FaceEmbedding) -> int:
            bb = _get_bbox_xyxy(fe)
            if not bb:
                return -1
            return int((bb[2] - bb[0]) * (bb[3] - bb[1]))

        out.sort(key=_area, reverse=True)
        return out
