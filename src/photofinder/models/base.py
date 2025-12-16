from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class FaceEmbedding:
    embedding: np.ndarray  # (D,)
    bbox_xyxy: Tuple[int, int, int, int]  # left, top, right, bottom

class FaceEmbedder:
    """Interface for face detection + embedding extraction."""
    name: str = "base"
    dim: int = 0

    def embed(self, bgr_image: np.ndarray) -> List[FaceEmbedding]:
        raise NotImplementedError
