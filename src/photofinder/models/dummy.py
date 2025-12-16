from __future__ import annotations
import numpy as np
from .base import FaceEmbedder, FaceEmbedding

class DummyEmbedder(FaceEmbedder):
    """Non-face baseline (for plumbing). NOT a real model."""
    name = "dummy"
    dim = 128

    def embed(self, bgr_image: np.ndarray):
        h, w = bgr_image.shape[:2]
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(self.dim,)).astype(np.float32)
        return [FaceEmbedding(embedding=emb, bbox_xyxy=(0, 0, w-1, h-1))]
