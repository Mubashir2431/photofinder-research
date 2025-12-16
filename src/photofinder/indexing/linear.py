from __future__ import annotations
import numpy as np

def l2_search(query: np.ndarray, mat: np.ndarray, top_k: int = 10):
    """Return (indices, distances) for L2 nearest neighbors."""
    # mat: (N, D), query: (D,)
    d = np.linalg.norm(mat - query[None, :], axis=1)
    k = min(top_k, d.shape[0])
    idx = np.argpartition(d, kth=k-1)[:k]
    idx = idx[np.argsort(d[idx])]
    return idx, d[idx]
