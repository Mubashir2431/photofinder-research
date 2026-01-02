from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def default_faiss_path(index_npz_path: str) -> str:
    """Given runs/.../index.npz -> runs/.../index.faiss"""
    return str(Path(index_npz_path).with_suffix(".faiss"))


def _coerce_out_path(out_path: str) -> Path:
    outp = Path(out_path)
    # If user passed a directory, write index.faiss inside it.
    if outp.exists() and outp.is_dir():
        outp = outp / "index.faiss"
    # Ensure .faiss suffix
    if outp.suffix.lower() != ".faiss":
        outp = outp.with_suffix(".faiss")
    outp.parent.mkdir(parents=True, exist_ok=True)
    return outp


def save_index(
    emb: np.ndarray,
    out_path: str,
    *,
    ann_type: str = "hnsw",   # flat | hnsw
    metric: str = "ip",       # ip | l2
    hnsw_m: int = 32,
    ef_construction: int = 200,
    normalize: bool = True,   # if metric=ip and you want cosine, normalize vectors
) -> str:
    """
    Build + save a FAISS index.

    - metric="ip" with normalize=True approximates cosine similarity.
    - metric="l2" uses squared L2 distances.
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is not installed. Install: pip install faiss-cpu") from e

    E = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
    if metric.lower() == "ip" and normalize:
        E = _l2_normalize_rows(E)

    d = int(E.shape[1])
    ann_type = ann_type.lower()
    metric = metric.lower()

    if ann_type == "flat":
        if metric == "ip":
            index = faiss.IndexFlatIP(d)
        elif metric == "l2":
            index = faiss.IndexFlatL2(d)
        else:
            raise ValueError("metric must be 'ip' or 'l2'")
    elif ann_type == "hnsw":
        if metric == "ip":
            index = faiss.IndexHNSWFlat(d, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        elif metric == "l2":
            index = faiss.IndexHNSWFlat(d, int(hnsw_m), faiss.METRIC_L2)
        else:
            raise ValueError("metric must be 'ip' or 'l2'")
        if hasattr(index, "hnsw"):
            index.hnsw.efConstruction = int(ef_construction)
    else:
        raise ValueError("ann_type must be 'flat' or 'hnsw'")

    index.add(E)

    outp = _coerce_out_path(out_path)
    faiss.write_index(index, str(outp))
    return str(outp)


def save_index_meta(faiss_path: str, meta: Dict[str, Any]) -> str:
    """
    Writes sidecar JSON next to .faiss to preserve build config.
    (e.g. index.faiss.meta.json)
    """
    import json

    p = Path(faiss_path)
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return str(meta_path)


def load_index(path: str, ef_search: int | None = None, nprobe: int | None = None, **_):
    """
    Load FAISS index. If path is a directory, looks for path/index.faiss.
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is not installed. Install: pip install faiss-cpu") from e

    p = Path(path)
    if p.exists() and p.is_dir():
        p = p / "index.faiss"

    if not p.exists():
        raise FileNotFoundError(f"FAISS index not found: {p}")

    index = faiss.read_index(str(p))

    if ef_search is not None and hasattr(index, "hnsw"):
        index.hnsw.efSearch = int(ef_search)

    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)

    return index


def search_index(index, q: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    q: (dim,) float32 -> runs FAISS search, returns (dist_or_score, indices)

    NOTE:
      - For IP indexes: higher is better (similarity).
      - For L2 indexes: lower is better (squared distance).
    """
    q = np.asarray(q, dtype=np.float32).reshape(1, -1)
    d_or_s, idx = index.search(q, int(topk))
    return d_or_s[0], idx[0]


def l2dist2_to_cos(d2: np.ndarray) -> np.ndarray:
    """
    Convert squared L2 distance between unit vectors to cosine similarity:
      ||u - v||^2 = 2 - 2cos  =>  cos = 1 - d2/2

    Only valid if both vectors are L2-normalized.
    """
    d2 = np.asarray(d2, dtype=np.float32)
    return 1.0 - 0.5 * d2
