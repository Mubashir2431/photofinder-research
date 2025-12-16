from __future__ import annotations
import json
import numpy as np
from typing import Any, Dict, List

def save_index_npz(path: str, emb: np.ndarray, meta: List[Dict[str, Any]]) -> None:
    # store meta as json string array for portability
    meta_json = np.array([json.dumps(m, ensure_ascii=False) for m in meta])
    np.savez_compressed(path, emb=emb.astype(np.float32), meta=meta_json)

def load_index_npz(path: str):
    data = np.load(path, allow_pickle=False)
    emb = data["emb"].astype(np.float32)
    meta = [json.loads(s) for s in data["meta"].tolist()]
    return emb, meta
