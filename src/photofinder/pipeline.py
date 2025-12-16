from __future__ import annotations
import os, time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from photofinder.datasets.imagefolder import load_imagefolder
from photofinder.models.registry import get_embedder
from photofinder.indexing.store import save_index_npz
from photofinder.utils import ensure_dir, get_run_info, write_json

def index_imagefolder(dataset_root: str, model_name: str, out_dir: str) -> str:
    """Index dataset_root/<label>/<img> into out_dir/index.npz"""
    ensure_dir(out_dir)
    runinfo = get_run_info()
    write_json(os.path.join(out_dir, "runinfo.json"), runinfo)

    embedder = get_embedder(model_name)
    samples = load_imagefolder(dataset_root)

    emb_list: List[np.ndarray] = []
    meta: List[Dict] = []

    t0 = time.time()
    for s in tqdm(samples, desc=f"Indexing ({model_name})"):
        img = cv2.imread(s.path)
        if img is None:
            continue
        faces = embedder.embed(img)
        # For a clean benchmark, keep ONLY the largest face (common simplification)
        if not faces:
            continue
        # choose largest bbox area
        best = max(faces, key=lambda f: (f.bbox_xyxy[2]-f.bbox_xyxy[0]) * (f.bbox_xyxy[3]-f.bbox_xyxy[1]))
        emb_list.append(best.embedding)
        meta.append({"path": s.path, "label": s.label, "bbox": best.bbox_xyxy})

    if not emb_list:
        raise RuntimeError("No embeddings produced. Check model installation and dataset format.")

    emb = np.stack(emb_list).astype(np.float32)
    index_path = os.path.join(out_dir, "index.npz")
    save_index_npz(index_path, emb, meta)

    timings = {
        "n_images_total": len(samples),
        "n_embeddings": int(emb.shape[0]),
        "embedder": model_name,
        "seconds_total": time.time() - t0,
        "seconds_per_image": (time.time() - t0) / max(1, len(samples)),
        "dim": int(emb.shape[1]),
    }
    write_json(os.path.join(out_dir, "timings.json"), timings)
    return index_path
