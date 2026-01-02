from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from photofinder.datasets.imagefolder import load_imagefolder
from photofinder.models.registry import get_embedder
from photofinder.indexing.store import save_index_npz
from photofinder.utils import ensure_dir, get_run_info, write_json


def _bbox_area_xyxy(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (np.linalg.norm(x) + 1e-12)


def index_imagefolder(
    dataset_root: str,
    model_name: str,
    out_dir: str,
    *,
    # Index-time knobs
    face_policy: str = "largest",      # largest | first | all
    det_upsample: int = 1,
    min_face_area: int = 0,
    max_faces: int = 5,                # only used for all
    fail_policy: str = "skip",         # skip | error
    metric: str = "cosine",            # cosine | l2 (logged; comparisons happen in cli/eval)
    normalize: bool = True,            # whether to store normalized embeddings
    # ArcFace-only knobs (ignored by other embedders)
    arcface_padding: float = 0.25,
    arcface_preproc: str = "insightface",
) -> str:
    """
    Index dataset_root/<label>/<img> into out_dir/index.npz.

    Any change to these parameters changes what vectors you store
    -> rebuild index.npz (and then rebuild index.faiss).
    """
    ensure_dir(out_dir)

    runinfo = get_run_info()
    cfg = {
        "dataset_root": dataset_root,
        "model": model_name,
        "face_policy": face_policy,
        "det_upsample": int(det_upsample),
        "min_face_area": int(min_face_area),
        "max_faces": int(max_faces),
        "fail_policy": fail_policy,
        "metric": metric,
        "normalize": bool(normalize),
        "arcface_padding": float(arcface_padding),
        "arcface_preproc": arcface_preproc,
    }
    write_json(os.path.join(out_dir, "runinfo.json"), runinfo)
    write_json(os.path.join(out_dir, "config.json"), cfg)

    embedder = get_embedder(
        model_name,
        det_upsample=int(det_upsample),
        arcface_padding=float(arcface_padding),
        arcface_preproc=str(arcface_preproc),
    )

    samples = load_imagefolder(dataset_root)

    emb_list: List[np.ndarray] = []
    meta: List[Dict] = []

    t0 = time.time()
    for s in tqdm(samples, desc=f"Indexing ({model_name})"):
        img = cv2.imread(s.path)
        if img is None:
            continue

        faces = embedder.embed(img)
        if not faces:
            if fail_policy == "error":
                raise RuntimeError(f"No face detected in image: {s.path}")
            continue

        # filter by min area
        keep = []
        for f in faces:
            if _bbox_area_xyxy(f.bbox_xyxy) >= int(min_face_area):
                keep.append(f)

        if not keep:
            if fail_policy == "error":
                raise RuntimeError(f"All detected faces were < min_face_area in image: {s.path}")
            continue

        fp = face_policy.lower()
        if fp == "first":
            selected = [keep[0]]
        elif fp == "largest":
            selected = [max(keep, key=lambda f: _bbox_area_xyxy(f.bbox_xyxy))]
        elif fp == "all":
            keep_sorted = sorted(keep, key=lambda f: _bbox_area_xyxy(f.bbox_xyxy), reverse=True)
            selected = keep_sorted[: max(1, int(max_faces))]
        else:
            raise ValueError("face_policy must be largest|first|all")

        for face_idx, f in enumerate(selected):
            v = np.asarray(f.embedding, dtype=np.float32).reshape(-1)
            if normalize:
                v = _l2_normalize_vec(v)

            emb_list.append(v)
            meta.append(
                {
                    "path": s.path,
                    "label": s.label,
                    "bbox": list(map(int, f.bbox_xyxy)),
                    "embedder": model_name,
                    "metric": metric,
                    "normalize": bool(normalize),
                    "face_policy": face_policy,
                    "det_upsample": int(det_upsample),
                    "face_idx": int(face_idx),
                }
            )

    if not emb_list:
        raise RuntimeError("No embeddings produced. Check model installation and dataset format.")

    emb = np.stack(emb_list).astype(np.float32)

    index_path = os.path.join(out_dir, "index.npz")
    save_index_npz(index_path, emb, meta)

    t_total = time.time() - t0
    timings = {
        "n_images_total": len(samples),
        "n_embeddings": int(emb.shape[0]),
        "embedder": model_name,
        "seconds_total": t_total,
        "seconds_per_image": t_total / max(1, len(samples)),
        "dim": int(emb.shape[1]),
    }
    write_json(os.path.join(out_dir, "timings.json"), timings)
    return index_path
