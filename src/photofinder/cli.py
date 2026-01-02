from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import typer
from rich import print

from photofinder.pipeline import index_imagefolder
from photofinder.indexing.store import load_index_npz
from photofinder.utils import ensure_dir, write_json

app = typer.Typer(no_args_is_help=True)

# -------------------------
# Helpers
# -------------------------


def _bbox_area_xyxy(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _largest_face(faces):
    return max(faces, key=lambda fe: _bbox_area_xyxy(fe.bbox_xyxy))


def _first_face(faces):
    return faces[0]


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (np.linalg.norm(x) + 1e-12)


def _infer_index_settings(meta: List[Dict]) -> Dict:
    """
    Try to infer config from index metadata. We intentionally keep this loose because
    old indices might not have all keys.
    """
    cfg: Dict = {}
    if meta and isinstance(meta[0], dict):
        m0 = meta[0]
        for k in ("embedder", "metric", "normalize", "face_policy", "det_upsample"):
            if k in m0:
                cfg[k] = m0[k]
    return cfg


def _prepare_for_metric(E: np.ndarray, q: np.ndarray, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (E2, q2) prepared for the chosen metric.
    - cosine: L2-normalize rows and query, then score = E2 @ q2
    - l2: leave as-is, distance^2 computed later
    """
    metric = metric.lower()
    if metric == "cosine":
        E2 = _l2_normalize_rows(E)
        q2 = _l2_normalize_vec(q)
        return E2, q2
    if metric == "l2":
        return E, q
    raise ValueError("metric must be 'cosine' or 'l2'")


def _score_bruteforce(E: np.ndarray, q: np.ndarray, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (sorted_idx, scores_or_dist) for the entire index.
    For cosine: higher score is better.
    For l2: lower distance is better (we return dist2).
    """
    E2, q2 = _prepare_for_metric(E, q, metric)
    if metric == "cosine":
        scores = E2 @ q2
        idx = np.argsort(-scores)
        return idx, scores
    # l2
    diff = E2 - q2.reshape(1, -1)
    dist2 = np.sum(diff * diff, axis=1)
    idx = np.argsort(dist2)
    return idx, dist2


def _format_score(metric: str, s: float) -> str:
    if metric == "l2":
        return f"dist2={s:.4f}"
    return f"score={s:.4f}"


# -------------------------
# Commands
# -------------------------


@app.command()
def index(
    dataset: str = typer.Option(..., help="Path to dataset root: root/<label>/<image>"),
    model: str = typer.Option("dummy", help="Model name (dummy, dlib_resnet_v1, arcface_onnx, etc.)"),
    out: str = typer.Option(..., help="Output folder (will create index.npz)"),
    # Index-time knobs
    face_policy: str = typer.Option("largest", help="largest | first | all"),
    det_upsample: int = typer.Option(1, help="Detector upsample (0/1/2). Higher = detects smaller faces, slower."),
    min_face_area: int = typer.Option(0, help="Skip faces smaller than this bbox area in pixels^2."),
    max_faces: int = typer.Option(5, help="Max faces per image when face_policy=all."),
    fail_policy: str = typer.Option("skip", help="skip | error (what to do if no face found)"),
    metric: str = typer.Option("cosine", help="cosine | l2 (how retrieval/search will compare vectors)"),
    normalize: str = typer.Option("on", help="on | off (whether to L2-normalize stored embeddings)"),
    # ArcFace-specific knobs (safe to pass for other models; they will be ignored)
    arcface_padding: float = typer.Option(0.25, help="ArcFace alignment padding (dlib get_face_chip padding)."),
    arcface_preproc: str = typer.Option("insightface", help="insightface | legacy (normalization formula)"),
):
    """
    Build an embedding index (index.npz).
    Any change in index-time knobs requires rebuilding:
      photofinder index ...
      photofinder build-ann ...
    """
    index_path = index_imagefolder(
        dataset_root=dataset,
        model_name=model,
        out_dir=out,
        face_policy=face_policy,
        det_upsample=det_upsample,
        min_face_area=min_face_area,
        max_faces=max_faces,
        fail_policy=fail_policy,
        metric=metric,
        normalize=(normalize.lower() == "on"),
        arcface_padding=arcface_padding,
        arcface_preproc=arcface_preproc,
    )
    print(f"[green]Saved:[/green] {index_path}")


@app.command("build-ann")
def build_ann_cmd(
    index: str = typer.Option(..., help="Path to index.npz"),
    out: Optional[str] = typer.Option(None, help="Optional output path (file or directory). Default: beside index.npz"),
    # ANN build-time knobs
    ann_type: str = typer.Option("hnsw", help="flat | hnsw"),
    faiss_metric: Optional[str] = typer.Option(None, help="ip | l2. If omitted, inferred from index metric."),
    hnsw_m: int = typer.Option(32, help="HNSW M parameter (graph degree)."),
    ef_construction: int = typer.Option(200, help="HNSW efConstruction (build-time accuracy vs build time)."),
):
    """
    Build a FAISS ANN index (index.faiss) from an existing index.npz.

    Changing these knobs requires rebuilding ONLY index.faiss (vectors unchanged):
      photofinder build-ann ...
    """
    emb, meta = load_index_npz(index)
    inferred = _infer_index_settings(meta)
    metric = str(inferred.get("metric", "cosine")).lower()

    # Map overall metric -> FAISS metric default
    fm = (faiss_metric or ("ip" if metric == "cosine" else "l2")).lower()

    from photofinder.indexing.ann_faiss import default_faiss_path, save_index, save_index_meta

    out_path = out or default_faiss_path(index)
    saved = save_index(
        emb=np.asarray(emb, dtype=np.float32),
        out_path=out_path,
        ann_type=ann_type,
        metric=fm,
        hnsw_m=hnsw_m,
        ef_construction=ef_construction,
        # For cosine/IP, we normalize inside save_index so ANN matches cosine.
        normalize=(metric == "cosine"),
    )
    save_index_meta(
        saved,
        {
            "ann_type": ann_type,
            "faiss_metric": fm,
            "hnsw_m": hnsw_m,
            "ef_construction": ef_construction,
            "index_metric": metric,
        },
    )
    print(f"[green]Saved:[/green] {saved}")


@app.command()
def search(
    index: str = typer.Option(..., help="Path to index.npz (built by photofinder index)"),
    query: str = typer.Option(..., help="Path to a query image"),
    topk: int = typer.Option(10, help="Number of nearest results to show"),
    # Search-time knobs
    backend: str = typer.Option("auto", help="auto | bruteforce | ann | both"),
    ef_search: int = typer.Option(64, help="If ANN present (HNSW), efSearch (higher=more accurate, slower)."),
    ann_k: int = typer.Option(200, help="How many ANN candidates to fetch before truncating/reranking."),
    rerank: str = typer.Option("on", help="on | off. If on, rerank ANN candidates with exact metric."),
    # Model / metric inference overrides
    model: Optional[str] = typer.Option(None, help="Optional override model (otherwise read from index meta)"),
    metric: Optional[str] = typer.Option(None, help="Optional override metric (otherwise read from index meta)"),
    query_face_policy: str = typer.Option("largest", help="largest | first (how to pick face in the QUERY image)"),
):
    """
    Search: embed ONE query image and return the most similar images from the index.

    - If backend=auto: uses ANN if index.faiss exists, else brute force.
    - If backend=both: prints ANN results and brute force results (debug/validation).
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("opencv-python is required for search. Install: pip install opencv-python") from e

    emb, meta = load_index_npz(index)
    inferred = _infer_index_settings(meta)

    model_name = model or inferred.get("embedder")
    if not model_name:
        raise RuntimeError("Could not infer model from index meta. Pass --model explicitly.")

    metric_name = (metric or inferred.get("metric") or "cosine").lower()
    if metric_name not in ("cosine", "l2"):
        raise RuntimeError("metric must be 'cosine' or 'l2'")

    from photofinder.models.registry import get_embedder

    # Reuse detector upsample from the index (so query-time matches index-time).
    embedder = get_embedder(model_name, det_upsample=int(inferred.get("det_upsample", 1)))

    img = cv2.imread(query)
    if img is None:
        raise FileNotFoundError(f"Could not read query image: {query}")

    faces = embedder.embed(img)
    if not faces:
        raise RuntimeError("No face detected in query image.")

    if query_face_policy == "first":
        q_face = _first_face(faces)
    else:
        q_face = _largest_face(faces)

    q_raw = np.asarray(q_face.embedding, dtype=np.float32).reshape(-1)

    # Precompute prepared index & query for exact scoring / reranking
    E_raw = np.asarray(emb, dtype=np.float32)
    E_prep, q_prep = _prepare_for_metric(E_raw, q_raw, metric_name)

    faiss_path = Path(index).with_suffix(".faiss")

    want_backend = backend.lower()
    if want_backend == "auto":
        want_backend = "ann" if faiss_path.exists() else "bruteforce"

    do_rerank = rerank.lower() == "on"

    def _print_results(title_backend: str, ids: np.ndarray, scores: np.ndarray):
        print(f"[cyan]Model:[/cyan] {model_name}")
        print(f"[cyan]Backend:[/cyan] {title_backend}")
        print(f"[cyan]Metric:[/cyan] {metric_name}")
        print(f"[cyan]Query:[/cyan] {query}")
        print(f"[cyan]Top-{len(ids)} results:[/cyan]\n")
        for rank, (i, s) in enumerate(zip(ids.tolist(), scores.tolist()), start=1):
            m = meta[i] if i < len(meta) else {}
            path = (m.get("path") if isinstance(m, dict) else None) or str(m)
            label = m.get("label") if isinstance(m, dict) else None
            if label:
                print(f"{rank:02d}. {_format_score(metric_name, float(s))}  label={label}  path={path}")
            else:
                print(f"{rank:02d}. {_format_score(metric_name, float(s))}  path={path}")

    def _exact_topk():
        idx_all, vals_all = _score_bruteforce(E_raw, q_raw, metric_name)
        ids = idx_all[: max(1, topk)]
        scores = vals_all[ids]
        return ids.astype(np.int64), scores.astype(np.float32)

    def _ann_topk():
        if not faiss_path.exists():
            raise FileNotFoundError(f"ANN index not found: {faiss_path}. Run: photofinder build-ann --index {index}")

        from photofinder.indexing.ann_faiss import load_index as load_faiss_index, search_index as faiss_search

        ann = load_faiss_index(str(faiss_path), ef_search=ef_search)
        d_or_s, ids = faiss_search(ann, q_prep, topk=max(ann_k, topk))

        # Filter invalid ids
        valid = ids >= 0
        ids = ids[valid].astype(np.int64)
        d_or_s = d_or_s[valid].astype(np.float32)

        # Rerank candidates with exact metric (recommended for correctness)
        if do_rerank:
            if metric_name == "cosine":
                cand_scores = (E_prep[ids] @ q_prep).astype(np.float32)
                order = np.argsort(-cand_scores)
                ids = ids[order]
                d_or_s = cand_scores[order]
            else:
                diff = E_prep[ids] - q_prep.reshape(1, -1)
                cand_dist2 = np.sum(diff * diff, axis=1).astype(np.float32)
                order = np.argsort(cand_dist2)
                ids = ids[order]
                d_or_s = cand_dist2[order]

        # Truncate
        ids = ids[: max(1, topk)]
        d_or_s = d_or_s[: max(1, topk)]
        return ids, d_or_s

    if want_backend == "bruteforce":
        ids, scores = _exact_topk()
        _print_results("bruteforce", ids, scores)
        return

    if want_backend == "ann":
        ids, scores = _ann_topk()
        _print_results("ann (faiss)", ids, scores)
        return

    if want_backend == "both":
        ids_a, scores_a = _ann_topk()
        _print_results(f"ann (faiss){' + rerank' if do_rerank else ''}", ids_a, scores_a)
        print("\n" + "-" * 80 + "\n")
        ids_b, scores_b = _exact_topk()
        _print_results("bruteforce", ids_b, scores_b)
        return

    raise RuntimeError("backend must be auto|bruteforce|ann|both")


@app.command("eval-retrieval")
def eval_retrieval_cmd(
    index: str = typer.Option(..., help="Path to index.npz"),
    out: str = typer.Option(..., help="Output folder for metrics json"),
    backend: str = typer.Option("bruteforce", help="bruteforce | ann | both"),
    top_k: int = typer.Option(10, help="Compute Rank1/Recall@k/MRR up to this k (10 typical)"),
    # ANN search-time knobs
    ann_k: int = typer.Option(200, help="How many ANN candidates to fetch per query (bigger = safer, slower)."),
    ef_search: int = typer.Option(64, help="HNSW efSearch."),
    rerank: str = typer.Option("on", help="on | off (rerank ANN candidates with exact metric)."),
    # Metric override
    metric: Optional[str] = typer.Option(None, help="Optional override metric (otherwise inferred from index meta)."),
):
    """
    Evaluate retrieval on identities with >=2 images.
    """
    ensure_dir(out)
    emb, meta = load_index_npz(index)
    inferred = _infer_index_settings(meta)

    metric_name = (metric or inferred.get("metric") or "cosine").lower()
    if metric_name not in ("cosine", "l2"):
        raise RuntimeError("metric must be 'cosine' or 'l2'")

    results: Dict[str, Dict] = {}

    from photofinder.eval.retrieval import eval_retrieval

    if backend in ("bruteforce", "both"):
        m = eval_retrieval(
            emb=np.asarray(emb, dtype=np.float32),
            meta=meta,
            top_k=top_k,
            metric=metric_name,
            ann_index=None,
        )
        results["bruteforce"] = m.__dict__
        write_json(os.path.join(out, "metrics_retrieval_bruteforce.json"), results["bruteforce"])
        print({"backend": "bruteforce", **results["bruteforce"]})

    if backend in ("ann", "both"):
        faiss_path = str(Path(index).with_suffix(".faiss"))
        from photofinder.indexing.ann_faiss import load_index as load_faiss_index

        ann = load_faiss_index(faiss_path, ef_search=ef_search)
        m = eval_retrieval(
            emb=np.asarray(emb, dtype=np.float32),
            meta=meta,
            top_k=top_k,
            metric=metric_name,
            ann_index=ann,
            ann_search_k=ann_k,
            rerank=(rerank.lower() == "on"),
        )
        ann_dict = m.__dict__ | {"ann_k": ann_k, "ef_search": ef_search, "rerank": rerank.lower()}
        results["ann"] = ann_dict
        write_json(os.path.join(out, "metrics_retrieval_ann.json"), ann_dict)
        print({"backend": "ann", **ann_dict})

    if backend == "both":
        write_json(os.path.join(out, "metrics_retrieval.json"), results)


if __name__ == "__main__":
    app()
