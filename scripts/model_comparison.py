"""Run the 4-model comparison (dlib, ArcFace, OpenCV SFace, MobileFaceNet) on one dataset.

For each model (sequentially, nothing else in parallel) this:
  1. builds the embedding index with photofinder.pipeline.index_imagefolder (largest face per image),
  2. evaluates exact (brute-force, cosine) retrieval: Rank-1, Recall@5/10, MRR,
  3. writes results/raw/<model>.json.
`--collect` merges raw/*.json into results/model_comparison.{csv,json}.

Throughput (images/s) = images in the dataset / wall-clock seconds of the indexing loop. The loop
covers JPEG decode + face detection + alignment + embedding + L2-normalisation, one image at a time
on CPU; it does NOT include loading the model weights or the retrieval evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

MODELS = ["dlib_resnet_v1", "arcface_onnx", "opencv_sface", "mobilefacenet_onnx"]


def cpu_name() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or platform.machine()


def run_model(model: str, dataset: str, out_root: Path, raw_dir: Path) -> dict:
    from photofinder.eval.retrieval import eval_retrieval
    from photofinder.indexing.store import load_index_npz
    from photofinder.pipeline import index_imagefolder

    out_dir = out_root / model
    t0 = time.time()
    index_path = index_imagefolder(dataset, model, str(out_dir))
    wall = time.time() - t0
    timings = json.loads((out_dir / "timings.json").read_text())
    emb, meta = load_index_npz(index_path)
    m = eval_retrieval(emb, meta, top_k=10, metric="cosine")
    n_img = timings["n_images_total"]
    rec = {
        "model": model,
        "rank1": m.rank1,
        "recall_at_5": m.recall_at_5,
        "recall_at_10": m.recall_at_10,
        "mrr": m.mrr,
        "n_queries": m.n_queries,
        "n_images": n_img,
        "n_embeddings": timings["n_embeddings"],
        "no_face_images": n_img - timings["n_embeddings"],
        "dim": timings["dim"],
        "index_seconds": timings["seconds_total"],
        "images_per_s": n_img / timings["seconds_total"],
        "seconds_per_image": timings["seconds_total"] / n_img,
        "wall_seconds_incl_model_load": wall,
        "dataset": dataset,
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{model}.json").write_text(json.dumps(rec, indent=2))
    return rec


def collect(raw_dir: Path, out_csv: Path, out_json: Path) -> list:
    recs = []
    for m in MODELS:
        p = raw_dir / f"{m}.json"
        if p.exists():
            recs.append(json.loads(p.read_text()))
    if not recs:
        raise SystemExit("no raw results found")
    keys = list(recs[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    out_json.write_text(json.dumps(recs, indent=2))
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="imagefolder root: <root>/<person>/<img>.jpg")
    ap.add_argument("--out-root", default="runs/model_comparison")
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--collect", action="store_true", help="only merge results/raw/*.json into CSV/JSON")
    a = ap.parse_args()
    res = Path(a.results)
    raw = res / "raw"
    if not a.collect:
        for m in a.models:
            rec = run_model(m, a.dataset, Path(a.out_root), raw)
            print(json.dumps(rec, indent=2), flush=True)
    collect(raw, res / "model_comparison.csv", res / "model_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
