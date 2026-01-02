from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

def run_cmd(cmd, cwd: Optional[Path] = None) -> float:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return dt

def read_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def find_index_for_ann_dir(ann_dir: Path) -> Path:
    # ann_dir example:
    # runs\sweeps\lfw\baseline\<model>\<base_run>\_ann\<ann_cfg>\index.faiss
    # we want:
    # runs\sweeps\lfw\baseline\<model>\<base_run>\index.npz
    # => go up two levels from ann_cfg dir to base_run dir
    # ann_dir is the ann_cfg directory (contains metrics_retrieval_ann.json)
    base_run_dir = ann_dir.parents[1]  # ...\<base_run>\_ann
    base_run_dir = base_run_dir.parent # ...\<base_run>
    idx = base_run_dir / "index.npz"
    if not idx.exists():
        raise FileNotFoundError(f"index.npz not found for ann run: {ann_dir}\nExpected: {idx}")
    return idx

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help=r"e.g. runs\sweeps\lfw")
    ap.add_argument("--ann-k", type=int, default=None, help="Override ann-k if you want")
    ap.add_argument("--ef-search", type=int, default=None, help="Override ef-search if you want")
    ap.add_argument("--rerank", choices=["on", "off"], default=None, help="Override rerank if you want")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.exists():
        raise FileNotFoundError(out_root)

    ann_metric_files = list(out_root.rglob("metrics_retrieval_ann.json"))
    ann_dirs = sorted({p.parent for p in ann_metric_files})

    patched = 0
    skipped = 0

    print(f"Found ANN run folders: {len(ann_dirs)}")

    for i, ann_dir in enumerate(ann_dirs, 1):
        timings_path = ann_dir / "timings.json"
        if timings_path.exists():
            skipped += 1
            continue

        idx_path = find_index_for_ann_dir(ann_dir)

        # Read existing ann metrics to infer params if you don't override
        m = read_json(ann_dir / "metrics_retrieval_ann.json")
        ann_k = args.ann_k if args.ann_k is not None else int(m.get("ann_k", 200))
        ef_search = args.ef_search if args.ef_search is not None else int(m.get("ef_search", 64))
        rerank = args.rerank if args.rerank is not None else str(m.get("rerank", "on"))

        print(f"\n[{i}/{len(ann_dirs)}] Patch: {ann_dir}")
        print(f"  index: {idx_path}")
        print(f"  ann_k={ann_k} ef_search={ef_search} rerank={rerank}")

        cmd = [
            "photofinder", "eval-retrieval",
            "--index", str(idx_path),
            "--out", str(ann_dir),
            "--backend", "ann",
            "--top-k", "10",
            "--ann-k", str(ann_k),
            "--ef-search", str(ef_search),
            "--rerank", rerank,
        ]

        if args.dry_run:
            print("  DRY RUN:", " ".join(cmd))
            continue

        dt = run_cmd(cmd)
        write_json(timings_path, {
            "patched": True,
            "eval_retrieval_ann_time_s": dt,
            "ann_k": ann_k,
            "ef_search": ef_search,
            "rerank": rerank,
        })
        patched += 1

    print(f"\nDone. patched={patched}, skipped_existing={skipped}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
