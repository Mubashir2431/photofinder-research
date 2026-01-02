#!/usr/bin/env python3
"""
PhotoFinder full sweep runner (v8)

What this does
- Runs *repeatable sweeps* across multiple embedders (models) and multiple knob configs.
- Builds embeddings (index.npz), evaluates brute-force retrieval, builds FAISS ANN (index.faiss),
  evaluates ANN retrieval, and logs everything to CSV + MD.

Why v8 (what it fixes vs v7)
- Uses SHORT, hash-based run folder names to avoid Windows path-length issues (common on OneDrive/Desktop).
- Creates ALL required directories explicitly.
- Writes CSV incrementally (flushes after each run) so you never lose 2 days of results.
- Safer formatting: never crashes when a metric is missing (None).

Run example (PowerShell)
python scripts\photofinder_full_sweep_v8.py `
  --dataset data\lfw\lfw_funneled `
  --out-root runs\sweeps\lfw `
  --models arcface_onnx dlib_resnet_v1 opencv_sface mobilefacenet_onnx `
  --phases baseline ann_knobs `
  --top-k 10 `
  --fast `
  --continue-on-error
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# Utilities
# ---------------------------

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def fmt_f(x: Any, nd: int = 4) -> str:
    """Safe float formatting (handles None/NaN/strings)."""
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def stable_hash(obj: Any, n: int = 10) -> str:
    """Stable short hash for configs (sorted JSON)."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd_live(cmd: List[str], cwd: Optional[Path] = None, dry_run: bool = False) -> float:
    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        return 0.0
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return dt


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------
# Config objects
# ---------------------------

@dataclass(frozen=True)
class IndexCfg:
    face_policy: str = "largest"
    det_upsample: int = 1
    min_face_area: int = 0
    max_faces: int = 5
    fail_policy: str = "skip"
    metric: str = "cosine"
    normalize: str = "on"
    arcface_padding: float = 0.25
    arcface_preproc: str = "insightface"

    def tag(self) -> str:
        # SHORT tag to keep Windows paths safe
        return f"idx_{stable_hash(asdict(self), 12)}"

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def index_cli_args(self, model: str) -> List[str]:
        # Passing arcface args is harmless (photofinder ignores if not applicable).
        return [
            "--face-policy", self.face_policy,
            "--det-upsample", str(self.det_upsample),
            "--min-face-area", str(self.min_face_area),
            "--max-faces", str(self.max_faces),
            "--fail-policy", self.fail_policy,
            "--metric", self.metric,
            "--normalize", self.normalize,
            "--arcface-padding", str(self.arcface_padding),
            "--arcface-preproc", self.arcface_preproc,
        ]


@dataclass(frozen=True)
class AnnCfg:
    ann_type: str = "hnsw"
    faiss_metric: Optional[str] = None  # None => let photofinder infer from index metric
    hnsw_m: int = 32
    ef_construction: int = 200
    ann_k: int = 500
    ef_search: int = 128
    rerank: str = "on"

    def tag(self) -> str:
        return f"ann_{stable_hash(asdict(self), 12)}"

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def build_cli_args(self) -> List[str]:
        args = [
            "--ann-type", self.ann_type,
            "--hnsw-m", str(self.hnsw_m),
            "--ef-construction", str(self.ef_construction),
        ]
        # Only pass faiss metric if explicitly set
        if self.faiss_metric:
            args += ["--faiss-metric", self.faiss_metric]
        return args

    def eval_cli_args(self, top_k: int) -> List[str]:
        # ann_k/ef_search/rerank/top_k are evaluation-time knobs (per your original sweep)
        return [
            "--backend", "ann",
            "--top-k", str(top_k),
            "--ann-k", str(self.ann_k),
            "--ef-search", str(self.ef_search),
            "--rerank", self.rerank,
        ]


# ---------------------------
# Sweep grids (same intent as v7)
# ---------------------------

def build_baseline_index_cfg(model: str) -> IndexCfg:
    return IndexCfg(
        face_policy="largest",
        det_upsample=1,
        min_face_area=0,
        max_faces=5,
        fail_policy="skip",
        metric="cosine",
        normalize="on",
        arcface_padding=0.25,
        arcface_preproc="insightface",
    )


def build_baseline_ann_cfg() -> AnnCfg:
    return AnnCfg(
        ann_type="hnsw",
        faiss_metric=None,
        hnsw_m=32,
        ef_construction=200,
        ann_k=500,
        ef_search=128,
        rerank="on",
    )


def make_index_knob_grid(model: str, fast: bool) -> List[IndexCfg]:
    """
    Index-time knob sweep (expensive: rebuilds embeddings).
    Kept small; matches v7 spirit.
    """
    base = build_baseline_index_cfg(model)

    det_upsamples = [1] if fast else [0, 1, 2]
    metrics = ["cosine"]
    normalizes = ["on"]

    if "arcface" in model:
        preprocs = ["insightface"] if fast else ["insightface", "legacy"]
        paddings = [0.25] if fast else [0.0, 0.25, 0.5]
    else:
        preprocs = [base.arcface_preproc]
        paddings = [base.arcface_padding]

    grid: List[IndexCfg] = []
    for du, met, norm, ap, pad in product(det_upsamples, metrics, normalizes, preprocs, paddings):
        grid.append(IndexCfg(
            face_policy=base.face_policy,
            det_upsample=int(du),
            min_face_area=base.min_face_area,
            max_faces=base.max_faces,
            fail_policy=base.fail_policy,
            metric=str(met),
            normalize=str(norm),
            arcface_padding=float(pad),
            arcface_preproc=str(ap),
        ))
    uniq = {json.dumps(asdict(c), sort_keys=True): c for c in grid}
    return list(uniq.values())


def make_ann_knob_grid(fast: bool) -> List[AnnCfg]:
    """
    ANN-time sweep (cheaper: no re-embedding).
    """
    if fast:
        hnsw_ms = [32]
        ef_cs = [200]
        ann_ks = [200, 500]
        ef_ss = [64, 128]
        reranks = ["on"]
    else:
        hnsw_ms = [16, 32]
        ef_cs = [100, 200, 400]
        ann_ks = [200, 500, 1000]
        ef_ss = [64, 128, 256]
        reranks = ["on", "off"]

    out: List[AnnCfg] = []
    for M, efC, k, efS, rr in product(hnsw_ms, ef_cs, ann_ks, ef_ss, reranks):
        out.append(AnnCfg(
            ann_type="hnsw",
            faiss_metric=None,
            hnsw_m=int(M),
            ef_construction=int(efC),
            ann_k=int(k),
            ef_search=int(efS),
            rerank=str(rr),
        ))
    uniq = {json.dumps(asdict(c), sort_keys=True): c for c in out}
    return list(uniq.values())


# ---------------------------
# Reporting (incremental CSV + final MD)
# ---------------------------

CSV_HEADER = [
    "timestamp", "phase", "model",
    "base_run_dir", "index_tag", "ann_tag",
    "index_time_s", "ann_build_time_s", "eval_bruteforce_time_s", "eval_ann_time_s",
    "bf_rank1", "bf_recall_at_5", "bf_recall_at_10", "bf_mrr", "bf_n_queries",
    "ann_rank1", "ann_recall_at_5", "ann_recall_at_10", "ann_mrr", "ann_n_queries",
    "index_cfg_json", "ann_cfg_json", "error",
]


class CsvLogger:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self._needs_header = not path.exists() or path.stat().st_size == 0
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=CSV_HEADER)
        if self._needs_header:
            self._w.writeheader()
            self._f.flush()

    def write_row(self, row: Dict[str, Any]) -> None:
        # Ensure all header keys exist
        out = {k: row.get(k, "") for k in CSV_HEADER}
        self._w.writerow(out)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def write_report_md(path: Path, args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)

    # Best per model by ANN rank1 then mrr then recall@10
    best: Dict[str, Dict[str, Any]] = {}
    for model in args.models:
        cand = [r for r in rows if r.get("model") == model and r.get("error", "") == "" and r.get("ann_rank1") not in ("", None)]
        if not cand:
            continue

        def key_fn(r: Dict[str, Any]) -> Tuple[float, float, float]:
            def f(k: str) -> float:
                try:
                    return float(r.get(k) or 0.0)
                except Exception:
                    return 0.0
            return (f("ann_rank1"), f("ann_mrr"), f("ann_recall_at_10"))

        best[model] = max(cand, key=key_fn)

    lines: List[str] = []
    lines.append("# PhotoFinder Sweep Report\n")
    lines.append(f"- Generated: {now_iso()}")
    lines.append(f"- Dataset: `{args.dataset}`")
    lines.append(f"- Out root: `{args.out_root}`")
    lines.append(f"- Models: {', '.join(args.models)}")
    lines.append(f"- Phases: {', '.join(args.phases)}")
    lines.append("")

    lines.append("## Best run per model (by ANN Rank-1)\n")
    lines.append("| Model | Phase | ANN Rank1 | ANN Recall@10 | ANN MRR | Base run dir | Index tag | ANN tag |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for model in args.models:
        r = best.get(model)
        if not r:
            lines.append(f"| {model} |  |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| {model} | {r.get('phase','')} | {fmt_f(r.get('ann_rank1'))} | {fmt_f(r.get('ann_recall_at_10'))} | {fmt_f(r.get('ann_mrr'))} | "
            f"`{r.get('base_run_dir','')}` | `{r.get('index_tag','')}` | `{r.get('ann_tag','')}` |"
        )
    lines.append("")

    lines.append("## All runs (compact)\n")
    lines.append("| Phase | Model | ANN Rank1 | ANN Recall@10 | ANN MRR | idx_s | ann_build_s | eval_bf_s | eval_ann_s | Index tag | ANN tag | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r.get('phase','')} | {r.get('model','')} | {fmt_f(r.get('ann_rank1'))} | {fmt_f(r.get('ann_recall_at_10'))} | {fmt_f(r.get('ann_mrr'))} | "
            f"{fmt_f(r.get('index_time_s'), 2)} | {fmt_f(r.get('ann_build_time_s'), 2)} | {fmt_f(r.get('eval_bruteforce_time_s'), 2)} | {fmt_f(r.get('eval_ann_time_s'), 2)} | "
            f"`{r.get('index_tag','')}` | `{r.get('ann_tag','')}` | {r.get('error','')} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def read_rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            out.append(row)
    return out


# ---------------------------
# PhotoFinder commands
# ---------------------------

def run_index(dataset: str, model: str, out_dir: Path, idx_cfg: IndexCfg, force: bool, dry_run: bool) -> float:
    ensure_dir(out_dir)
    idx_npz = out_dir / "index.npz"
    if idx_npz.exists() and not force:
        print(f"  ✓ index exists, skipping: {idx_npz}")
        return 0.0

    cmd = ["photofinder", "index", "--dataset", dataset, "--model", model, "--out", str(out_dir)] + idx_cfg.index_cli_args(model)
    print(f"  → RUN: {' '.join(cmd)}")
    return run_cmd_live(cmd, dry_run=dry_run)


def run_eval_bruteforce(index_npz: Path, out_dir: Path, top_k: int, force: bool, dry_run: bool) -> Tuple[float, Optional[Dict[str, Any]]]:
    ensure_dir(out_dir)
    m_path = out_dir / "metrics.json"
    if m_path.exists() and not force:
        print(f"  ✓ bf metrics exists, skipping: {m_path}")
        return 0.0, load_json(m_path)

    cmd = ["photofinder", "eval-retrieval", "--index", str(index_npz), "--out", str(out_dir), "--backend", "bruteforce", "--top-k", str(top_k)]
    print(f"  → RUN: {' '.join(cmd)}")
    dt = run_cmd_live(cmd, dry_run=dry_run)
    return dt, load_json(m_path)


def ensure_ann_subrun_has_index(base_run_dir: Path, ann_dir: Path) -> Path:
    ensure_dir(ann_dir)
    src = base_run_dir / "index.npz"
    dst = ann_dir / "index.npz"
    if not src.exists():
        raise FileNotFoundError(f"Missing base index: {src}")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    return dst


def run_build_ann(ann_dir: Path, idx_npz: Path, ann_cfg: AnnCfg, force: bool, dry_run: bool) -> float:
    ensure_dir(ann_dir)
    faiss_path = ann_dir / "index.faiss"
    if faiss_path.exists() and not force:
        print(f"  ✓ ann index exists, skipping: {faiss_path}")
        return 0.0

    # IMPORTANT: pass a FILE path (not just a directory) so photofinder doesn't rely on implicit dirs.
    cmd = ["photofinder", "build-ann", "--index", str(idx_npz), "--out", str(faiss_path)] + ann_cfg.build_cli_args()
    print(f"  → RUN: {' '.join(cmd)}")
    return run_cmd_live(cmd, dry_run=dry_run)


def run_eval_ann(ann_dir: Path, idx_npz: Path, ann_cfg: AnnCfg, top_k: int, force: bool, dry_run: bool) -> Tuple[float, Optional[Dict[str, Any]]]:
    ensure_dir(ann_dir)
    m_path = ann_dir / "metrics.json"
    if m_path.exists() and not force:
        print(f"  ✓ ann metrics exists, skipping: {m_path}")
        return 0.0, load_json(m_path)

    cmd = ["photofinder", "eval-retrieval", "--index", str(idx_npz), "--out", str(ann_dir)] + ann_cfg.eval_cli_args(top_k)
    print(f"  → RUN: {' '.join(cmd)}")
    dt = run_cmd_live(cmd, dry_run=dry_run)
    return dt, load_json(m_path)


# ---------------------------
# Main sweep
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Run PhotoFinder sweeps with robust logging.")
    ap.add_argument("--dataset", required=True, help="Dataset root, e.g. data\\lfw\\lfw_funneled")
    ap.add_argument("--out-root", required=True, help="Run output root, e.g. runs\\sweeps\\lfw")
    ap.add_argument("--models", nargs="+", required=True, help="Models/embedders to run")
    ap.add_argument("--phases", nargs="+", default=["baseline", "ann_knobs"], choices=["baseline", "index_knobs", "ann_knobs"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--fast", action="store_true", help="Smaller grids for quick iteration")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    ap.add_argument("--out-csv", default=None, help="Override summary CSV path")
    ap.add_argument("--out-md", default=None, help="Override summary MD path")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    out_csv = Path(args.out_csv) if args.out_csv else (out_root / "summary_results.csv")
    out_md = Path(args.out_md) if args.out_md else (out_root / "summary_results.md")

    logger = CsvLogger(out_csv)

    def log_row(row: Dict[str, Any]) -> None:
        logger.write_row(row)

    try:
        # Sweep
        for phase in args.phases:
            print(f"\n=== PHASE: {phase} ===\n")

            for mi, model in enumerate(args.models, start=1):
                print(f"[{mi}/{len(args.models)}] MODEL: {model}")

                # Choose index configs for this phase
                if phase == "baseline":
                    index_cfgs = [build_baseline_index_cfg(model)]
                    ann_cfgs = [build_baseline_ann_cfg()]
                elif phase == "index_knobs":
                    index_cfgs = make_index_knob_grid(model, fast=args.fast)
                    ann_cfgs = [build_baseline_ann_cfg()]
                elif phase == "ann_knobs":
                    index_cfgs = [build_baseline_index_cfg(model)]
                    ann_cfgs = make_ann_knob_grid(fast=args.fast)
                else:
                    raise ValueError(f"Unknown phase: {phase}")

                total = len(index_cfgs) * len(ann_cfgs)
                run_i = 0

                for idx_cfg in index_cfgs:
                    index_tag = idx_cfg.tag()
                    base_run_dir = out_root / phase / model / index_tag
                    ensure_dir(base_run_dir)

                    # Persist config next to run for traceability
                    (base_run_dir / "index_cfg.json").write_text(json.dumps(idx_cfg.to_json(), indent=2), encoding="utf-8")

                    # 1) Index
                    try:
                        t_index = run_index(args.dataset, model, base_run_dir, idx_cfg, force=args.force, dry_run=args.dry_run)
                    except Exception as e:
                        row = {
                            "timestamp": now_iso(),
                            "phase": phase,
                            "model": model,
                            "base_run_dir": str(base_run_dir),
                            "index_tag": index_tag,
                            "ann_tag": "",
                            "index_time_s": t_index if "t_index" in locals() else "",
                            "ann_build_time_s": "",
                            "eval_bruteforce_time_s": "",
                            "eval_ann_time_s": "",
                            "index_cfg_json": json.dumps(idx_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                            "ann_cfg_json": "",
                            "error": f"index: {e}",
                        }
                        log_row(row)
                        print(f"  ✗ ERROR (index): {e}")
                        if not args.continue_on_error:
                            raise
                        continue

                    idx_npz = base_run_dir / "index.npz"

                    # 2) BF Eval
                    try:
                        t_bf, bf = run_eval_bruteforce(idx_npz, base_run_dir, args.top_k, force=args.force, dry_run=args.dry_run)
                    except Exception as e:
                        row = {
                            "timestamp": now_iso(),
                            "phase": phase,
                            "model": model,
                            "base_run_dir": str(base_run_dir),
                            "index_tag": index_tag,
                            "ann_tag": "",
                            "index_time_s": t_index,
                            "ann_build_time_s": "",
                            "eval_bruteforce_time_s": "",
                            "eval_ann_time_s": "",
                            "bf_rank1": "",
                            "bf_recall_at_5": "",
                            "bf_recall_at_10": "",
                            "bf_mrr": "",
                            "bf_n_queries": "",
                            "index_cfg_json": json.dumps(idx_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                            "ann_cfg_json": "",
                            "error": f"eval_bf: {e}",
                        }
                        log_row(row)
                        print(f"  ✗ ERROR (eval_bf): {e}")
                        if not args.continue_on_error:
                            raise
                        bf = None
                        t_bf = 0.0

                    # 3) ANN sweeps
                    for ann_cfg in ann_cfgs:
                        run_i += 1
                        ann_tag = ann_cfg.tag()

                        ann_dir = base_run_dir / "_ann" / ann_tag
                        ensure_dir(ann_dir)

                        (ann_dir / "ann_cfg.json").write_text(json.dumps(ann_cfg.to_json(), indent=2), encoding="utf-8")

                        # Copy base index into ann subrun for photofinder compatibility
                        try:
                            ann_idx_npz = ensure_ann_subrun_has_index(base_run_dir, ann_dir)
                        except Exception as e:
                            row = {
                                "timestamp": now_iso(),
                                "phase": phase,
                                "model": model,
                                "base_run_dir": str(base_run_dir),
                                "index_tag": index_tag,
                                "ann_tag": ann_tag,
                                "index_time_s": t_index,
                                "ann_build_time_s": "",
                                "eval_bruteforce_time_s": t_bf,
                                "eval_ann_time_s": "",
                                "bf_rank1": (bf or {}).get("rank1", ""),
                                "bf_recall_at_5": (bf or {}).get("recall_at_5", ""),
                                "bf_recall_at_10": (bf or {}).get("recall_at_10", ""),
                                "bf_mrr": (bf or {}).get("mrr", ""),
                                "bf_n_queries": (bf or {}).get("n_queries", ""),
                                "ann_rank1": "",
                                "ann_recall_at_5": "",
                                "ann_recall_at_10": "",
                                "ann_mrr": "",
                                "ann_n_queries": "",
                                "index_cfg_json": json.dumps(idx_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                                "ann_cfg_json": json.dumps(ann_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                                "error": f"copy_index_to_ann: {e}",
                            }
                            log_row(row)
                            print(f"  ✗ ERROR (copy_ann_index): {e}")
                            if not args.continue_on_error:
                                raise
                            continue

                        # Build ANN
                        try:
                            t_build = run_build_ann(ann_dir, ann_idx_npz, ann_cfg, force=args.force, dry_run=args.dry_run)
                        except Exception as e:
                            row = {
                                "timestamp": now_iso(),
                                "phase": phase,
                                "model": model,
                                "base_run_dir": str(base_run_dir),
                                "index_tag": index_tag,
                                "ann_tag": ann_tag,
                                "index_time_s": t_index,
                                "ann_build_time_s": "",
                                "eval_bruteforce_time_s": t_bf,
                                "eval_ann_time_s": "",
                                "bf_rank1": (bf or {}).get("rank1", ""),
                                "bf_recall_at_5": (bf or {}).get("recall_at_5", ""),
                                "bf_recall_at_10": (bf or {}).get("recall_at_10", ""),
                                "bf_mrr": (bf or {}).get("mrr", ""),
                                "bf_n_queries": (bf or {}).get("n_queries", ""),
                                "ann_rank1": "",
                                "ann_recall_at_5": "",
                                "ann_recall_at_10": "",
                                "ann_mrr": "",
                                "ann_n_queries": "",
                                "index_cfg_json": json.dumps(idx_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                                "ann_cfg_json": json.dumps(ann_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                                "error": f"build_ann: {e}",
                            }
                            log_row(row)
                            print(f"  ✗ ERROR (build_ann): {e}")
                            if not args.continue_on_error:
                                raise
                            continue

                        # Eval ANN
                        try:
                            t_ann, ann = run_eval_ann(ann_dir, ann_idx_npz, ann_cfg, args.top_k, force=args.force, dry_run=args.dry_run)
                        except Exception as e:
                            row = {
                                "timestamp": now_iso(),
                                "phase": phase,
                                "model": model,
                                "base_run_dir": str(base_run_dir),
                                "index_tag": index_tag,
                                "ann_tag": ann_tag,
                                "index_time_s": t_index,
                                "ann_build_time_s": t_build,
                                "eval_bruteforce_time_s": t_bf,
                                "eval_ann_time_s": "",
                                "bf_rank1": (bf or {}).get("rank1", ""),
                                "bf_recall_at_5": (bf or {}).get("recall_at_5", ""),
                                "bf_recall_at_10": (bf or {}).get("recall_at_10", ""),
                                "bf_mrr": (bf or {}).get("mrr", ""),
                                "bf_n_queries": (bf or {}).get("n_queries", ""),
                                "ann_rank1": "",
                                "ann_recall_at_5": "",
                                "ann_recall_at_10": "",
                                "ann_mrr": "",
                                "ann_n_queries": "",
                                "index_cfg_json": json.dumps(idx_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                                "ann_cfg_json": json.dumps(ann_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                                "error": f"eval_ann: {e}",
                            }
                            log_row(row)
                            print(f"  ✗ ERROR (eval_ann): {e}")
                            if not args.continue_on_error:
                                raise
                            continue

                        # Success row
                        row = {
                            "timestamp": now_iso(),
                            "phase": phase,
                            "model": model,
                            "base_run_dir": str(base_run_dir),
                            "index_tag": index_tag,
                            "ann_tag": ann_tag,
                            "index_time_s": t_index,
                            "ann_build_time_s": t_build,
                            "eval_bruteforce_time_s": t_bf,
                            "eval_ann_time_s": t_ann,
                            "bf_rank1": (bf or {}).get("rank1", ""),
                            "bf_recall_at_5": (bf or {}).get("recall_at_5", ""),
                            "bf_recall_at_10": (bf or {}).get("recall_at_10", ""),
                            "bf_mrr": (bf or {}).get("mrr", ""),
                            "bf_n_queries": (bf or {}).get("n_queries", ""),
                            "ann_rank1": (ann or {}).get("rank1", ""),
                            "ann_recall_at_5": (ann or {}).get("recall_at_5", ""),
                            "ann_recall_at_10": (ann or {}).get("recall_at_10", ""),
                            "ann_mrr": (ann or {}).get("mrr", ""),
                            "ann_n_queries": (ann or {}).get("n_queries", ""),
                            "index_cfg_json": json.dumps(idx_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                            "ann_cfg_json": json.dumps(ann_cfg.to_json(), separators=(",", ":"), sort_keys=True),
                            "error": "",
                        }
                        log_row(row)

                        # Nice progress print
                        print(f"  ✓ DONE [{run_i}/{total}] idx={index_tag} ann={ann_tag} "
                              f"ANN rank1={fmt_f(row['ann_rank1'])} recall@10={fmt_f(row['ann_recall_at_10'])}")

        # Final MD report from CSV
        rows = read_rows_from_csv(out_csv)
        write_report_md(out_md, args, rows)
        print(f"\nSaved CSV: {out_csv}\nSaved MD : {out_md}")

    finally:
        logger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
