#!/usr/bin/env python3
"""
PhotoFinder sweep runner (stable on Windows).

Key idea (fixes the ANN FileNotFoundError you hit):
- `photofinder eval-retrieval --backend ann` ALWAYS looks for `index.faiss` next to the `--index index.npz`.
- Therefore, for each ANN config we create a dedicated folder that contains:
    - index.npz  (copied/hardlinked from the base run)
    - index.faiss (built into that same folder)
  and we run eval on that folder's index.npz.
This makes the CLI happy and avoids missing `index.faiss` / `metrics_retrieval_ann.json` problems.

Outputs (written incrementally so you never lose work):
- <out_root>/summary_results.csv
- <out_root>/summary_results.jsonl
- <out_root>/summary_results.md
- <out_root>/best_by_model.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# Small helpers
# ---------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def fmt_f(x: Any, nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def safe_tag(s: str) -> str:
    # Windows-safe-ish folder tags
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    s = s.replace(" ", "_")
    s = s.replace("__", "_")
    return s


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()


def run_cmd_live(cmd: List[str], cwd: Optional[Path] = None) -> float:
    """
    Runs a command and streams output live (so tqdm/progress stays visible).
    Returns wall time seconds.
    """
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return dt


def ensure_photofinder_available() -> None:
    try:
        run_cmd_live(["photofinder", "--help"])
    except Exception as e:
        raise RuntimeError(
            "Couldn't run `photofinder --help`. Make sure:\n"
            "  1) Your .venv is activated\n"
            "  2) `photofinder` is installed and on PATH\n"
            f"Original error: {e}"
        )


def try_hardlink_or_copy(src: Path, dst: Path) -> None:
    """
    Prefer hardlink (fast, no extra disk) but fallback to copy on failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


# ---------------------------
# Config dataclasses
# ---------------------------

@dataclass(frozen=True)
class IndexCfg:
    face_policy: str = "largest"
    det_upsample: int = 1
    min_face_area: int = 0
    max_faces: int = 5
    fail_policy: str = "skip"
    metric: str = "cosine"       # cosine | l2
    normalize: str = "on"        # on | off
    arcface_padding: float = 0.25
    arcface_preproc: str = "insightface"  # insightface | legacy

    def to_cli_args(self) -> List[str]:
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

    def tag(self) -> str:
        return safe_tag(
            f"fp_{self.face_policy}_du_{self.det_upsample}"
            f"_mfa_{self.min_face_area}_mf_{self.max_faces}"
            f"_fail_{self.fail_policy}_met_{self.metric}_norm_{self.normalize}"
            f"_ap_{self.arcface_preproc}_pad_{self.arcface_padding}"
        )


@dataclass(frozen=True)
class AnnCfg:
    ann_type: str = "hnsw"     # flat | hnsw
    faiss_metric: Optional[str] = None  # ip | l2 | None(infer)
    hnsw_m: int = 32
    ef_construction: int = 200
    ann_k: int = 500
    ef_search: int = 128
    rerank: str = "on"

    def build_cli_args(self) -> List[str]:
        args = [
            "--ann-type", self.ann_type,
            "--hnsw-m", str(self.hnsw_m),
            "--ef-construction", str(self.ef_construction),
        ]
        if self.faiss_metric:
            args += ["--faiss-metric", self.faiss_metric]
        return args

    def eval_cli_args(self, top_k: int) -> List[str]:
        return [
            "--top-k", str(top_k),
            "--ann-k", str(self.ann_k),
            "--ef-search", str(self.ef_search),
            "--rerank", self.rerank,
        ]

    def tag(self, top_k: int) -> str:
        fm = self.faiss_metric or "infer"
        return safe_tag(
            f"ann_{self.ann_type}_fm_{fm}_M_{self.hnsw_m}_efC_{self.ef_construction}"
            f"_k_{self.ann_k}_efS_{self.ef_search}_rr_{self.rerank}_top_{top_k}"
        )


@dataclass
class RunResult:
    timestamp: str
    phase: str
    model: str
    dataset: str
    run_dir: str            # base index folder
    index_tag: str
    ann_tag: str

    index_cfg: Dict[str, Any]
    ann_cfg: Dict[str, Any]

    # timings
    index_time_s: Optional[float] = None
    ann_build_time_s: Optional[float] = None
    eval_bruteforce_time_s: Optional[float] = None
    eval_ann_time_s: Optional[float] = None

    # metrics
    bruteforce: Optional[Dict[str, Any]] = None
    ann: Optional[Dict[str, Any]] = None

    error: str = ""


# ---------------------------
# Photofinder steps
# ---------------------------

def expected_index_paths(run_dir: Path) -> Tuple[Path, Path]:
    return run_dir / "index.npz", run_dir / "metrics_retrieval_bruteforce.json"


def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_index(dataset: Path, model: str, run_dir: Path, index_cfg: IndexCfg, force: bool) -> float:
    idx_path, _ = expected_index_paths(run_dir)
    if idx_path.exists() and not force:
        print(f"  ✓ index exists, skipping: {idx_path}")
        return 0.0

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["photofinder", "index", "--dataset", str(dataset), "--model", model, "--out", str(run_dir)]
    cmd += index_cfg.to_cli_args()

    print(f"  → RUN: {' '.join(cmd)}")
    return run_cmd_live(cmd)


def run_eval_bruteforce(run_dir: Path, top_k: int, force: bool) -> Tuple[float, Dict[str, Any]]:
    idx_path, m_path = expected_index_paths(run_dir)
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing index.npz at {idx_path}")

    if m_path.exists() and not force:
        print(f"  ✓ bruteforce metrics exist, skipping: {m_path}")
        return 0.0, load_metrics(m_path)

    cmd = [
        "photofinder", "eval-retrieval",
        "--index", str(idx_path),
        "--out", str(run_dir),
        "--backend", "bruteforce",
        "--top-k", str(top_k),
    ]
    print(f"  → RUN: {' '.join(cmd)}")
    dt = run_cmd_live(cmd)
    return dt, load_metrics(m_path)


def ann_subrun_dir(base_run_dir: Path, ann_cfg: AnnCfg, top_k: int) -> Path:
    return base_run_dir / "_ann" / ann_cfg.tag(top_k)


def ensure_ann_subrun_has_index(base_run_dir: Path, ann_dir: Path) -> Path:
    """
    Make sure ann_dir has index.npz (hardlink/copy from base_run_dir/index.npz).
    Returns path to ann_dir/index.npz.
    """
    src = base_run_dir / "index.npz"
    if not src.exists():
        raise FileNotFoundError(f"Missing base index.npz at {src}")
    ann_dir.mkdir(parents=True, exist_ok=True)
    dst = ann_dir / "index.npz"
    try_hardlink_or_copy(src, dst)
    return dst


def run_build_ann(base_run_dir: Path, ann_cfg: AnnCfg, top_k: int, force: bool) -> Tuple[float, Path]:
    """
    Builds ANN inside ann_dir so that eval-retrieval can find index.faiss next to index.npz.
    Returns (time_s, ann_dir).
    """
    ann_dir = ann_subrun_dir(base_run_dir, ann_cfg, top_k)
    idx_npz = ensure_ann_subrun_has_index(base_run_dir, ann_dir)
    faiss_path = ann_dir / "index.faiss"

    if faiss_path.exists() and not force:
        print(f"  ✓ faiss exists, skipping: {faiss_path}")
        return 0.0, ann_dir

    cmd = ["photofinder", "build-ann", "--index", str(idx_npz), "--out", str(ann_dir)]
    cmd += ann_cfg.build_cli_args()
    print(f"  → RUN: {' '.join(cmd)}")
    dt = run_cmd_live(cmd)
    return dt, ann_dir


def run_eval_ann(base_run_dir: Path, ann_cfg: AnnCfg, ann_dir: Path, top_k: int, force: bool) -> Tuple[float, Dict[str, Any]]:
    """
    Eval ANN by pointing `--index` to ann_dir/index.npz so the CLI loads ann_dir/index.faiss.
    """
    idx_npz = ann_dir / "index.npz"
    m_path = ann_dir / "metrics_retrieval_ann.json"

    if m_path.exists() and not force:
        print(f"  ✓ ann metrics exist, skipping: {m_path}")
        return 0.0, load_metrics(m_path)

    cmd = [
        "photofinder", "eval-retrieval",
        "--index", str(idx_npz),
        "--out", str(ann_dir),
        "--backend", "ann",
    ] + ann_cfg.eval_cli_args(top_k)

    print(f"  → RUN: {' '.join(cmd)}")
    dt = run_cmd_live(cmd)
    return dt, load_metrics(m_path)


# ---------------------------
# Sweep grids
# ---------------------------

def build_baseline_index_cfg(model: str) -> IndexCfg:
    # Keep your established baseline. ArcFace knobs matter only for arcface models,
    # but passing them is harmless for others (photofinder ignores irrelevant).
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
        faiss_metric=None,   # let photofinder infer from index metric
        hnsw_m=32,
        ef_construction=200,
        ann_k=500,
        ef_search=128,
        rerank="on",
    )


def make_index_knob_grid(model: str, fast: bool) -> List[IndexCfg]:
    """
    Index-time knob sweep (EXPENSIVE: rebuilds embeddings).
    Keep small unless you're sure you want a big run.
    """
    base = build_baseline_index_cfg(model)

    det_upsamples = [1] if fast else [0, 1, 2]

    # Keep metric + normalize fixed for now to keep comparisons fair.
    metrics = ["cosine"]
    normalizes = ["on"]

    # ArcFace-related knobs only meaningful for arcface models
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
            metric=met,
            normalize=norm,
            arcface_padding=float(pad),
            arcface_preproc=ap,
        ))

    # de-dup
    uniq = {asdict(c).__repr__(): c for c in grid}
    return list(uniq.values())


def make_ann_knob_grid(fast: bool) -> List[AnnCfg]:
    """
    ANN-time sweep (CHEAPER: doesn't rebuild embeddings).
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
    uniq = {asdict(c).__repr__(): c for c in out}
    return list(uniq.values())


# ---------------------------
# Reporting
# ---------------------------

CSV_HEADER = [
    "timestamp", "phase", "model", "base_run_dir", "index_tag", "ann_tag",
    "index_time_s", "ann_build_time_s", "eval_bruteforce_time_s", "eval_ann_time_s",
    "bf_rank1", "bf_recall_at_5", "bf_recall_at_10", "bf_mrr", "bf_n_queries",
    "ann_rank1", "ann_recall_at_5", "ann_recall_at_10", "ann_mrr", "ann_n_queries",
    "index_cfg_json", "ann_cfg_json", "error",
]


def write_report_md(path: Path, args: argparse.Namespace, rows: List[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Pick best per model by ANN rank1, then mrr, then recall@10
    best: Dict[str, RunResult] = {}
    for model in args.models:
        candidates = [r for r in rows if (r.model == model and r.ann and not r.error)]
        if not candidates:
            continue

        def key_fn(r: RunResult) -> Tuple[float, float, float]:
            an = r.ann or {}
            return (float(an.get("rank1", 0.0)), float(an.get("mrr", 0.0)), float(an.get("recall_at_10", 0.0)))

        best[model] = max(candidates, key=key_fn)

    lines: List[str] = []
    lines.append("# PhotoFinder Sweep Report\n")
    lines.append(f"- Generated: {now_iso()}")
    lines.append(f"- Dataset: `{args.dataset}`")
    lines.append(f"- Out root: `{args.out_root}`")
    lines.append(f"- Models: {', '.join(args.models)}")
    lines.append(f"- Phases: {', '.join(args.phases)}")
    lines.append("")

    # Best-by-model table
    lines.append("## Best run per model (by ANN Rank-1)\n")
    lines.append("| Model | Phase | ANN Rank1 | ANN Recall@10 | ANN MRR | Base run dir | Index tag | ANN tag |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for model in args.models:
        r = best.get(model)
        if not r:
            lines.append(f"| {model} |  |  |  |  |  |  |  |")
            continue
        an = r.ann or {}
        lines.append(
            f"| {model} | {r.phase} | {fmt_f(an.get('rank1'))} | {fmt_f(an.get('recall_at_10'))} | {fmt_f(an.get('mrr'))} | "
            f"`{r.run_dir}` | `{r.index_tag}` | `{r.ann_tag}` |"
        )
    lines.append("")

    # Full results (compact)
    lines.append("## All runs (compact)\n")
    lines.append("| Phase | Model | ANN Rank1 | ANN Recall@10 | ANN MRR | idx_s | ann_build_s | eval_bf_s | eval_ann_s | Index tag | ANN tag | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in rows:
        an = r.ann or {}
        lines.append(
            f"| {r.phase} | {r.model} | {fmt_f(an.get('rank1'))} | {fmt_f(an.get('recall_at_10'))} | {fmt_f(an.get('mrr'))} | "
            f"{fmt_f(r.index_time_s, 2)} | {fmt_f(r.ann_build_time_s, 2)} | {fmt_f(r.eval_bruteforce_time_s, 2)} | {fmt_f(r.eval_ann_time_s, 2)} | "
            f"`{r.index_tag}` | `{r.ann_tag}` | {r.error or ''} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_best_json(path: Path, args: argparse.Namespace, rows: List[RunResult]) -> None:
    best_by_model: Dict[str, Any] = {}
    for model in args.models:
        candidates = [r for r in rows if (r.model == model and r.ann and not r.error)]
        if not candidates:
            continue

        def key_fn(r: RunResult) -> Tuple[float, float, float]:
            an = r.ann or {}
            return (float(an.get("rank1", 0.0)), float(an.get("mrr", 0.0)), float(an.get("recall_at_10", 0.0)))

        best = max(candidates, key=key_fn)
        best_by_model[model] = {
            "phase": best.phase,
            "base_run_dir": best.run_dir,
            "index_tag": best.index_tag,
            "ann_tag": best.ann_tag,
            "index_cfg": best.index_cfg,
            "ann_cfg": best.ann_cfg,
            "timings_s": {
                "index": best.index_time_s,
                "ann_build": best.ann_build_time_s,
                "eval_bruteforce": best.eval_bruteforce_time_s,
                "eval_ann": best.eval_ann_time_s,
            },
            "metrics": {
                "bruteforce": best.bruteforce,
                "ann": best.ann,
            },
        }
    write_json(path, best_by_model)


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset root: root/<label>/<image>")
    ap.add_argument("--out-root", required=True, help="Root folder for sweep outputs")
    ap.add_argument("--models", nargs="+", default=["dlib_resnet_v1", "arcface_onnx", "opencv_sface", "mobilefacenet_onnx"])
    ap.add_argument(
        "--phases",
        nargs="+",
        default=["baseline", "index_knobs", "ann_knobs"],
        choices=["baseline", "index_knobs", "ann_knobs"],
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--fast", action="store_true", help="Smaller grids for quick testing")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    ap.add_argument("--continue-on-error", action="store_true", help="Keep going after errors")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ensure_photofinder_available()

    summary_csv = out_root / "summary_results.csv"
    summary_jsonl = out_root / "summary_results.jsonl"
    report_md = out_root / "summary_results.md"
    best_json = out_root / "best_by_model.json"

    # Write header if new CSV
    new_csv = not summary_csv.exists()
    f_csv = summary_csv.open("a", newline="", encoding="utf-8")
    w_csv = csv.writer(f_csv)
    if new_csv:
        w_csv.writerow(CSV_HEADER)
        f_csv.flush()

    rows: List[RunResult] = []

    def write_row(rr: RunResult) -> None:
        bf = rr.bruteforce or {}
        an = rr.ann or {}
        w_csv.writerow([
            rr.timestamp, rr.phase, rr.model, rr.run_dir, rr.index_tag, rr.ann_tag,
            rr.index_time_s, rr.ann_build_time_s, rr.eval_bruteforce_time_s, rr.eval_ann_time_s,
            bf.get("rank1"), bf.get("recall_at_5"), bf.get("recall_at_10"), bf.get("mrr"), bf.get("n_queries"),
            an.get("rank1"), an.get("recall_at_5"), an.get("recall_at_10"), an.get("mrr"), an.get("n_queries"),
            json.dumps(rr.index_cfg, sort_keys=True),
            json.dumps(rr.ann_cfg, sort_keys=True),
            rr.error
        ])
        f_csv.flush()
        append_jsonl(summary_jsonl, asdict(rr))

    baseline_ann_cfg = build_baseline_ann_cfg()

    # ---------------------------
    # Phase A: baseline
    # ---------------------------
    if "baseline" in args.phases:
        print("\n=== PHASE: baseline ===")
        for i, model in enumerate(args.models, start=1):
            idx_cfg = build_baseline_index_cfg(model)
            base_run_dir = out_root / "baseline" / model / idx_cfg.tag()
            print(f"\n[{i}/{len(args.models)}] MODEL: {model}")

            rr = RunResult(
                timestamp=now_iso(),
                phase="baseline",
                model=model,
                dataset=str(dataset),
                run_dir=str(base_run_dir),
                index_tag=idx_cfg.tag(),
                ann_tag=baseline_ann_cfg.tag(args.top_k),
                index_cfg=asdict(idx_cfg),
                ann_cfg=asdict(baseline_ann_cfg),
            )

            try:
                rr.index_time_s = run_index(dataset, model, base_run_dir, idx_cfg, force=args.force)
                rr.eval_bruteforce_time_s, rr.bruteforce = run_eval_bruteforce(base_run_dir, top_k=args.top_k, force=args.force)

                rr.ann_build_time_s, ann_dir = run_build_ann(base_run_dir, baseline_ann_cfg, args.top_k, force=args.force)
                rr.eval_ann_time_s, rr.ann = run_eval_ann(base_run_dir, baseline_ann_cfg, ann_dir, args.top_k, force=args.force)

            except Exception as e:
                rr.error = str(e)
                print(f"  ✗ ERROR: {e}")
                rows.append(rr)
                write_row(rr)
                if not args.continue_on_error:
                    f_csv.close()
                    write_report_md(report_md, args, rows)
                    write_best_json(best_json, args, rows)
                    return 1
                else:
                    continue

            rows.append(rr)
            write_row(rr)

    # ---------------------------
    # Phase B: index-time knobs
    # ---------------------------
    if "index_knobs" in args.phases:
        print("\n=== PHASE: index_knobs (rebuilds embeddings) ===")
        for i, model in enumerate(args.models, start=1):
            grid = make_index_knob_grid(model, fast=args.fast)
            print(f"\n[{i}/{len(args.models)}] MODEL: {model}")
            print(f"  Index configs: {len(grid)}")
            for j, idx_cfg in enumerate(grid, start=1):
                run_dir = out_root / "index_knobs" / model / idx_cfg.tag()
                print(f"\n  ({j}/{len(grid)}) index_cfg: {idx_cfg.tag()}")

                rr = RunResult(
                    timestamp=now_iso(),
                    phase="index_knobs",
                    model=model,
                    dataset=str(dataset),
                    run_dir=str(run_dir),
                    index_tag=idx_cfg.tag(),
                    ann_tag=baseline_ann_cfg.tag(args.top_k),
                    index_cfg=asdict(idx_cfg),
                    ann_cfg=asdict(baseline_ann_cfg),
                )

                try:
                    rr.index_time_s = run_index(dataset, model, run_dir, idx_cfg, force=args.force)
                    rr.eval_bruteforce_time_s, rr.bruteforce = run_eval_bruteforce(run_dir, top_k=args.top_k, force=args.force)

                    rr.ann_build_time_s, ann_dir = run_build_ann(run_dir, baseline_ann_cfg, args.top_k, force=args.force)
                    rr.eval_ann_time_s, rr.ann = run_eval_ann(run_dir, baseline_ann_cfg, ann_dir, args.top_k, force=args.force)

                except Exception as e:
                    rr.error = str(e)
                    print(f"    ✗ ERROR: {e}")
                    rows.append(rr)
                    write_row(rr)
                    if not args.continue_on_error:
                        f_csv.close()
                        write_report_md(report_md, args, rows)
                        write_best_json(best_json, args, rows)
                        return 1
                    else:
                        continue

                rows.append(rr)
                write_row(rr)

    # ---------------------------
    # Phase C: ANN knobs (cheaper)
    # ---------------------------
    if "ann_knobs" in args.phases:
        print("\n=== PHASE: ann_knobs (rebuilds ANN only) ===")
        ann_grid = make_ann_knob_grid(fast=args.fast)
        print(f"ANN configs: {len(ann_grid)}")

        for i, model in enumerate(args.models, start=1):
            idx_cfg = build_baseline_index_cfg(model)
            base_run_dir = out_root / "baseline" / model / idx_cfg.tag()

            print(f"\n[{i}/{len(args.models)}] MODEL: {model}")

            # Ensure baseline vectors + bruteforce exist (reuse for every ann_cfg)
            try:
                _ = run_index(dataset, model, base_run_dir, idx_cfg, force=False)
                bf_dt, bf_metrics = run_eval_bruteforce(base_run_dir, top_k=args.top_k, force=False)
            except Exception as e:
                rr = RunResult(
                    timestamp=now_iso(),
                    phase="ann_knobs_base_check",
                    model=model,
                    dataset=str(dataset),
                    run_dir=str(base_run_dir),
                    index_tag=idx_cfg.tag(),
                    ann_tag="",
                    index_cfg=asdict(idx_cfg),
                    ann_cfg={},
                    error=str(e),
                )
                print(f"  ✗ ERROR preparing baseline vectors: {e}")
                rows.append(rr)
                write_row(rr)
                if not args.continue_on_error:
                    f_csv.close()
                    write_report_md(report_md, args, rows)
                    write_best_json(best_json, args, rows)
                    return 1
                else:
                    continue

            # For each ann cfg, build+eval into its own ann_dir (contains index.npz + index.faiss)
            for j, ann_cfg in enumerate(ann_grid, start=1):
                print(f"\n  ({j}/{len(ann_grid)}) ann_cfg: {ann_cfg.tag(args.top_k)}")

                rr = RunResult(
                    timestamp=now_iso(),
                    phase="ann_knobs",
                    model=model,
                    dataset=str(dataset),
                    run_dir=str(base_run_dir),
                    index_tag=idx_cfg.tag(),
                    ann_tag=ann_cfg.tag(args.top_k),
                    index_cfg=asdict(idx_cfg),
                    ann_cfg=asdict(ann_cfg),
                )

                try:
                    rr.bruteforce = bf_metrics
                    rr.eval_bruteforce_time_s = bf_dt

                    rr.ann_build_time_s, ann_dir = run_build_ann(base_run_dir, ann_cfg, args.top_k, force=args.force)
                    rr.eval_ann_time_s, rr.ann = run_eval_ann(base_run_dir, ann_cfg, ann_dir, args.top_k, force=args.force)

                except Exception as e:
                    rr.error = str(e)
                    print(f"    ✗ ERROR: {e}")
                    rows.append(rr)
                    write_row(rr)
                    if not args.continue_on_error:
                        f_csv.close()
                        write_report_md(report_md, args, rows)
                        write_best_json(best_json, args, rows)
                        return 1
                    else:
                        continue

                rows.append(rr)
                write_row(rr)

    # finalize
    f_csv.close()
    write_report_md(report_md, args, rows)
    write_best_json(best_json, args, rows)

    print("\n✅ Sweep complete.")
    print(f"CSV:  {summary_csv}")
    print(f"MD:   {report_md}")
    print(f"BEST: {best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
