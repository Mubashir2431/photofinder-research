#!/usr/bin/env python3
"""
photofinder_full_sweep_v3.py

Fixes the two issues you hit:
1) `photofinder eval-retrieval --backend ann` always loads FAISS from:
      Path(index_npz).with_suffix(".faiss")
   i.e., it expects `index.faiss` BESIDE `index.npz`.
   So this script always ensures `run_dir/index.faiss` exists before ANN eval.

2) Summary/CSV should not crash when some runs are missing metrics.
   This script can (a) run sweeps with progress, (b) repair missing ANN metrics,
   and (c) summarize existing runs into a clean CSV + Markdown report.

Works on Windows + PowerShell. Uses only stdlib.

Examples
--------
# 1) Run baseline + index_knobs + ann_knobs (resume-safe)
python scripts\photofinder_full_sweep_v3.py sweep `
  --dataset data\lfw\lfw_funneled `
  --out-root runs\sweeps\lfw `
  --models dlib_resnet_v1 arcface_onnx opencv_sface mobilefacenet_onnx `
  --phases baseline index_knobs ann_knobs

# 2) Summarize whatever is already in runs\sweeps\lfw (FAST)
python scripts\photofinder_full_sweep_v3.py summarize `
  --run-root runs\sweeps\lfw `
  --out runs\sweeps\lfw\summary_v3.csv

# 3) Repair: for every index.npz under run-root, build index.faiss beside it and compute ANN metrics (NO re-index)
python scripts\photofinder_full_sweep_v3.py repair `
  --run-root runs\sweeps\lfw `
  --ann-k 500 --ef-search 128 --rerank on
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------
# Utilities
# ---------------------------

def p(*args: Any) -> None:
    print(*args, flush=True)


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, float]:
    """
    Run a command, letting it print its own progress (tqdm etc).
    Returns (exit_code, elapsed_seconds).
    """
    t0 = time.perf_counter()
    # Important: DO NOT capture stdout/stderr so the CLI progress is visible.
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode, time.perf_counter() - t0


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def fmt4(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def relparts(root: Path, pth: Path) -> List[str]:
    try:
        return list(pth.relative_to(root).parts)
    except Exception:
        return list(pth.parts)


# ---------------------------
# Config models
# ---------------------------

@dataclass(frozen=True)
class IndexCfg:
    face_policy: str = "largest"     # largest | first | all
    det_upsample: int = 1            # 0/1/2
    min_face_area: int = 0
    max_faces: int = 5
    fail_policy: str = "skip"        # skip | error
    metric: str = "cosine"           # cosine | l2
    normalize: str = "on"            # on | off
    arcface_padding: float = 0.25
    arcface_preproc: str = "insightface"  # insightface | legacy (as your CLI shows)

    def tag(self) -> str:
        # match your previous folder naming style
        return (
            f"fp_{self.face_policy}"
            f"_du_{self.det_upsample}"
            f"_m_{self.metric}"
            f"_n_{self.normalize}"
            f"_ap_{self.arcface_preproc}"
            f"_pad_{self.arcface_padding}"
        )


@dataclass(frozen=True)
class AnnBuildCfg:
    ann_type: str = "hnsw"  # flat | hnsw
    faiss_metric: str = "infer"  # ip | l2 | infer
    hnsw_m: int = 32
    ef_construction: int = 200

    def tag(self) -> str:
        return f"ann_{self.ann_type}_fm_{self.faiss_metric}_M_{self.hnsw_m}_efC_{self.ef_construction}"


@dataclass(frozen=True)
class AnnEvalCfg:
    ann_k: int = 500
    ef_search: int = 128
    rerank: str = "on"  # on | off

    def tag(self) -> str:
        return f"k_{self.ann_k}_efS_{self.ef_search}_rr_{self.rerank}"


@dataclass
class Metrics:
    rank1: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    mrr: Optional[float] = None
    n_queries: Optional[int] = None


@dataclass
class SweepRow:
    # identity
    phase: str
    model: str
    run_dir: str
    index_tag: str
    ann_build_tag: str
    ann_eval_tag: str

    # knobs (expanded)
    face_policy: str
    det_upsample: int
    min_face_area: int
    max_faces: int
    fail_policy: str
    metric: str
    normalize: str
    arcface_padding: float
    arcface_preproc: str

    ann_type: str
    faiss_metric: str
    hnsw_m: int
    ef_construction: int
    ann_k: int
    ef_search: int
    rerank: str

    # timings (wall clock)
    index_time_s: Optional[float] = None
    eval_brut_time_s: Optional[float] = None
    ann_build_time_s: Optional[float] = None
    eval_ann_time_s: Optional[float] = None

    # metrics
    brut_rank1: Optional[float] = None
    brut_recall_at_5: Optional[float] = None
    brut_recall_at_10: Optional[float] = None
    brut_mrr: Optional[float] = None
    brut_n_queries: Optional[int] = None

    ann_rank1: Optional[float] = None
    ann_recall_at_5: Optional[float] = None
    ann_recall_at_10: Optional[float] = None
    ann_mrr: Optional[float] = None
    ann_n_queries: Optional[int] = None

    # status
    error: str = ""


def load_metrics_file(path: Path) -> Metrics:
    if not path.exists():
        return Metrics()
    d = read_json(path)
    return Metrics(
        rank1=safe_float(d.get("rank1")),
        recall_at_5=safe_float(d.get("recall_at_5")),
        recall_at_10=safe_float(d.get("recall_at_10")),
        mrr=safe_float(d.get("mrr")),
        n_queries=int(d["n_queries"]) if d.get("n_queries") is not None else None,
    )


# ---------------------------
# Defaults: sweeps
# ---------------------------

def default_index_cfgs_for_phase(phase: str) -> List[IndexCfg]:
    base = IndexCfg()
    if phase == "baseline":
        return [base]
    if phase == "index_knobs":
        # Keep it small: varies only the detector upsample (the biggest knob)
        return [
            IndexCfg(det_upsample=0),
            IndexCfg(det_upsample=1),
            IndexCfg(det_upsample=2),
        ]
    if phase == "ann_knobs":
        # Use baseline embedding index; ANN knobs will vary
        return [base]
    return [base]


def default_ann_build_cfgs_for_phase(phase: str) -> List[AnnBuildCfg]:
    # Building FAISS is cheap compared to indexing, but keep it reasonable.
    if phase == "ann_knobs":
        return [
            AnnBuildCfg(ann_type="hnsw", faiss_metric="infer", hnsw_m=16, ef_construction=100),
            AnnBuildCfg(ann_type="hnsw", faiss_metric="infer", hnsw_m=32, ef_construction=200),
            AnnBuildCfg(ann_type="hnsw", faiss_metric="infer", hnsw_m=64, ef_construction=400),
        ]
    return [AnnBuildCfg()]


def default_ann_eval_cfgs_for_phase(phase: str) -> List[AnnEvalCfg]:
    if phase == "ann_knobs":
        # A compact but informative sweep.
        cfgs: List[AnnEvalCfg] = []
        for ann_k in (100, 200, 500, 1000):
            for ef_s in (32, 64, 128, 256):
                cfgs.append(AnnEvalCfg(ann_k=ann_k, ef_search=ef_s, rerank="on"))
        # One no-rerank config (speed baseline)
        cfgs.append(AnnEvalCfg(ann_k=500, ef_search=128, rerank="off"))
        return cfgs
    return [AnnEvalCfg()]


# ---------------------------
# Core: commands
# ---------------------------

def cmd_index(dataset: str, model: str, out_dir: Path, cfg: IndexCfg) -> List[str]:
    return [
        "photofinder",
        "index",
        "--dataset", dataset,
        "--model", model,
        "--out", str(out_dir),
        "--face-policy", cfg.face_policy,
        "--det-upsample", str(cfg.det_upsample),
        "--min-face-area", str(cfg.min_face_area),
        "--max-faces", str(cfg.max_faces),
        "--fail-policy", cfg.fail_policy,
        "--metric", cfg.metric,
        "--normalize", cfg.normalize,
        "--arcface-padding", str(cfg.arcface_padding),
        "--arcface-preproc", cfg.arcface_preproc,
    ]


def cmd_eval_brut(index_npz: Path, out_dir: Path, top_k: int) -> List[str]:
    return [
        "photofinder",
        "eval-retrieval",
        "--index", str(index_npz),
        "--out", str(out_dir),
        "--backend", "bruteforce",
        "--top-k", str(top_k),
    ]


def cmd_build_ann(index_npz: Path, out_faiss: Path, b: AnnBuildCfg) -> List[str]:
    cmd = [
        "photofinder",
        "build-ann",
        "--index", str(index_npz),
        "--out", str(out_faiss),
        "--ann-type", b.ann_type,
        "--hnsw-m", str(b.hnsw_m),
        "--ef-construction", str(b.ef_construction),
    ]
    if b.faiss_metric and b.faiss_metric != "infer":
        cmd += ["--faiss-metric", b.faiss_metric]
    return cmd


def cmd_eval_ann(index_npz: Path, out_dir: Path, top_k: int, e: AnnEvalCfg, metric_override: Optional[str] = None) -> List[str]:
    cmd = [
        "photofinder",
        "eval-retrieval",
        "--index", str(index_npz),
        "--out", str(out_dir),
        "--backend", "ann",
        "--top-k", str(top_k),
        "--ann-k", str(e.ann_k),
        "--ef-search", str(e.ef_search),
        "--rerank", e.rerank,
    ]
    if metric_override:
        cmd += ["--metric", metric_override]
    return cmd


# ---------------------------
# Mode: sweep
# ---------------------------

def sweep(
    dataset: str,
    out_root: Path,
    models: List[str],
    phases: List[str],
    top_k: int,
    resume: bool,
    force: bool,
    fail_fast: bool,
) -> Tuple[Path, Path]:
    ensure_dir(out_root)
    csv_path = out_root / "sweep_results_v3.csv"
    md_path = out_root / "sweep_report_v3.md"

    # We'll append rows as we go (so Ctrl+C still leaves data).
    fieldnames = list(asdict(SweepRow(
        phase="", model="", run_dir="", index_tag="", ann_build_tag="", ann_eval_tag="",
        face_policy="", det_upsample=0, min_face_area=0, max_faces=0, fail_policy="",
        metric="", normalize="", arcface_padding=0.0, arcface_preproc="",
        ann_type="", faiss_metric="", hnsw_m=0, ef_construction=0, ann_k=0, ef_search=0, rerank="",
    )).keys())

    existing_uids: set[str] = set()
    if csv_path.exists() and resume and not force:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    uid = (row.get("phase","") + "|" + row.get("model","") + "|" + row.get("run_dir","") + "|" +
                           row.get("index_tag","") + "|" + row.get("ann_build_tag","") + "|" + row.get("ann_eval_tag",""))
                    existing_uids.add(uid)
        except Exception:
            pass

    def append_row(w: csv.DictWriter, rr: SweepRow) -> None:
        uid = rr.phase + "|" + rr.model + "|" + rr.run_dir + "|" + rr.index_tag + "|" + rr.ann_build_tag + "|" + rr.ann_eval_tag
        if uid in existing_uids and resume and not force:
            return
        w.writerow(asdict(rr))
        existing_uids.add(uid)

    p("Checking photofinder CLI...")
    subprocess.run(["photofinder", "--help"])

    total_models = len(models)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()

        for phase in phases:
            p(f"\n=== PHASE: {phase} ===")
            idx_cfgs = default_index_cfgs_for_phase(phase)
            ann_build_cfgs = default_ann_build_cfgs_for_phase(phase)
            ann_eval_cfgs = default_ann_eval_cfgs_for_phase(phase)

            for mi, model in enumerate(models, start=1):
                p(f"\n[{mi}/{total_models}] MODEL: {model}")
                p(f"  Index configs: {len(idx_cfgs)} | ANN build cfgs: {len(ann_build_cfgs)} | ANN eval cfgs: {len(ann_eval_cfgs)}")

                for ic_i, idx_cfg in enumerate(idx_cfgs, start=1):
                    index_tag = idx_cfg.tag()
                    run_dir = out_root / phase / model / index_tag
                    ensure_dir(run_dir)

                    index_npz = run_dir / "index.npz"
                    brut_metrics_path = run_dir / "metrics_retrieval_bruteforce.json"

                    # ---- INDEX ----
                    idx_time: Optional[float] = None
                    if force or (not index_npz.exists()):
                        p(f"\n  ({ic_i}/{len(idx_cfgs)}) index_cfg: {index_tag}")
                        cmd = cmd_index(dataset, model, run_dir, idx_cfg)
                        p("  → RUN:", " ".join(cmd))
                        code, idx_time = run_cmd(cmd)
                        if code != 0:
                            rr = SweepRow(
                                phase=phase, model=model, run_dir=str(run_dir), index_tag=index_tag,
                                ann_build_tag="", ann_eval_tag="",
                                **asdict(idx_cfg),
                                ann_type="", faiss_metric="", hnsw_m=0, ef_construction=0, ann_k=0, ef_search=0, rerank="",
                                index_time_s=idx_time,
                                error=f"index failed ({code})",
                            )
                            append_row(writer, rr)
                            f.flush()
                            if fail_fast:
                                raise SystemExit(1)
                            continue
                    else:
                        p(f"\n  ({ic_i}/{len(idx_cfgs)}) index_cfg: {index_tag} (resume: index exists)")

                    # ---- EVAL BRUTEFORCE ----
                    brut_time: Optional[float] = None
                    brut_m = Metrics()
                    if force or (not brut_metrics_path.exists()):
                        cmd = cmd_eval_brut(index_npz, run_dir, top_k=top_k)
                        p("  → RUN:", " ".join(cmd))
                        code, brut_time = run_cmd(cmd)
                        if code != 0:
                            rr = SweepRow(
                                phase=phase, model=model, run_dir=str(run_dir), index_tag=index_tag,
                                ann_build_tag="", ann_eval_tag="",
                                **asdict(idx_cfg),
                                ann_type="", faiss_metric="", hnsw_m=0, ef_construction=0, ann_k=0, ef_search=0, rerank="",
                                index_time_s=idx_time,
                                eval_brut_time_s=brut_time,
                                error=f"eval bruteforce failed ({code})",
                            )
                            append_row(writer, rr)
                            f.flush()
                            if fail_fast:
                                raise SystemExit(1)
                            continue
                    brut_m = load_metrics_file(brut_metrics_path)

                    # ---- ANN SWEEP ----
                    # For baseline/index_knobs we still run ONE ann config (default) so you get both backends.
                    # For ann_knobs phase, we run full ann_build_cfgs × ann_eval_cfgs.
                    ann_build_iter = ann_build_cfgs
                    ann_eval_iter = ann_eval_cfgs

                    ann_total = len(ann_build_iter) * len(ann_eval_iter)
                    ann_done = 0

                    for bcfg in ann_build_iter:
                        ann_build_tag = bcfg.tag()
                        for ecfg in ann_eval_iter:
                            ann_eval_tag = ecfg.tag()
                            ann_done += 1

                            ann_dir = run_dir / "_ann" / f"{ann_build_tag}_{ann_eval_tag}"
                            ensure_dir(ann_dir)

                            # Paths:
                            # - store the built FAISS in ann_dir for reproducibility
                            # - copy it beside index.npz as run_dir/index.faiss so eval-retrieval can load it.
                            ann_faiss = ann_dir / "index.faiss"
                            run_faiss = run_dir / "index.faiss"
                            ann_metrics_path = ann_dir / "metrics_retrieval_ann.json"

                            p(f"  [ANN {ann_done}/{ann_total}] {ann_build_tag} {ann_eval_tag}")

                            ann_build_time: Optional[float] = None
                            if force or (not ann_faiss.exists()):
                                cmd = cmd_build_ann(index_npz, ann_faiss, bcfg)
                                p("    → RUN:", " ".join(cmd))
                                code, ann_build_time = run_cmd(cmd)
                                if code != 0:
                                    rr = SweepRow(
                                        phase=phase, model=model, run_dir=str(run_dir), index_tag=index_tag,
                                        ann_build_tag=ann_build_tag, ann_eval_tag=ann_eval_tag,
                                        **asdict(idx_cfg),
                                        ann_type=bcfg.ann_type, faiss_metric=bcfg.faiss_metric, hnsw_m=bcfg.hnsw_m, ef_construction=bcfg.ef_construction,
                                        ann_k=ecfg.ann_k, ef_search=ecfg.ef_search, rerank=ecfg.rerank,
                                        index_time_s=idx_time, eval_brut_time_s=brut_time,
                                        ann_build_time_s=ann_build_time,
                                        brut_rank1=brut_m.rank1, brut_recall_at_5=brut_m.recall_at_5, brut_recall_at_10=brut_m.recall_at_10, brut_mrr=brut_m.mrr, brut_n_queries=brut_m.n_queries,
                                        error=f"build-ann failed ({code})",
                                    )
                                    append_row(writer, rr)
                                    f.flush()
                                    if fail_fast:
                                        raise SystemExit(1)
                                    continue
                            else:
                                # even if we skip building, keep ann_build_time empty
                                pass

                            # Copy to beside index.npz (required by eval-retrieval ann)
                            try:
                                shutil.copyfile(ann_faiss, run_faiss)
                            except Exception as ex:
                                rr = SweepRow(
                                    phase=phase, model=model, run_dir=str(run_dir), index_tag=index_tag,
                                    ann_build_tag=ann_build_tag, ann_eval_tag=ann_eval_tag,
                                    **asdict(idx_cfg),
                                    ann_type=bcfg.ann_type, faiss_metric=bcfg.faiss_metric, hnsw_m=bcfg.hnsw_m, ef_construction=bcfg.ef_construction,
                                    ann_k=ecfg.ann_k, ef_search=ecfg.ef_search, rerank=ecfg.rerank,
                                    index_time_s=idx_time, eval_brut_time_s=brut_time, ann_build_time_s=ann_build_time,
                                    brut_rank1=brut_m.rank1, brut_recall_at_5=brut_m.recall_at_5, brut_recall_at_10=brut_m.recall_at_10, brut_mrr=brut_m.mrr, brut_n_queries=brut_m.n_queries,
                                    error=f"copy faiss beside index failed: {ex}",
                                )
                                append_row(writer, rr)
                                f.flush()
                                if fail_fast:
                                    raise SystemExit(1)
                                continue

                            ann_eval_time: Optional[float] = None
                            if force or (not ann_metrics_path.exists()):
                                cmd = cmd_eval_ann(index_npz, ann_dir, top_k=top_k, e=ecfg)
                                p("    → RUN:", " ".join(cmd))
                                code, ann_eval_time = run_cmd(cmd)
                                if code != 0:
                                    rr = SweepRow(
                                        phase=phase, model=model, run_dir=str(run_dir), index_tag=index_tag,
                                        ann_build_tag=ann_build_tag, ann_eval_tag=ann_eval_tag,
                                        **asdict(idx_cfg),
                                        ann_type=bcfg.ann_type, faiss_metric=bcfg.faiss_metric, hnsw_m=bcfg.hnsw_m, ef_construction=bcfg.ef_construction,
                                        ann_k=ecfg.ann_k, ef_search=ecfg.ef_search, rerank=ecfg.rerank,
                                        index_time_s=idx_time, eval_brut_time_s=brut_time, ann_build_time_s=ann_build_time, eval_ann_time_s=ann_eval_time,
                                        brut_rank1=brut_m.rank1, brut_recall_at_5=brut_m.recall_at_5, brut_recall_at_10=brut_m.recall_at_10, brut_mrr=brut_m.mrr, brut_n_queries=brut_m.n_queries,
                                        error=f"eval ann failed ({code})",
                                    )
                                    append_row(writer, rr)
                                    f.flush()
                                    if fail_fast:
                                        raise SystemExit(1)
                                    continue

                            ann_m = load_metrics_file(ann_metrics_path)

                            rr = SweepRow(
                                phase=phase, model=model, run_dir=str(run_dir), index_tag=index_tag,
                                ann_build_tag=ann_build_tag, ann_eval_tag=ann_eval_tag,
                                **asdict(idx_cfg),
                                ann_type=bcfg.ann_type, faiss_metric=bcfg.faiss_metric, hnsw_m=bcfg.hnsw_m, ef_construction=bcfg.ef_construction,
                                ann_k=ecfg.ann_k, ef_search=ecfg.ef_search, rerank=ecfg.rerank,
                                index_time_s=idx_time, eval_brut_time_s=brut_time, ann_build_time_s=ann_build_time, eval_ann_time_s=ann_eval_time,
                                brut_rank1=brut_m.rank1, brut_recall_at_5=brut_m.recall_at_5, brut_recall_at_10=brut_m.recall_at_10, brut_mrr=brut_m.mrr, brut_n_queries=brut_m.n_queries,
                                ann_rank1=ann_m.rank1, ann_recall_at_5=ann_m.recall_at_5, ann_recall_at_10=ann_m.recall_at_10, ann_mrr=ann_m.mrr, ann_n_queries=ann_m.n_queries,
                            )
                            append_row(writer, rr)
                            f.flush()

    # Write a small markdown report from the CSV we just produced
    summarize(run_root=out_root, out_csv=csv_path, out_md=md_path)
    return csv_path, md_path


# ---------------------------
# Mode: summarize
# ---------------------------

def summarize(run_root: Path, out_csv: Optional[Path] = None, out_md: Optional[Path] = None) -> Tuple[Path, Path]:
    run_root = Path(run_root)
    out_csv = Path(out_csv) if out_csv else (run_root / "summary_v3.csv")
    out_md = Path(out_md) if out_md else (run_root / "summary_v3.md")

    p(f"\n[SUMMARIZE] Scanning for index.npz under: {run_root}")
    indices = sorted(run_root.rglob("index.npz"))
    p(f"[SUMMARIZE] Found {len(indices)} index.npz files.")

    rows: List[Dict[str, Any]] = []

    for idx in indices:
        run_dir = idx.parent
        parts = relparts(run_root, run_dir)
        phase = parts[0] if len(parts) >= 1 else ""
        model = parts[1] if len(parts) >= 2 else ""
        index_tag = parts[2] if len(parts) >= 3 else run_dir.name

        cfg_path = run_dir / "config.json"
        cfg = read_json(cfg_path) if cfg_path.exists() else {}

        idx_cfg = {
            "face_policy": cfg.get("face_policy"),
            "det_upsample": cfg.get("det_upsample"),
            "min_face_area": cfg.get("min_face_area"),
            "max_faces": cfg.get("max_faces"),
            "fail_policy": cfg.get("fail_policy"),
            "metric": cfg.get("metric"),
            "normalize": "on" if cfg.get("normalize") else ("off" if cfg.get("normalize") is False else None),
            "arcface_padding": cfg.get("arcface_padding"),
            "arcface_preproc": cfg.get("arcface_preproc"),
        }

        # timings produced by indexing (if present)
        t_path = run_dir / "timings.json"
        timings = read_json(t_path) if t_path.exists() else {}

        brut = load_metrics_file(run_dir / "metrics_retrieval_bruteforce.json")

        # ANN results can be:
        #   - run_dir/metrics_retrieval_ann.json (if you ran repair mode),
        #   - or many under run_dir/_ann/*/metrics_retrieval_ann.json (from sweep).
        ann_files: List[Path] = []
        direct_ann = run_dir / "metrics_retrieval_ann.json"
        if direct_ann.exists():
            ann_files.append(direct_ann)
        ann_files += sorted(run_dir.glob("_ann/*/metrics_retrieval_ann.json"))

        if not ann_files:
            rows.append({
                "phase": phase,
                "model": model or cfg.get("model"),
                "run_dir": str(run_dir),
                "index_tag": index_tag,
                **idx_cfg,
                "ann_run_dir": "",
                "ann_build_tag": "",
                "ann_eval_tag": "",
                "ann_k": None,
                "ef_search": None,
                "rerank": None,
                "hnsw_m": None,
                "ef_construction": None,
                "brut_rank1": brut.rank1,
                "brut_recall_at_5": brut.recall_at_5,
                "brut_recall_at_10": brut.recall_at_10,
                "brut_mrr": brut.mrr,
                "brut_n_queries": brut.n_queries,
                "ann_rank1": None,
                "ann_recall_at_5": None,
                "ann_recall_at_10": None,
                "ann_mrr": None,
                "ann_n_queries": None,
                "timings_seconds_total": timings.get("seconds_total"),
                "timings_n_images_total": timings.get("n_images_total"),
            })
            continue

        for af in ann_files:
            ann_dir = af.parent
            ann_tag = ann_dir.name
            # If it looks like: ann_hnsw_fm_infer_M_32_efC_200_k_500_efS_128_rr_on
            # we can parse a few values (best-effort).
            ann_k = ef_search = None
            rerank = None
            hnsw_m = efC = None
            try:
                # split after "..._M_.._efC_.._k_.._efS_.._rr_.."
                tokens = ann_tag.split("_")
                # quick parse by scanning tokens
                for i, tok in enumerate(tokens):
                    if tok == "M" and i + 1 < len(tokens):
                        hnsw_m = int(tokens[i + 1])
                    if tok == "efC" and i + 1 < len(tokens):
                        efC = int(tokens[i + 1])
                    if tok == "k" and i + 1 < len(tokens):
                        ann_k = int(tokens[i + 1])
                    if tok == "efS" and i + 1 < len(tokens):
                        ef_search = int(tokens[i + 1])
                    if tok == "rr" and i + 1 < len(tokens):
                        rerank = tokens[i + 1]
            except Exception:
                pass

            ann = load_metrics_file(af)

            rows.append({
                "phase": phase,
                "model": model or cfg.get("model"),
                "run_dir": str(run_dir),
                "index_tag": index_tag,
                **idx_cfg,
                "ann_run_dir": str(ann_dir),
                "ann_build_tag": "",  # can be derived from ann_tag if you want
                "ann_eval_tag": ann_tag,
                "ann_k": ann_k,
                "ef_search": ef_search,
                "rerank": rerank,
                "hnsw_m": hnsw_m,
                "ef_construction": efC,
                "brut_rank1": brut.rank1,
                "brut_recall_at_5": brut.recall_at_5,
                "brut_recall_at_10": brut.recall_at_10,
                "brut_mrr": brut.mrr,
                "brut_n_queries": brut.n_queries,
                "ann_rank1": ann.rank1,
                "ann_recall_at_5": ann.recall_at_5,
                "ann_recall_at_10": ann.recall_at_10,
                "ann_mrr": ann.mrr,
                "ann_n_queries": ann.n_queries,
                "timings_seconds_total": timings.get("seconds_total"),
                "timings_n_images_total": timings.get("n_images_total"),
            })

    ensure_dir(out_csv.parent)
    # Write clean CSV
    if rows:
        cols = sorted({k for r in rows for k in r.keys()})
    else:
        cols = []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write a simple markdown report: best brut + best ann per model
    best_by_model: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        m = str(r.get("model") or "")
        if not m:
            continue
        # best bruteforce (by brut_rank1)
        br = safe_float(r.get("brut_rank1"))
        ar = safe_float(r.get("ann_rank1"))
        cur = best_by_model.get(m, {"best_brut": None, "best_ann": None})
        if br is not None:
            if cur["best_brut"] is None or br > safe_float(cur["best_brut"].get("brut_rank1")):
                cur["best_brut"] = r
        if ar is not None:
            if cur["best_ann"] is None or ar > safe_float(cur["best_ann"].get("ann_rank1")):
                cur["best_ann"] = r
        best_by_model[m] = cur

    lines: List[str] = []
    lines.append(f"# Photofinder Sweep Summary\n")
    lines.append(f"- Run root: `{run_root}`\n")
    lines.append(f"- Indices found: **{len(indices)}**\n")
    lines.append(f"- Rows written: **{len(rows)}**\n\n")
    lines.append("## Best per model\n")
    lines.append("| Model | Best Brut Rank1 | Best Ann Rank1 | Best Ann cfg | |\n")
    lines.append("|---|---:|---:|---|---|\n")
    for m, d in sorted(best_by_model.items()):
        bb = d["best_brut"]
        ba = d["best_ann"]
        bbv = safe_float(bb.get("brut_rank1")) if bb else None
        bav = safe_float(ba.get("ann_rank1")) if ba else None
        ann_cfg = (ba.get("ann_eval_tag") or ba.get("ann_run_dir") or "") if ba else ""
        lines.append(f"| {m} | {fmt4(bbv)} | {fmt4(bav)} | {ann_cfg} | |\n")

    ensure_dir(out_md.parent)
    out_md.write_text("".join(lines), encoding="utf-8")

    p(f"[SUMMARIZE] Wrote:\n  - {out_csv}\n  - {out_md}")
    return out_csv, out_md


# ---------------------------
# Mode: repair (fill missing ANN metrics beside index.npz)
# ---------------------------

def repair(
    run_root: Path,
    ann_k: int,
    ef_search: int,
    rerank: str,
    top_k: int,
    force: bool,
) -> None:
    run_root = Path(run_root)
    p(f"\n[REPAIR] Scanning for index.npz under: {run_root}")
    indices = sorted(run_root.rglob("index.npz"))
    p(f"[REPAIR] Found {len(indices)} indices.")

    # default build config
    build_cfg = AnnBuildCfg()
    eval_cfg = AnnEvalCfg(ann_k=ann_k, ef_search=ef_search, rerank=rerank)

    for i, idx in enumerate(indices, start=1):
        run_dir = idx.parent
        run_faiss = run_dir / "index.faiss"
        ann_metrics = run_dir / "metrics_retrieval_ann.json"

        p(f"\n[REPAIR {i}/{len(indices)}] {run_dir}")

        if force or not run_faiss.exists():
            cmd = cmd_build_ann(idx, run_faiss, build_cfg)
            p("  → RUN:", " ".join(cmd))
            code, _t = run_cmd(cmd)
            if code != 0:
                p("  ✗ build-ann failed; skipping.")
                continue
        else:
            p("  (resume) index.faiss exists")

        if force or not ann_metrics.exists():
            cmd = cmd_eval_ann(idx, run_dir, top_k=top_k, e=eval_cfg)
            p("  → RUN:", " ".join(cmd))
            code, _t = run_cmd(cmd)
            if code != 0:
                p("  ✗ eval ann failed; skipping.")
                continue
        else:
            p("  (resume) metrics_retrieval_ann.json exists")


# ---------------------------
# CLI
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # sweep
    ap_s = sub.add_parser("sweep", help="Run sweeps with progress + incremental CSV + report.")
    ap_s.add_argument("--dataset", required=True, help="Dataset root (root/<label>/<image>)")
    ap_s.add_argument("--out-root", required=True, help="Output root, e.g., runs\\sweeps\\lfw")
    ap_s.add_argument("--models", nargs="+", required=True, help="Models (names recognized by photofinder)")
    ap_s.add_argument("--phases", nargs="+", default=["baseline"], choices=["baseline", "index_knobs", "ann_knobs"], help="Which phases to run")
    ap_s.add_argument("--top-k", type=int, default=10)
    ap_s.add_argument("--resume", action="store_true", default=True)
    ap_s.add_argument("--force", action="store_true", help="Re-run everything (overwrites/redo).")
    ap_s.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")

    # summarize
    ap_sum = sub.add_parser("summarize", help="Scan an existing run-root and write a clean summary CSV + MD.")
    ap_sum.add_argument("--run-root", required=True, help="Root folder to scan, e.g., runs\\sweeps\\lfw")
    ap_sum.add_argument("--out", default=None, help="Output CSV path (default: <run-root>\\summary_v3.csv)")

    # repair
    ap_r = sub.add_parser("repair", help="For each index.npz under run-root: ensure index.faiss beside it, then compute ANN metrics in-place.")
    ap_r.add_argument("--run-root", required=True)
    ap_r.add_argument("--ann-k", type=int, default=500)
    ap_r.add_argument("--ef-search", type=int, default=128)
    ap_r.add_argument("--rerank", default="on", choices=["on", "off"])
    ap_r.add_argument("--top-k", type=int, default=10)
    ap_r.add_argument("--force", action="store_true", help="Overwrite/recompute ann metrics.")

    args = ap.parse_args()

    if args.cmd == "sweep":
        csv_path, md_path = sweep(
            dataset=args.dataset,
            out_root=Path(args.out_root),
            models=args.models,
            phases=args.phases,
            top_k=args.top_k,
            resume=args.resume,
            force=args.force,
            fail_fast=args.fail_fast,
        )
        p("\nDONE.")
        p(f"- CSV: {csv_path}")
        p(f"- Report: {md_path}")
        return 0

    if args.cmd == "summarize":
        out_csv = Path(args.out) if args.out else None
        summarize(run_root=Path(args.run_root), out_csv=out_csv)
        return 0

    if args.cmd == "repair":
        repair(
            run_root=Path(args.run_root),
            ann_k=args.ann_k,
            ef_search=args.ef_search,
            rerank=args.rerank,
            top_k=args.top_k,
            force=args.force,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
