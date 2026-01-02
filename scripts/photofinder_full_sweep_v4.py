#!/usr/bin/env python3
r"""photofinder_full_sweep_v4.py

One script to:
  • run multi-model sweeps (baseline + index-time knobs + ANN knobs)
  • record timings (index / ANN build / bruteforce eval / ANN eval)
  • write results incrementally to CSV so you don't lose work on crashes
  • repair missing outputs after an interrupted sweep
  • generate a structured summary (CSV + Markdown)

USAGE (PowerShell examples):
  python scripts\photofinder_full_sweep_v4.py sweep `
    --dataset data\lfw\lfw_funneled `
    --out-root runs\sweeps\lfw `
    --models dlib_resnet_v1 arcface_onnx opencv_sface mobilefacenet_onnx `
    --phases baseline index_knobs ann_knobs

  # After an interrupted sweep:
  python scripts\photofinder_full_sweep_v4.py repair `
    --run-root runs\sweeps\lfw `
    --ann-type hnsw --hnsw-m 32 --ef-construction 200 `
    --ann-k 500 --ef-search 128 --rerank on --top-k 10

  python scripts\photofinder_full_sweep_v4.py summarize --run-root runs\sweeps\lfw

Notes:
  • `photofinder eval-retrieval --backend ann` always looks for index.faiss *beside* index.npz.
    This script builds per-ANN-config indices in a subfolder and then copies the chosen one
    to <run_dir>\index.faiss right before running the ANN eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------
# Helpers
# ---------------------------


def now_s() -> float:
    return time.perf_counter()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_rel(p: Path) -> str:
    try:
        return str(p)
    except Exception:
        return p.as_posix()


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    capture: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, float, str]:
    """Run command.

    If capture=True: stream stdout/stderr to console AND also return captured text.
    Returns: (returncode, elapsed_seconds, captured_text)
    """

    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"  → RUN: {cmd_str}")

    t0 = now_s()

    if not capture:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
        dt = now_s() - t0
        return proc.returncode, dt, ""

    # capture while streaming
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None

    buf: List[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        buf.append(line)

    rc = proc.wait()
    dt = now_s() - t0
    return rc, dt, "".join(buf)


_JSON_RE = re.compile(r"\{[\s\S]*?\}\s*$", re.MULTILINE)


def capture_last_json(text: str) -> Optional[Dict[str, Any]]:
    """Find the last JSON-ish dict printed in stdout and parse it."""
    # photofinder prints python dicts with single quotes (not strict json)
    # So we accept either JSON or python-literal dict.
    # Strategy: find last {...} block, then try json.loads, else ast.literal_eval.
    m = _JSON_RE.search(text.strip())
    if not m:
        return None

    blob = m.group(0).strip()

    # try strict json first
    try:
        return json.loads(blob)
    except Exception:
        pass

    # try python literal
    try:
        import ast

        obj = ast.literal_eval(blob)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def fmt_f(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


# ---------------------------
# Configs
# ---------------------------


@dataclass
class IndexCfg:
    face_policy: str = "largest"
    det_upsample: int = 1
    min_face_area: int = 0
    max_faces: int = 5
    fail_policy: str = "skip"
    metric: str = "cosine"  # cosine | l2
    normalize: str = "on"  # on | off
    arcface_padding: float = 0.25
    arcface_preproc: str = "insightface"

    def id(self) -> str:
        # keep reasonably short for Windows paths
        return (
            f"fp_{self.face_policy}"
            f"_du_{self.det_upsample}"
            f"_m_{self.metric}"
            f"_n_{self.normalize}"
            f"_ap_{self.arcface_preproc}"
            f"_pad_{self.arcface_padding:g}"
        )


@dataclass
class AnnBuildCfg:
    ann_type: str = "hnsw"  # hnsw | flat
    faiss_metric: str = "infer"  # infer | ip | l2
    hnsw_m: int = 32
    ef_construction: int = 200

    def id(self) -> str:
        return f"ann_{self.ann_type}_fm_{self.faiss_metric}_M_{self.hnsw_m}_efC_{self.ef_construction}"


@dataclass
class AnnEvalCfg:
    ann_k: int = 500
    ef_search: int = 128
    rerank: str = "on"  # on | off
    top_k: int = 10

    def id(self) -> str:
        return f"k_{self.ann_k}_efS_{self.ef_search}_rr_{self.rerank}_top_{self.top_k}"


@dataclass
class RunResult:
    phase: str
    model: str
    cfg_id: str
    run_dir: str

    # timings
    index_time_s: Optional[float] = None
    ann_build_time_s: Optional[float] = None
    brut_eval_time_s: Optional[float] = None
    ann_eval_time_s: Optional[float] = None

    # bruteforce metrics
    brut_rank1: Optional[float] = None
    brut_recall_at_5: Optional[float] = None
    brut_recall_at_10: Optional[float] = None
    brut_mrr: Optional[float] = None
    brut_n_queries: Optional[int] = None

    # ann metrics
    ann_rank1: Optional[float] = None
    ann_recall_at_5: Optional[float] = None
    ann_recall_at_10: Optional[float] = None
    ann_mrr: Optional[float] = None
    ann_n_queries: Optional[int] = None

    # config details (for reporting)
    index_cfg: Optional[Dict[str, Any]] = None
    ann_build_cfg: Optional[Dict[str, Any]] = None
    ann_eval_cfg: Optional[Dict[str, Any]] = None

    error: Optional[str] = None


# ---------------------------
# photofinder command builders
# ---------------------------


def cmd_index(dataset: Path, model: str, out_dir: Path, cfg: IndexCfg) -> List[str]:
    return [
        "photofinder",
        "index",
        "--dataset",
        str(dataset),
        "--model",
        model,
        "--out",
        str(out_dir),
        "--face-policy",
        cfg.face_policy,
        "--det-upsample",
        str(cfg.det_upsample),
        "--min-face-area",
        str(cfg.min_face_area),
        "--max-faces",
        str(cfg.max_faces),
        "--fail-policy",
        cfg.fail_policy,
        "--metric",
        cfg.metric,
        "--normalize",
        cfg.normalize,
        "--arcface-padding",
        str(cfg.arcface_padding),
        "--arcface-preproc",
        cfg.arcface_preproc,
    ]


def cmd_build_ann(index_npz: Path, out_faiss: Path, bcfg: AnnBuildCfg) -> List[str]:
    cmd = [
        "photofinder",
        "build-ann",
        "--index",
        str(index_npz),
        "--out",
        str(out_faiss),
        "--ann-type",
        bcfg.ann_type,
        "--hnsw-m",
        str(bcfg.hnsw_m),
        "--ef-construction",
        str(bcfg.ef_construction),
    ]
    if bcfg.faiss_metric != "infer":
        cmd += ["--faiss-metric", bcfg.faiss_metric]
    return cmd


def cmd_eval(index_npz: Path, out_dir: Path, backend: str, ecfg: AnnEvalCfg) -> List[str]:
    cmd = [
        "photofinder",
        "eval-retrieval",
        "--index",
        str(index_npz),
        "--out",
        str(out_dir),
        "--backend",
        backend,
        "--top-k",
        str(ecfg.top_k),
    ]
    if backend == "ann":
        cmd += [
            "--ann-k",
            str(ecfg.ann_k),
            "--ef-search",
            str(ecfg.ef_search),
            "--rerank",
            ecfg.rerank,
        ]
    return cmd


# ---------------------------
# Result I/O (incremental)
# ---------------------------


CSV_FIELDS = [
    "phase",
    "model",
    "cfg_id",
    "run_dir",
    "index_time_s",
    "ann_build_time_s",
    "brut_eval_time_s",
    "ann_eval_time_s",
    "brut_rank1",
    "brut_recall_at_5",
    "brut_recall_at_10",
    "brut_mrr",
    "brut_n_queries",
    "ann_rank1",
    "ann_recall_at_5",
    "ann_recall_at_10",
    "ann_mrr",
    "ann_n_queries",
    "index_cfg",
    "ann_build_cfg",
    "ann_eval_cfg",
    "error",
]


def append_csv(path: Path, rr: RunResult) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        row = asdict(rr)
        # JSON-encode nested dicts for easy reading
        for k in ("index_cfg", "ann_build_cfg", "ann_eval_cfg"):
            if row.get(k) is not None:
                row[k] = json.dumps(row[k], ensure_ascii=False)
        w.writerow({k: row.get(k) for k in CSV_FIELDS})
        f.flush()


def write_summary_md(path: Path, rows: List[RunResult]) -> None:
    ensure_dir(path.parent)

    # sort for readability
    rows = sorted(rows, key=lambda r: (r.phase, r.model, r.cfg_id))

    with path.open("w", encoding="utf-8") as f:
        f.write("# Photofinder Sweep Summary\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # group by (phase, model)
        cur = None
        for r in rows:
            key = (r.phase, r.model)
            if key != cur:
                cur = key
                f.write(f"\n## Phase: {r.phase} — Model: {r.model}\n\n")
                f.write(
                    "| cfg_id | brut Rank1 | brut R@5 | brut R@10 | ann Rank1 | ann R@5 | ann R@10 | index_s | ann_build_s | brut_eval_s | ann_eval_s | error |\n"
                )
                f.write(
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
                )

            f.write(
                "| "
                + " | ".join(
                    [
                        r.cfg_id,
                        fmt_f(r.brut_rank1),
                        fmt_f(r.brut_recall_at_5),
                        fmt_f(r.brut_recall_at_10),
                        fmt_f(r.ann_rank1),
                        fmt_f(r.ann_recall_at_5),
                        fmt_f(r.ann_recall_at_10),
                        fmt_f(r.index_time_s, 1),
                        fmt_f(r.ann_build_time_s, 1),
                        fmt_f(r.brut_eval_time_s, 1),
                        fmt_f(r.ann_eval_time_s, 1),
                        (r.error or ""),
                    ]
                )
                + " |\n"
            )


# ---------------------------
# Sweep logic
# ---------------------------


def default_index_cfgs() -> List[IndexCfg]:
    # Baseline knob values are the photofinder defaults.
    # Keep the sweep small enough to be runnable.
    return [
        IndexCfg(det_upsample=0),
        IndexCfg(det_upsample=1),
        IndexCfg(det_upsample=2),
    ]


def default_ann_build_cfgs() -> List[AnnBuildCfg]:
    return [
        AnnBuildCfg(ann_type="hnsw", faiss_metric="infer", hnsw_m=32, ef_construction=200),
        AnnBuildCfg(ann_type="hnsw", faiss_metric="infer", hnsw_m=48, ef_construction=200),
        AnnBuildCfg(ann_type="hnsw", faiss_metric="infer", hnsw_m=32, ef_construction=400),
    ]


def default_ann_eval_cfgs() -> List[AnnEvalCfg]:
    return [
        AnnEvalCfg(ann_k=200, ef_search=64, rerank="on", top_k=10),
        AnnEvalCfg(ann_k=500, ef_search=128, rerank="on", top_k=10),
        AnnEvalCfg(ann_k=800, ef_search=256, rerank="on", top_k=10),
    ]


def sweep_one(
    *,
    phase: str,
    dataset: Path,
    out_root: Path,
    model: str,
    idx_cfg: IndexCfg,
    bcfg: Optional[AnnBuildCfg],
    ecfg: AnnEvalCfg,
    force: bool,
    csv_path: Path,
    rows: List[RunResult],
) -> None:
    cfg_id = idx_cfg.id()
    run_dir = out_root / phase / model / cfg_id
    ensure_dir(run_dir)

    rr = RunResult(phase=phase, model=model, cfg_id=cfg_id, run_dir=str(run_dir))
    rr.index_cfg = asdict(idx_cfg)
    rr.ann_eval_cfg = asdict(ecfg)
    rr.ann_build_cfg = asdict(bcfg) if bcfg else None

    index_npz = run_dir / "index.npz"

    # ---- Index ----
    if force or not index_npz.exists():
        rc, dt, _ = run_cmd(cmd_index(dataset, model, run_dir, idx_cfg), capture=False)
        rr.index_time_s = dt
        if rc != 0:
            rr.error = f"index failed (rc={rc})"
            append_csv(csv_path, rr)
            rows.append(rr)
            return
    else:
        rr.index_time_s = None

    # ---- Bruteforce eval ----
    brut_json_path = run_dir / "eval_bruteforce.json"
    if force or not brut_json_path.exists():
        rc, dt, out = run_cmd(cmd_eval(index_npz, run_dir, "bruteforce", ecfg), capture=True)
        rr.brut_eval_time_s = dt
        if rc != 0:
            rr.error = f"bruteforce eval failed (rc={rc})"
            append_csv(csv_path, rr)
            rows.append(rr)
            return
        m = capture_last_json(out)
        if m:
            rr.brut_rank1 = float(m.get("rank1")) if m.get("rank1") is not None else None
            rr.brut_recall_at_5 = float(m.get("recall_at_5")) if m.get("recall_at_5") is not None else None
            rr.brut_recall_at_10 = float(m.get("recall_at_10")) if m.get("recall_at_10") is not None else None
            rr.brut_mrr = float(m.get("mrr")) if m.get("mrr") is not None else None
            rr.brut_n_queries = int(m.get("n_queries")) if m.get("n_queries") is not None else None
            with brut_json_path.open("w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
        else:
            rr.error = "bruteforce eval produced no metrics"
    else:
        try:
            m = json.loads(brut_json_path.read_text(encoding="utf-8"))
            rr.brut_rank1 = float(m.get("rank1")) if m.get("rank1") is not None else None
            rr.brut_recall_at_5 = float(m.get("recall_at_5")) if m.get("recall_at_5") is not None else None
            rr.brut_recall_at_10 = float(m.get("recall_at_10")) if m.get("recall_at_10") is not None else None
            rr.brut_mrr = float(m.get("mrr")) if m.get("mrr") is not None else None
            rr.brut_n_queries = int(m.get("n_queries")) if m.get("n_queries") is not None else None
        except Exception:
            pass

    # ---- ANN build + ANN eval ----
    if bcfg is not None:
        ann_dir = run_dir / "_ann" / (bcfg.id() + "_" + ecfg.id())
        ensure_dir(ann_dir)

        ann_faiss = ann_dir / "index.faiss"
        run_faiss = run_dir / "index.faiss"  # this is what photofinder eval-retrieval expects
        ann_json_path = ann_dir / "eval_ann.json"

        # Build ANN
        if force or not ann_faiss.exists():
            rc, dt, _ = run_cmd(cmd_build_ann(index_npz, ann_faiss, bcfg), capture=False)
            rr.ann_build_time_s = dt
            if rc != 0:
                rr.error = f"build-ann failed (rc={rc})"
                append_csv(csv_path, rr)
                rows.append(rr)
                return

        # Copy beside index.npz for eval
        try:
            # overwrite any existing
            if run_faiss.exists():
                run_faiss.unlink()
            # use copyfile to preserve Windows compatibility
            import shutil

            shutil.copyfile(ann_faiss, run_faiss)
        except Exception as e:
            rr.error = f"failed to copy index.faiss beside index.npz: {e}"
            append_csv(csv_path, rr)
            rows.append(rr)
            return

        # ANN eval
        if force or not ann_json_path.exists():
            rc, dt, out = run_cmd(cmd_eval(index_npz, ann_dir, "ann", ecfg), capture=True)
            rr.ann_eval_time_s = dt
            if rc != 0:
                rr.error = f"ann eval failed (rc={rc})"
                append_csv(csv_path, rr)
                rows.append(rr)
                return
            m = capture_last_json(out)
            if m:
                rr.ann_rank1 = float(m.get("rank1")) if m.get("rank1") is not None else None
                rr.ann_recall_at_5 = float(m.get("recall_at_5")) if m.get("recall_at_5") is not None else None
                rr.ann_recall_at_10 = float(m.get("recall_at_10")) if m.get("recall_at_10") is not None else None
                rr.ann_mrr = float(m.get("mrr")) if m.get("mrr") is not None else None
                rr.ann_n_queries = int(m.get("n_queries")) if m.get("n_queries") is not None else None
                with ann_json_path.open("w", encoding="utf-8") as f:
                    json.dump(m, f, indent=2)
            else:
                rr.error = "ann eval produced no metrics"
        else:
            try:
                m = json.loads(ann_json_path.read_text(encoding="utf-8"))
                rr.ann_rank1 = float(m.get("rank1")) if m.get("rank1") is not None else None
                rr.ann_recall_at_5 = float(m.get("recall_at_5")) if m.get("recall_at_5") is not None else None
                rr.ann_recall_at_10 = float(m.get("recall_at_10")) if m.get("recall_at_10") is not None else None
                rr.ann_mrr = float(m.get("mrr")) if m.get("mrr") is not None else None
                rr.ann_n_queries = int(m.get("n_queries")) if m.get("n_queries") is not None else None
            except Exception:
                pass

    # Append row after each run so you don't lose progress
    append_csv(csv_path, rr)
    rows.append(rr)


def do_sweep(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    phases = args.phases
    models = args.models

    # output files
    csv_path = out_root / "results_incremental.csv"
    md_path = out_root / "summary.md"

    rows: List[RunResult] = []

    ecfg = AnnEvalCfg(ann_k=args.ann_k, ef_search=args.ef_search, rerank=args.rerank, top_k=args.top_k)

    for phase in phases:
        print(f"\n=== PHASE: {phase} ===\n")

        if phase == "baseline":
            idx_cfgs = [IndexCfg()]
            ann_build_cfgs = [AnnBuildCfg(ann_type=args.ann_type, faiss_metric=args.faiss_metric, hnsw_m=args.hnsw_m, ef_construction=args.ef_construction)]

            for mi, model in enumerate(models, 1):
                print(f"[{mi}/{len(models)}] MODEL: {model}")
                sweep_one(
                    phase=phase,
                    dataset=dataset,
                    out_root=out_root,
                    model=model,
                    idx_cfg=idx_cfgs[0],
                    bcfg=ann_build_cfgs[0],
                    ecfg=ecfg,
                    force=args.force,
                    csv_path=csv_path,
                    rows=rows,
                )

        elif phase == "index_knobs":
            idx_cfgs = default_index_cfgs()
            bcfg = AnnBuildCfg(ann_type=args.ann_type, faiss_metric=args.faiss_metric, hnsw_m=args.hnsw_m, ef_construction=args.ef_construction)

            for mi, model in enumerate(models, 1):
                print(f"[{mi}/{len(models)}] MODEL: {model}")
                print(f"  Index configs: {len(idx_cfgs)}\n")
                for ci, idx_cfg in enumerate(idx_cfgs, 1):
                    print(f"  ({ci}/{len(idx_cfgs)}) index_cfg: {idx_cfg.id()}")
                    sweep_one(
                        phase=phase,
                        dataset=dataset,
                        out_root=out_root,
                        model=model,
                        idx_cfg=idx_cfg,
                        bcfg=bcfg,
                        ecfg=ecfg,
                        force=args.force,
                        csv_path=csv_path,
                        rows=rows,
                    )

        elif phase == "ann_knobs":
            # keep index fixed at baseline, sweep ANN build/eval knobs
            idx_cfg = IndexCfg()
            ann_build_cfgs = default_ann_build_cfgs()
            ann_eval_cfgs = default_ann_eval_cfgs()

            for mi, model in enumerate(models, 1):
                print(f"[{mi}/{len(models)}] MODEL: {model}")
                print(f"  ANN build configs: {len(ann_build_cfgs)}")
                print(f"  ANN eval configs : {len(ann_eval_cfgs)}\n")

                for bi, bcfg in enumerate(ann_build_cfgs, 1):
                    for ei, ecfg2 in enumerate(ann_eval_cfgs, 1):
                        print(
                            f"  ({bi}/{len(ann_build_cfgs)}) build_cfg: {bcfg.id()} | ({ei}/{len(ann_eval_cfgs)}) eval_cfg: {ecfg2.id()}"
                        )
                        sweep_one(
                            phase=phase,
                            dataset=dataset,
                            out_root=out_root,
                            model=model,
                            idx_cfg=idx_cfg,
                            bcfg=bcfg,
                            ecfg=ecfg2,
                            force=args.force,
                            csv_path=csv_path,
                            rows=rows,
                        )

        else:
            print(f"Unknown phase: {phase}")
            return 2

    write_summary_md(md_path, rows)
    print(f"\nSaved incremental CSV: {csv_path}")
    print(f"Saved summary markdown: {md_path}")
    return 0


# ---------------------------
# Summarize / Repair
# ---------------------------


def iter_run_dirs(run_root: Path) -> Iterable[Path]:
    # Any directory that contains index.npz is a run.
    for p in run_root.rglob("index.npz"):
        yield p.parent


def load_existing_result(run_dir: Path) -> RunResult:
    # phase/model/cfg_id are inferred from path layout if possible
    parts = run_dir.parts
    phase = "unknown"
    model = "unknown"
    cfg_id = run_dir.name

    # heuristic: .../<phase>/<model>/<cfg_id>
    if len(parts) >= 3:
        phase = parts[-3]
        model = parts[-2]

    rr = RunResult(phase=phase, model=model, cfg_id=cfg_id, run_dir=str(run_dir))

    brut_path = run_dir / "eval_bruteforce.json"
    if brut_path.exists():
        try:
            m = json.loads(brut_path.read_text(encoding="utf-8"))
            rr.brut_rank1 = float(m.get("rank1")) if m.get("rank1") is not None else None
            rr.brut_recall_at_5 = float(m.get("recall_at_5")) if m.get("recall_at_5") is not None else None
            rr.brut_recall_at_10 = float(m.get("recall_at_10")) if m.get("recall_at_10") is not None else None
            rr.brut_mrr = float(m.get("mrr")) if m.get("mrr") is not None else None
            rr.brut_n_queries = int(m.get("n_queries")) if m.get("n_queries") is not None else None
        except Exception:
            pass

    # find ann results (best effort: load the first eval_ann.json under _ann)
    ann_jsons = list(run_dir.glob("_ann/**/eval_ann.json"))
    if ann_jsons:
        ann_path = sorted(ann_jsons)[0]
        try:
            m = json.loads(ann_path.read_text(encoding="utf-8"))
            rr.ann_rank1 = float(m.get("rank1")) if m.get("rank1") is not None else None
            rr.ann_recall_at_5 = float(m.get("recall_at_5")) if m.get("recall_at_5") is not None else None
            rr.ann_recall_at_10 = float(m.get("recall_at_10")) if m.get("recall_at_10") is not None else None
            rr.ann_mrr = float(m.get("mrr")) if m.get("mrr") is not None else None
            rr.ann_n_queries = int(m.get("n_queries")) if m.get("n_queries") is not None else None
        except Exception:
            pass

    return rr


def do_summarize(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    out_csv = run_root / "summary_results.csv"
    out_md = run_root / "summary_results.md"

    rows = [load_existing_result(d) for d in iter_run_dirs(run_root)]

    # overwrite output csv
    if out_csv.exists():
        out_csv.unlink()
    for r in rows:
        append_csv(out_csv, r)

    write_summary_md(out_md, rows)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Runs found: {len(rows)}")
    return 0


def repair_runs(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    out_csv = run_root / "repaired_results.csv"
    out_md = run_root / "repaired_summary.md"

    # Build cfg to (re)create a *single* ANN index per run (so summarize has something consistent)
    bcfg = AnnBuildCfg(
        ann_type=args.ann_type,
        faiss_metric=args.faiss_metric,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
    )
    ecfg = AnnEvalCfg(ann_k=args.ann_k, ef_search=args.ef_search, rerank=args.rerank, top_k=args.top_k)

    # fresh outputs
    if out_csv.exists():
        out_csv.unlink()

    rows: List[RunResult] = []

    run_dirs = sorted(iter_run_dirs(run_root))
    print(f"Found {len(run_dirs)} run folders under {run_root}")

    for i, run_dir in enumerate(run_dirs, 1):
        print(f"\n[{i}/{len(run_dirs)}] REPAIR: {run_dir}")
        rr = load_existing_result(run_dir)

        index_npz = run_dir / "index.npz"
        if not index_npz.exists():
            rr.error = "missing index.npz"
            append_csv(out_csv, rr)
            rows.append(rr)
            continue

        # Bruteforce metrics
        brut_json_path = run_dir / "eval_bruteforce.json"
        if args.force or not brut_json_path.exists():
            rc, dt, out = run_cmd(cmd_eval(index_npz, run_dir, "bruteforce", ecfg), capture=True)
            rr.brut_eval_time_s = dt
            if rc == 0:
                m = capture_last_json(out)
                if m:
                    rr.brut_rank1 = float(m.get("rank1")) if m.get("rank1") is not None else None
                    rr.brut_recall_at_5 = float(m.get("recall_at_5")) if m.get("recall_at_5") is not None else None
                    rr.brut_recall_at_10 = float(m.get("recall_at_10")) if m.get("recall_at_10") is not None else None
                    rr.brut_mrr = float(m.get("mrr")) if m.get("mrr") is not None else None
                    rr.brut_n_queries = int(m.get("n_queries")) if m.get("n_queries") is not None else None
                    brut_json_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
            else:
                rr.error = f"bruteforce eval failed (rc={rc})"

        # ANN build + eval
        ann_dir = run_dir / "_ann" / (bcfg.id() + "_" + ecfg.id())
        ensure_dir(ann_dir)
        ann_faiss = ann_dir / "index.faiss"
        run_faiss = run_dir / "index.faiss"
        ann_json_path = ann_dir / "eval_ann.json"

        if args.force or not ann_faiss.exists():
            rc, dt, _ = run_cmd(cmd_build_ann(index_npz, ann_faiss, bcfg), capture=False)
            rr.ann_build_time_s = dt
            if rc != 0:
                rr.error = (rr.error or "") + f" | build-ann failed (rc={rc})"

        # Ensure index.faiss is beside index.npz
        if ann_faiss.exists():
            try:
                import shutil

                if run_faiss.exists():
                    run_faiss.unlink()
                shutil.copyfile(ann_faiss, run_faiss)
            except Exception as e:
                rr.error = (rr.error or "") + f" | copy index.faiss failed: {e}"

        if args.force or not ann_json_path.exists():
            if run_faiss.exists():
                rc, dt, out = run_cmd(cmd_eval(index_npz, ann_dir, "ann", ecfg), capture=True)
                rr.ann_eval_time_s = dt
                if rc == 0:
                    m = capture_last_json(out)
                    if m:
                        rr.ann_rank1 = float(m.get("rank1")) if m.get("rank1") is not None else None
                        rr.ann_recall_at_5 = float(m.get("recall_at_5")) if m.get("recall_at_5") is not None else None
                        rr.ann_recall_at_10 = float(m.get("recall_at_10")) if m.get("recall_at_10") is not None else None
                        rr.ann_mrr = float(m.get("mrr")) if m.get("mrr") is not None else None
                        rr.ann_n_queries = int(m.get("n_queries")) if m.get("n_queries") is not None else None
                        ann_json_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
                else:
                    rr.error = (rr.error or "") + f" | ann eval failed (rc={rc})"
            else:
                rr.error = (rr.error or "") + " | missing run_dir/index.faiss for ANN eval"

        # record config info
        rr.ann_build_cfg = asdict(bcfg)
        rr.ann_eval_cfg = asdict(ecfg)

        append_csv(out_csv, rr)
        rows.append(rr)

    write_summary_md(out_md, rows)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_md}")
    return 0


# ---------------------------
# CLI
# ---------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="photofinder_full_sweep_v4.py",
        description="Run / repair / summarize photofinder sweeps.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # sweep
    ps = sub.add_parser("sweep", help="Run sweep (baseline / index_knobs / ann_knobs).")
    ps.add_argument("--dataset", required=True)
    ps.add_argument("--out-root", required=True)
    ps.add_argument("--models", nargs="+", required=True)
    ps.add_argument(
        "--phases",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "index_knobs", "ann_knobs"],
    )
    ps.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")

    # baseline + default ann params
    ps.add_argument("--ann-type", default="hnsw", choices=["hnsw", "flat"])
    ps.add_argument("--faiss-metric", default="infer", choices=["infer", "ip", "l2"])
    ps.add_argument("--hnsw-m", type=int, default=32)
    ps.add_argument("--ef-construction", type=int, default=200)

    # eval params
    ps.add_argument("--ann-k", type=int, default=500)
    ps.add_argument("--ef-search", type=int, default=128)
    ps.add_argument("--rerank", default="on", choices=["on", "off"])
    ps.add_argument("--top-k", type=int, default=10)

    # summarize
    pm = sub.add_parser("summarize", help="Scan run-root and write consolidated CSV/MD.")
    pm.add_argument("--run-root", required=True)

    # repair
    pr = sub.add_parser("repair", help="Fill missing eval/ANN artifacts for existing runs.")
    pr.add_argument("--run-root", required=True)
    pr.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")

    pr.add_argument("--ann-type", default="hnsw", choices=["hnsw", "flat"])
    pr.add_argument("--faiss-metric", default="infer", choices=["infer", "ip", "l2"])
    pr.add_argument("--hnsw-m", type=int, default=32)
    pr.add_argument("--ef-construction", type=int, default=200)

    pr.add_argument("--ann-k", type=int, default=500)
    pr.add_argument("--ef-search", type=int, default=128)
    pr.add_argument("--rerank", default="on", choices=["on", "off"])
    pr.add_argument("--top-k", type=int, default=10)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "sweep":
        return do_sweep(args)
    if args.cmd == "summarize":
        return do_summarize(args)
    if args.cmd == "repair":
        return repair_runs(args)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
