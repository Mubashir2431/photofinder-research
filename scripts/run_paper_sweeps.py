from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_photofinder_cmd() -> List[str]:
    """
    Prefer the console script `photofinder`.
    Fallback: `python -m photofinder.cli`
    """
    try:
        subprocess.run(
            ["photofinder", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return ["photofinder"]
    except Exception:
        return [sys.executable, "-m", "photofinder.cli"]


def _banner(msg: str) -> None:
    print("\n" + "=" * 110)
    print(msg)
    print("=" * 110 + "\n", flush=True)


def run_stream(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, float]:
    """
    Run a subprocess while streaming stdout/stderr live to the console AND writing to log file.
    This fixes the "blank terminal" problem.
    """
    t0 = time.perf_counter()

    log_f = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8")

    def tee(line: str) -> None:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_f:
            log_f.write(line)

    try:
        # Merge stderr into stdout so tqdm/errors are visible live.
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        tee("CMD:\n" + " ".join(cmd) + "\n\n")

        assert p.stdout is not None
        for line in p.stdout:
            tee(line)

        rc = p.wait()
        dt = time.perf_counter() - t0
        tee(f"\n[exit_code={rc}] elapsed_seconds={dt:.2f}\n")
        return rc, dt
    finally:
        if log_f:
            log_f.close()


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class IndexExp:
    name: str
    args: List[str]  # args after: index --dataset ... --model ... --out ...


@dataclass
class AnnBuildExp:
    name: str
    args: List[str]  # args after: build-ann --index ... (we will add index/out)


@dataclass
class EvalExp:
    name: str
    args: List[str]  # args after: eval-retrieval --index ... --out ... (we will add index/out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"data\lfw\lfw_funneled", help="Dataset root in imagefolder format")
    ap.add_argument("--out-root", default=r"runs\paper\lfw_funneled", help="Root folder for paper runs")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--ann-k", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=["arcface_onnx", "dlib_resnet_v1"])
    ap.add_argument("--quick", action="store_true", help="Run fewer experiments (sanity sweep)")
    ap.add_argument("--resume", action="store_true", help="Skip steps if outputs already exist")
    args = ap.parse_args()

    photofinder = find_photofinder_cmd()
    dataset = args.dataset
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Ensure live output even if some tools buffer
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    _banner(f"Paper sweeps starting\ndataset={dataset}\nout_root={out_root}\nmodels={args.models}\nquick={args.quick}\nresume={args.resume}")

    # ----------------------------
    # Index-time sweeps (rebuild index.npz)
    # ----------------------------
    index_exps: List[IndexExp] = [
        IndexExp(
            "baseline",
            [
                "--face-policy", "largest",
                "--det-upsample", "1",
                "--min-face-area", "0",
                "--fail-policy", "skip",
                "--metric", "cosine",
                "--normalize", "on",
            ],
        ),
        IndexExp(
            "upsample0",
            ["--face-policy", "largest", "--det-upsample", "0", "--metric", "cosine", "--normalize", "on"],
        ),
        IndexExp(
            "upsample2",
            ["--face-policy", "largest", "--det-upsample", "2", "--metric", "cosine", "--normalize", "on"],
        ),
        IndexExp(
            "minarea900",
            ["--face-policy", "largest", "--det-upsample", "1", "--min-face-area", "900", "--metric", "cosine", "--normalize", "on"],
        ),
        IndexExp(
            "face_all_5",
            ["--face-policy", "all", "--max-faces", "5", "--det-upsample", "1", "--metric", "cosine", "--normalize", "on"],
        ),
    ]

    # ArcFace-only sweeps (padding/preproc)
    arcface_only_exps: List[IndexExp] = [
        IndexExp(
            "arcface_pad010",
            ["--face-policy", "largest", "--det-upsample", "1", "--metric", "cosine", "--normalize", "on",
             "--arcface-padding", "0.10", "--arcface-preproc", "insightface"],
        ),
        IndexExp(
            "arcface_pad040",
            ["--face-policy", "largest", "--det-upsample", "1", "--metric", "cosine", "--normalize", "on",
             "--arcface-padding", "0.40", "--arcface-preproc", "insightface"],
        ),
        IndexExp(
            "arcface_legacy_preproc",
            ["--face-policy", "largest", "--det-upsample", "1", "--metric", "cosine", "--normalize", "on",
             "--arcface-padding", "0.25", "--arcface-preproc", "legacy"],
        ),
    ]

    if args.quick:
        index_exps = [index_exps[0], index_exps[2]]  # baseline + upsample2
        arcface_only_exps = [arcface_only_exps[0]]

    # ----------------------------
    # ANN build sweeps (rebuild index.faiss only)
    # ----------------------------
    ann_build_exps: List[AnnBuildExp] = [
        AnnBuildExp("hnsw_m32_efc200", ["--ann-type", "hnsw", "--hnsw-m", "32", "--ef-construction", "200"]),
        AnnBuildExp("hnsw_m16_efc200", ["--ann-type", "hnsw", "--hnsw-m", "16", "--ef-construction", "200"]),
        AnnBuildExp("hnsw_m64_efc200", ["--ann-type", "hnsw", "--hnsw-m", "64", "--ef-construction", "200"]),
        AnnBuildExp("flat_exact", ["--ann-type", "flat"]),
    ]
    if args.quick:
        ann_build_exps = [ann_build_exps[0], ann_build_exps[-1]]

    # ----------------------------
    # Eval sweeps (no rebuild)
    # ----------------------------
    eval_exps: List[EvalExp] = [
        EvalExp(
            "eval_default_rerank_on",
            ["--backend", "both", "--top-k", str(args.topk), "--ann-k", str(args.ann_k),
             "--ef-search", str(args.ef_search), "--rerank", "on"],
        ),
        EvalExp(
            "eval_ann_weak_rerank_off",
            ["--backend", "ann", "--top-k", str(args.topk), "--ann-k", "50",
             "--ef-search", "16", "--rerank", "off"],
        ),
        EvalExp(
            "eval_ann_strong_rerank_off",
            ["--backend", "ann", "--top-k", str(args.topk), "--ann-k", "200",
             "--ef-search", "128", "--rerank", "off"],
        ),
    ]
    if args.quick:
        eval_exps = [eval_exps[0], eval_exps[1]]

    summary_rows: List[Dict[str, Any]] = []

    for model in args.models:
        model_root = out_root / model
        model_root.mkdir(parents=True, exist_ok=True)

        this_index_exps = list(index_exps)
        if model == "arcface_onnx":
            this_index_exps += arcface_only_exps

        for ix in this_index_exps:
            exp_dir = model_root / ix.name
            exp_dir.mkdir(parents=True, exist_ok=True)

            index_npz = exp_dir / "index.npz"

            # 1) INDEX
            if args.resume and index_npz.exists():
                _banner(f"[RESUME] SKIP index: {model}/{ix.name} (index.npz already exists)")
                dt_index = 0.0
            else:
                _banner(f"[1/3] INDEX  model={model}  exp={ix.name}")
                cmd_index = photofinder + [
                    "index",
                    "--dataset", dataset,
                    "--model", model,
                    "--out", str(exp_dir),
                ] + ix.args

                code, dt_index = run_stream(cmd_index, log_path=exp_dir / "logs" / "01_index.log", env=env)
                if code != 0:
                    print(f"[FAIL] index {model}/{ix.name} (see {exp_dir / 'logs' / '01_index.log'})", flush=True)
                    continue

            if not index_npz.exists():
                print(f"[FAIL] index.npz missing for {model}/{ix.name}", flush=True)
                continue

            # 2) ANN BUILD SWEEPS
            for ab in ann_build_exps:
                ann_dir = exp_dir / "ann" / ab.name
                ann_dir.mkdir(parents=True, exist_ok=True)

                target_npz = ann_dir / "index.npz"
                target_faiss = ann_dir / "index.faiss"

                if not target_npz.exists():
                    shutil.copy2(index_npz, target_npz)

                if args.resume and target_faiss.exists():
                    _banner(f"[RESUME] SKIP build-ann: {model}/{ix.name}/{ab.name} (index.faiss exists)")
                    dt_build = 0.0
                else:
                    _banner(f"[2/3] BUILD-ANN  model={model}  exp={ix.name}  ann={ab.name}")
                    cmd_build = photofinder + ["build-ann", "--index", str(target_npz)] + ab.args

                    code_b, dt_build = run_stream(cmd_build, log_path=ann_dir / "logs" / "02_build_ann.log", env=env)
                    if code_b != 0:
                        print(f"[FAIL] build-ann {model}/{ix.name}/{ab.name} (see {ann_dir / 'logs' / '02_build_ann.log'})", flush=True)
                        continue

                # 3) EVAL SWEEPS
                for ev in eval_exps:
                    out_eval = ann_dir / "eval" / ev.name
                    out_eval.mkdir(parents=True, exist_ok=True)

                    bf_json = out_eval / "metrics_retrieval_bruteforce.json"
                    ann_json = out_eval / "metrics_retrieval_ann.json"

                    if args.resume and (bf_json.exists() or ann_json.exists()):
                        _banner(f"[RESUME] SKIP eval: {model}/{ix.name}/{ab.name}/{ev.name} (metrics exist)")
                        dt_eval = 0.0
                    else:
                        _banner(f"[3/3] EVAL  model={model}  exp={ix.name}  ann={ab.name}  eval={ev.name}")
                        cmd_eval = photofinder + [
                            "eval-retrieval",
                            "--index", str(target_npz),
                            "--out", str(out_eval),
                        ] + ev.args

                        code_e, dt_eval = run_stream(cmd_eval, log_path=out_eval / "logs" / "03_eval.log", env=env)
                        if code_e != 0:
                            print(f"[FAIL] eval {model}/{ix.name}/{ab.name}/{ev.name} (see {out_eval / 'logs' / '03_eval.log'})", flush=True)
                            continue

                    row: Dict[str, Any] = {
                        "model": model,
                        "index_exp": ix.name,
                        "ann_build": ab.name,
                        "eval_exp": ev.name,
                        "seconds_index": dt_index,
                        "seconds_build_ann": dt_build,
                        "seconds_eval": dt_eval,
                        "dataset": dataset,
                        "index_npz": str(target_npz),
                        "out_eval": str(out_eval),
                    }

                    cfg_path = exp_dir / "config.json"
                    if cfg_path.exists():
                        row.update({f"cfg_{k}": v for k, v in read_json(cfg_path).items()})

                    if bf_json.exists():
                        m = read_json(bf_json)
                        for k, v in m.items():
                            row[f"bf_{k}"] = v

                    if ann_json.exists():
                        m = read_json(ann_json)
                        for k, v in m.items():
                            row[f"ann_{k}"] = v

                    write_json(out_eval / "sweep_config.json", {
                        "model": model,
                        "dataset": dataset,
                        "index_exp": ix.name,
                        "ann_build": ab.name,
                        "eval_exp": ev.name,
                        "index_args": ix.args,
                        "ann_build_args": ab.args,
                        "eval_args": ev.args,
                    })

                    summary_rows.append(row)
                    print(f"[OK] {model} | {ix.name} | {ab.name} | {ev.name}", flush=True)

    # Write summary.csv
    if summary_rows:
        out_csv = out_root / "summary.csv"
        keys: List[str] = sorted({k for r in summary_rows for k in r.keys()})
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
        _banner(f"Saved summary: {out_csv}")
    else:
        _banner("No results collected.")


if __name__ == "__main__":
    main()
