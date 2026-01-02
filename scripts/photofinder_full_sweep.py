#!/usr/bin/env python3
r"""
photofinder_full_sweep.py

Runs a reproducible sweep over:
  1) baseline (index + bruteforce eval + ANN build + ANN eval)
  2) index_knobs (rebuild embeddings with a few index-time knobs)
  3) ann_knobs (keep embeddings fixed; vary only ANN build/search knobs)

IMPORTANT implementation detail:
  photofinder eval-retrieval --backend ann ALWAYS loads FAISS from:
      Path(index_npz).with_suffix(".faiss")   ->  <run_dir>/index.faiss
  So this script ALWAYS builds ANN in-place (next to index.npz).
  It then COPIES that index.faiss into a per-config archive folder.

Example (PowerShell):
  python scripts\photofinder_full_sweep.py `
    --dataset data\lfw\lfw_funneled `
    --out-root runs\sweeps\lfw `
    --models dlib_resnet_v1 arcface_onnx opencv_sface mobilefacenet_onnx `
    --phases baseline index_knobs ann_knobs

Outputs:
  <out-root>/sweep_results.csv
  <out-root>/sweep_results.jsonl
  <out-root>/sweep_summary.md
  plus per-run folders that contain index.npz, index.faiss, and metrics*.json

"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Config dataclasses
# -----------------------------

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
        ap = f"{self.arcface_padding:.2f}".rstrip("0").rstrip(".")
        return (
            f"fp_{self.face_policy}"
            f"_du_{self.det_upsample}"
            f"_mfa_{self.min_face_area}"
            f"_mf_{self.max_faces}"
            f"_fail_{self.fail_policy}"
            f"_met_{self.metric}"
            f"_norm_{self.normalize}"
            f"_ap_{self.arcface_preproc}"
            f"_pad_{ap}"
        )

    def to_cli(self) -> List[str]:
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
class AnnBuildCfg:
    ann_type: str = "hnsw"
    faiss_metric: Optional[str] = None  # "ip" or "l2" or None (infer)
    hnsw_m: int = 32
    ef_construction: int = 200

    def tag(self) -> str:
        fm = self.faiss_metric or "infer"
        return f"ann_{self.ann_type}_fm_{fm}_M_{self.hnsw_m}_efC_{self.ef_construction}"

    def to_cli(self) -> List[str]:
        args = ["--ann-type", self.ann_type, "--hnsw-m", str(self.hnsw_m), "--ef-construction", str(self.ef_construction)]
        if self.faiss_metric:
            args += ["--faiss-metric", self.faiss_metric]
        return args


@dataclass(frozen=True)
class AnnSearchCfg:
    ann_k: int = 500
    ef_search: int = 128
    rerank: str = "on"  # on/off

    def tag(self) -> str:
        return f"k_{self.ann_k}_efS_{self.ef_search}_rr_{self.rerank}"

    def to_cli(self) -> List[str]:
        return ["--ann-k", str(self.ann_k), "--ef-search", str(self.ef_search), "--rerank", self.rerank]


@dataclass
class ResultRow:
    phase: str
    model: str
    run_dir: str

    index_cfg: Dict
    ann_build_cfg: Optional[Dict]
    ann_search_cfg: Optional[Dict]

    # timings
    t_index_s: float
    t_eval_brut_s: float
    t_build_ann_s: float
    t_eval_ann_s: float

    # metrics bruteforce
    brut_rank1: Optional[float] = None
    brut_recall_at_5: Optional[float] = None
    brut_recall_at_10: Optional[float] = None
    brut_mrr: Optional[float] = None
    brut_n_queries: Optional[int] = None

    # metrics ann
    ann_rank1: Optional[float] = None
    ann_recall_at_5: Optional[float] = None
    ann_recall_at_10: Optional[float] = None
    ann_mrr: Optional[float] = None
    ann_n_queries: Optional[int] = None


# -----------------------------
# Helpers
# -----------------------------

def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> float:
    """Run command and stream output to console (so you see progress bars). Returns elapsed seconds."""
    t0 = time.perf_counter()
    proc = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None)
    t1 = time.perf_counter()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return t1 - t0


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def expected_paths(run_dir: Path) -> Tuple[Path, Path]:
    return run_dir / "index.npz", run_dir / "index.faiss"


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_read_metrics(out_dir: Path, ann: bool) -> Dict:
    fname = "metrics_ann.json" if ann else "metrics.json"
    p = out_dir / fname
    if not p.exists():
        return {}
    try:
        return read_json(p)
    except Exception:
        return {}


def write_jsonl(path: Path, rows: List[ResultRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[ResultRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_summary_md(path: Path, rows: List[ResultRow]) -> None:
    baseline = [r for r in rows if r.phase == "baseline"]
    src = baseline if baseline else rows

    per_model: Dict[str, ResultRow] = {}
    for r in src:
        cur = per_model.get(r.model)
        if cur is None or (r.brut_rank1 or 0) > (cur.brut_rank1 or 0):
            per_model[r.model] = r

    lines: List[str] = []
    lines.append("# Photofinder Sweep Summary\n\n")
    lines.append(f"Total runs: **{len(rows)}**\n\n")
    lines.append("## Best baseline per model (bruteforce)\n\n")
    lines.append("| Model | Rank@1 | Recall@5 | Recall@10 | MRR | Index time | Eval time |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for model, r in sorted(per_model.items()):
        lines.append(
            f"| {model} | {r.brut_rank1:.4f} | {r.brut_recall_at_5:.4f} | {r.brut_recall_at_10:.4f} | "
            f"{r.brut_mrr:.4f} | {r.t_index_s/60:.1f} min | {r.t_eval_brut_s/60:.1f} min |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


# -----------------------------
# Sweep runners
# -----------------------------

def run_index(dataset: Path, model: str, run_dir: Path, idx: IndexCfg, force: bool) -> float:
    index_npz, _ = expected_paths(run_dir)
    if index_npz.exists() and not force:
        return 0.0

    safe_mkdir(run_dir)
    cmd = ["photofinder", "index", "--dataset", str(dataset), "--model", model, "--out", str(run_dir)]
    cmd += idx.to_cli()
    print(f"  → RUN: {' '.join(cmd)}")
    return run_cmd(cmd)


def run_eval_bruteforce(run_dir: Path, top_k: int) -> float:
    index_npz, _ = expected_paths(run_dir)
    cmd = ["photofinder", "eval-retrieval", "--index", str(index_npz), "--out", str(run_dir), "--backend", "bruteforce", "--top-k", str(top_k)]
    print(f"  → RUN: {' '.join(cmd)}")
    return run_cmd(cmd)


def run_build_ann_inplace(run_dir: Path, annb: AnnBuildCfg, force: bool) -> Tuple[float, Path]:
    """
    Build FAISS in-place at <run_dir>/index.faiss (required by photofinder eval-retrieval backend=ann),
    then copy to an archive folder and return the archive path.
    """
    index_npz, faiss_inplace = expected_paths(run_dir)
    ann_archive_dir = run_dir / "_ann" / annb.tag()
    safe_mkdir(ann_archive_dir)
    faiss_archive = ann_archive_dir / "index.faiss"

    if faiss_archive.exists() and (not force):
        if (not faiss_inplace.exists()) or (faiss_inplace.stat().st_size != faiss_archive.stat().st_size):
            shutil.copy2(faiss_archive, faiss_inplace)
        return 0.0, faiss_archive

    cmd = ["photofinder", "build-ann", "--index", str(index_npz)]
    cmd += annb.to_cli()
    print(f"  → RUN: {' '.join(cmd)}")
    t = run_cmd(cmd)

    if not faiss_inplace.exists():
        raise FileNotFoundError(f"Expected FAISS index missing after build: {faiss_inplace}")

    shutil.copy2(faiss_inplace, faiss_archive)
    return t, faiss_archive


def run_eval_ann(run_dir: Path, ann_archive: Path, anns: AnnSearchCfg, top_k: int) -> float:
    index_npz, faiss_inplace = expected_paths(run_dir)

    if (not faiss_inplace.exists()) or (faiss_inplace.stat().st_size != ann_archive.stat().st_size):
        shutil.copy2(ann_archive, faiss_inplace)

    out_dir = ann_archive.parent
    cmd = ["photofinder", "eval-retrieval", "--index", str(index_npz), "--out", str(out_dir), "--backend", "ann", "--top-k", str(top_k)]
    cmd += anns.to_cli()
    print(f"  → RUN: {' '.join(cmd)}")
    return run_cmd(cmd)


# -----------------------------
# Knob grids
# -----------------------------

def baseline_index_cfg() -> IndexCfg:
    return IndexCfg()


def index_knob_grid(quick: bool) -> List[IndexCfg]:
    det = [0, 1] if quick else [0, 1, 2]
    min_area = [0] if quick else [0, 400]
    arc_pre = ["insightface"] if quick else ["insightface", "legacy"]
    cfgs: List[IndexCfg] = []
    for du, mfa, ap in itertools.product(det, min_area, arc_pre):
        cfgs.append(IndexCfg(det_upsample=du, min_face_area=mfa, arcface_preproc=ap))
    uniq = []
    seen = set()
    for c in cfgs:
        if c.tag() not in seen:
            seen.add(c.tag())
            uniq.append(c)
    return uniq


def ann_knob_grid(quick: bool) -> Tuple[List[AnnBuildCfg], List[AnnSearchCfg]]:
    builds = [AnnBuildCfg(hnsw_m=32, ef_construction=200)]
    if not quick:
        builds.append(AnnBuildCfg(hnsw_m=16, ef_construction=100))

    if quick:
        searches = [AnnSearchCfg(ann_k=200, ef_search=64, rerank="on")]
    else:
        searches = [
            AnnSearchCfg(ann_k=200, ef_search=64, rerank="on"),
            AnnSearchCfg(ann_k=500, ef_search=64, rerank="on"),
            AnnSearchCfg(ann_k=200, ef_search=128, rerank="on"),
            AnnSearchCfg(ann_k=500, ef_search=128, rerank="on"),
        ]
    return builds, searches


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--out-root", required=True, type=str)
    ap.add_argument("--models", nargs="+", default=["dlib_resnet_v1", "arcface_onnx", "opencv_sface", "mobilefacenet_onnx"])
    ap.add_argument("--phases", nargs="+", default=["baseline", "index_knobs", "ann_knobs"],
                    choices=["baseline", "index_knobs", "ann_knobs"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_root = Path(args.out_root)
    safe_mkdir(out_root)

    rows: List[ResultRow] = []

    if "baseline" in args.phases:
        print("\n=== PHASE: baseline ===\n")
        idx = baseline_index_cfg()
        ann_builds, ann_searches = ann_knob_grid(quick=True)
        annb = ann_builds[0]
        anns = ann_searches[0]

        for mi, model in enumerate(args.models, start=1):
            run_dir = out_root / "baseline" / model / idx.tag()

            print(f"[{mi}/{len(args.models)}] MODEL: {model}")
            t_index = run_index(dataset, model, run_dir, idx, force=args.force)
            t_eval_b = run_eval_bruteforce(run_dir, args.top_k)
            t_build, ann_archive = run_build_ann_inplace(run_dir, annb, force=args.force)
            t_eval_a = run_eval_ann(run_dir, ann_archive, anns, args.top_k)

            brut = maybe_read_metrics(run_dir, ann=False)
            annm = maybe_read_metrics(ann_archive.parent, ann=True)

            rows.append(
                ResultRow(
                    phase="baseline",
                    model=model,
                    run_dir=str(run_dir),
                    index_cfg=asdict(idx),
                    ann_build_cfg=asdict(annb),
                    ann_search_cfg=asdict(anns),
                    t_index_s=t_index,
                    t_eval_brut_s=t_eval_b,
                    t_build_ann_s=t_build,
                    t_eval_ann_s=t_eval_a,
                    brut_rank1=brut.get("rank1"),
                    brut_recall_at_5=brut.get("recall_at_5"),
                    brut_recall_at_10=brut.get("recall_at_10"),
                    brut_mrr=brut.get("mrr"),
                    brut_n_queries=brut.get("n_queries"),
                    ann_rank1=annm.get("rank1"),
                    ann_recall_at_5=annm.get("recall_at_5"),
                    ann_recall_at_10=annm.get("recall_at_10"),
                    ann_mrr=annm.get("mrr"),
                    ann_n_queries=annm.get("n_queries"),
                )
            )
            print("")

    if "index_knobs" in args.phases:
        print("\n=== PHASE: index_knobs (rebuilds embeddings) ===\n")
        cfgs = index_knob_grid(quick=args.quick)

        for mi, model in enumerate(args.models, start=1):
            print(f"[{mi}/{len(args.models)}] MODEL: {model}")
            print(f"  Index configs: {len(cfgs)}\n")

            for ci, idx in enumerate(cfgs, start=1):
                run_dir = out_root / "index_knobs" / model / idx.tag()
                print(f"  ({ci}/{len(cfgs)}) index_cfg: {idx.tag()}")

                t_index = run_index(dataset, model, run_dir, idx, force=args.force)
                t_eval_b = run_eval_bruteforce(run_dir, args.top_k)

                brut = maybe_read_metrics(run_dir, ann=False)

                rows.append(
                    ResultRow(
                        phase="index_knobs",
                        model=model,
                        run_dir=str(run_dir),
                        index_cfg=asdict(idx),
                        ann_build_cfg=None,
                        ann_search_cfg=None,
                        t_index_s=t_index,
                        t_eval_brut_s=t_eval_b,
                        t_build_ann_s=0.0,
                        t_eval_ann_s=0.0,
                        brut_rank1=brut.get("rank1"),
                        brut_recall_at_5=brut.get("recall_at_5"),
                        brut_recall_at_10=brut.get("recall_at_10"),
                        brut_mrr=brut.get("mrr"),
                        brut_n_queries=brut.get("n_queries"),
                    )
                )
                print("")

            print("")

    if "ann_knobs" in args.phases:
        print("\n=== PHASE: ann_knobs (keeps embeddings fixed) ===\n")
        idx = baseline_index_cfg()
        ann_builds, ann_searches = ann_knob_grid(quick=args.quick)

        for mi, model in enumerate(args.models, start=1):
            run_dir = out_root / "ann_knobs" / model / idx.tag()

            print(f"[{mi}/{len(args.models)}] MODEL: {model}")
            t_index = run_index(dataset, model, run_dir, idx, force=args.force)
            t_eval_b = run_eval_bruteforce(run_dir, args.top_k)
            brut = maybe_read_metrics(run_dir, ann=False)

            for bi, annb in enumerate(ann_builds, start=1):
                print(f"  ANN build ({bi}/{len(ann_builds)}): {annb.tag()}")
                t_build, ann_archive = run_build_ann_inplace(run_dir, annb, force=args.force)

                for si, anns in enumerate(ann_searches, start=1):
                    print(f"    ANN eval ({si}/{len(ann_searches)}): {anns.tag()}")
                    t_eval_a = run_eval_ann(run_dir, ann_archive, anns, args.top_k)
                    annm = maybe_read_metrics(ann_archive.parent, ann=True)

                    rows.append(
                        ResultRow(
                            phase="ann_knobs",
                            model=model,
                            run_dir=str(run_dir),
                            index_cfg=asdict(idx),
                            ann_build_cfg=asdict(annb),
                            ann_search_cfg=asdict(anns),
                            t_index_s=t_index if (bi == 1 and si == 1) else 0.0,
                            t_eval_brut_s=t_eval_b if (bi == 1 and si == 1) else 0.0,
                            t_build_ann_s=t_build if si == 1 else 0.0,
                            t_eval_ann_s=t_eval_a,
                            brut_rank1=brut.get("rank1"),
                            brut_recall_at_5=brut.get("recall_at_5"),
                            brut_recall_at_10=brut.get("recall_at_10"),
                            brut_mrr=brut.get("mrr"),
                            brut_n_queries=brut.get("n_queries"),
                            ann_rank1=annm.get("rank1"),
                            ann_recall_at_5=annm.get("recall_at_5"),
                            ann_recall_at_10=annm.get("recall_at_10"),
                            ann_mrr=annm.get("mrr"),
                            ann_n_queries=annm.get("n_queries"),
                        )
                    )
                print("")
            print("")

    if rows:
        csv_path = out_root / "sweep_results.csv"
        jsonl_path = out_root / "sweep_results.jsonl"
        md_path = out_root / "sweep_summary.md"

        write_csv(csv_path, rows)
        write_jsonl(jsonl_path, rows)
        write_summary_md(md_path, rows)

        print("\n=== DONE ===")
        print(f"Saved CSV:   {csv_path}")
        print(f"Saved JSONL: {jsonl_path}")
        print(f"Saved MD:    {md_path}")
    else:
        print("No runs executed (rows empty).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
