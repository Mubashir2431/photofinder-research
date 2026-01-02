#!/usr/bin/env python3
"""
make_sweep_summary_v2.py

Robust summary builder for PhotoFinder sweeps.

Supports layouts like:
  runs/<phase>/<model>/<run_tag>/index.npz
  runs/<phase>/<model>/<run_tag>/metrics_retrieval_bruteforce.json
  runs/<phase>/<model>/<run_tag>/_ann/<ann_tag>/metrics_retrieval_ann.json

Where <run_tag> can be idx_<hash> or a long config folder name,
and <ann_tag> can be ann_<hash> or ann_hnsw_...
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # If a file somehow contains python-dict-like text, try a soft fallback
        try:
            txt = path.read_text(encoding="utf-8").strip()
            # Last resort: convert single quotes to double quotes (best-effort)
            txt2 = txt.replace("'", '"')
            return json.loads(txt2)
        except Exception:
            return None


def fmt_float(x: Any, nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def safe_replace(tmp: Path, final: Path) -> Path:
    """
    Replace final with tmp. If final is locked (PermissionError),
    write to an alternate timestamped filename instead.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.replace(final)
        return final
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = final.with_name(f"{final.stem}_{ts}{final.suffix}")
        tmp.replace(alt)
        print(f"WARNING: Could not overwrite locked file:\n  {final}\nWrote instead:\n  {alt}")
        return alt


def find_index_run_dirs(run_root: Path) -> List[Path]:
    """
    Index runs are folders that contain index.npz but are NOT inside _ann.
    """
    out = []
    for p in run_root.rglob("index.npz"):
        if "_ann" in p.parts:
            continue
        out.append(p.parent)
    # de-dupe & stable sort
    out = sorted(set(out))
    return out


def phase_model_from_rel(run_root: Path, run_dir: Path) -> Tuple[str, str, str]:
    """
    Try to infer phase/model/run_tag from run_dir relative path.
    Expected: <phase>/<model>/<run_tag>/...
    """
    rel = run_dir.resolve().relative_to(run_root.resolve())
    parts = rel.parts
    phase = parts[0] if len(parts) >= 1 else ""
    model = parts[1] if len(parts) >= 2 else ""
    run_tag = "/".join(parts[2:]) if len(parts) >= 3 else run_dir.name
    return phase, model, run_tag


def flatten_cfg(cfg: Optional[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    if not cfg:
        return {f"{prefix}_{k}": "" for k in []}
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        out[f"{prefix}_{k}"] = v
    return out


def pick_index_cfg(run_dir: Path) -> Dict[str, Any]:
    # prefer index_cfg.json if present, else config.json, else runinfo.json
    for name in ("index_cfg.json", "config.json", "runinfo.json"):
        cfg = read_json(run_dir / name)
        if cfg:
            return cfg
    return {}


def pick_ann_cfg(ann_dir: Path) -> Dict[str, Any]:
    for name in ("ann_cfg.json", "config.json", "runinfo.json"):
        cfg = read_json(ann_dir / name)
        if cfg:
            return cfg
    return {}


def read_metrics(run_dir: Path, which: str) -> Dict[str, Any]:
    # which = "bruteforce" or "ann"
    p = run_dir / f"metrics_retrieval_{which}.json"
    m = read_json(p)
    return m or {}


def discover_ann_dirs(run_dir: Path) -> List[Path]:
    ann_root = run_dir / "_ann"
    if not ann_root.exists():
        return []
    return sorted([p for p in ann_root.iterdir() if p.is_dir()])


def row_from(run_root: Path, run_dir: Path, ann_dir: Optional[Path]) -> Dict[str, Any]:
    phase, model, run_tag = phase_model_from_rel(run_root, run_dir)

    idx_cfg = pick_index_cfg(run_dir)
    bf = read_metrics(run_dir, "bruteforce")

    ann_tag = ""
    ann_cfg: Dict[str, Any] = {}
    annm: Dict[str, Any] = {}

    if ann_dir is not None:
        ann_tag = ann_dir.name
        ann_cfg = pick_ann_cfg(ann_dir)
        annm = read_metrics(ann_dir, "ann")

    # pull common metric keys safely
    def pull(m: Dict[str, Any], key: str) -> Any:
        return m.get(key, None)

    # store a compact error if present
    err_txt = ""
    for cand in (run_dir / "error.txt", run_dir / "ERROR.txt"):
        if cand.exists():
            err_txt = cand.read_text(encoding="utf-8", errors="ignore")[:500].strip()
            break
    if not err_txt and ann_dir is not None:
        for cand in (ann_dir / "error.txt", ann_dir / "ERROR.txt"):
            if cand.exists():
                err_txt = cand.read_text(encoding="utf-8", errors="ignore")[:500].strip()
                break

    return {
        "phase": phase,
        "model": model,
        "run_tag": run_tag,
        "ann_tag": ann_tag,
        # bruteforce metrics
        "bf_rank1": pull(bf, "rank1"),
        "bf_recall_at_5": pull(bf, "recall_at_5"),
        "bf_recall_at_10": pull(bf, "recall_at_10"),
        "bf_mrr": pull(bf, "mrr"),
        "bf_n_queries": pull(bf, "n_queries"),
        # ann metrics
        "ann_rank1": pull(annm, "rank1"),
        "ann_recall_at_5": pull(annm, "recall_at_5"),
        "ann_recall_at_10": pull(annm, "recall_at_10"),
        "ann_mrr": pull(annm, "mrr"),
        "ann_n_queries": pull(annm, "n_queries"),
        # include config blobs (kept as JSON text so you don’t lose knobs)
        "index_cfg_json": json.dumps(idx_cfg, ensure_ascii=False),
        "ann_cfg_json": json.dumps(ann_cfg, ensure_ascii=False) if ann_dir else "",
        "error": err_txt,
        "run_dir": str(run_dir),
        "ann_dir": str(ann_dir) if ann_dir else "",
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    # stable column order
    fieldnames = [
        "phase", "model", "run_tag", "ann_tag",
        "bf_rank1", "bf_recall_at_5", "bf_recall_at_10", "bf_mrr", "bf_n_queries",
        "ann_rank1", "ann_recall_at_5", "ann_recall_at_10", "ann_mrr", "ann_n_queries",
        "index_cfg_json", "ann_cfg_json",
        "error", "run_dir", "ann_dir",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return safe_replace(tmp, path)


def write_md(path: Path, rows: List[Dict[str, Any]], run_root: Path) -> Path:
    # best per model by ann_rank1 (ignore missing)
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        m = r["model"]
        v = r.get("ann_rank1", None)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if m not in best or float(best[m]["ann_rank1"]) < fv:
            best[m] = r

    lines: List[str] = []
    lines.append("# PhotoFinder Sweep Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Rows: {len(rows)}")
    lines.append("")
    lines.append("## Best run per model (by ANN Rank-1)")
    lines.append("")
    lines.append("| Model | Phase | ANN Rank1 | ANN Recall@10 | ANN MRR | Run tag | ANN tag |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for model in sorted(set(r["model"] for r in rows)):
        r = best.get(model)
        if not r:
            lines.append(f"| {model} |  |  |  |  |  |  |")
        else:
            lines.append(
                f"| {model} | {r['phase']} | {fmt_float(r.get('ann_rank1'))} | "
                f"{fmt_float(r.get('ann_recall_at_10'))} | {fmt_float(r.get('ann_mrr'))} | "
                f"{r['run_tag']} | {r['ann_tag']} |"
            )

    lines.append("")
    lines.append("## All runs")
    lines.append("")
    lines.append("| Phase | Model | BF Rank1 | ANN Rank1 | ANN Recall@10 | Run tag | ANN tag | Error |")
    lines.append("|---|---|---:|---:|---:|---|---|---|")
    for r in rows:
        err = (r.get("error") or "")
        if len(err) > 60:
            err = err[:57] + "..."
        lines.append(
            f"| {r['phase']} | {r['model']} | {fmt_float(r.get('bf_rank1'))} | "
            f"{fmt_float(r.get('ann_rank1'))} | {fmt_float(r.get('ann_recall_at_10'))} | "
            f"{r['run_tag']} | {r['ann_tag']} | {err} |"
        )

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return safe_replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="Sweep run root, e.g. runs\\sweeps\\lfw")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--out-md", required=True, help="Output MD path")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    idx_dirs = find_index_run_dirs(run_root)

    rows: List[Dict[str, Any]] = []
    for run_dir in idx_dirs:
        ann_dirs = discover_ann_dirs(run_dir)
        if not ann_dirs:
            rows.append(row_from(run_root, run_dir, None))
        else:
            for ann_dir in ann_dirs:
                rows.append(row_from(run_root, run_dir, ann_dir))

    rows = sorted(rows, key=lambda r: (r["phase"], r["model"], r["run_tag"], r["ann_tag"]))

    csv_path = write_csv(out_csv, rows)
    md_path = write_md(out_md, rows, run_root)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote MD : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
