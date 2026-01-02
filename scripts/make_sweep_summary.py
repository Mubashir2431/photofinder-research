from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def fnum(x: Optional[float]) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: Any) -> str:
        if v is None or v == "":
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    # compact leaderboard-style table
    header = [
        "phase", "model", "index_cfg", "ann_cfg",
        "bf_rank1", "bf_recall@5", "bf_recall@10", "bf_mrr",
        "ann_rank1", "ann_recall@5", "ann_recall@10", "ann_mrr",
        "ann_k", "ef_search", "rerank"
    ]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(h)) for h in header) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="e.g. runs\\sweeps\\lfw")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--phases", nargs="*", default=["baseline", "ann_knobs"])
    args = ap.parse_args()

    run_root = Path(args.run_root)
    phases = args.phases

    rows: List[Dict[str, Any]] = []

    # Scan only baseline + ann_knobs to avoid picking up your extra 20 index.npz files
    for phase in phases:
        phase_dir = run_root / phase
        if not phase_dir.exists():
            continue

        for model_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
            model = model_dir.name

            for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                index_npz = run_dir / "index.npz"
                if not index_npz.exists():
                    continue

                index_cfg = run_dir.name

                # bruteforce metrics are stored in the RUN DIR (same place as index.npz)
                bf_path = run_dir / "metrics_retrieval_bruteforce.json"
                bf = read_json(bf_path) if bf_path.exists() else {}

                bf_rank1 = fnum(bf.get("rank1"))
                bf_r5 = fnum(bf.get("recall_at_5"))
                bf_r10 = fnum(bf.get("recall_at_10"))
                bf_mrr = fnum(bf.get("mrr"))

                # ANN metrics live in _ann/<ann_cfg>/metrics_retrieval_ann.json
                ann_root = run_dir / "_ann"
                if ann_root.exists():
                    ann_dirs = sorted([p for p in ann_root.iterdir() if p.is_dir()])
                else:
                    ann_dirs = []

                # If there are no ann dirs, still emit a row for bruteforce-only
                if not ann_dirs:
                    rows.append({
                        "phase": phase,
                        "model": model,
                        "index_cfg": index_cfg,
                        "ann_cfg": "",
                        "bf_rank1": bf_rank1,
                        "bf_recall_at_5": bf_r5,
                        "bf_recall_at_10": bf_r10,
                        "bf_mrr": bf_mrr,
                        "ann_rank1": None,
                        "ann_recall_at_5": None,
                        "ann_recall_at_10": None,
                        "ann_mrr": None,
                        "ann_k": None,
                        "ef_search": None,
                        "rerank": None,
                    })
                    continue

                for ann_dir in ann_dirs:
                    ann_cfg = ann_dir.name
                    ann_path = ann_dir / "metrics_retrieval_ann.json"
                    ann = read_json(ann_path) if ann_path.exists() else {}

                    rows.append({
                        "phase": phase,
                        "model": model,
                        "index_cfg": index_cfg,
                        "ann_cfg": ann_cfg,
                        "bf_rank1": bf_rank1,
                        "bf_recall_at_5": bf_r5,
                        "bf_recall_at_10": bf_r10,
                        "bf_mrr": bf_mrr,
                        "ann_rank1": fnum(ann.get("rank1")),
                        "ann_recall_at_5": fnum(ann.get("recall_at_5")),
                        "ann_recall_at_10": fnum(ann.get("recall_at_10")),
                        "ann_mrr": fnum(ann.get("mrr")),
                        "ann_k": ann.get("ann_k"),
                        "ef_search": ann.get("ef_search"),
                        "rerank": ann.get("rerank"),
                    })

    # Sort for readability
    rows.sort(key=lambda r: (r["phase"], r["model"], r["index_cfg"], r["ann_cfg"]))

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    fieldnames = [
        "phase", "model", "index_cfg", "ann_cfg",
        "bf_rank1", "bf_recall_at_5", "bf_recall_at_10", "bf_mrr",
        "ann_rank1", "ann_recall_at_5", "ann_recall_at_10", "ann_mrr",
        "ann_k", "ef_search", "rerank"
    ]
    write_csv(out_csv, rows, fieldnames)
    write_md(out_md, rows)

    print(f"✅ Wrote {len(rows)} rows")
    print(f"CSV: {out_csv}")
    print(f"MD : {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
