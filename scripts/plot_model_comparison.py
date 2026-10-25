"""Bar chart of results/model_comparison.csv: Rank-1 (left) and throughput on a log axis (right)."""
from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "dlib_resnet_v1": "dlib",
    "arcface_onnx": "ArcFace\n(R100)",
    "opencv_sface": "OpenCV\nSFace",
    "mobilefacenet_onnx": "MobileFaceNet",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/model_comparison.csv")
    ap.add_argument("--out", default="results/model_comparison.png")
    ap.add_argument("--title", default="LFW subset, CPU only")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, newline="")))
    names = [LABELS.get(r["model"], r["model"]) for r in rows]
    r1 = [100 * float(r["rank1"]) for r in rows]
    ips = [float(r["images_per_s"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    c = "#3b6ea5"
    b1 = ax1.bar(names, r1, color=c)
    ax1.set_ylim(min(r1) - 3, 100)
    ax1.set_ylabel("Rank-1 (%)")
    ax1.set_title("Rank-1 accuracy")
    ax1.bar_label(b1, fmt="%.2f", padding=2)
    b2 = ax2.bar(names, ips, color=c)
    ax2.set_yscale("log")
    ax2.set_ylabel("images / s (detect + embed, 1 CPU box)")
    ax2.set_title("Throughput (log scale)")
    ax2.bar_label(b2, fmt="%.1f", padding=2)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
