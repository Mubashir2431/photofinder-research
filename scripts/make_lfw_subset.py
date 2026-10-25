"""Create a deterministic LFW subset in the imagefolder layout (root/<person>/<img>.jpg).

Identities are visited in sorted order; for each identity the first
``--per-person`` images (sorted by filename) are copied. Identities with fewer
than ``--min-per-person`` images are skipped, so every kept identity can act as
a retrieval query.

Example (all 1,680 LFW identities with >=2 images, capped at 3 images each -> 4,261 images;
this is the protocol of results/model_comparison.csv):
    python scripts/make_lfw_subset.py --src <lfw_funneled> --dst <out> --people 0 --per-person 3
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from typing import List, Optional


def make_subset(src: str, dst: str, people: int = 100, per_person: int = 3, min_per_person: int = 2) -> dict:
    os.makedirs(dst, exist_ok=True)
    names = [p for p in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, p))]

    copied = 0
    kept_people = 0
    for person in names:
        if people > 0 and kept_people >= people:
            break
        imgs = sorted(glob.glob(os.path.join(src, person, "*.jpg")))
        if len(imgs) < min_per_person:
            continue
        out_dir = os.path.join(dst, person)
        os.makedirs(out_dir, exist_ok=True)
        for img in imgs[:per_person]:
            shutil.copy2(img, os.path.join(out_dir, os.path.basename(img)))
            copied += 1
        kept_people += 1

    return {"people": kept_people, "images": copied, "dst": dst}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create a deterministic LFW subset (imagefolder layout).")
    ap.add_argument("--src", default=os.path.join("data", "lfw", "lfw_funneled"))
    ap.add_argument("--dst", default=os.path.join("data", "lfw_small", "images"))
    ap.add_argument("--people", type=int, default=100, help="Max identities to keep (0 = all).")
    ap.add_argument("--per-person", type=int, default=3, help="Max images per identity.")
    ap.add_argument("--min-per-person", type=int, default=2, help="Skip identities with fewer images.")
    args = ap.parse_args(argv)

    info = make_subset(args.src, args.dst, args.people, args.per_person, args.min_per_person)
    print("People kept:", info["people"])
    print("Images copied:", info["images"])
    print("Out:", info["dst"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
