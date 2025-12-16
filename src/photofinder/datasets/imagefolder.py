from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

@dataclass
class Sample:
    path: str
    label: str  # person_id

def load_imagefolder(root: str) -> List[Sample]:
    """Expects root/<label>/<image>.*"""
    samples: List[Sample] = []
    for label in sorted(os.listdir(root)):
        p = os.path.join(root, label)
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            if fn.lower().endswith(IMG_EXT):
                samples.append(Sample(path=os.path.join(p, fn), label=label))
    if not samples:
        raise FileNotFoundError(f"No images found in imagefolder dataset at: {root}")
    return samples
