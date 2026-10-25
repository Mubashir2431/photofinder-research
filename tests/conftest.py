from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def load_script(name: str):
    """Import a file from scripts/ as a module (scripts/ is not a package)."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pf_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def clustered_vectors():
    """20 identities x 4 images, tight clusters in 32-D -> perfect retrieval expected."""
    rng = np.random.default_rng(42)
    n_id, per_id, dim = 20, 4, 32
    centers = rng.normal(size=(n_id, dim)).astype(np.float32)
    emb, meta = [], []
    for i in range(n_id):
        for j in range(per_id):
            emb.append(centers[i] + 0.01 * rng.normal(size=dim).astype(np.float32))
            meta.append({"label": f"id{i:02d}", "path": f"id{i:02d}/{j}.jpg"})
    return np.stack(emb).astype(np.float32), meta


@pytest.fixture
def tiny_imagefolder(tmp_path):
    """Two identities x 3 synthetic images (no faces needed: used with the dummy model)."""
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "images"
    for person, color in (("alice", 80), ("bob", 200)):
        (root / person).mkdir(parents=True)
        for i in range(3):
            img = np.full((64, 64, 3), color, dtype=np.uint8)
            img[:, : 10 * (i + 1)] = 30 * i
            assert cv2.imwrite(str(root / person / f"{i}.png"), img)
    return root
