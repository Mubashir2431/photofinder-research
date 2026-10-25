"""Index build/search parity: exact (numpy / FAISS flat) vs FAISS HNSW on small random vectors."""
from __future__ import annotations

import json

import numpy as np
import pytest

from photofinder.eval.retrieval import eval_retrieval
from photofinder.indexing.linear import l2_search
from photofinder.indexing.store import load_index_npz, save_index_npz

faiss = pytest.importorskip("faiss")

from photofinder.indexing import ann_faiss  # noqa: E402


def _exact_topk_ip(E: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(Q @ E.T), axis=1)[:, :k]


@pytest.fixture
def random_unit_vectors():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(1000, 32)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    Q = rng.normal(size=(50, 32)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return E, Q


def test_l2_search_matches_argsort():
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(200, 16)).astype(np.float32)
    q = rng.normal(size=16).astype(np.float32)
    idx, d = l2_search(q, mat, top_k=10)
    ref = np.argsort(np.linalg.norm(mat - q, axis=1))[:10]
    assert idx.tolist() == ref.tolist()
    assert np.all(np.diff(d) >= 0)


def test_flat_index_equals_exact(tmp_path, random_unit_vectors):
    E, Q = random_unit_vectors
    path = ann_faiss.save_index(E, str(tmp_path / "flat.faiss"), ann_type="flat", metric="ip")
    index = ann_faiss.load_index(path)
    for qi in range(len(Q)):
        _, ids = ann_faiss.search_index(index, Q[qi], 10)
        assert ids.tolist() == _exact_topk_ip(E, Q[qi : qi + 1], 10)[0].tolist()


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_hnsw_recall_vs_exact(tmp_path, random_unit_vectors, metric):
    E, Q = random_unit_vectors
    path = ann_faiss.save_index(
        E, str(tmp_path / f"hnsw_{metric}.faiss"), ann_type="hnsw", metric=metric, hnsw_m=16, ef_construction=200
    )
    index = ann_faiss.load_index(path, ef_search=256)
    assert index.hnsw.efSearch == 256
    k = 10
    exact = _exact_topk_ip(E, Q, k)  # on unit vectors, IP order == L2 order
    hits = 0
    for qi in range(len(Q)):
        _, ids = ann_faiss.search_index(index, Q[qi], k)
        hits += len(set(ids.tolist()) & set(exact[qi].tolist()))
    recall = hits / (len(Q) * k)
    assert recall >= 0.95, recall


def test_hnsw_ip_normalizes_inputs(tmp_path, random_unit_vectors):
    E, Q = random_unit_vectors
    scaled = E * 7.5  # cosine must not depend on vector norms
    path = ann_faiss.save_index(scaled, str(tmp_path / "x.faiss"), ann_type="flat", metric="ip", normalize=True)
    index = ann_faiss.load_index(path)
    _, ids = ann_faiss.search_index(index, Q[0], 5)
    assert ids.tolist() == _exact_topk_ip(E, Q[:1], 5)[0].tolist()


def test_ann_eval_with_rerank_matches_bruteforce(tmp_path, clustered_vectors):
    emb, meta = clustered_vectors
    path = ann_faiss.save_index(emb, str(tmp_path / "idx.faiss"), ann_type="hnsw", metric="ip", hnsw_m=16)
    index = ann_faiss.load_index(path, ef_search=128)
    bf = eval_retrieval(emb, meta, top_k=10, metric="cosine")
    ann = eval_retrieval(emb, meta, top_k=10, metric="cosine", ann_index=index, ann_search_k=len(emb), rerank=True)
    assert ann == bf
    assert bf.rank1 == 1.0


def test_save_index_dir_and_suffix_handling(tmp_path, random_unit_vectors):
    E, _ = random_unit_vectors
    out_dir = tmp_path / "ann"
    out_dir.mkdir()
    p1 = ann_faiss.save_index(E, str(out_dir), ann_type="flat")
    assert p1.endswith("index.faiss")
    p2 = ann_faiss.save_index(E, str(tmp_path / "noext"), ann_type="flat")
    assert p2.endswith(".faiss")
    # load_index accepts a directory
    assert ann_faiss.load_index(str(out_dir)).ntotal == len(E)
    meta_path = ann_faiss.save_index_meta(p1, {"hnsw_m": 8})
    assert json.loads(open(meta_path).read()) == {"hnsw_m": 8}


def test_bad_ann_args(tmp_path, random_unit_vectors):
    E, _ = random_unit_vectors
    with pytest.raises(ValueError):
        ann_faiss.save_index(E, str(tmp_path / "a.faiss"), ann_type="ivf")
    with pytest.raises(ValueError):
        ann_faiss.save_index(E, str(tmp_path / "b.faiss"), ann_type="hnsw", metric="cos")
    with pytest.raises(FileNotFoundError):
        ann_faiss.load_index(str(tmp_path / "missing.faiss"))


def test_default_faiss_path_and_l2_to_cos():
    assert ann_faiss.default_faiss_path("runs/x/index.npz").replace("\\", "/") == "runs/x/index.faiss"
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    d2 = np.sum((a - b) ** 2)
    assert ann_faiss.l2dist2_to_cos(d2) == pytest.approx(float(a @ b))


def test_npz_store_roundtrip(tmp_path):
    emb = np.random.default_rng(3).normal(size=(5, 4)).astype(np.float64)
    meta = [{"label": "é", "path": f"{i}.jpg", "bbox": [0, 0, 1, 1]} for i in range(5)]
    p = str(tmp_path / "index.npz")
    save_index_npz(p, emb, meta)
    emb2, meta2 = load_index_npz(p)
    assert emb2.dtype == np.float32
    np.testing.assert_allclose(emb2, emb.astype(np.float32))
    assert meta2 == meta
