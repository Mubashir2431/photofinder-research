"""Rank-1 / Recall@K / MRR on hand-made toy cases where the answer is known."""
from __future__ import annotations

import numpy as np
import pytest

from photofinder.eval.retrieval import RetrievalMetrics, eval_retrieval


def _m(label):
    return {"label": label}


def test_perfect_retrieval():
    # Two well-separated identities with two images each.
    emb = np.array([[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]], dtype=np.float32)
    meta = [_m("a"), _m("a"), _m("b"), _m("b")]
    m = eval_retrieval(emb, meta, top_k=10)
    assert m == RetrievalMetrics(rank1=1.0, recall_at_5=1.0, recall_at_10=1.0, mrr=1.0, n_queries=4)


def test_known_ranks_l2():
    # 1-D embeddings so the ranking is obvious under L2 distance.
    #   idx: 0    1    2    3    4    5
    #   pos: 0.0  1.0  0.4  0.5  0.6  0.7
    #   lbl: q    q    x    y    z    w
    # Query 0: neighbours by distance -> 2(0.4) 3 4 5 1  => first positive at rank 5
    # Query 1: neighbours by distance -> 5(0.3) 4 3 2 0  => first positive at rank 5
    emb = np.array([[0.0], [1.0], [0.4], [0.5], [0.6], [0.7]], dtype=np.float32)
    meta = [_m("q"), _m("q"), _m("x"), _m("y"), _m("z"), _m("w")]
    m = eval_retrieval(emb, meta, top_k=10, metric="l2")
    assert m.n_queries == 2  # singleton identities are gallery-only distractors
    assert m.rank1 == 0.0
    assert m.recall_at_5 == 1.0
    assert m.recall_at_10 == 1.0
    assert m.mrr == pytest.approx(1 / 5)


def test_mixed_ranks_cosine():
    # Query 0 (label a) finds its positive (1) at rank 1.
    # Query 1 (label a): nearest is 2 (label b), then 0 -> rank 2.
    # Queries 2,3 (label b): ranks 1 each (2<->3 nearly identical).
    emb = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.7, 0.71],
            [0.69, 0.72],
        ],
        dtype=np.float32,
    )
    meta = [_m("a"), _m("a"), _m("b"), _m("b")]
    m = eval_retrieval(emb, meta, top_k=10, metric="cosine")
    assert m.n_queries == 4
    # query 0: sims to 1=.8, 2=.70, 3=.69 -> positive first => rank 1
    # query 1: sims 0=.8, 2=.986, 3=.984 -> rank 3
    assert m.rank1 == pytest.approx(3 / 4)
    assert m.recall_at_5 == 1.0
    assert m.mrr == pytest.approx((1 + 1 / 3 + 1 + 1) / 4)


def test_recall_at_k_boundaries():
    # Query 0's positive is the 7th nearest -> counts for R@10 but not R@5.
    emb = np.array([[0.0]] + [[0.1 * i] for i in range(1, 7)] + [[0.75]], dtype=np.float32)
    meta = [_m("q")] + [_m(f"d{i}") for i in range(6)] + [_m("q")]
    m = eval_retrieval(emb, meta, top_k=10, metric="l2")
    # query 0: six distractors (0.1..0.6) come first, then 0.75 -> rank 7
    # query 7 (0.75): distances .15 .25 .35 .45 .55 .65 to distractors, .75 to q -> rank 7
    assert m.n_queries == 2
    assert m.rank1 == 0.0
    assert m.recall_at_5 == 0.0
    assert m.recall_at_10 == 1.0
    assert m.mrr == pytest.approx(1 / 7)


def test_no_queries_returns_zeros():
    emb = np.eye(3, dtype=np.float32)
    meta = [_m("a"), _m("b"), _m("c")]
    m = eval_retrieval(emb, meta)
    assert m == RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0)


def test_bad_metric_raises():
    with pytest.raises(ValueError):
        eval_retrieval(np.eye(2, dtype=np.float32), [_m("a"), _m("a")], metric="hamming")


def test_cosine_is_scale_invariant(clustered_vectors):
    emb, meta = clustered_vectors
    scaled = emb * np.linspace(0.1, 10, len(emb), dtype=np.float32)[:, None]
    a = eval_retrieval(emb, meta, metric="cosine")
    b = eval_retrieval(scaled, meta, metric="cosine")
    assert a == b
    assert a.rank1 == 1.0
