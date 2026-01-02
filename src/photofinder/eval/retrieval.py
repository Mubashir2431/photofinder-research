from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class RetrievalMetrics:
    rank1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    n_queries: int


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _group_by_label(meta: Sequence[Any]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        if isinstance(m, dict) and "label" in m:
            groups[str(m["label"])].append(i)
    return groups


def eval_retrieval(
    emb: np.ndarray,
    meta: Sequence[Any],
    *,
    top_k: int = 10,
    metric: str = "cosine",                 # cosine | l2
    ann_index: Optional[object] = None,     # a loaded FAISS index
    ann_search_k: int = 200,                # ANN candidate depth
    rerank: bool = True,                    # rerank ANN candidates with exact metric
) -> RetrievalMetrics:
    """
    Evaluate retrieval (rank1/recall@k/MRR) on identities with >=2 images.
    """
    metric = metric.lower()
    if metric not in ("cosine", "l2"):
        raise ValueError("metric must be 'cosine' or 'l2'")

    E_raw = np.asarray(emb, dtype=np.float32)
    groups = _group_by_label(meta)

    query_indices: List[int] = []
    positives_by_q: Dict[int, set[int]] = {}

    for _, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for q in idxs:
            query_indices.append(q)
            positives_by_q[q] = set(idxs) - {q}

    if not query_indices:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0)

    use_ann = ann_index is not None
    if use_ann:
        from photofinder.indexing.ann_faiss import search_index as faiss_search

    # Prepare embeddings for exact scoring
    if metric == "cosine":
        E = _l2_normalize_rows(E_raw)
    else:
        E = E_raw

    hits_rank1 = hits_at5 = hits_at10 = 0
    rr_sum = 0.0

    k_fetch = max(int(ann_search_k), int(top_k) + 50)

    for q_idx in query_indices:
        pos = positives_by_q[q_idx]
        q = E[q_idx]

        if use_ann:
            _d, idxs = faiss_search(ann_index, q, k_fetch)
            cand = [int(i) for i in idxs.tolist() if int(i) >= 0 and int(i) != q_idx]

            if rerank and cand:
                cand_arr = np.asarray(cand, dtype=np.int64)
                if metric == "cosine":
                    scores = (E[cand_arr] @ q).astype(np.float32)
                    order = np.argsort(-scores)
                    cand = cand_arr[order].tolist()
                else:
                    diff = E[cand_arr] - q.reshape(1, -1)
                    dist2 = np.sum(diff * diff, axis=1).astype(np.float32)
                    order = np.argsort(dist2)
                    cand = cand_arr[order].tolist()

            idxs_ranked = cand
        else:
            if metric == "cosine":
                scores_all = E @ q
                idxs_ranked = [int(i) for i in np.argsort(-scores_all) if int(i) != q_idx]
            else:
                diff = E - q.reshape(1, -1)
                dist2 = np.sum(diff * diff, axis=1)
                idxs_ranked = [int(i) for i in np.argsort(dist2) if int(i) != q_idx]

        first_rank = None
        for r, i in enumerate(idxs_ranked[: max(k_fetch, top_k)], start=1):
            if i in pos:
                first_rank = r
                break

        if first_rank is not None:
            rr_sum += 1.0 / first_rank
            if first_rank == 1:
                hits_rank1 += 1
            if first_rank <= 5:
                hits_at5 += 1
            if first_rank <= 10:
                hits_at10 += 1

    n = len(query_indices)
    return RetrievalMetrics(
        rank1=hits_rank1 / n,
        recall_at_5=hits_at5 / n,
        recall_at_10=hits_at10 / n,
        mrr=rr_sum / n,
        n_queries=n,
    )
