from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from photofinder.indexing.linear import l2_search

@dataclass
class RetrievalMetrics:
    rank1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    n_queries: int

def eval_retrieval(emb: np.ndarray, meta: List[Dict], top_k: int = 10) -> RetrievalMetrics:
    # Build label -> indices
    label_to_idx = defaultdict(list)
    for i, m in enumerate(meta):
        label_to_idx[m["label"]].append(i)

    # Only evaluate queries where at least 2 images exist for the identity
    valid_queries = [i for i, m in enumerate(meta) if len(label_to_idx[m["label"]]) >= 2]
    if not valid_queries:
        raise ValueError("No identities with >=2 images. Retrieval eval needs at least 2 per label.")

    r1 = r5 = r10 = 0
    mrr_sum = 0.0

    for qi in valid_queries:
        q = emb[qi]
        idx, _dist = l2_search(q, emb, top_k=top_k + 1)  # +1 to allow self-filter
        idx = [int(j) for j in idx if int(j) != qi][:top_k]

        true_label = meta[qi]["label"]
        hits = [k for k, j in enumerate(idx, start=1) if meta[j]["label"] == true_label]

        if hits:
            first = hits[0]
            mrr_sum += 1.0 / first
            if first == 1:
                r1 += 1
            if first <= 5:
                r5 += 1
            if first <= 10:
                r10 += 1

    n = len(valid_queries)
    return RetrievalMetrics(
        rank1=r1 / n,
        recall_at_5=r5 / n,
        recall_at_10=r10 / n,
        mrr=mrr_sum / n,
        n_queries=n,
    )
