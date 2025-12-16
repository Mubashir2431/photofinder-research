# Claims → Evidence (fill this before writing results)

## Claim A: Caching / pre-indexing reduces query latency
- Evidence: timing table (no-cache vs cached) + hardware + n queries
- Artifact: runs/<dataset>/<model>/timings.json, query_timings.csv

## Claim B: Model X outperforms Model Y on retrieval
- Evidence: Rank-1, Recall@k, MRR on public dataset(s) with fixed protocol
- Artifact: metrics_retrieval.json + seed + dataset card

## Claim C: ANN indexing achieves sublinear query time without degrading recall
- Evidence: latency vs N plots (linear vs HNSW/FAISS) + Recall@k differences
- Artifact: results/leaderboard.csv + plots
