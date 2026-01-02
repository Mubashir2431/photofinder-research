````md
# PhotoFinder Research Bench

A reproducible template for **offline face-based photo retrieval** experiments.

## What this repo gives you
- Multiple embedding models (plug-in style)
- Folder-based datasets + public benchmarks (e.g., LFW)
- Retrieval evaluation (Rank-1, Recall@K, MRR, etc.)
- ANN indexing (FAISS / HNSW) + rerank testing
- Sweep runner for **baseline + ANN knobs**
- Clean artifacts (JSON/CSV/MD) suitable for a research paper

> **Do not commit private face datasets.** Use public benchmarks (like LFW) or keep private datasets local.

---

## Repo structure
- `src/photofinder/` — core library + `photofinder` CLI
- `scripts/` — sweep runner + summary builder
- `runs/` — outputs (indexes, ANN files, metrics, summaries) **(do not commit)**
- `paper/` — claim → evidence table, outline, figures

---

## Quick start (CPU-only)
### 1) Create & activate venv

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install

```bash
pip install -U pip
pip install -e ".[base]"
```

Optional extras (only if your repo defines them):

```bash
pip install -e ".[dlib,onnx,hnsw]"
```

---

## Dataset format

### Simple folder dataset (recommended)

Expected layout:

```
data/<dataset_name>/images/<person_id>/<image_file>.jpg
```

Example:

```
data/my_event/images/abdul/IMG_001.jpg
data/my_event/images/abdul/IMG_002.jpg
data/my_event/images/hamza/IMG_010.jpg
```

### LFW funneled layout (benchmark)

If you use the standard LFW funneled structure:

```
data/lfw/lfw_funneled/<person_id>/<image_file>.jpg
```

---

## Core CLI (single run)

### 1) Build an embedding index

```bash
photofinder index --dataset data/lfw/lfw_funneled --model arcface_onnx --out runs/lfw_arcface
```

### 2) Evaluate retrieval (bruteforce)

```bash
photofinder eval-retrieval --index runs/lfw_arcface/index.npz --out runs/lfw_arcface --backend bruteforce --top-k 10
```

### 3) Build ANN (FAISS / HNSW)

```bash
photofinder build-ann --index runs/lfw_arcface/index.npz --out runs/lfw_arcface/_ann --ann-type hnsw --hnsw-m 32 --ef-construction 200
```

### 4) Evaluate retrieval (ANN)

```bash
photofinder eval-retrieval --index runs/lfw_arcface/index.npz --out runs/lfw_arcface/_ann --backend ann --top-k 10 --ann-k 500 --ef-search 128 --rerank on
```

---

## Sweeps (baseline + ANN knobs)

The sweep script automates:

* multiple models
* index build
* bruteforce retrieval eval
* ANN build + ANN eval across knob sets
* consolidated summary report (CSV + MD)

### Run a sweep (example)

```powershell
python scripts\photofinder_full_sweep_v8.py `
  --dataset data\lfw\lfw_funneled `
  --out-root runs\sweeps\lfw `
  --models arcface_onnx dlib_resnet_v1 opencv_sface mobilefacenet_onnx `
  --phases baseline ann_knobs `
  --top-k 10 `
  --fast `
  --continue-on-error
```

### Phases

* `baseline`
  Builds `index.npz` and runs bruteforce retrieval eval once per model.

* `ann_knobs`
  Builds FAISS ANN indexes and evaluates retrieval across ANN knob combinations (HNSW params, `ef_search`, `rerank`, etc.).

---

## Build a sweep summary (CSV + MD)

After runs exist under `runs/sweeps/<dataset>/...`:

```powershell
python scripts\make_sweep_summary.py `
  --run-root runs\sweeps\lfw `
  --out-csv runs\sweeps\lfw\summary_results.csv `
  --out-md runs\sweeps\lfw\summary_results.md
```

### Windows note (important)

If `summary_results.csv` is open in Excel, you may get:
`PermissionError: [Errno 13] Permission denied`

Fix: close the CSV and rerun.

---

## Outputs (what gets saved)

Each run folder typically contains:

* `index.npz` — embeddings + metadata
* `metrics_retrieval_bruteforce.json` — bruteforce retrieval metrics
* `_ann/.../index.faiss` — ANN index file
* `_ann/.../metrics_retrieval_ann.json` — ANN retrieval metrics

Sweep-level summaries:

* `summary_results.csv` — all runs in one table
* `summary_results.md` — human-readable report

---

## Git hygiene

### Recommended `.gitignore`

```gitignore
runs/
*.npz
*.faiss
*.onnx
```

---

## License

MIT (template). Replace if needed.

```
::contentReference[oaicite:0]{index=0}
```

