# PhotoFinder Research Bench (template)

A reproducible repo template for **offline face-based photo retrieval** experiments:
- Multiple embedding models (plug-in architecture)
- Multiple datasets (folder-based adapters)
- Retrieval + verification evaluation
- Speed/memory measurements
- Clean experiment artifacts (CSV/JSON) suitable for a paper

> This is a template. You should **not** commit private face datasets. Use public benchmarks (e.g., LFW) or keep private datasets local.

## Quick start (CPU-only baseline)
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -e ".[base]"
```

### Dataset format (simple)
This template expects:
```
data/<dataset_name>/images/<person_id>/<image_file>.jpg
```

### Run an experiment
1) Index a dataset:
```bash
photofinder index --dataset data/lfw/images --model dummy --out runs/lfw_dummy
```

2) Evaluate retrieval:
```bash
photofinder eval-retrieval --index runs/lfw_dummy/index.npz --out runs/lfw_dummy
```

## Add real models
- Dlib embedding wrapper is included, but you must supply paths to the dlib model files via env vars:
  - `DLIB_SHAPE_PREDICTOR_PATH`
  - `DLIB_FACE_REC_MODEL_PATH`

Install extras:
```bash
pip install -e ".[dlib,hnsw]"
```

## Repo organization
- `src/photofinder/` core library
- `configs/experiments/` YAML experiment configs (model+dataset+indexer)
- `runs/<dataset>/<model>/` (your outputs; keep large artifacts out of git)
- `paper/` claim→evidence table + outline

## License
MIT (template). Replace if needed.
