"""Experiment config parsing and sweep-grid construction."""
from __future__ import annotations

import json

import pytest
import yaml

from conftest import REPO_ROOT, load_script


def test_example_experiment_config_parses():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "experiments" / "lfw_dummy.yaml").read_text())
    assert set(cfg) == {"dataset_root", "model", "out_dir"}
    assert cfg["model"] == "dummy"


@pytest.fixture(scope="module")
def sweep():
    return load_script("photofinder_full_sweep_v8")


def test_full_ann_grid_is_108_per_model(sweep):
    grid = sweep.make_ann_knob_grid(fast=False)
    # 2 M x 3 efConstruction x 3 ann_k x 3 efSearch x 2 rerank
    assert len(grid) == 108
    assert len({c.tag() for c in grid}) == 108  # tags are unique
    assert {c.hnsw_m for c in grid} == {16, 32}
    assert {c.rerank for c in grid} == {"on", "off"}


def test_fast_ann_grid(sweep):
    assert len(sweep.make_ann_knob_grid(fast=True)) == 4


def test_baseline_ann_cfg_is_in_full_grid(sweep):
    assert sweep.build_baseline_ann_cfg() in sweep.make_ann_knob_grid(fast=False)


def test_index_knob_grid_sizes(sweep):
    assert len(sweep.make_index_knob_grid("arcface_onnx", fast=False)) == 3 * 2 * 3
    assert len(sweep.make_index_knob_grid("opencv_sface", fast=False)) == 3
    assert len(sweep.make_index_knob_grid("opencv_sface", fast=True)) == 1


def test_tags_are_stable_and_short(sweep):
    a = sweep.IndexCfg()
    b = sweep.IndexCfg()
    assert a.tag() == b.tag()
    assert a.tag().startswith("idx_") and len(a.tag()) == 16
    assert sweep.IndexCfg(det_upsample=2).tag() != a.tag()
    assert sweep.stable_hash({"x": 1, "y": 2}) == sweep.stable_hash({"y": 2, "x": 1})


def test_cli_arg_rendering(sweep):
    idx = sweep.IndexCfg(det_upsample=0)
    args = idx.index_cli_args("dummy")
    assert args[args.index("--det-upsample") + 1] == "0"
    ann = sweep.AnnCfg(hnsw_m=16, ef_search=256, rerank="off")
    assert ann.build_cli_args()[:2] == ["--ann-type", "hnsw"]
    assert "--faiss-metric" not in ann.build_cli_args()
    assert "--faiss-metric" in sweep.AnnCfg(faiss_metric="l2").build_cli_args()
    ev = ann.eval_cli_args(top_k=10)
    assert ev[ev.index("--ef-search") + 1] == "256"
    assert ev[ev.index("--rerank") + 1] == "off"
    assert json.loads(json.dumps(ann.to_json()))["hnsw_m"] == 16


def test_fmt_f_handles_missing(sweep):
    assert sweep.fmt_f(None) == ""
    assert sweep.fmt_f("0.5") == "0.5000"
    assert sweep.fmt_f("n/a") == "n/a"


def test_csv_logger_roundtrip(sweep, tmp_path):
    p = tmp_path / "s.csv"
    log = sweep.CsvLogger(p)
    log.write_row({"model": "dummy", "ann_rank1": 0.9, "unknown_key": "dropped"})
    log.close()
    rows = sweep.read_rows_from_csv(p)
    assert rows[0]["model"] == "dummy"
    assert rows[0]["ann_rank1"] == "0.9"
    assert "unknown_key" not in rows[0]
