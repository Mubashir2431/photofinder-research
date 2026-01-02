#!/usr/bin/env python3
"""Aggregate run result JSON files under `runs/` into a single CSV.

Usage:
  python scripts/aggregate_runs.py --runs-dir runs --out runs/summary_results.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path


def load_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def gather_rows(runs_dir: Path):
    rows = []
    all_keys = set()

    for p in runs_dir.rglob('*.json'):
        name = p.name
        if name.startswith('metrics_') or name in ('metrics_retrieval.json', 'runinfo.json', 'config.json', 'timings.json'):
            rel = p.relative_to(runs_dir)
            parts = rel.parts
            # try to extract category/dataset/model from path parts
            category = parts[0] if len(parts) > 0 else ''
            dataset = parts[1] if len(parts) > 1 else ''
            model = parts[2] if len(parts) > 2 else ''

            data = load_json(p) or {}

            row = {
                'run_path': str(p.parent),
                'file': name,
                'category': category,
                'dataset': dataset,
                'model': model,
            }

            # flatten top-level JSON keys into row (prefix keys with file type to avoid collisions)
            prefix = name.replace('.json', '')
            if isinstance(data, dict):
                for k, v in data.items():
                    key = f'{prefix}.{k}'
                    # convert non-primitive to JSON string
                    if isinstance(v, (dict, list)):
                        row[key] = json.dumps(v, ensure_ascii=False)
                    else:
                        row[key] = v
                    all_keys.add(key)

            rows.append(row)

    return rows, sorted(all_keys)


def write_csv(rows, keys, out_path: Path):
    fixed = ['run_path', 'file', 'category', 'dataset', 'model']
    header = fixed + keys
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            # ensure all header keys exist
            row = {k: r.get(k, '') for k in header}
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default='runs')
    ap.add_argument('--out', default='runs/summary_results.csv')
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out = Path(args.out)

    rows, keys = gather_rows(runs_dir)
    write_csv(rows, keys, out)
    print(f'Wrote {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()
