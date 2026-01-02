#!/usr/bin/env python3
"""More robust sweep aggregator that fully flattens nested JSON (including timings).

Produces `runs/sweeps_summary_full.csv` by default.
"""
import argparse
import csv
import json
from pathlib import Path


TARGET_JSON_NAMES = ('metrics_retrieval.json', 'runinfo.json', 'config.json', 'timings.json')


def load_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def flatten(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            flatten(key, v, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value


def gather_rows(sweeps_dir: Path):
    rows = []
    keys = set()
    sweeps_dir = Path(sweeps_dir)
    if not sweeps_dir.exists():
        return rows, []

    for sweep in sorted(sweeps_dir.iterdir()):
        if not sweep.is_dir():
            continue
        sweep_name = sweep.name
        for run_type in ('baseline', 'ann_knobs'):
            d = sweep / run_type
            if not d.exists():
                continue
            for j in d.rglob('*.json'):
                name = j.name
                if not (name.startswith('metrics_') or name in TARGET_JSON_NAMES):
                    continue

                try:
                    rel = j.parent.relative_to(d)
                    rel_parts = rel.parts
                except Exception:
                    rel_parts = []

                model = rel_parts[0] if len(rel_parts) > 0 else ''
                variant = '/'.join(rel_parts) if len(rel_parts) > 0 else ''

                data = load_json(j) or {}
                row = {
                    'sweep': sweep_name,
                    'run_type': run_type,
                    'model': model,
                    'variant': variant,
                    'run_path': str(j.parent),
                    'json_file': name,
                }

                if isinstance(data, dict):
                    flat = {}
                    flatten('', data, flat)
                    for k, v in flat.items():
                        col = f"{name.replace('.json','')}.{k}"
                        row[col] = v
                        keys.add(col)

                rows.append(row)

    return rows, sorted(keys)


def write_csv(rows, keys, out_path: Path):
    fixed = ['sweep', 'run_type', 'model', 'variant', 'run_path', 'json_file']
    header = fixed + keys
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in header})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweeps-dir', default='runs/sweeps')
    ap.add_argument('--out', default='runs/sweeps_summary_full.csv')
    args = ap.parse_args()
    rows, keys = gather_rows(Path(args.sweeps_dir))
    write_csv(rows, keys, Path(args.out))
    print(f'Wrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()
