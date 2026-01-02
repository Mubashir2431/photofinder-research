#!/usr/bin/env python3
"""Aggregate sweep results (baseline and ann_knobs) into a single CSV.

This script scans `runs/sweeps/<sweep>/(baseline|ann_knobs)` and collects
JSON files including `metrics_*.json`, `metrics_retrieval.json`,
`runinfo.json`, `config.json`, and `timings.json`. Nested JSON structures are
recursively flattened so timing fields and other nested evaluations are preserved.

Usage:
  python scripts/aggregate_sweeps.py --sweeps-dir runs/sweeps --out runs/sweeps_summary.csv
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
    """Recursively flatten dicts; lists are JSON-encoded to preserve contents."""
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            flatten(key, v, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value


def gather_sweep_rows(sweeps_dir: Path):
    rows = []
    all_keys = set()

    if not sweeps_dir.exists():
        return rows, []

    # iterate each sweep folder under sweeps_dir
    for sweep_dir in sweeps_dir.iterdir():
        if not sweep_dir.is_dir():
            continue
        sweep_name = sweep_dir.name

        for run_type in ('baseline', 'ann_knobs'):
            run_type_dir = sweep_dir / run_type
            if not run_type_dir.exists():
                continue

            # find all target JSON files under this run_type (including nested)
            for j in run_type_dir.rglob('*.json'):
                name = j.name
                if not (name.startswith('metrics_') or name in TARGET_JSON_NAMES):
                    continue

                # determine model and variant from path relative to run_type_dir
                try:
                    rel = j.parent.relative_to(run_type_dir)
                    rel_parts = rel.parts
                except Exception:
                    rel_parts = []

                model = rel_parts[0] if len(rel_parts) > 0 else ''
                variant = '/'.join(rel_parts) if len(rel_parts) > 0 else ''

                data = load_json(j) or {}

                base_row = {
                    'sweep': sweep_name,
                    'run_type': run_type,
                    'model': model,
                    'variant': variant,
                    'run_path': str(j.parent),
                    'json_file': name,
                }

                # flatten JSON contents with a prefix of file stem
                prefix = name.replace('.json', '')
                if isinstance(data, dict):
                    flat = {}
                    flatten('', data, flat)
                    for k, v in flat.items():
                        key = f'{prefix}.{k}'
                        base_row[key] = v
                        all_keys.add(key)

                rows.append(base_row)

    return rows, sorted(all_keys)


def write_csv(rows, keys, out_path: Path):
    fixed = ['sweep', 'run_type', 'model', 'variant', 'run_path', 'json_file']
    header = fixed + keys
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, '') for k in header}
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweeps-dir', default='runs/sweeps')
    ap.add_argument('--out', default='runs/sweeps_summary.csv')
    args = ap.parse_args()

    sweeps_dir = Path(args.sweeps_dir)
    out = Path(args.out)

    rows, keys = gather_sweep_rows(sweeps_dir)
    write_csv(rows, keys, out)
    print(f'Wrote {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Aggregate sweep results (baseline and ann_knobs) into a single CSV.

This script scans `runs/sweeps/<sweep>/(baseline|ann_knobs)` and collects
JSON files including `metrics_*.json`, `metrics_retrieval.json`,
`runinfo.json`, `config.json`, and `timings.json`. Nested JSON structures are
recursively flattened so timing fields and other nested evaluations are preserved.

Usage:
  python scripts/aggregate_sweeps.py --sweeps-dir runs/sweeps --out runs/sweeps_summary.csv
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
    """Recursively flatten dicts; lists are JSON-encoded to preserve contents."""
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            flatten(key, v, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value


def gather_sweep_rows(sweeps_dir: Path):
    rows = []
    all_keys = set()

    if not sweeps_dir.exists():
        return rows, []

    # iterate each sweep folder under sweeps_dir
    for sweep_dir in sweeps_dir.iterdir():
        if not sweep_dir.is_dir():
            continue
        sweep_name = sweep_dir.name

        for run_type in ('baseline', 'ann_knobs'):
            run_type_dir = sweep_dir / run_type
            if not run_type_dir.exists():
                continue

            # find all target JSON files under this run_type (including nested)
            for j in run_type_dir.rglob('*.json'):
                name = j.name
                if not (name.startswith('metrics_') or name in TARGET_JSON_NAMES):
                    continue

                # determine model and variant from path relative to run_type_dir
                try:
                    rel = j.parent.relative_to(run_type_dir)
                    rel_parts = rel.parts
                except Exception:
                    rel_parts = []

                model = rel_parts[0] if len(rel_parts) > 0 else ''
                variant = '/'.join(rel_parts) if len(rel_parts) > 0 else ''

                data = load_json(j) or {}

                base_row = {
                    'sweep': sweep_name,
                    'run_type': run_type,
                    'model': model,
                    'variant': variant,
                    'run_path': str(j.parent),
                    'json_file': name,
                }

                # flatten JSON contents with a prefix of file stem
                prefix = name.replace('.json', '')
                if isinstance(data, dict):
                    flat = {}
                    flatten('', data, flat)
                    for k, v in flat.items():
                        key = f'{prefix}.{k}'
                        base_row[key] = v
                        all_keys.add(key)

                rows.append(base_row)

    return rows, sorted(all_keys)


def write_csv(rows, keys, out_path: Path):
    fixed = ['sweep', 'run_type', 'model', 'variant', 'run_path', 'json_file']
    header = fixed + keys
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, '') for k in header}
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweeps-dir', default='runs/sweeps')
    ap.add_argument('--out', default='runs/sweeps_summary.csv')
    args = ap.parse_args()

    sweeps_dir = Path(args.sweeps_dir)
    out = Path(args.out)

    rows, keys = gather_sweep_rows(sweeps_dir)
    write_csv(rows, keys, out)
    print(f'Wrote {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Aggregate sweep results (baseline and ann_knobs) into a single CSV.

Scans `runs/sweeps/**/(baseline|ann_knobs)` and extracts JSON files
like `metrics_*.json`, `runinfo.json`, `config.json`, and `timings.json`.
Writes a flattened CSV suitable for research analysis.

Usage:
  python scripts/aggregate_sweeps.py --sweeps-dir runs/sweeps --out runs/sweeps_summary.csv
"""
import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None
    def flatten(prefix: str, value, out: dict):
        """Recursively flatten dicts; lists are JSON-encoded."""
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else k
                flatten(key, v, out)
        elif isinstance(value, list):
            out[prefix] = json.dumps(value, ensure_ascii=False)
        else:
            out[prefix] = value


def gather_sweep_rows(sweeps_dir: Path):
    rows = []
    all_keys = set()

    # find baseline or ann_knobs directories under sweeps_dir
    for sweep_sub in sweeps_dir.iterdir():
        if not sweep_sub.is_dir():
            continue
    # iterate directories looking for baseline or ann_knobs
    for p in sweeps_dir.rglob('*'):
        if not p.is_dir():
            continue
        if p.name not in ('baseline', 'ann_knobs'):
            continue

        # determine sweep name and parent structure
        rel = p.relative_to(sweeps_dir)
        parts = rel.parts
        sweep_name = parts[0] if len(parts) > 0 else ''
        run_type = p.name

        # inside this directory, there may be multiple model-run subfolders
        for run_dir in p.rglob('*'):
            if not run_dir.is_dir():
                continue

            # consider JSON files directly inside run_dir
            for j in run_dir.glob('*.json'):
                name = j.name
                if not (name.startswith('metrics_') or name in ('metrics_retrieval.json', 'runinfo.json', 'config.json', 'timings.json')):
                    continue

                data = load_json(j) or {}
                row = {
                    'sweep': sweep_name,
                    'run_type': run_type,
                    'run_path': str(j.parent),
                    'json_file': name,
                }

                prefix = name.replace('.json', '')
                if isinstance(data, dict):
                    for k, v in data.items():
                        key = f'{prefix}.{k}'
                        if isinstance(v, (dict, list)):
                            row[key] = json.dumps(v, ensure_ascii=False)
                        else:
                            row[key] = v
                        all_keys.add(key)

                rows.append(row)
                            'model': model,
                            'variant': variant,

    return rows, sorted(all_keys)


def write_csv(rows, keys, out_path: Path):
    fixed = ['sweep', 'run_type', 'run_path', 'json_file']
    header = fixed + keys
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, '') for k in header}
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweeps-dir', default='runs/sweeps')
    ap.add_argument('--out', default='runs/sweeps_summary.csv')
    args = ap.parse_args()

    sweeps_dir = Path(args.sweeps_dir)
    out = Path(args.out)

    rows, keys = gather_sweep_rows(sweeps_dir)
    write_csv(rows, keys, out)
    print(f'Wrote {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()
