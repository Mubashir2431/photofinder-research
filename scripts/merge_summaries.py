#!/usr/bin/env python3
"""Merge two summary CSVs into a single CSV for analysis.

This script reads two CSV files, unions their columns, adds a `source`
column indicating origin, and writes `runs/merged_summary.csv` by default.

Usage:
  python scripts/merge_summaries.py --a runs/summary_results.csv --b runs/sweeps_summary.csv --out runs/merged_summary.csv
"""
import argparse
import csv
from pathlib import Path


def read_csv(path: Path):
    rows = []
    with path.open('r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows, (r.fieldnames or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', default='runs/merged_summary.csv')
    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    out_path = Path(args.out)

    a_rows, a_fields = read_csv(a_path)
    b_rows, b_fields = read_csv(b_path)

    # union of fields and ensure `source` column
    all_fields = []
    for f in (a_fields or []) + (b_fields or []):
        if f not in all_fields:
            all_fields.append(f)
    if 'source' not in all_fields:
        all_fields.insert(0, 'source')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()

        for r in a_rows:
            row = {k: r.get(k, '') for k in all_fields}
            row['source'] = 'all'
            w.writerow(row)

        for r in b_rows:
            row = {k: r.get(k, '') for k in all_fields}
            row['source'] = 'sweeps'
            w.writerow(row)

    print(f'Wrote merged CSV to {out_path} ({len(a_rows)+len(b_rows)} rows)')


if __name__ == '__main__':
    main()
