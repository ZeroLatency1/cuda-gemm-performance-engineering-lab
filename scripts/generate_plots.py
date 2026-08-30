#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'results' / 'experiments.csv'
OUT = ROOT / 'results' / 'plots'
OUT.mkdir(parents=True, exist_ok=True)

if not CSV.exists() or CSV.stat().st_size == 0:
    raise SystemExit('No experiment history exists; no plots generated.')

with CSV.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if r.get('status') == 'PASS' and r.get('gflops') not in ('', 'null', None)]
if not rows:
    raise SystemExit('No measured PASS records exist yet; no plots generated.')

# Plot only from actual stored measurements. No values are synthesized.
groups = defaultdict(list)
for r in rows:
    try:
        g = float(r['gflops'])
        if g <= 0: continue
        label = f"{r['kernel']} ({r['dtype']})"
        groups[label].append((int(r['M']), int(r['N']), int(r['K']), g))
    except (ValueError, KeyError):
        continue

for label, points in sorted(groups.items()):
    points.sort(key=lambda x: x[0] * x[1] * x[2])
    x = [f"{m}x{n}x{k}" for m, n, k, _ in points]
    y = [g for _, _, _, g in points]
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('GFLOPS (from median kernel latency)')
    plt.xlabel('Workload MxNxK')
    plt.title(f'{label} — stored measurements only')
    plt.tight_layout()
    safe = label.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(OUT / f'gflops_{safe}.png', dpi=160)
    plt.close()

# Cross-kernel comparison is useful only for a common workload.
common = defaultdict(dict)
for r in rows:
    key = (r.get('M'), r.get('N'), r.get('K'), r.get('dtype'))
    common[key][r.get('kernel')] = float(r['gflops'])
for key, values in common.items():
    if len(values) < 2: continue
    labels = list(values.keys())
    heights = [values[k] for k in labels]
    workload = 'x'.join(key[:3]) + '_' + key[3]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, heights)
    plt.ylabel('GFLOPS (median latency)')
    plt.xlabel('Kernel')
    plt.title(f'Kernel comparison — {workload}')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(OUT / f'kernel_comparison_{workload}.png', dpi=160)
    plt.close()

print(f'Generated measured-data plots under {OUT}')
