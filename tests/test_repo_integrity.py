#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEADER = [
    'experiment_id','timestamp','hostname','os','wsl_status','experiment_name','git_commit',
    'gpu','gpu_uuid','compute_capability','driver','cuda_runtime','cuda_toolkit','compiler',
    'cmake_version','kernel','kernel_variant','dtype','M','N','K','warmup','iterations','seed',
    'median_ms','p95_ms','min_ms','gflops','h2d_ms','d2h_ms','end_to_end_ms','end_to_end_gflops',
    'latency_delta_ms','latency_delta_pct','throughput_delta_gflops','throughput_delta_pct',
    'verification_status','max_abs_error','max_rel_error','status','cuda_errors','runtime_errors',
    'environment_warnings','parent_experiment_id','baseline_experiment_id',
    'optimization_description','notes'
]

csv_path = ROOT / 'results' / 'experiments.csv'
with csv_path.open(newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)
assert header == EXPECTED_HEADER, 'experiment CSV header mismatch'

ids = [r[0] for r in rows]
assert len(ids) == len(set(ids)), 'duplicate experiment IDs in CSV'
for row in rows:
    assert len(row) == len(EXPECTED_HEADER), f'CSV row has {len(row)} columns, expected {len(EXPECTED_HEADER)}'

jsonl = ROOT / 'results' / 'experiments.jsonl'
json_ids = []
if jsonl.exists():
    for line in jsonl.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        assert set(EXPECTED_HEADER).issubset(obj.keys())
        json_ids.append(obj['experiment_id'])
assert json_ids == ids, 'CSV and JSONL experiment histories disagree'

raw_dir = ROOT / 'results' / 'raw'
raw_ids = []
if raw_dir.exists():
    for path in sorted(raw_dir.glob('EXP-*.json')):
        obj = json.loads(path.read_text(encoding='utf-8'))
        assert obj['experiment_id'] == path.stem, f'{path.name} experiment_id mismatch'
        raw_ids.append(obj['experiment_id'])
assert raw_ids == ids, 'raw experiment artifacts disagree with indexed history'

assert 'set(CMAKE_CUDA_ARCHITECTURES 89' in (ROOT / 'CMakeLists.txt').read_text()
assert (ROOT / 'src' / 'benchmark.cu').read_text().count('struct BenchmarkResult') == 0

subprocess.run([sys.executable, str(ROOT / 'scripts' / 'validate_results.py')], check=True)
print('Repository integrity PASS')
