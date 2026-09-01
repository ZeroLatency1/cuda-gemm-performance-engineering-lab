#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
CSV = RESULTS / 'experiments.csv'
JSONL = RESULTS / 'experiments.jsonl'
RAW = RESULTS / 'raw'
PROFILES = RESULTS / 'profiling.jsonl'

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


def die(msg: str) -> None:
    print(f'RESULT VALIDATION FAIL: {msg}', file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if not CSV.exists(): die(f'missing {CSV}')
    if not JSONL.exists(): die(f'missing {JSONL}')
    if not RAW.exists(): die(f'missing {RAW}')

    with CSV.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if header != EXPECTED_HEADER:
        die('CSV header mismatch')
    ids = [row[0] for row in rows]
    if len(ids) != len(set(ids)):
        die('duplicate experiment IDs in CSV')
    for idx, row in enumerate(rows, start=2):
        if len(row) != len(EXPECTED_HEADER):
            die(f'CSV row {idx} has {len(row)} columns, expected {len(EXPECTED_HEADER)}')

    json_rows = []
    for line_no, line in enumerate(JSONL.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            die(f'JSONL line {line_no} is invalid JSON: {exc}')
        if not set(EXPECTED_HEADER).issubset(obj):
            die(f'JSONL line {line_no} is missing schema fields')
        json_rows.append(obj)
    json_ids = [obj['experiment_id'] for obj in json_rows]
    if json_ids != ids:
        die('CSV and JSONL histories disagree or are out of order')

    raw_ids = []
    for path in sorted(RAW.glob('EXP-*.json')):
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            die(f'{path.name} is invalid JSON: {exc}')
        exp_id = obj.get('experiment_id')
        if exp_id != path.stem:
            die(f'{path.name} contains experiment_id={exp_id!r}')
        raw_ids.append(exp_id)
    if raw_ids != sorted(ids, key=lambda x: int(x[4:])):
        die('raw experiment artifacts do not exactly match the experiment IDs')

    known = set(ids)
    if PROFILES.exists():
        for line_no, line in enumerate(PROFILES.read_text(encoding='utf-8').splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            exp_id = obj.get('experiment_id')
            if exp_id not in known:
                die(f'profiling.jsonl line {line_no} references unknown experiment {exp_id}')
            report = ROOT / obj.get('report', '')
            if not report.exists():
                die(f'profiling.jsonl line {line_no} references missing report {report}')
            if obj.get('performance_claim_eligible') is not False:
                die(f'profile {obj.get("profile_id")} is not explicitly marked performance_claim_eligible=false')

            # Never allow unresolved shell placeholders into newly-created
            # profiling records. Empty lineage fields are valid; literal
            # ${...} placeholders are not.
            for field in (
                'parent_experiment_id',
                'baseline_experiment_id',
                'optimization_description',
            ):
                value = obj.get(field)
                if isinstance(value, str) and '${' in value:
                    die(
                        f'profiling.jsonl line {line_no} contains unresolved '
                        f'shell placeholder in {field}: {value!r}'
                    )

    print(f'Result validation PASS: {len(ids)} experiments, {len(raw_ids)} raw artifacts')


if __name__ == '__main__':
    main()
