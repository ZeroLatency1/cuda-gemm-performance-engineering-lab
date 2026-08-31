#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'project_state.json'
STATUS_MIRROR = ROOT / 'results' / 'project_status.json'
CSV = ROOT / 'results' / 'experiments.csv'
JSONL = ROOT / 'results' / 'experiments.jsonl'
PROFILES = ROOT / 'results' / 'profiling.jsonl'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--build', choices=['PASS', 'FAIL'])
    p.add_argument('--correctness', choices=['PASS', 'FAIL'])
    p.add_argument('--benchmark', choices=['PASS', 'FAIL'])
    p.add_argument('--profiling', choices=['PASS', 'LIMITED', 'FAIL'])
    p.add_argument('--documentation', choices=['PASS', 'IN_PROGRESS'])
    return p.parse_args()


def load_rows():
    if JSONL.exists() and JSONL.stat().st_size:
        rows = []
        for line in JSONL.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if not CSV.exists() or CSV.stat().st_size == 0:
        return []
    with CSV.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_profiled_experiment_ids():
    ids = set()
    if not PROFILES.exists():
        return ids
    for line in PROFILES.read_text(encoding='utf-8').splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj.get('performance_claim_eligible') is False and obj.get('experiment_id'):
                ids.add(obj['experiment_id'])
    return ids


def workload_key(row):
    return '|'.join([
        str(row.get('gpu_uuid') or row.get('gpu', 'UNKNOWN')),
        str(row.get('gpu', 'UNKNOWN')),
        str(row.get('compute_capability', 'UNKNOWN')),
        str(row.get('driver', 'UNKNOWN')),
        str(row.get('cuda_runtime', 'UNKNOWN')),
        str(row.get('cuda_toolkit', 'UNKNOWN')),
        str(row.get('dtype', 'UNKNOWN')),
        str(row.get('M', '')), str(row.get('N', '')), str(row.get('K', '')),
        str(row.get('seed', '')),
    ])


def compact_result(row):
    return {
        'experiment_id': row.get('experiment_id'),
        'kernel': row.get('kernel'),
        'kernel_variant': row.get('kernel_variant'),
        'dtype': row.get('dtype'),
        'M': int(row['M']), 'N': int(row['N']), 'K': int(row['K']),
        'median_ms': float(row['median_ms']),
        'p95_ms': float(row['p95_ms']),
        'min_ms': float(row['min_ms']),
        'gflops': float(row['gflops']),
    }


def main():
    args = parse_args()
    state = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {
        'project': 'CUDA GEMM Performance Engineering Lab',
        'status': {},
        'current_best_kernel': None,
        'best_verified_result': None,
        'best_verified_by_workload': {},
        'latest_experiment_id': None,
        'last_successful_build': None,
        'last_successful_correctness_run': None,
        'last_successful_profile': None,
        'unsupported_features': [],
        'known_limitations': [],
        'known_regressions': [],
        'open_investigations': [],
    }
    state['state_schema_version'] = 2
    status = state.setdefault('status', {})
    if args.build:
        status['build'] = args.build
    if args.correctness:
        status['correctness'] = args.correctness
    if args.benchmark:
        status['benchmark'] = args.benchmark
    if args.profiling:
        status['profiling'] = args.profiling
    if args.documentation:
        status['documentation'] = args.documentation

    rows = load_rows()
    profiled_ids = load_profiled_experiment_ids()
    if rows:
        state['latest_experiment_id'] = rows[-1].get('experiment_id') or state.get('latest_experiment_id')

        eligible = []
        for row in rows:
            if row.get('experiment_id') in profiled_ids:
                continue
            if row.get('status') != 'PASS' or row.get('verification_status') != 'PASS':
                continue
            try:
                if float(row.get('gflops', '0')) <= 0:
                    continue
                eligible.append(row)
            except (TypeError, ValueError):
                continue

        best_by_workload = {}
        for row in eligible:
            key = workload_key(row)
            incumbent = best_by_workload.get(key)
            if incumbent is None or float(row['gflops']) > float(incumbent['gflops']):
                best_by_workload[key] = row
        state['best_verified_by_workload'] = {
            key: compact_result(row) for key, row in sorted(best_by_workload.items())
        }

        # Preserve the legacy single-best fields for compatibility, but scope them
        # to the workload of the latest verified experiment rather than all workloads.
        latest_verified = None
        for row in reversed(eligible):
            latest_verified = row
            break
        if latest_verified is not None:
            key = workload_key(latest_verified)
            best = best_by_workload[key]
            state['current_best_kernel'] = best.get('kernel')
            state['best_verified_result'] = compact_result(best)
        else:
            state['current_best_kernel'] = None
            state['best_verified_result'] = None

    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    if args.build == 'PASS':
        state['last_successful_build'] = now
    if args.correctness == 'PASS':
        state['last_successful_correctness_run'] = now
    if args.profiling == 'PASS':
        state['last_successful_profile'] = now

    STATE.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    STATUS_MIRROR.parent.mkdir(parents=True, exist_ok=True)
    STATUS_MIRROR.write_text(json.dumps({
        'canonical_state': '../project_state.json',
        'project': state.get('project'),
        'status': state.get('status', {}),
        'state_schema_version': state.get('state_schema_version', 2),
        'latest_experiment_id': state.get('latest_experiment_id'),
        'current_best_kernel': state.get('current_best_kernel'),
        'best_verified_result': state.get('best_verified_result'),
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
