#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'project_state.json'
CSV = ROOT / 'results' / 'experiments.csv'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--build', choices=['PASS','FAIL'])
    p.add_argument('--correctness', choices=['PASS','FAIL'])
    p.add_argument('--benchmark', choices=['PASS','FAIL'])
    p.add_argument('--profiling', choices=['PASS','LIMITED','FAIL'])
    p.add_argument('--documentation', choices=['PASS','IN_PROGRESS'])
    return p.parse_args()


def load_rows():
    if not CSV.exists() or CSV.stat().st_size == 0:
        return []
    with CSV.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    state = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {
        'project': 'CUDA GEMM Performance Engineering Lab',
        'status': {}, 'current_best_kernel': None, 'best_verified_result': None,
        'latest_experiment_id': None, 'last_successful_build': None,
        'last_successful_correctness_run': None, 'unsupported_features': [],
        'known_limitations': [], 'known_regressions': [], 'open_investigations': []
    }
    status = state.setdefault('status', {})
    if args.build: status['build'] = args.build
    if args.correctness: status['correctness'] = args.correctness
    if args.benchmark: status['benchmark'] = args.benchmark
    if args.profiling: status['profiling'] = args.profiling
    if args.documentation: status['documentation'] = args.documentation

    rows = load_rows()
    if rows:
        state['latest_experiment_id'] = rows[-1].get('experiment_id') or state.get('latest_experiment_id')
        verified = [r for r in rows if r.get('status') == 'PASS' and r.get('verification_status') == 'PASS' and r.get('gflops') not in ('', 'null', None)]
        if verified:
            best = max(verified, key=lambda r: float(r['gflops']))
            state['current_best_kernel'] = best.get('kernel')
            state['best_verified_result'] = {
                'experiment_id': best.get('experiment_id'),
                'kernel': best.get('kernel'),
                'dtype': best.get('dtype'),
                'M': int(best['M']), 'N': int(best['N']), 'K': int(best['K']),
                'median_ms': float(best['median_ms']), 'gflops': float(best['gflops'])
            }

    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    if args.build == 'PASS': state['last_successful_build'] = now
    if args.correctness == 'PASS': state['last_successful_correctness_run'] = now
    STATE.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
