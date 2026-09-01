#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tests/test_repo_integrity.py
python3 scripts/validate_results.py
python3 -m py_compile scripts/update_project_state.py scripts/validate_results.py scripts/generate_plots.py tests/test_repo_integrity.py
for f in scripts/*.sh; do bash -n "$f"; done
printf 'Static repository tests PASS. Target CUDA execution still requires the specified RTX 4060 environment.\n'
