#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

KERNEL="register"
M=2048
N=2048
K=2048
DTYPE="fp32"
WARMUP=0
ITERATIONS=1
SECTION_SET="basic"
LAUNCH_COUNT=1
EXPERIMENT_NAME=""
PARENT_ID="${PARENT_ID:-}"
BASELINE_ID="${BASELINE_ID:-}"
OPT_DESCRIPTION="${OPT_DESCRIPTION:-}"
NCU_BIN=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_profile.sh [options]

Profiles one benchmark configuration with Nsight Compute. Profiling timings are
for diagnosis only and are not eligible for performance comparisons.

Options:
  --kernel <name>                    Kernel to profile (default: register)
  --M <int> --N <int> --K <int>      GEMM dimensions (default: 2048 each)
  --dtype <fp32|fp16>                Data type (default: fp32)
  --warmup <int>                     Benchmark warmups (default: 0)
  --iterations <int>                 Benchmark iterations (default: 1)
  --set <name>                       Nsight Compute section set (default: basic)
  --launch-count <int>               NCU launch count (default: 1)
  --experiment-name <name>           Name for the profiling experiment
  --parent-experiment-id <id>        Parent optimization experiment
  --baseline-experiment-id <id>      Baseline experiment for lineage
  --optimization-description <txt>   Hypothesis/change description
  --ncu-binary <path>                Explicit ncu executable
  --help                             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel) KERNEL="$2"; shift 2 ;;
    --M) M="$2"; shift 2 ;;
    --N) N="$2"; shift 2 ;;
    --K) K="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --set) SECTION_SET="$2"; shift 2 ;;
    --launch-count) LAUNCH_COUNT="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --parent-experiment-id) PARENT_ID="$2"; shift 2 ;;
    --baseline-experiment-id) BASELINE_ID="$2"; shift 2 ;;
    --optimization-description) OPT_DESCRIPTION="$2"; shift 2 ;;
    --ncu-binary) NCU_BIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in M N K WARMUP ITERATIONS LAUNCH_COUNT; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer" >&2
    exit 2
  fi
done
if (( M == 0 || N == 0 || K == 0 )); then
  echo "ERROR: M, N, and K must be positive" >&2
  exit 2
fi
if (( ITERATIONS == 0 || LAUNCH_COUNT == 0 )); then
  echo "ERROR: iterations and launch-count must be positive" >&2
  exit 2
fi
if [[ "$DTYPE" != "fp32" && "$DTYPE" != "fp16" ]]; then
  echo "ERROR: dtype must be fp32 or fp16" >&2
  exit 2
fi

find_ncu() {
  if [[ -n "$NCU_BIN" ]]; then
    [[ -x "$NCU_BIN" ]] || { echo "ERROR: ncu binary is not executable: $NCU_BIN" >&2; return 1; }
    printf '%s\n' "$NCU_BIN"
    return 0
  fi
  if command -v ncu >/dev/null 2>&1; then
    command -v ncu
    return 0
  fi
  local candidate
  for candidate in \
    "/usr/local/NVIDIA-Nsight-Compute/target/linux-desktop-glibc_2_11_3-x64/ncu" \
    "/usr/local/NVIDIA-Nsight-Compute-2026.2/target/linux-desktop-glibc_2_11_3-x64/ncu" \
    "/opt/nvidia/nsight-compute/2024.3.2/target/linux-desktop-glibc_2_11_3-x64/ncu"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! NCU_BIN="$(find_ncu)"; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  LIMIT="profiles/limitations/PROFILING_${STAMP}.txt"
  mkdir -p profiles/limitations
  {
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tool=nsight_compute"
    echo "status=UNAVAILABLE"
    echo "command=ncu --set $SECTION_SET --launch-count $LAUNCH_COUNT ./build/cuda_gemm_benchmark --M $M --N $N --K $K --kernel $KERNEL --dtype $DTYPE"
    echo "error=ncu not found in PATH or known installation locations"
  } | tee "$LIMIT"
  python3 scripts/update_project_state.py --profiling LIMITED
  exit 0
fi

if [[ ! -x build/cuda_gemm_benchmark ]]; then
  "$PWD/scripts/build.sh"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TMP_DIR="profiles/.pending_${STAMP}_$$"
TMP_OUT="$TMP_DIR/profile"
LOG="$TMP_DIR/profile.log"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ -z "$EXPERIMENT_NAME" ]]; then
  EXPERIMENT_NAME="profile_${KERNEL}_${M}x${N}x${K}_${STAMP}"
fi

CMD=(
  "$NCU_BIN"
  --set "$SECTION_SET"
  --launch-count "$LAUNCH_COUNT"
  -o "$TMP_OUT"
  ./build/cuda_gemm_benchmark
  --M "$M" --N "$N" --K "$K"
  --kernel "$KERNEL"
  --dtype "$DTYPE"
  --warmup "$WARMUP"
  --iterations "$ITERATIONS"
  --no-verify
  --experiment-name "$EXPERIMENT_NAME"
)
[[ -n "$PARENT_ID" ]] && CMD+=(--parent-experiment-id "$PARENT_ID")
[[ -n "$BASELINE_ID" ]] && CMD+=(--baseline-experiment-id "$BASELINE_ID")
[[ -n "$OPT_DESCRIPTION" ]] && CMD+=(--optimization-description "$OPT_DESCRIPTION")

printf 'Profiling command: '
printf '%q ' "${CMD[@]}"
printf '\n'

set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

EXP_ID="$(grep -oE 'Experiment recorded: EXP-[0-9]+' "$LOG" | tail -1 | sed 's/.* //')"
REPORT="$TMP_OUT.ncu-rep"

if (( rc != 0 )); then
  mkdir -p profiles/limitations
  FAILURE_LOG="profiles/limitations/PROFILING_${STAMP}.log"
  cp "$LOG" "$FAILURE_LOG" 2>/dev/null || true
  LIMIT="profiles/limitations/PROFILING_${STAMP}.txt"
  {
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tool=nsight_compute"
    echo "status=LIMITED_OR_FAILED"
    echo "ncu_version=$($NCU_BIN --version 2>/dev/null | tail -1 || echo UNKNOWN)"
    echo "kernel=$KERNEL"
    echo "workload=${M}x${N}x${K}"
    echo "dtype=$DTYPE"
    echo "experiment_id=${EXP_ID:-UNKNOWN}"
    echo "exit_code=$rc"
    echo "command=$(printf '%q ' "${CMD[@]}")"
    echo "log=${FAILURE_LOG}"
  } | tee "$LIMIT"
  python3 scripts/update_project_state.py --profiling LIMITED
  exit 0
fi

if [[ -z "$EXP_ID" || ! -f "$REPORT" ]]; then
  mkdir -p profiles/limitations
  FAILURE_LOG="profiles/limitations/PROFILING_${STAMP}.log"
  cp "$LOG" "$FAILURE_LOG" 2>/dev/null || true
  LIMIT="profiles/limitations/PROFILING_${STAMP}.txt"
  {
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tool=nsight_compute"
    echo "status=LIMITED_OR_FAILED"
    echo "error=NCU exited successfully but experiment ID or .ncu-rep artifact was missing"
    echo "experiment_id=${EXP_ID:-UNKNOWN}"
    echo "report=$REPORT"
    echo "log=$FAILURE_LOG"
  } | tee "$LIMIT"
  python3 scripts/update_project_state.py --profiling LIMITED
  exit 0
fi

PROFILE_DIR="profiles/$EXP_ID"
mkdir -p "$PROFILE_DIR"
mv "$REPORT" "$PROFILE_DIR/profile.ncu-rep"
mv "$LOG" "$PROFILE_DIR/profile.log"

NCU_VERSION="$($NCU_BIN --version 2>/dev/null | tail -1 || echo UNKNOWN)"
REPORT_PATH="profiles/$EXP_ID/profile.ncu-rep"
COMMAND_TEXT="$(printf '%q ' "${CMD[@]}")"
PROFILE_META="$PROFILE_DIR/metadata.json"
python3 - "$PROFILE_META" <<PY
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
obj = {
  "profiled_experiment_id": "${EXP_ID}",
  "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
  "profiler": "nsight-compute",
  "profiler_version": ${NCU_VERSION@Q},
  "kernel": "${KERNEL}",
  "dtype": "${DTYPE}",
  "M": int("${M}"), "N": int("${N}"), "K": int("${K}"),
  "warmup": int("${WARMUP}"), "iterations": int("${ITERATIONS}"),
  "section_set": "${SECTION_SET}",
  "launch_count": int("${LAUNCH_COUNT}"),
  "parent_experiment_id": "${PARENT_ID}",
  "baseline_experiment_id": "${BASELINE_ID}",
  "optimization_description": ${OPT_DESCRIPTION@Q},
  "report": "${REPORT_PATH}",
  "command": ${COMMAND_TEXT@Q},
  "performance_claim_eligible": False,
  "notes": "Nsight Compute instrumentation materially perturbs benchmark timing; use this artifact for kernel diagnosis, not performance comparisons."
}
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY

mkdir -p results
python3 - "$EXP_ID" "$REPORT_PATH" "$PROFILE_META" "$NCU_VERSION" "$KERNEL" "$DTYPE" "$M" "$N" "$K" "$PARENT_ID" "$BASELINE_ID" "$OPT_DESCRIPTION" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path.cwd()
row = {
    "profile_id": "PROFILE-" + sys.argv[1],
    "experiment_id": sys.argv[1],
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "profiler": "nsight-compute",
    "profiler_version": sys.argv[4],
    "kernel": sys.argv[5],
    "dtype": sys.argv[6],
    "M": int(sys.argv[7]), "N": int(sys.argv[8]), "K": int(sys.argv[9]),
    "report": sys.argv[2],
    "metadata": sys.argv[3],
    "parent_experiment_id": sys.argv[10],
    "baseline_experiment_id": sys.argv[11],
    "optimization_description": sys.argv[12],
    "performance_claim_eligible": False,
}
path = root / "results" / "profiling.jsonl"
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, separators=(",", ":")) + "\n")
PY

python3 scripts/update_project_state.py --profiling PASS
printf 'Nsight Compute profiling completed: experiment=%s report=%s\n' "$EXP_ID" "$REPORT_PATH"
