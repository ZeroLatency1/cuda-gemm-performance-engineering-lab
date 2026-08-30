#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d .git ]; then
  git init
fi
git add .
if ! git diff --cached --quiet; then
  git commit -m "Initialize CUDA GEMM performance engineering lab"
fi
printf 'Git repository initialized at %s\n' "$PWD"
