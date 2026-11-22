#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
EXAMPLE="${EXAMPLE:-$REPO_ROOT/examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py}"
MNK="${MNK:-512,1024,64}"

if [ ! -x "$VENV/bin/run-iket" ]; then
  echo "run-iket not found in $VENV; run ./setup.sh first" >&2
  exit 1
fi
if [ ! -f "$EXAMPLE" ]; then
  echo "IKET example not found: $EXAMPLE" >&2
  exit 1
fi

export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="$ROOT/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$TMPDIR"

exec run-iket --output-dir "$ROOT/iket_output" --clobber \
  profile --postprocess all -- \
  python "$EXAMPLE" --mnk "$MNK"
