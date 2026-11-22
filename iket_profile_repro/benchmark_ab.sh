#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
EXAMPLE="${EXAMPLE:-$REPO_ROOT/examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py}"
MNK="${MNK:-512,1024,64}"
REPS="${REPS:-2}"
BUILD="$ROOT/build/ab"
RESULTS="$ROOT/results/ab_metrics_reproduced.txt"

if [ ! -x "$VENV/bin/run-iket" ]; then
  echo "run-iket not found in $VENV; run ./setup.sh first" >&2
  exit 1
fi

export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="$ROOT/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$BUILD" "$TMPDIR"

python "$ROOT/make_minimal.py" --input "$EXAMPLE" --output "$BUILD/fp16_gemm_4_iket_min.py"
: > "$RESULTS"

for kind in full min; do
  if [ "$kind" = full ]; then
    script="$EXAMPLE"
  else
    script="$BUILD/fp16_gemm_4_iket_min.py"
  fi
  for rep in $(seq 1 "$REPS"); do
    out="$BUILD/${kind}_${rep}"
    run-iket --output-dir "$out" --clobber profile --postprocess all -- \
      python "$script" --mnk "$MNK" >/dev/null 2>&1
    python "$ROOT/summarize_iket.py" "$kind" "$rep" "$out" | tee -a "$RESULTS"
  done
done

echo "Wrote $RESULTS"
