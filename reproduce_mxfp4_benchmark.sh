#!/usr/bin/env bash
set -euo pipefail

# Reproduce the B200 CuTe MXFP4 benchmark.
# Override ENV_ROOT if the temporary uv environment is elsewhere.
ENV_ROOT="${ENV_ROOT:-/dev/shm/cutlass_mxfp4_bench.0QgEVu}"
PYTHON="${ENV_ROOT}/.venv/bin/python"
SITE="${ENV_ROOT}/.venv/lib/python3.12/site-packages"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
MODE="${1:-benchmark}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing uv environment: ${PYTHON}" >&2
  echo "Set ENV_ROOT to the directory containing the benchmark .venv." >&2
  exit 1
fi

export CUDA_HOME
export CUDA_INSTALL_PATH="${CUDA_HOME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="${ENV_ROOT}/.venv/bin:${CUDA_HOME}/bin:${PATH}"

CUDA_LIBS="$(find "${SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${CUDA_LIBS}${SITE}/nvidia_cutlass_dsl/cu12/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

case "${MODE}" in
  production)
    exec "${PYTHON}" \
      examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_production.py \
      --mnk 8192,8192,1024 \
      --output_dtype fp16 \
      --benchmark
    ;;
  simple)
    exec "${PYTHON}" \
      examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_simple.py \
      --benchmark
    ;;
  benchmark)
    exec "${PYTHON}" "${ENV_ROOT}/benchmark.py"
    ;;
  compare)
    exec "${PYTHON}" "${ENV_ROOT}/compare_dense_gemm.py"
    ;;
  correctness)
    exec "${PYTHON}" \
      examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py \
      --a_dtype Float4E2M1FN \
      --b_dtype Float4E2M1FN \
      --sf_dtype Float8E8M0FNU \
      --sf_vec_size 32 \
      --c_dtype Float16 \
      --a_major k \
      --b_major k \
      --c_major n \
      --mma_tiler_mn 128,128 \
      --cluster_shape_mn 1,1 \
      --mnkl 512,512,256,1 \
      --warmup_iterations 2 \
      --iterations 5
    ;;
  large)
    exec "${PYTHON}" \
      examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py \
      --a_dtype Float4E2M1FN \
      --b_dtype Float4E2M1FN \
      --sf_dtype Float8E8M0FNU \
      --sf_vec_size 32 \
      --c_dtype Float16 \
      --a_major k \
      --b_major k \
      --c_major n \
      --mma_tiler_mn 256,128 \
      --cluster_shape_mn 2,1 \
      --mnkl 8192,8192,1024,1 \
      --warmup_iterations 10 \
      --iterations 50 \
      --skip_ref_check
    ;;
  *)
    echo "Usage: $0 [production|simple|benchmark|compare|correctness|large]" >&2
    exit 2
    ;;
esac
