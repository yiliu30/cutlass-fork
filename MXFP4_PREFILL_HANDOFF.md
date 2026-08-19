# Blackwell MXFP4 Prefill GEMM Handoff

Date: 2026-08-17

## Executive summary

This checkout contains two MXFP4 CuTe DSL paths:

1. `dense_blockscaled_gemm_simple.py` — an educational single-warp kernel.
2. `dense_blockscaled_gemm_production.py` — the production-facing prefill API.

The production API reuses the existing warp-specialized persistent Blackwell
kernel and fixes the first deployment configuration to:

```text
A/B:       Float4E2M1FN (MXFP4)
Scales:    Float8E8M0FNU (UE8M0), vector size 32
MMA tile:  128 x 256
Cluster:   1 x 1
CTA group: ONE
Output:    FP16 or BF16
Workload:  prefill, L=1
```

The latest B200 benchmark run measured:

```text
MXFP4 production: 86.78 us, 1583.7 TFLOP/s
BF16 torch.mm:    92.93 us, 1479.0 TFLOP/s
Speedup:          1.071x
```

Timings vary with GPU state; the production command enforces a 5% maximum
slowdown relative to BF16 `torch.mm`.

## Files and ownership

| File | Purpose |
| --- | --- |
| `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_production.py` | Production API, validation, packing helpers, compile cache, benchmark CLI |
| `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py` | Warp-specialized persistent device implementation used by the production API |
| `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_simple.py` | Educational reference implementation |
| `test/examples/CuTeDSL/sm_100a/test_dense_blockscaled_gemm_production.py` | GPU correctness and API tests |
| `reproduce_mxfp4_benchmark.sh` | Environment setup and reproducible commands |

The production wrapper is intentionally separate from the educational kernel.
Future kernel optimizations should preserve the production API contract unless
the model integration is updated at the same time.

## Environment

The validated machine is an NVIDIA B200. The driver reports CUDA 13.0, while
the working toolkit is `/usr/local/cuda-12.8`; the host does not need a system
`nvcc` because the environment uses the toolkit path explicitly.

The existing uv environment is:

```text
/dev/shm/cutlass_mxfp4_bench.0QgEVu/.venv
```

The reproduction script sets the required `PATH`, `CUDA_HOME`,
`CUDA_INSTALL_PATH`, and `LD_LIBRARY_PATH`. Override the environment when the
temporary directory changes:

```bash
ENV_ROOT=/path/to/cutlass_mxfp4_bench \
CUDA_HOME=/usr/local/cuda-12.8 \
./reproduce_mxfp4_benchmark.sh production
```

## Production API

Import the wrapper from the repository’s CuTe DSL example path:

```python
import torch

from cute.blackwell.kernel.blockscaled_gemm.dense_blockscaled_gemm_production import (
    Mxfp4PrefillGemm,
)

gemm = Mxfp4PrefillGemm(output_dtype=torch.float16)

# A and B are already-packed FP4 tensors. Scales are already in MMA layout.
c = gemm(a_fp4, b_fp4, a_scales, b_scales, (M, N, K))
```

For repeated model calls, retain the `Mxfp4PrefillGemm` object. It caches
compiled variants by device, logical shape, and output dtype. The cache is
bounded to 16 variants by default.

An existing output buffer may be reused:

```python
c = torch.empty((M, N), device="cuda", dtype=torch.float16)
gemm(a_fp4, b_fp4, a_scales, b_scales, (M, N, K), out=c)
```

The call is asynchronous with respect to the supplied/current CUDA stream. It
does not perform a synchronization or silently call `torch.mm`.

## Input contract

The caller owns quantization, packing, padding, and fallback selection.

### A/B operands

- Logical A shape: `(M, K)`, K-major.
- Logical B shape: `(N, K)`, K-major; the operation computes `A @ B.T`.
- Storage: prepacked `Float4E2M1FN` values, commonly represented by
  `torch.float4_e2m1fn_x2` or byte storage.
- A/B storage must be contiguous and 16-byte aligned.
- `pack_mxfp4_values(values)` is provided for one-time setup conversion from a
  CUDA FP32/FP16 matrix. Do not call it inside the GEMM hot path.

### Scale operands

- Logical scale shape before packing: `(rows, ceil(K / 32))`.
- Dtype: `torch.float8_e8m0fnu`.
- `a_scales` has M rows; `b_scales` has N rows.
- The GEMM call receives the Blackwell MMA-packed scale layout, not the logical
  row/block layout.
- `pack_mxfp4_scales(scales, rows, K)` performs the one-time layout conversion.
- Scale storage must be 32-byte aligned.

The scale conversion helper uses a host source buffer internally. This is
intentional and avoids a CuTe JIT pointer-space fault observed when a CUDA
source tensor was passed directly to the repository conversion routine.

### Shape and output requirements

The first production variant requires:

```text
M % 128 == 0
N % 256 == 0
K % 256 == 0
```

Only `L=1` is exposed by the production wrapper. A prefill runtime should
flatten its active tokens into M and pad before calling the kernel. Unsupported
inputs raise `ValueError`/`TypeError`; the wrapper does not implement an
internal fallback.

Supported output dtypes are `torch.float16` and `torch.bfloat16`. The output
must be contiguous and 16-byte aligned when supplied by the caller.

## Reproduction commands

### Production smoke test and benchmark

```bash
./reproduce_mxfp4_benchmark.sh production
```

This runs the target shape `(8192, 8192, 1024)` with 10 warmups and 50 timed
iterations, then compares against BF16 `torch.mm`.

Run a smaller smoke test directly:

```bash
ENV_ROOT=/dev/shm/cutlass_mxfp4_bench.0QgEVu
"$ENV_ROOT/.venv/bin/python" \
  examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_production.py \
  --mnk 512,512,256 --output_dtype fp16
```

Test BF16 output:

```bash
"$ENV_ROOT/.venv/bin/python" \
  examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_production.py \
  --mnk 128,256,256 --output_dtype bf16
```

### Educational kernel

```bash
./reproduce_mxfp4_benchmark.sh simple
```

The educational kernel uses a different single-warp implementation and is not
the production path.

## Validation status

Validated on the B200 environment:

- Production FP16 smoke test at `(8192,8192,1024)`: PASS.
- Production FP16 correctness at `(128,256,256)`: PASS.
- Production BF16 correctness at `(128,256,256)`: PASS.
- Non-unit scale correctness (`SFA=2`, `SFB=1`): PASS.
- Caller-provided output buffer reuse: PASS.
- Compiled-variant cache reuse: PASS.
- Invalid unpadded shape rejection: PASS.
- Production benchmark parity gate: PASS within 5% of BF16 `torch.mm`.

The repository environment did not include the `pytest` executable. The GPU
correctness cases were therefore also run through a direct CUDA harness. The
pytest test file remains available for environments with the test dependencies:

```bash
python -m pytest -q \
  test/examples/CuTeDSL/sm_100a/test_dense_blockscaled_gemm_production.py \
  -m L0
```

## Important limitations

This is production-quality for the defined prefill contract, but it is not a
drop-in replacement for every LLM `torch.mm` call:

- It currently supports FP4 x FP4, not the common FP4-weight x BF16-activation
  path.
- Decode/small-M latency is not optimized.
- There is no bias, activation, residual, or other fused epilogue.
- There is no implicit padding or runtime fallback.
- Cluster multicast and two-CTA MMA are deferred; the current cluster is
  `(1,1)`.
- The API is a Python CuTe DSL example and is not yet packaged as a C++/Torch
  custom operator.
- The caller must ensure that the model’s weight and activation scale layout
  matches the documented MMA-packed representation.

## Recommended next steps

1. Integrate `Mxfp4PrefillGemm` into one prefill linear layer with explicit
   padding and a caller-owned fallback.
2. Replace synthetic inputs with the model’s actual FP4 weight and scale
   packing, then compare outputs against the model’s BF16 path.
3. Add an offline packer/cache for persistent model weights so packing never
   occurs during inference.
4. Benchmark a representative prefill matrix suite, not only the 8192-shape
   smoke case.
5. Add a separate FP4-weight x BF16-activation kernel before attempting decode
   integration.
6. Consider `(256,128)` with `(2,1)` clusters and cluster multicast only after
   the one-CTA API is integrated and profiled.
7. Package the API as a Torch custom operator if Python/JIT overhead becomes a
   deployment constraint.

