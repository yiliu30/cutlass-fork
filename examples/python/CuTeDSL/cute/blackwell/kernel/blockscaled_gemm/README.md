# Blackwell Dense Block-Scaled GEMM Variants

This directory contains related CuTe DSL examples for persistent dense
block-scaled GEMM on NVIDIA Blackwell (`SM100`):

| Example | Main design goal |
| --- | --- |
| [`dense_blockscaled_gemm_persistent.py`](dense_blockscaled_gemm_persistent.py) | General-purpose persistent GEMM baseline |
| [`dense_blockscaled_gemm_persistent_amax.py`](dense_blockscaled_gemm_persistent_amax.py) | Fuse an absolute-maximum reduction into the GEMM epilogue |
| [`dense_blockscaled_gemm_persistent_prefetch.py`](dense_blockscaled_gemm_persistent_prefetch.py) | Add explicit TMA data prefetch hints |
| [`dense_blockscaled_gemm_simple.py`](dense_blockscaled_gemm_simple.py) | Fixed, educational MXFP4 pipeline |
| [`dense_blockscaled_gemm_production.py`](dense_blockscaled_gemm_production.py) | Production-facing MXFP4 prefill API |

These are not five different matrix-multiplication algorithms. They share the
same Blackwell execution model and explore different extensions to it.

For production integration details, see the repository-level
[`MXFP4_PREFILL_HANDOFF.md`](../../../../../../../MXFP4_PREFILL_HANDOFF.md).

## Shared execution model

The kernels compute a block-scaled GEMM of the form

```text
C = (A * SFA) x (B * SFB)
```

where A and B are low-precision values and SFA/SFB are block scale factors.
The main execution pipeline is:

```text
Persistent tile scheduler
        |
        +-- TMA warp:       GMEM -> SMEM
        |                   A, B, SFA, SFB
        |
        +-- MMA warp:       SMEM -> TMEM
        |                   tcgen05 block-scaled MMA
        |
        +-- Epilogue warps: TMEM -> registers -> SMEM -> GMEM
                            output conversion and C store
```

The baseline implementation documents this organization in the kernel
overview at [`dense_blockscaled_gemm_persistent.py`](dense_blockscaled_gemm_persistent.py#L55).

The important mechanisms are:

- **Warp specialization:** warp 5 performs TMA loads, warp 4 performs MMA,
  and warps 0--3 perform the epilogue.
- **Staged pipelines:** A/B/scale-factor tiles reside in circular shared-memory
  stages. Producers acquire an empty stage, issue TMA loads, and consumers wait
  for the stage to become full.
- **Persistent scheduling:** a CTA remains active and processes multiple output
  tiles using a persistent tile scheduler.
- **Cluster multicast:** the cluster can multicast A, B, or scale-factor tiles
  to reduce duplicate global-memory traffic.

The baseline TMA producer/consumer pipeline is visible at
[`dense_blockscaled_gemm_persistent.py#L1015`](dense_blockscaled_gemm_persistent.py#L1015).

All three examples also prefetch TMA descriptors when the kernel starts. That
descriptor setup should not be confused with the explicit future-data prefetch
added by the prefetch variant.

## Persistent baseline

[`dense_blockscaled_gemm_persistent.py`](dense_blockscaled_gemm_persistent.py)
is the general-purpose implementation.

Its design priorities are:

- broad block-scaled dtype and layout support;
- support for different A and B dtypes where the MMA configuration allows it;
- standard asynchronous TMA/UMMA pipeline behavior;
- persistent tile scheduling and warp specialization;
- optional fused elementwise epilogue operations.

For each K tile, the TMA warp loads A, B, SFA, and SFB into a shared-memory
stage. The MMA warp waits for that stage, moves scale factors to tensor memory,
and launches the Blackwell block-scaled MMA operation. The accumulator remains
in tensor memory until the epilogue warps convert and store it.

This is the best default reference kernel when the application needs raw GEMM
performance or the broadest set of supported configurations.

## Amax variant

[`dense_blockscaled_gemm_persistent_amax.py`](dense_blockscaled_gemm_persistent_amax.py)
adds a fused absolute-maximum calculation to the epilogue.

The epilogue performs both operations while the accumulator is already in
registers:

```text
accumulator -> output conversion/store
            -> abs(value) -> max reduction -> global Amax scalar
```

The reduction hierarchy is:

1. Each thread computes a maximum over its accumulator values.
2. Each epilogue warp performs a warp-level maximum reduction.
3. One value per warp is written to shared memory.
4. One thread computes the block maximum.
5. A global atomic max combines results from all output tiles and CTAs.

The implementation is at
[`dense_blockscaled_gemm_persistent_amax.py#L1335`](dense_blockscaled_gemm_persistent_amax.py#L1335).
The Amax is calculated from the higher-precision accumulator before conversion
to the output dtype, which is useful for quantization and requantization
pipelines.

### Amax trade-offs

The fused reduction avoids launching a second kernel to scan the output, but it
adds reduction instructions, synchronization, shared-memory traffic, and a
global atomic operation. The Amax version also has a more specialized interface:
A and B must use the same dtype.

Therefore, this variant should be selected when the application needs Amax,
not merely because its standalone GEMM timing happens to be similar to the
baseline.

## Prefetch variant

[`dense_blockscaled_gemm_persistent_prefetch.py`](dense_blockscaled_gemm_persistent_prefetch.py)
adds explicit prefetch hints for future A, B, SFA, and SFB tiles.

There are two prefetch phases:

- **Initial prefetch:** issue hints for the first few K tiles to prime the
  pipeline.
- **Rolling prefetch:** while loading the current K tile, issue hints for a
  future tile.

The initial prefetch code is at
[`dense_blockscaled_gemm_persistent_prefetch.py#L1071`](dense_blockscaled_gemm_persistent_prefetch.py#L1071),
and the rolling prefetch code is at
[`dense_blockscaled_gemm_persistent_prefetch.py#L1141`](dense_blockscaled_gemm_persistent_prefetch.py#L1141).

The prefetch distance is configured as follows:

```text
None -> use num_ab_stage automatically
0    -> disable explicit data prefetch
N>0  -> prefetch N K tiles ahead
```

This design is intended to hide global-memory latency:

```text
early prefetch hint -> later TMA load -> MMA consumes the data
```

Explicit prefetch is not automatically beneficial. It can add instructions,
cache pressure, and TMA control overhead. If the existing staged TMA pipeline
already hides memory latency, the extra hints can make the kernel slower.

## Important implementation difference

The Amax and Prefetch files are alternative implementations rather than
minimal one-line feature switches on top of the baseline. In particular, their
MMA mainloops explicitly set the SFA/SFB tensor-memory fields for each K block
before calling `cute.gemm`; the baseline passes the scale-factor tensors as
MMA operands.

Baseline style:

```python
cute.gemm(
    tiled_mma,
    tCtAcc,
    [tCrA[tile_crd], tCtSFA],
    [tCrB[tile_crd], tCtSFB],
    tCtAcc,
)
```

Explicit scale-factor setup in the Amax variant:

```python
tiled_mma.set(tcgen05.Field.SFA, ...)
tiled_mma.set(tcgen05.Field.SFB, ...)
cute.gemm(tiled_mma, tCtAcc, tCrA[...], tCrB[...], tCtAcc)
```

See [`dense_blockscaled_gemm_persistent_amax.py#L1165`](dense_blockscaled_gemm_persistent_amax.py#L1165)
and [`dense_blockscaled_gemm_persistent_prefetch.py#L1347`](dense_blockscaled_gemm_persistent_prefetch.py#L1347).
Consequently, measured differences include both the intended feature and some
implementation/plumbing differences between the examples.

## Fixed educational MXFP4 kernel

[`dense_blockscaled_gemm_simple.py`](dense_blockscaled_gemm_simple.py) is the
learning-oriented version. It fixes the problem to MXFP4 (`Float4E2M1FN` with
`Float8E8M0FNU` scales, vector size 32), a `128x256x256` tile, one CTA per
output tile, four shared-memory stages, and a direct FP16 epilogue. It keeps
the important operations visible in one file:

1. TMA loads A, B, SFA, and SFB into staged shared memory.
2. Scale factors move from shared memory to tensor memory.
3. `tcgen05` performs the block-scaled MMA into a tensor-memory accumulator.
4. The accumulator is converted to FP16 and stored to C.

The host reference explicitly expands each 32-element scale block. This makes
the correctness check easy to follow and avoids hiding the reference in a
library scaled-matmul call. The benchmark compares the kernel with BF16
`torch.mm`; the default acceptance margin is 10% because this intentionally
single-warp teaching kernel is not the production warp-specialized kernel.

From the repository root, run:

```bash
./reproduce_mxfp4_benchmark.sh simple
```

For a quick correctness-only run, invoke the example directly:

```bash
${ENV_ROOT:-/dev/shm/cutlass_mxfp4_bench.0QgEVu}/.venv/bin/python \
  examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_simple.py \
  --mnkl 512,512,256,1
```

The strict comparison can be requested with `--parity_tolerance 0.0`.

## Production MXFP4 prefill API

[`dense_blockscaled_gemm_production.py`](dense_blockscaled_gemm_production.py)
wraps the warp-specialized persistent kernel with a model-facing API. The first
production configuration deliberately uses one-CTA clusters:

```text
MXFP4: E2M1 x E2M1
Scale: UE8M0, vector size 32
MMA tile: 128x256
Cluster: 1x1
CTA group: ONE
Output: FP16 or BF16
```

It keeps TMA/UMMA staging, tensor-memory accumulation, warp specialization,
persistent tile scheduling, stream-aware launches, and compiled-variant
caching. It defers cluster multicast, two-CTA MMA, decode-specific kernels,
and fused epilogues.

The hot path accepts already-packed FP4 A/B tensors and already-formatted
UE8M0 scales. Use `pack_mxfp4_values` and `pack_mxfp4_scales` during model
setup, not inside every GEMM call:

```python
from cute.blackwell.kernel.blockscaled_gemm.dense_blockscaled_gemm_production import (
    Mxfp4PrefillGemm,
    pack_mxfp4_scales,
    pack_mxfp4_values,
)

gemm = Mxfp4PrefillGemm(output_dtype=torch.float16)
c = gemm(a_fp4, b_fp4, a_scales, b_scales, (M, N, K))
```

Inputs must be padded for the selected tile and K-block sizes. Unsupported
inputs raise validation errors; the API does not silently fall back to
`torch.mm`. Reproduce the production smoke test with:

```bash
./reproduce_mxfp4_benchmark.sh production
```

The production benchmark uses 10 warmups and 50 iterations and reports the
median latency against BF16 `torch.mm`.

## Comparison from the B200 benchmark

The three variants were run with the same configuration:

```text
M=N=8192, K=1024, L=1
A/B: Float4E2M1FN
Scale: Float8E8M0FNU, vector size 32
C: Float16
MMA tile: 256x128
Cluster: 2x1
10 warmups, 50 iterations
```

Measured results:

| Variant | Median | Effective GEMM rate | Relative to baseline |
| --- | ---: | ---: | ---: |
| Persistent | 40.15 us | 3,423 TFLOP/s | 1.00x |
| Persistent + Amax | 39.82 us | 3,451 TFLOP/s | 1.01x |
| Persistent + Prefetch, auto | 49.74 us | 2,763 TFLOP/s | 0.81x |

The Amax result is effectively tied with the baseline for this large,
compute-heavy shape; that does not imply the reduction has zero cost. The
automatic prefetch configuration was about 19% slower, indicating that the
baseline pipeline already hid enough latency for this particular workload.
Prefetch distance should be tuned per shape rather than assumed to improve
performance.

## Which kernel should be used?

| Requirement | Recommended variant |
| --- | --- |
| Raw GEMM performance and broad support | Persistent baseline |
| GEMM output plus Amax for quantization | Amax variant |
| A demonstrably memory-latency-bound workload | Prefetch variant, tuned experimentally |

To reproduce the three-way comparison from the repository root:

```bash
./reproduce_mxfp4_benchmark.sh compare
```
