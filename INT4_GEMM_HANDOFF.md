# INT4 BMG GEMM Handoff

Date: 2026-08-18
Repo: `/home/yiliu4/workspace/sycl-tla`
Build dir used for validation: `build-icpx2`

## Goal

Optimize `examples/00_bmg_gemm/00_bmg_gemm_int4` against the fixed INT8 baseline:

- Baseline: `00_bmg_gemm_int8 = 176.36 TOPS`
- Target: `1.6x` INT8 = `282.18 TOPS`
- Achieved: `~314 TOPS`, about `1.78x` INT8

## Main Result

The promoted signed INT4 target is:

- Executable: `build-icpx2/examples/00_bmg_gemm/00_bmg_gemm_int4`
- Input/output: `int4_t x int4_t -> int32_t`
- Tile shape: `Shape<_256, _256, _128>`
- Subgroup layout: default `8x4`
- Pipeline stages: `2`
- `B` operand layout: `ColumnMajor`
- Launch property: `sycl::ext::intel::experimental::grf_size<256>`
- IGC option: `allowDecompose2DBlockFuncs=0`

The important tuning point was making signed INT4 use `ColumnMajor` for `B` with `K=128`. Row-major `B` variants were much slower.

## Files Changed

- `examples/00_bmg_gemm/00_bmg_gemm.cpp`
  - Added macro-driven dtype support for BF16 default, INT8, signed INT4, and unsigned UINT4.
  - Added macro-driven tile shape controls: `BMG_GEMM_TILE_M/N/K`.
  - Added `BMG_GEMM_LAYOUT_B_COLUMN`.
  - Added `BMG_GEMM_SG_LAYOUT_4X8` for tuning variants.
  - Added `BMG_GEMM_PIPELINE_STAGES`.
  - Added K-alignment checks: INT4 requires `K % 64 == 0`, INT8 requires `K % 32 == 0`.
  - Prints `TOPS` for integer GEMM variants.

- `examples/00_bmg_gemm/CMakeLists.txt`
  - Added `00_bmg_gemm_int8`.
  - Added promoted `00_bmg_gemm_int4`.
  - Added tuning variants for signed INT4 and unsigned UINT4.
  - Applies `allowDecompose2DBlockFuncs=0` to integer variants.

- `include/cutlass/gemm/device/gemm_universal_adapter.h`
  - Added Intel large GRF launch property for SYCL Intel targets:
    `sycl::ext::intel::experimental::grf_size<256>`.
  - This was required to recover INT8 performance and avoid register-spill-driven slowdown.

## Build Environment

Validated with:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
```

Compiler observed:

```text
Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)
```

Configure command used:

```bash
cmake -S . -B build-icpx2 -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER=icx \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_PROFILER=OFF \
  -DCUTLASS_ENABLE_PERFORMANCE=OFF \
  -DBENCHMARK_ENABLE_TESTING=OFF \
  -DBENCHMARK_ENABLE_GTEST_TESTS=OFF \
  -DBENCHMARK_USE_BUNDLED_GTEST=OFF
```

Build command:

```bash
cmake --build build-icpx2 --target 00_bmg_gemm 00_bmg_gemm_int8 00_bmg_gemm_int4 -j 16
```

## Verification Commands

Correctness:

```bash
./build-icpx2/examples/00_bmg_gemm/00_bmg_gemm \
  --m=256 --n=256 --k=256 --l=1 --iterations=0 --verify=1

./build-icpx2/examples/00_bmg_gemm/00_bmg_gemm_int8 \
  --m=256 --n=256 --k=512 --l=1 --iterations=0 --verify=1

./build-icpx2/examples/00_bmg_gemm/00_bmg_gemm_int4 \
  --m=256 --n=256 --k=512 --l=1 --iterations=0 --verify=1
```

All three passed.

Performance:

```bash
./build-icpx2/examples/00_bmg_gemm/00_bmg_gemm_int8 \
  --m=5120 --n=4096 --k=4096 --l=1 --iterations=100 --verify=0

./build-icpx2/examples/00_bmg_gemm/00_bmg_gemm_int4 \
  --m=5120 --n=4096 --k=4096 --l=1 --iterations=100 --verify=0
```

Final measured results:

```text
00_bmg_gemm_int8:
  176.336 TOPS, 0.9743 ms

00_bmg_gemm_int4 promoted run 1:
  313.960 TOPS, 0.5472 ms

00_bmg_gemm_int4 promoted run 2:
  314.213 TOPS, 0.5468 ms

00_bmg_gemm_int4 promoted run 3:
  314.034 TOPS, 0.5471 ms
```

Ratio against requested baseline:

```text
314.0 / 176.36 = 1.78x
```

Apple-to-apple summary for the final comparison:

| Kernel | Data Type | Tile | B Layout | Perf | Time | Ratio vs INT8 |
|---|---|---|---|---:|---:|---:|
| `00_bmg_gemm_int8` | `int8_t x int8_t -> int32_t` | `256x256x32` | RowMajor | `176.336 TOPS` | `0.9743 ms` | `1.00x` |
| `00_bmg_gemm_int4` | `int4_t x int4_t -> int32_t` | `256x256x128` | ColumnMajor | `314.034 TOPS` | `0.5471 ms` | `1.78x` |

Apple-to-apple INT4 layout comparison:

| Kernel | Data Type | Tile | B Layout | Perf | Time | Ratio vs RowMajor B |
|---|---|---|---|---:|---:|---:|
| `00_bmg_gemm_int4_s4_256x256x128` | `int4_t x int4_t -> int32_t` | `256x256x128` | RowMajor | `137.549 TOPS` | `1.2490 ms` | `1.00x` |
| `00_bmg_gemm_int4_s4_256x256x128_colb` | `int4_t x int4_t -> int32_t` | `256x256x128` | ColumnMajor | `314.125 TOPS` | `0.5469 ms` | `2.28x` |

This isolates the effect of `BMG_GEMM_LAYOUT_B_COLUMN`: same signed INT4 data type,
same problem size, same tile shape, same pipeline depth, and same subgroup layout.
The ColumnMajor-B path is much faster because it aligns the B operand memory layout
with the Xe DPAS/VNNI-friendly B register layout used by `XE_DPAS_TT`.

What `ColumnMajor B` changes in this kernel:

| Aspect | RowMajor B | ColumnMajor B | Why it matters |
|---|---|---|---|
| CUTLASS tag | `cutlass::layout::RowMajor` | `cutlass::layout::ColumnMajor` | Selected by `BMG_GEMM_LAYOUT_B_COLUMN` in `00_bmg_gemm.cpp`. |
| CuTe stride for B modes `[N,K,L]` | `Stride<Int<1>, int64_t, int64_t>` | `Stride<int64_t, Int<1>, int64_t>` | Defined by `TagToStrideB` in `include/cutlass/detail/layout.hpp`. |
| Contiguous memory mode | `N` is contiguous | `K` is contiguous | The Xe DPAS B operand layout is VNNI-like over K for each N block. |
| 2D block copy selection | `find_x_mode(gstride)` picks the `N` mode as the 2D-copy contiguous axis | `find_x_mode(gstride)` picks the `K` mode as the 2D-copy contiguous axis | The autoselected `get_block_2d_copy_B<void>` path derives its copy atom/layout from this stride. |
| Reorder call in mainloop | `reorder(tBrB, tCrB)` is still called | `reorder(tBrB, tCrB)` is still called | The code path does not literally remove `reorder`; the win is from feeding it a B copy fragment that better matches the DPAS B fragment layout. |
| Expected effect | More expensive layout conversion and/or less efficient B block-load pattern | Cheaper B feed path into the DPAS/VNNI register layout | This is consistent with the measured `2.28x` speedup for the same INT4 tile. |

Important nuance: this benchmark proves the ColumnMajor-B path is much faster, but
it does not by itself prove that `reorder(tBrB, tCrB)` becomes a complete no-op.
The source still contains the reorder call in `include/cutlass/gemm/collective/xe_mma.hpp`.
To prove the exact reorder instruction difference, inspect the generated VISA/assembly
or collect compiler/profiler instruction metrics for the two variants.

## Tuning Data

Representative sweep on `M=5120, N=4096, K=4096, iterations=100`:

```text
00_bmg_gemm_int8                         176.313 TOPS
00_bmg_gemm_int4                         162.893 TOPS  old default, 256x128x128 row-B
00_bmg_gemm_int4_s4_256x256x64           183.743 TOPS
00_bmg_gemm_int4_s4_256x256x128          137.549 TOPS
00_bmg_gemm_int4_s4_128x256x128          110.132 TOPS
00_bmg_gemm_int4_s4_256x128x128_sg4x8    189.602 TOPS
00_bmg_gemm_int4_s4_256x256x128_colb     314.125 TOPS
00_bmg_gemm_int4_u4_256x256x128          136.381 TOPS
00_bmg_gemm_int4_u4_256x256x128_colb     314.279 TOPS
```

Conclusion: the performance win is primarily from the `BMG_GEMM_LAYOUT_B_COLUMN` path for INT4, not signed vs unsigned.

## Profiling Notes

- Built-in timing was used for the final measurements.
- `unitrace` was not available in the tested environment after sourcing oneAPI, so no PTI counter CSVs were collected.
- Compiler output should still be watched for spill diagnostics. Earlier INT8 performance was poor until `grf_size<256>` was added to the Intel SYCL kernel properties.
- The final promoted INT4 build did not show the earlier explicit spill warning in the captured output; it did show IGC retry-manager recompilation warnings, followed by successful build.

## PR Notes

Before opening the PR:

1. Review whether `grf_size<256>` should apply to all Intel SYCL GEMM launches or be gated for Xe/BMG XMX kernels only.
2. Decide whether to keep all tuning variant targets in `CMakeLists.txt` or reduce to only:
   - `00_bmg_gemm_int8`
   - `00_bmg_gemm_int4`
   - optionally one `u4` diagnostic target.
3. Run formatting or the repo-preferred style check if one is required by CI.
4. Include the benchmark table above in the PR description.

Suggested PR summary:

```text
Add INT8 and INT4 BMG GEMM examples and tune signed INT4 to 1.78x INT8 baseline.

The promoted INT4 path uses int4_t x int4_t -> int32_t with tile 256x256x128,
ColumnMajor B, and large GRF launch properties. On BMG G21, it measures about
314 TOPS versus the fixed INT8 baseline of 176.36 TOPS.
```

## Current Git State

Expected modified files:

```text
M examples/00_bmg_gemm/00_bmg_gemm.cpp
M examples/00_bmg_gemm/CMakeLists.txt
M include/cutlass/gemm/device/gemm_universal_adapter.h
```

This handoff doc is additional local documentation for the optimization work.
