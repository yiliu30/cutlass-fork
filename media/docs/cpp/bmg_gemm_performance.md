# BMG Performance Benchmarks

This report records performance measurements for the `00_bmg_gemm` example on
an Intel Arc Pro B70 GPU. The example computes a row-major BF16 x BF16 GEMM
with FP32 accumulation and FP32 output.

## Test environment

| Component | Value |
|---|---|
| GPU | Intel Arc Pro B70 Graphics |
| Architecture reported by `sycl-ls` | `intel_gpu_bmg_g31` |
| Execution units | 256 |
| GPU maximum clock | 2800 MHz |
| GPU memory | 32656 MiB |
| Compiler | IntelLLVM 2026.1.0 |
| Driver | 1.15.38646 |
| Example | `examples/00_bmg_gemm/00_bmg_gemm.cpp` |

The example was configured and built for the GPU's native target:

```bash
source /opt/intel/oneapi/setvars.sh
export CC=icx
export CXX=icpx
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=100000000000

cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
  -DCUTLASS_SYCL_RUNNING_CI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx
ninja 00_bmg_gemm
```

`CMAKE_BUILD_TYPE=Release` is essential. A build without it achieved only about
4.3 TFLOP/s at `5120x4096x4096` and generated heavily spilling device code.
The Release build generated a GEMM kernel with zero reported spills.

`IGC_VISAOptions="-perfmodel"` is optional. It enables detailed compiler
performance-model diagnostics but is not required for optimized execution.

Correctness was checked with:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
./examples/00_bmg_gemm/00_bmg_gemm \
  --m=1024 --n=1024 --k=1024 --iterations=1 --verify=1
```

The result passed.

## Square problem sweep

Each result is the average reported by the example over 50 timed iterations.
Verification was disabled during timing.

| M = N = K | Time (ms) | Performance (TFLOP/s) |
|---:|---:|---:|
| 256 | 0.0512 | 0.656 |
| 512 | 0.0299 | 8.990 |
| 768 | 0.0513 | 17.654 |
| 1024 | 0.0813 | 26.403 |
| 1536 | 0.2496 | 29.038 |
| 2048 | 0.3111 | 55.228 |
| 3072 | 0.5012 | 115.696 |
| 4096 | 0.9386 | 146.430 |
| 5120 | 1.8710 | 143.472 |
| 6144 | 3.4384 | 134.907 |
| 8192 | 8.6472 | 127.153 |
| 10240 | 17.2405 | 124.560 |
| 12288 | 28.8153 | 128.781 |
| 16384 | 68.3629 | 128.668 |

Small problems do not expose enough parallel work to saturate the GPU.
Throughput rises rapidly from 2048 through 4096, peaks in the 4096-4864
region, and settles near 125-135 TFLOP/s for large square problems.

## Peak-region refinement

Shapes from 3072 through 5120 were measured three times with 50 timed
iterations per run. The table preserves all three measurements to show
run-to-run variation.

| M = N = K | Run 1 | Run 2 | Run 3 | Best (TFLOP/s) |
|---:|---:|---:|---:|---:|
| 3072 | 99.168 | 115.404 | 115.615 | 115.615 |
| 3328 | 115.594 | 118.183 | 115.780 | 118.183 |
| 3584 | 110.139 | 122.783 | 121.359 | 122.783 |
| 3840 | 141.966 | 128.458 | 128.184 | 141.966 |
| 4096 | 147.482 | 152.500 | 146.988 | 152.500 |
| 4352 | 138.633 | 139.472 | 139.151 | 139.472 |
| 4608 | 142.576 | 141.508 | 140.099 | 142.576 |
| 4864 | 154.211 | 145.859 | 145.363 | **154.211** |
| 5120 | 143.081 | 143.321 | 143.185 | 143.321 |

The best square result observed was 154.211 TFLOP/s at `4864x4864x4864`.
The variation in repeated measurements indicates that clock, power, thermal,
or concurrent system activity should be controlled when comparing small
performance differences.

## Rectangular result

The example's default shape achieved a higher observed result than the square
sweep:

| M | N | K | Time (ms) | Performance (TFLOP/s) |
|---:|---:|---:|---:|---:|
| 5120 | 4096 | 4096 | 1.0075 | **170.514** |

This was the peak observed result in this test session. It is a measured kernel
throughput for one shape and environment, not a hardware specification.

## Flash-attention comparison

The SYCL-TLA `06_bmg_flash_attention` prefill example was compared with
`ark.sdpa` from a local ARK 0.15.0 development build. ARK was built from
`/root/auto-round/auto_round_extension/ark` for `intel_gpu_bmg_g31` using
PyTorch 2.13.0+xpu and the same oneAPI 2026.1 toolchain and IGC optimization
environment described above.

Both implementations ran non-causal, contiguous prefill attention with:

| Parameter | Value |
|---|---:|
| Batch | 1 |
| Q heads | 40 |
| KV heads | 40 |
| Q sequence length | 32768 |
| KV sequence length | 32768 |
| QK head dimension | 128 |
| VO head dimension | 128 |
| Input type | BF16 |
| Warm-up iterations | 5 |
| Timed iterations | 20 |

The SYCL-TLA command was:

```bash
./examples/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_bfloat16_t_hdim128 \
  --batch=1 \
  --num_heads_q=40 \
  --num_heads_kv=40 \
  --seq_len_qo=32768 \
  --seq_len_kv=32768 \
  --head_size_qk=128 \
  --head_size_vo=128 \
  --scheduler=Individual \
  --warmup=5 \
  --iterations=20 \
  --verify=0
```

Three independent timing runs produced:

| Implementation | Run | Time (ms) | Performance (TFLOP/s) |
|---|---:|---:|---:|
| SYCL-TLA example 06 | 1 | 395.5224 | 55.598 |
| SYCL-TLA example 06 | 2 | 396.1328 | 55.512 |
| SYCL-TLA example 06 | 3 | 396.2457 | 55.496 |
| ARK `ark.sdpa` | 1 | 181.4514 | 121.191 |
| ARK `ark.sdpa` | 2 | 182.6182 | 120.416 |
| ARK `ark.sdpa` | 3 | 182.8944 | 120.235 |

| Implementation | Average time (ms) | Average performance (TFLOP/s) | Relative speed |
|---|---:|---:|---:|
| SYCL-TLA example 06 | 395.9670 | 55.535 | 1.00x |
| ARK `ark.sdpa` | **182.3213** | **120.614** | **2.17x** |

The 32K benchmark omitted reference verification because materializing or
computing the full reference attention problem is prohibitively expensive. A
smaller BF16 case with the same 40-head and 128-dimension configuration was
compared against PyTorch SDPA. It produced a maximum absolute difference of
0.003906 and a mean absolute difference of 0.000026; all output values were
finite.

This is not a completely output-equivalent comparison. ARK returns BF16 output,
whereas example 06 is instantiated with FP32 output. ARK therefore performs
less output conversion and writes half as many output bytes. The two attention
matrix multiplications and the FLOP accounting are otherwise equivalent.

### Flash-attention performance root cause

Further investigation found that the primary performance difference is a
register-pressure regression in the current SYCL-TLA FMHA mainloop. It is not
caused by a missing benchmark environment variable.

ARK's local extension was built against SYCL-TLA commit `fedbba40`, while the
standalone measurements used this repository at commit `91e5bd73`. ARK also
uses a private `SDPAFwdMainloop` derived from the older implementation:

```text
/root/auto-round/auto_round_extension/ark/
  auto_round_kernel/wrapper/include/stla/xe_sdpa_fwd_mainloop.hpp
```

The current mainloop is:

```text
applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp
```

The current implementation keeps additional data live across the KV loop:

- `tSrQ_arr[DTiles]` retains reordered Q fragments so they can be reused.
- `prepared_k[DTiles]` retains prepared K copy payloads.
- `prepared_v[VTiles]` retains prepared V copy payloads.
- Partial softmax reduction state has additional live ranges.

These changes avoid repeated payload construction and Q loads, but for head
dimension 128 they increase register lifetime and pressure enough to make IGC
spill. At sequence length 32768, spill costs inside the loop are repeated over
1024 KV blocks (`32768 / 32`).

IGC VISA performance-model diagnostics for the current attention kernel
variants reported:

```text
Spill Size = 3968-7616 bytes
# Spills   = 88-100
# Fills    = 46-62
Dynamic spill/fill instructions = 3.1-5.7%
```

The exact ARK BF16, head-dimension-128, non-causal kernel reported:

```text
Spill Size = 0
# Spills   = 0
# Fills    = 0
```

Controlled experiments separated the output conversion cost from the mainloop
cost:

| Implementation | Output type | Average time (ms) | Performance (TFLOP/s) |
|---|---|---:|---:|
| Current SYCL-TLA | FP32 | 395.967 | 55.535 |
| Current SYCL-TLA, experimental output change | BF16 | 374.511 | 58.718 |
| ARK `ark.sdpa` | BF16 | 182.321 | 120.614 |

Changing the current example to BF16 output improved throughput by only about
6%, leaving most of the gap intact. The source was restored to its original
FP32 output configuration after this experiment.

The following possible causes were also checked:

| Candidate | Result |
|---|---|
| `IGC_RemoveUnusedIdImplicitLocalIDs=0` | No improvement; approximately 55.2 TFLOP/s |
| FP32 versus BF16 output | Approximately 6% impact, not the primary cause |
| QK/PV tile shapes | Equivalent between the compared kernels |
| Pipeline depth | Two stages in both kernels |
| Launch grid | Equivalent: 5120 work-groups for this problem |
| Causal masking | Disabled in both kernels |
| Sequence chunking | Disabled in both kernels |
| Scheduler overhead | Current specialization removes unnecessary div/mod operations |

The principal optimization target is therefore the live-register footprint of
the current mainloop. Q-fragment reuse and prepared K/V payloads should be
retained only if they can be scheduled without spilling, or selectively
disabled for configurations such as BF16 head dimension 128 where their
register cost exceeds the saved load and payload-construction work.

## Historical commit comparison

The repository was switched to detached commit
`fedbba404c6fb634554b7e5aec9a463710b6f952`, the SYCL-TLA revision used by the
ARK build. Both examples were rebuilt from a clean directory in Release mode
for `intel_gpu_bmg_g31`, using the same compiler and optimization environment
as the preceding tests.

### Flash attention at `fedbba40`

The same BF16, batch-1, 40-head, 32768-token, head-dimension-128, non-causal
prefill workload produced:

| Run | Time (ms) | Performance (TFLOP/s) |
|---:|---:|---:|
| 1 | 173.0173 | 127.098 |
| 2 | 173.3316 | 126.868 |
| 3 | 173.2416 | 126.934 |
| **Average** | **173.1968** | **126.967** |

The historical example passed a smaller correctness run with batch 1, 40
heads, sequence length 512, and head dimension 128.

| Implementation | Average time (ms) | Average performance (TFLOP/s) | Relative to current |
|---|---:|---:|---:|
| Current SYCL-TLA `91e5bd73` | 395.9670 | 55.535 | 1.00x |
| ARK `ark.sdpa` | 182.3213 | 120.614 | 2.17x |
| Historical SYCL-TLA `fedbba40` | **173.1968** | **126.967** | **2.29x** |

The historical standalone example is approximately 5.3% faster than ARK even
though the example writes FP32 output and ARK writes BF16 output. This confirms
that the large FMHA slowdown was introduced after `fedbba40`; it is not an
inherent limitation of the example, hardware, or benchmark procedure.

### GEMM at `fedbba40`

The historical `00_bmg_gemm` target was measured three times per shape with 50
timed iterations. Verification passed at `1024x1024x1024`.

| Shape (M x N x K) | Run | Time (ms) | Performance (TFLOP/s) |
|---|---:|---:|---:|
| 4096 x 4096 x 4096 | 1 | 0.8054 | 170.641 |
| 4096 x 4096 x 4096 | 2 | 0.8043 | 170.891 |
| 4096 x 4096 x 4096 | 3 | 0.8052 | 170.687 |
| 5120 x 4096 x 4096 | 1 | 1.1268 | 152.460 |
| 5120 x 4096 x 4096 | 2 | 1.1183 | 153.624 |
| 5120 x 4096 x 4096 | 3 | 1.1167 | 153.844 |
| 8192 x 8192 x 8192 | 1 | 8.6585 | 126.987 |
| 8192 x 8192 x 8192 | 2 | 8.6772 | 126.712 |
| 8192 x 8192 x 8192 | 3 | 8.6440 | 127.199 |

| Shape (M x N x K) | Average time (ms) | Average performance (TFLOP/s) |
|---|---:|---:|
| 4096 x 4096 x 4096 | **0.8050** | **170.740** |
| 5120 x 4096 x 4096 | **1.1206** | **153.309** |
| 8192 x 8192 x 8192 | **8.6599** | **126.966** |

Unlike flash attention, these GEMM results do not show a uniform historical
advantage. The historical commit is faster for the measured 4096-cubed case,
similar at 8192 cubed, and below the best current-commit result previously
observed for the default rectangular shape. GEMM measurements around the peak
also showed clock-related run-to-run variation, so these data do not establish
a GEMM regression between the two commits.
