# BMG Benchmarks at SYCL-TLA Commit `fedbba40`

This report records GEMM and flash-attention performance at SYCL-TLA commit
`fedbba404c6fb634554b7e5aec9a463710b6f952`, the revision used by the compared
ARK build.

## Environment

| Component | Value |
|---|---|
| GPU | Intel Arc Pro B70 Graphics |
| GPU architecture | `intel_gpu_bmg_g31` |
| GPU execution units | 256 |
| GPU maximum clock | 2800 MHz |
| GPU memory | 32656 MiB |
| Compiler | IntelLLVM 2026.1.0 |
| Driver | 1.15.38646 |
| SYCL-TLA commit | `fedbba404c6fb634554b7e5aec9a463710b6f952` |
| Commit subject | Add LSE output configuration to FMHAFwdEpilogue and kernel parameters |

The repository was checked out in detached-HEAD mode:

```bash
git switch --detach fedbba404c6fb634554b7e5aec9a463710b6f952
```

The examples were built from a clean Release configuration:

```bash
source /opt/intel/oneapi/setvars.sh
export CC=icx
export CXX=icpx
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export IGC_RemoveUnusedIdImplicitLocalIDs=0
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"

rm -rf build
mkdir build
cd build
cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
  -DCUTLASS_SYCL_RUNNING_CI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx
```

`IGC_VISAOptions="-perfmodel"` was not enabled during timed measurements.

## Flash-attention prefill

The BF16 prefill target was built with:

```bash
ninja 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128
```

The benchmark configuration was:

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
| Output type | FP32 |
| Causal mask | Disabled |
| Variable sequence length | Disabled |
| Scheduler | Individual |
| Warm-up iterations | 5 |
| Timed iterations | 20 |

Command:

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

Results:

| Run | Time (ms) | Performance (TFLOP/s) |
|---:|---:|---:|
| 1 | 173.0173 | 127.098 |
| 2 | 173.3316 | 126.868 |
| 3 | 173.2416 | 126.934 |
| **Average** | **173.1968** | **126.967** |

A smaller correctness run with batch 1, 40 heads, sequence length 512, and
head dimension 128 passed.

### Comparison

| Implementation | Output type | Average time (ms) | Average performance (TFLOP/s) | Relative speed |
|---|---|---:|---:|---:|
| Current SYCL-TLA `91e5bd73` | FP32 | 395.9670 | 55.535 | 1.00x |
| ARK `ark.sdpa` | BF16 | 182.3213 | 120.614 | 2.17x |
| Historical SYCL-TLA `fedbba40` | FP32 | **173.1968** | **126.967** | **2.29x** |

The historical standalone example is approximately 5.3% faster than ARK
despite writing FP32 output instead of BF16 output. This confirms that the
large standalone FMHA slowdown was introduced after `fedbba40`.

### `unitrace` profile

The historical kernel was profiled with Intel PTI `unitrace` 2.4.0. Device
timing measured the FMHA kernel at 163.128 ms. It launched 5120 workgroups with
a `{1, 128, 40}` grid, `{256, 1, 1}` local size, SIMD16 execution, 256-GRF
mode, no SLM, and no reported private or spill memory.

#### Install `unitrace`

The oneAPI installation on this system provided PTI runtime components but not
the `unitrace` executable. The tool was therefore built from Intel's `pti-gpu`
repository at commit `c71e8316e19bb5316157b9046d877b5eff0e262c`
(`unitrace` 2.4.0):

```bash
source /opt/intel/oneapi/setvars.sh

git clone https://github.com/intel/pti-gpu.git
cd pti-gpu
git checkout c71e8316e19bb5316157b9046d877b5eff0e262c

cmake -S tools/unitrace -B tools/unitrace/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_WITH_L0=1 \
  -DBUILD_WITH_XPTI=1 \
  -DBUILD_WITH_MPI=0 \
  -DBUILD_WITH_ITT=0 \
  -DBUILD_WITH_OMP=0 \
  -DBUILD_WITH_OPENCL=0
cmake --build tools/unitrace/build

export UNITRACE="$PWD/tools/unitrace/build/unitrace"
"$UNITRACE" --version
```

This minimal configuration retains the Level Zero backend required for device
timing, hardware metrics, and EU stall sampling. XPTI is retained for
SYCL/Unified Runtime tracing. MPI, ITT, OpenMP, and OpenCL tracing are not
needed for this standalone SYCL benchmark.

The GPU metrics runtime must expose the requested metric groups. These commands
can be used to confirm device discovery and find the available group names:

```bash
"$UNITRACE" --device-list
"$UNITRACE" --metric-list
```

#### Profiling commands

Run from the SYCL-TLA `build` directory after sourcing oneAPI. The benchmark
environment and command were:

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export IGC_RemoveUnusedIdImplicitLocalIDs=0
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"

APP=./examples/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_bfloat16_t_hdim128
ARGS="--batch=1 --num_heads_q=40 --num_heads_kv=40 \
--seq_len_qo=32768 --seq_len_kv=32768 \
--head_size_qk=128 --head_size_vo=128 \
--scheduler=Individual --warmup=5 --iterations=20 --verify=0"
```

Capture kernel timing, launch shape, SIMD width, GRF mode, SLM, and spill
information:

```bash
"$UNITRACE" --device-timing --verbose \
  --output fmha-device-timing \
  "$APP" $ARGS
```

Collect per-kernel ComputeBasic counters:

```bash
"$UNITRACE" --metric-query --group ComputeBasic \
  --output fmha-compute-basic.metrics \
  "$APP" $ARGS
```

Collect instruction-address EU stall samples:

```bash
"$UNITRACE" --stall-sampling \
  --output fmha-stalls.metrics \
  "$APP" $ARGS
```

`unitrace` appends the process ID to output file names. Metric collection adds
profiling overhead, so use these runs for bottleneck analysis rather than
benchmark timing. Device timing also includes warm-up and initialization
kernels; identify the demangled `XeFMHAFwdKernel` row when reading its report.

For instruction-to-source correlation, compile with
`-gline-tables-only`, dump the exact kernel shader, and analyze the matching
stall file:

```bash
mkdir -p igc-dump
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$PWD/igc-dump" \
  "$UNITRACE" --stall-sampling \
  --output fmha-stalls-with-shader.metrics \
  "$APP" $ARGS
```

The shader dump and stall samples must come from the same binary and compiler
configuration because instruction addresses can change between builds.

#### BKC: map a stall hotspot to source

Use the following best-known configuration (BKC) when exact source attribution
is required. The important requirements are to generate device debug metadata
at AOT build time and to collect the shader dump and stall samples from the
same binary.

1. Create an isolated artifact directory and Python environment:

```bash
export PROFILE_DIR="$PWD/unitrace-fmha-profile"
mkdir -p "$PROFILE_DIR/igc-dump"

python3 -m venv "$PROFILE_DIR/venv"
"$PROFILE_DIR/venv/bin/pip" install \
  "pandas>=2.2.1" "matplotlib>=3.8"
```

2. Configure the Release build with full debug metadata. IntelLLVM 2026.1
   ignores `-gline-tables-only` for `spir64_gen`, so use `-g` while retaining
   `-O3`:

```bash
source /opt/intel/oneapi/setvars.sh --force
export CC=icx
export CXX=icpx

cmake -S . -B build -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
  -DCUTLASS_SYCL_RUNNING_CI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -g"
```

3. Force recompilation of the AOT device image with IGC shader dumping
   enabled. Setting dump variables only while running the executable is
   insufficient for an AOT `spir64_gen` build.

```bash
export IGC_RemoveUnusedIdImplicitLocalIDs=0
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir="$PROFILE_DIR/igc-dump"

ninja -C build -t clean 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128
ninja -C build 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128
```

Confirm that the dump contains `.asm`, `.dat`, `.visa.ll`, and `.zeinfo`
files—not only `HardwareCaps.txt` and `SIPKernelDump.bin`:

```bash
find "$PROFILE_DIR/igc-dump" -type f \
  \( -name "*.asm" -o -name "*.dat" -o -name "*.zeinfo" \) |
  head
```

4. Collect stalls from that exact executable:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export UNITRACE=/path/to/pti-gpu/tools/unitrace/build/unitrace
export APP=./build/examples/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_bfloat16_t_hdim128

"$UNITRACE" --stall-sampling \
  --output "$PROFILE_DIR/fmha-stalls.metrics" \
  "$APP" \
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

Use the PID-suffixed device-metrics file printed by `unitrace`, not its
separate host log. For example:

```text
unitrace-fmha-profile/fmha-stalls.metrics.metrics.<pid>.metrics
```

5. Join sampled instruction pointers with the matching shader dump:

```bash
export ANALYZER=/path/to/pti-gpu/tools/unitrace/scripts/metrics/analyzeperfmetrics.py
export METRICS="$PROFILE_DIR/fmha-stalls.metrics.metrics.<pid>.metrics"

"$PROFILE_DIR/venv/bin/python" "$ANALYZER" \
  --shaderdump "$PROFILE_DIR/igc-dump" \
  --report "$PROFILE_DIR/fmha-stall-report.txt" \
  --output "$PROFILE_DIR/fmha-stall-chart.pdf" \
  "$METRICS"
```

The short forms accepted by PTI 2.4.0 are `-s`, `-r`, and `-o`:

```bash
"$PROFILE_DIR/venv/bin/python" "$ANALYZER" \
  -s "$PROFILE_DIR/igc-dump" \
  -r "$PROFILE_DIR/fmha-stall-report.txt" \
  -o "$PROFILE_DIR/fmha-stall-chart.pdf" \
  "$METRICS"
```

6. Verify the analyzer selected the intended specialization and shader entry:

```bash
grep -m1 "^Kernel:" "$PROFILE_DIR/fmha-stall-report.txt"
grep -m1 "^Assembly with instruction addresses:" \
  "$PROFILE_DIR/fmha-stall-report.txt"
grep -n -A12 -B2 "000020B0" "$PROFILE_DIR/fmha-stall-report.txt"
```

For this workload, the expected kernel contains
`FMHAProblemShape<false>` and `FMHAFwdMainloop<..., false, false, false, ...>`.
It maps to SIMD16 shader `entry_0009`. The report must identify `0x20b0` as
`sync.allwr`, its producers as Q/K block loads and BF16 DPAS instructions, and
the source locations as:

```text
applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp:361-366
include/cute/arch/copy_xe_2d.hpp:82,130
include/cute/arch/mma_xe.hpp:137
```

If source lines are absent, do not infer attribution from instruction
addresses alone. Check that:

- the target was rebuilt after enabling `IGC_ShaderDumpEnable`;
- `-g` appears in the compile command from
  `ninja -C build -t commands <target>`;
- the stall samples and shader dump came from the same rebuilt executable;
- the selected `.zeinfo` entry has the exact kernel template booleans, SIMD
  width, and GRF count;
- the analyzer consumed the device-metrics file rather than the host log.

After profiling, restore the normal Release flags and rebuild the target:

```bash
cmake -S . -B build -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"
ninja -C build -t clean 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128
ninja -C build 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128
```

The ComputeBasic metric query reported:

| Metric | Value |
|---|---:|
| GPU busy | 100.00% |
| Average clock | 2643 MHz |
| XVE active | 66.35% |
| XVE stall | 32.62% |
| XVE thread occupancy | 98.92% |
| Multiple-pipe active | 17.56% |
| ALU0 utilization | 21.30% |
| ALU1 utilization | 24.64% |
| ALU2/XMX utilization | 77.29% |
| Shared-function access hold | 0.58% |
| L3 stall | 0.037% |
| GPU memory read | 10.16 GB/s |
| GPU memory write | 4.20 GB/s |
| Memory request queue full | 0.00% |

Instruction-level sampling collected 123407 classified stall events:

| Stall class | Events | Share |
|---|---:|---:|
| Scoreboard ID dependency (`SbidStall`) | 50796 | 41.16% |
| Instruction-distance dependency (`DistStall`) | 44995 | 36.46% |
| Execution-pipe contention (`PipeStall`) | 19781 | 16.03% |
| Instruction fetch | 6038 | 4.89% |
| Control | 1714 | 1.39% |
| Synchronization | 83 | 0.07% |

The largest scoreboard hotspot was IP `0x20b0`, followed by `0x1ff0`,
`0x1f98`, `0x2048`, and `0x2088`. A second collection from a matching
full-debug build, analyzed against its IGC shader dump, attributed this region
to the QK GEMM loop:

| Instruction | Operation | Exact source attribution |
|---|---|---|
| `0x1ee0` | Transposed 2D load | `copy_xe_2d.hpp:130`; K copy initiated by `xe_fmha_fwd_mainloop.hpp:362` |
| `0x1f98` | Wait for load and DPAS dependencies | DPAS declared at `mma_xe.hpp:137`; QK GEMM initiated by `xe_fmha_fwd_mainloop.hpp:366` |
| `0x1fb0` | BF16 DPAS into FP32 accumulator | `mma_xe.hpp:137`; QK GEMM at `xe_fmha_fwd_mainloop.hpp:366` |
| `0x1ff0` | DPAS dependency wait | Same QK GEMM |
| `0x2000` | Transposed 2D load | `copy_xe_2d.hpp:130`; K copy at `xe_fmha_fwd_mainloop.hpp:362` |
| `0x2040` | BF16 DPAS into FP32 accumulator | `mma_xe.hpp:137`; QK GEMM at `xe_fmha_fwd_mainloop.hpp:366` |
| `0x2048` | DPAS dependency wait | Same QK GEMM |
| `0x2080` | BF16 DPAS into FP32 accumulator | `mma_xe.hpp:137`; QK GEMM at `xe_fmha_fwd_mainloop.hpp:366` |
| `0x2088` | DPAS dependency wait | Same QK GEMM |
| `0x2090` | Non-transposed BF16 2D load | `copy_xe_2d.hpp:82`; Q copy initiated by `xe_fmha_fwd_mainloop.hpp:361` |
| `0x20b0` | Wait for four outstanding dependencies | QK GEMM at `xe_fmha_fwd_mainloop.hpp:366` |

At `0x20b0`, `sync.allwr ($20,$21,$23,$25)` waits simultaneously for DPAS
instructions at `0x2040` and `0x2080`, a transposed K load at `0x2000`, and a Q
load at `0x2090`. The compiler then emits another four-instruction DPAS group
at `0x20c0`-`0x20d8`. The dominant hotspot is therefore the unrolled
`cute::gemm(mma_qk, tSrQ, tSrK, tSrS)` call at
`applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp:366`,
not softmax, PV GEMM, a workgroup barrier, or output storage.

The source attribution required rebuilding with `-g`; IntelLLVM 2026.1 warns
that `-gline-tables-only` is ignored for the `spir64_gen` device target.
PTI's generated address-tagged assembly and report are retained as
`fmha-igc-dump-debug/*entry_0009.asm.ip` and
`fmha-stall-debug-report.txt` in the session artifacts.

The profile rules out DRAM bandwidth, cache misses, barriers, low occupancy,
and clock throttling as primary limits. The historical kernel is
execution/dependency limited: XMX/ALU2 is busy, but only 17.56% of cycles overlap
multiple pipes and 77.62% of sampled stalls are scoreboard or instruction
dependency stalls.

Optimization experiments should therefore be prioritized as follows:

1. Reschedule the QK loop at `xe_fmha_fwd_mainloop.hpp:360-366` so independent
   DPAS accumulator chains cover the latency of the Q and K 2D loads instead of
   converging at the four-way wait at `0x20b0`.
2. Increase the Q/K software-pipeline distance or move independent payload
   preparation between the DPAS groups, subject to retaining zero spills.
3. Sweep `VTiles`, `DTiles`, and subgroup decomposition while rejecting any
   variant that spills; the current commit's 88-100 spills are already known to
   erase the historical kernel's advantage.
4. Preserve asynchronous K/V prefetch distance, but do not add more live
   payloads unless compiler diagnostics and `unitrace` still report zero spill
   memory.
5. Treat barrier and DRAM optimizations as low priority because synchronization
   is 0.07% of sampled stalls and external traffic is only 14.36 GB/s.

## GEMM

The BF16-input, FP32-accumulation/output GEMM target was built with:

```bash
ninja 00_bmg_gemm
```

Each shape was measured three times with 50 timed iterations and verification
disabled during timing.

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

Summary:

| Shape (M x N x K) | Average time (ms) | Average performance (TFLOP/s) |
|---|---:|---:|
| 4096 x 4096 x 4096 | **0.8050** | **170.740** |
| 5120 x 4096 x 4096 | **1.1206** | **153.309** |
| 8192 x 8192 x 8192 | **8.6599** | **126.966** |

Correctness passed at `1024x1024x1024`.

Unlike flash attention, GEMM does not show a uniform historical advantage. The
historical commit is faster for the measured 4096-cubed case, similar at 8192
cubed, and below the best current-commit result previously observed for the
default rectangular shape. The available measurements therefore do not
establish a GEMM performance regression between these commits.
