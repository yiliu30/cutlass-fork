# CUTLASS CuTe DSL IKET profiling reproducer

This directory reproduces the CUTLASS CuTe DSL IKET profiling guide example:

- Guide: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html
- Example: `examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py`

The reproducer was validated on an NVIDIA B200 host with eight B200 GPUs.

## Environment

- uv 0.13.9
- Python 3.12.15
- CUDA Toolkit 12.8 (`/usr/local/cuda-12.8`)
- `nvidia-cutlass-dsl[tools]==4.9.0.dev1`
- `torch==2.10.0+cu128`
- Exact package set: `requirements.lock`

`nvidia-cutlass-dsl-tools`, which provides the `run-iket` executable, was only
published for `4.9.0.dev1` at the time of this run. Stable `nvidia-cutlass-dsl`
`4.8.1` does not provide the `tools` extra.

The login environment on the validation host defaults to CUDA 12.0. The scripts
explicitly put `/usr/local/cuda-12.8/bin` and `/usr/local/cuda-12.8/lib64`
first so that both `nvcc` and the runtime libraries match the CUDA 12.8 PyTorch
wheel.

## Setup

```bash
cd iket_profile_repro
./setup.sh
```

`setup.sh` installs a project-local `uv` under `.uv/`, creates `.venv/`, and
installs `requirements.lock`. No shell profile changes are made.

## Run the documented example

```bash
./run.sh
```

The default problem size is `512,1024,64`, matching the guide's command. Override
it with `MNK`:

```bash
MNK=8192,8192,8192 ./run.sh
```

Outputs are written to `iket_output/`:

- `*.pftrace` - open at https://ui.perfetto.dev/
- `*.trace.json` - machine-readable event data
- `*.html` - local trace viewer
- `*.pftrace.gz` - compressed trace

The run defaults to `CUDA_VISIBLE_DEVICES=0`.

## Committed results

The committed trace in `results/` is from:

```text
--mnk 512,1024,64
CUDA_VISIBLE_DEVICES=0
```

It contains one kernel launch with:

- 2,832 IKET ranges
- 112 warp lifetimes
- range names including `kernel_e2e`, `prologue`, `tma_main`, `mma_main`,
  `mma_k_tile`, `mma_issue`, and the epilogue ranges

Open `results/iket_pid_0x4cd4d.pftrace` in Perfetto, or inspect
`results/iket_pid_0x4cd4d.trace.json` programmatically.

## IKET overhead A/B test

IKET events are stripped by default. When IKET lowering is enabled by
`run-iket`, instrumentation changes the generated kernel and has measurable
overhead.

`benchmark_ab.sh` follows the guide's recommendation: compare trace-reported
kernel durations between a minimal one-range variant and the full event set.

```bash
./benchmark_ab.sh
```

The script:

1. Generates `build/ab/fp16_gemm_4_iket_min.py` from the upstream example,
   keeping only the outer `kernel_e2e` range.
2. Runs the full and minimal variants twice each.
3. Writes `results/ab_metrics_reproduced.txt`.

Committed measurements from two runs each:

| Variant | Ranges | `kernel_e2e` median | `kernel_e2e` mean | Warp lifetime median |
| --- | ---: | ---: | ---: | ---: |
| Full instrumentation | 2,832 | 3.50 us | 2.89 us | 3.92 us |
| Minimal instrumentation | 112 | 2.76 us | 2.37 us | 3.22 us |

The extra event sites add roughly **22-27%** to the trace-reported kernel
duration for this small GEMM. This percentage is workload-specific. The delta
isolates the extra ranges; total IKET cost is higher because fixed per-kernel
entry/exit overhead is not represented as an IKET range.

For production performance, run without `run-iket` or `CUTE_DSL_COMPILER_OPT=iket`.
Use instrumented traces for relative phase analysis, not as production latency
or throughput benchmarks.

## Scripts

- `setup.sh` - create the local uv environment
- `run.sh` - run the documented Perfetto/JSON profile command
- `make_minimal.py` - generate the one-range A/B variant
- `benchmark_ab.sh` - run the full-vs-minimal overhead comparison
- `summarize_iket.py` - summarize trace-reported kernel durations
- `profile_uninstrumented.py` - optional torch profiler sanity check of the
  uninstrumented kernel
