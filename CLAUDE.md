# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUTLASS 4.4.1 is NVIDIA's header-only C++ template library for high-performance GEMM and related linear algebra on CUDA GPUs. It spans Volta (SM70) through Blackwell (SM100/SM120) architectures. CUTLASS 4.x adds **CuTe DSL**, a Python-native kernel programming interface built on CuTe abstractions that compiles to PTX via MLIR — no C++ required.

This clone lives inside a SageAttention3 kernel study workspace. The parent directory (`../`) contains Phase 1-5 study guides mapping CuTe concepts to the SA3 Blackwell kernel.

## Build Commands

### C++ (CMake) — builds examples, tests, profiler

```bash
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="90a"        # Hopper (use "80" for Ampere, "100a" for Blackwell SM100, "120a" for Blackwell GeForce)
make -j$(nproc)
```

Key CMake variables:
- `CUTLASS_NVCC_ARCHS` — semicolon-separated arch list (e.g. `"80;90a;100a;120a"`)
- `CUTLASS_TEST_LEVEL` — 0=sanity, 1=release, 2=exhaustive
- `CUTLASS_ENABLE_TESTS` / `CUTLASS_ENABLE_EXAMPLES` / `CUTLASS_ENABLE_PROFILER` — toggle build targets
- `CUTLASS_LIBRARY_KERNELS` — wildcard filter for profiler kernel instantiation (e.g. `cutlass_tensorop_s*gemm_f16_*`)
- `CUTLASS_ENABLE_HEADERS_ONLY=ON` — skip all build targets, header-only use

### Run Tests

```bash
cd build
make test_unit -j$(nproc)                              # all unit tests
./test/unit/cute/core/cutlass_test_unit_cute_core       # single test binary
./test/unit/cute/core/cutlass_test_unit_cute_core --gtest_filter="*Layout*"  # single test case
make test_unit_gemm -j$(nproc)                         # GEMM tests only
```

Tests use Google Test. Binaries mirror the namespace hierarchy under `test/unit/`.

### CUTLASS Profiler

```bash
make cutlass_profiler -j16
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*gemm_f16_* --m=3456 --n=4096 --k=4096
```

### Python Packages (3 separate packages)

```bash
# 1. PyCuTe — pure-Python CuTe layout algebra (no GPU needed)
pip install -e python/ --no-build-isolation
# or: cd python && pip install -e . (uses setup_pycute.py for pycute package + cutlass_library + cutlass_cppgen)

# 2. CuTe DSL — Python kernel compiler (nvidia-cutlass-dsl, requires GPU + CUDA 12.x)
cd python/CuTeDSL
./setup.sh              # CUDA 12 (default)
./setup.sh --cu13       # CUDA 13
# This installs nvidia-cutlass-dsl wheel + dependencies

# 3. PyCuTe standalone
cd python && pip install -e . --config-settings="--build-option=--pycute"
```

### Run CuTeDSL Examples

```bash
# Notebooks (Jupyter)
cd examples/python/CuTeDSL/notebooks
jupyter notebook hello_world.ipynb

# Standalone scripts
python examples/python/CuTeDSL/blackwell/dense_gemm.py
python examples/python/CuTeDSL/hopper/fmha.py
python examples/python/CuTeDSL/blackwell/fmha.py
```

## Architecture

### Two Eras: CUTLASS 2.x vs 3.x

**CUTLASS 2.x** (still present): `threadblock/` → `warp/` → `thread/` hierarchy with explicit tile iterators. Lives in `include/cutlass/gemm/threadblock/`, `warp/`, `thread/`.

**CUTLASS 3.x** (current): CuTe-based with three-layer GEMM decomposition:
- **Device layer** (`gemm/device/`): `GemmUniversalAdapter` — host-side launch wrapper
- **Kernel layer** (`gemm/kernel/`): tile scheduling, grid dimensions, kernel entry point
- **Collective layer** (`gemm/collective/`): the mainloop MMA — files named by arch like `sm90_mma_tma_gmma_ss_warpspecialized.hpp`

The `CollectiveBuilder` (`collective_builder.hpp`) auto-selects the right collective based on arch, data types, and tile shape.

### CuTe Core (`include/cute/`)

The layout algebra engine underpinning CUTLASS 3.x:
- **`layout.hpp`** / **`stride.hpp`** — `Layout<Shape, Stride>` — the fundamental abstraction mapping logical coordinates to memory offsets
- **`tensor.hpp`** — `Tensor = pointer + Layout` — a view over data
- **`atom/mma_atom.hpp`** — `Mma_Atom` / `TiledMma` — wraps hardware MMA instructions (HMMA, GMMA, UMMA, WGMMA) with layout information
- **`atom/copy_atom.hpp`** — `Copy_Atom` / `TiledCopy` — wraps copy instructions (TMA, cp.async, ldmatrix)
- **`algorithm/`** — `copy()`, `gemm()`, `axpby()`, etc. operating on CuTe tensors
- **`arch/`** — raw PTX wrappers organized by SM generation (`mma_sm90_gmma.hpp`, `copy_sm90_tma.hpp`, `mma_sm120.hpp`, etc.)

Layout composition (functional composition of layouts) is how CuTe derives thread↔data mappings, swizzled shared memory layouts, and TMA descriptors.

### Architecture-Specific Code Organization

Files are named with SM generation suffix throughout the codebase:
- `sm70` = Volta, `sm75` = Turing, `sm80` = Ampere, `sm89` = Ada, `sm90` = Hopper
- `sm100` = Blackwell datacenter (B200), `sm103` = Blackwell B300, `sm110` = DRIVE Thor
- `sm120` = Blackwell GeForce (RTX 50x0), `sm121` = DGX Spark

The `a` suffix (e.g. `90a`, `100a`) denotes architecture-accelerated features (GMMA, TMA, UMMA) whose PTX is not forward-compatible.

### CuTe DSL (`python/CuTeDSL/cutlass/`)

Python-native kernel compiler producing PTX via TVM/MLIR. Key modules:
- `cutlass.cute` — Python CuTe: `Layout`, `Tensor`, `copy()`, `gemm()`, atoms
- `cutlass.cute.arch` — PTX instruction wrappers (barriers, fences, MMA ops)
- `cutlass.cute.nvgpu` — GPU-specific helpers (TMA descriptors, TMEM allocator)
- `cutlass.cute.experimental` — higher-level APIs: fragment-free copy/dot, auto TMA descriptors, pipeline abstractions
- `cutlass.torch` — PyTorch integration (launch kernels from torch, auto-generate torch bindings)
- `cutlass.jax` — JAX integration

CuTeDSL examples are in `examples/python/CuTeDSL/` organized by arch: `hopper/`, `blackwell/`, `blackwell_geforce/`, `ampere/`, `notebooks/`.

### Key C++ Examples for Kernel Study

| Example | Architecture | Topic |
|---------|-------------|-------|
| `48_hopper_warp_specialized_gemm` | SM90 | Warp specialization pattern |
| `77_blackwell_fmha` | SM100 | Flash attention with CUTLASS 3.x |
| `88_hopper_fmha` | SM90 | Flash attention on Hopper |
| `93_blackwell_low_latency_gqa` | SM100 | GQA with flash decoding + cluster reduction |
| `70_blackwell_gemm` | SM100 | Basic Blackwell GEMM |
| `72_blackwell_narrow_precision_gemm` | SM100 | FP4/FP8 block-scaled GEMM |

### Warp Specialization Pattern (SM90+)

Critical pattern for high-performance kernels on Hopper/Blackwell:
- **Producer warpgroups**: issue TMA loads into shared memory, signal via barriers
- **Consumer warpgroups**: execute MMA (GMMA/WGMMA/UMMA) from shared memory, accumulate in registers
- Coordinated via `NamedBarrier` and `PipelineTmaAsync` (async pipeline with multiple stages)
- Register allocation split: producers get ~24 registers, consumers get ~232 registers (`warpgroup_reg_alloc<N>()`)

### Pipeline Abstractions (`include/cutlass/pipeline/`)

- `sm90_pipeline.hpp` — `PipelineTmaAsync<Stages>` for Hopper (TMA + barrier-based producer-consumer)
- `sm100_pipeline.hpp` — Blackwell pipelines (support for TMEM, programmatic dependent launch)

## Context: SageAttention3 Kernel Study

The parent directory (`../`) contains study guides that walk through the SA3 Blackwell kernel (`sageattention3_blackwell/sageattn3/blackwell/`) using CuTe concepts from this repo. The SA3 kernel uses:
- Custom SM120 NVFP4 MMA atom (CuTe `Mma_Atom` wrapping block-scaled FP4 WGMMA)
- TMA loads with 3-stage async pipeline
- Warp specialization (1 producer + 1 consumer warpgroup)
- Fused online softmax with FP4 quantization in the mainloop


### Dev env info
python: /home/yiliu7/workspace/envs/sage-local/bin/python

### Fcous
The project is forked from the cutlass for study purpose, the user do not have much background 
in cutlass/cute, but know the basic of cuda/ptx programming. Please Take this in mind ALWAYS!