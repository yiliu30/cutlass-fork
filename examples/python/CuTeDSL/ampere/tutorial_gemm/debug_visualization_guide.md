# CuTe DSL Debug & Visualization Tools for GEMM Kernels

A practical reference for inspecting layouts, data movement, and kernel behavior
at every stage of a CuTe DSL GEMM — from host-side layout construction through
GPU execution to PTX generation.

---

## 1. Compile-Time Introspection (`@cute.jit` Host Side)

These tools work inside `@cute.jit` functions where layouts, tiled MMA, and
tiled copy objects are constructed. They execute at compile time (JIT trace time)
and print to stdout on the host.

### Layout Properties

```python
@cute.jit
def host_gemm(mA, mB, mC):
    sA_layout = cute.make_layout((128, 32, 3), stride=(1, 128, 128 * 32))

    print(sA_layout)                # Full layout structure
    print(sA_layout.type)           # Compile-time type (shape + stride encoding)
    print(cute.size(sA_layout))     # Total number of elements
    print(cute.shape(sA_layout))    # Shape tuple, e.g. (128, 32, 3)
    print(cute.stride(sA_layout))   # Stride tuple, e.g. (1, 128, 4096)
    print(cute.rank(sA_layout))     # Number of modes (3 for the above)
    print(cute.depth(sA_layout))    # Nesting depth of hierarchical layouts
    print(cute.cosize(sA_layout))   # Range of the layout mapping (max offset + 1)
```

### TiledMma / TiledCopy Type Inspection

```python
    print(tiled_mma.type)           # MMA atom + thread layout + value layout
    print(tiled_copy_A.type)        # Copy atom + thread/value decomposition
```

This is particularly useful for verifying that `make_tiled_mma` and
`make_tiled_copy_tv` produced the expected configuration before launching.

### LaTeX Visualization (TiKZ Diagrams)

```python
    cute.print_latex(sA_layout)          # Layout as a colored grid
    cute.print_latex_tv(tiled_copy_A)    # Thread-Value decomposition diagram
    cute.print_latex_tv(tiled_mma)       # MMA thread-value mapping
```

**How to use:** These output LaTeX/TiKZ source code to stdout. Save it to a `.tex`
file, compile with `pdflatex`, and you get a visual diagram showing which thread
owns which data elements — color-coded by thread ID.

Example workflow:
```bash
python step4_tensorop_gemm.py --mnk 128,128,128 2>&1 | grep -A 1000 'begin{tikz' > layout.tex
pdflatex layout.tex
open layout.pdf
```

---

## 2. Runtime Debugging (`@cute.kernel` GPU Side)

### `cute.printf` — GPU Thread Printing

```python
@cute.kernel
def gemm_kernel(mA, mB, mC, ...):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # ALWAYS guard with thread/block ID checks to avoid output flood
    if tidx == 0 and bidx == 0 and bidy == 0:
        cute.printf("Block (0,0), Thread 0\n")
        cute.printf("k_tile_count = %d\n", k_tile_count)
        cute.printf("num_k_blocks = %d\n", num_k_blocks)
```

**Key notes:**
- Uses C-style format strings: `%d` (int), `%f` (float), `%x` (hex), `%u` (unsigned)
- **Always** guard with `if tidx == 0 and bidx == 0` or you'll get 128+ lines per print
- Output may be interleaved across warps; use `cute.arch.sync_threads()` before
  printf for deterministic ordering within a block
- Flushed on `cudaDeviceSynchronize()` (i.e., `torch.cuda.synchronize()`)

### `cute.print_tensor` — Dump Tensor Data + Layout

```python
    if tidx == 0 and bidx == 0:
        cute.print_tensor(tCrC)                 # Register fragment values + layout
        cute.print_tensor(sA)                    # Shared memory tile contents
        cute.print_tensor(gA[None, None, 0])     # First k-tile of global A
```

This prints both **data values** and **layout metadata**. Extremely useful for
verifying that data lands in the right location after G2S, S2R, or epilogue copies.

### Runtime Assertions

```python
from cutlass.cute import assert_

@cute.kernel
def gemm_kernel(...):
    assert_(k_tile_count > 0, "k_tile_count must be positive")
    assert_(num_k_blocks == 2, "Expected 2 k-blocks for BK=32, MMA_K=16")
```

Must be enabled at compile time:
```python
compiled = cute.compile(host_gemm, mA, mB, mC, enable_assertions=True)
```

---

## 3. Thread-Data Mapping Visualization

### Identity Tensor Trick

The most powerful technique for understanding how threads map to output elements:

```python
@cute.kernel
def debug_mapping_kernel(mC, tiled_mma):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Create identity tensor: value at position (i,j) encodes (i,j)
    cC = cute.make_identity_tensor(cute.make_shape(BM, BN))

    thr_mma = tiled_mma.get_slice(tidx)
    tCcC = thr_mma.partition_C(cC)  # Thread's view of identity

    if tidx == 0 and bidx == 0:
        cute.print_tensor(tCcC)  # Shows which (m,n) coordinates thread 0 owns
    if tidx == 1 and bidx == 0:
        cute.print_tensor(tCcC)  # Compare with thread 1
```

This reveals the exact output element coordinates each thread is responsible
for — critical for understanding MMA partitioning and epilogue copy patterns.

### Partition Shape Inspection

```python
    if tidx == 0 and bidx == 0:
        cute.printf("tCrA mode-2 size (k_blocks) = %d\n", cute.size(tCrA, mode=[2]))
        cute.printf("tCsA mode-3 size (stages)   = %d\n", cute.size(tCsA, mode=[3]))
        cute.printf("tAgA mode-3 size (k_tiles)   = %d\n", cute.size(tAgA, mode=[3]))
```

---

## 4. Compiler & IR Inspection (Environment Variables)

Set these before running your script to inspect compilation internals:

| Variable | Effect |
|----------|--------|
| `CUTE_DSL_PRINT_IR=1` | Print full MLIR intermediate representation |
| `CUTE_DSL_KEEP_PTX=1` | Save generated PTX assembly to disk |
| `CUTE_DSL_KEEP_CUBIN=1` | Save generated CUBIN binary to disk |
| `CUTE_DSL_DRYRUN=1` | Compile only — don't execute (fast syntax check) |
| `CUTE_DSL_GENERATE_LINE_INFO=1` | Add source line info for NCU correlation |
| `CUTE_DSL_PRINT_PASS_TIMING=1` | Print compiler pass names and durations |

### Usage Examples

```bash
# Inspect MLIR IR to understand lowering
CUTE_DSL_PRINT_IR=1 python step4_tensorop_gemm.py --mnk 128,128,128

# Save PTX to check instruction selection
CUTE_DSL_KEEP_PTX=1 python step4_tensorop_gemm.py --mnk 128,128,128
grep -E "mma\.|ldmatrix|cp\.async" *.ptx   # Find key instructions

# Fast compile-only check (no GPU execution)
CUTE_DSL_DRYRUN=1 python step5_vectorized_gemm.py --mnk 128,128,128
```

### Programmatic Compile Options

```python
compiled = cute.compile(
    host_gemm, mA, mB, mC,
    generate_line_info=True,    # For NCU source correlation
    keep_ptx=True,              # Save PTX file
    enable_assertions=True,     # Enable runtime assert_()
)
```

---

## 5. Profiling Integration

### NVIDIA Nsight Compute (NCU)

```bash
# Full profile with source correlation
CUTE_DSL_GENERATE_LINE_INFO=1 ncu --set full \
    python step5_vectorized_gemm.py --mnk 512,512,512
```

### Built-in Benchmarking

```python
from cutlass.cute import benchmark

ms = benchmark(compiled, mA, mB, mC, warmup=10, repeat=100)
flops = 2 * M * N * K / (ms * 1e-3) / 1e12  # TFLOPS
print(f"  {ms:.3f} ms, {flops:.1f} TFLOPS")
```

---

## 6. Practical Debugging Recipes for the Tutorial Steps

### Recipe 1: "Is my data landing in the right SMEM location?" (Steps 2-5)

```python
# After G2S copy completes, before MMA:
cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()
if tidx == 0 and bidx == 0:
    cute.print_tensor(sA)   # Dump entire SMEM tile for A
    cute.print_tensor(sB)   # Dump entire SMEM tile for B
```

### Recipe 2: "What does my accumulator contain after 1 k-tile?" (All steps)

```python
for k_tile in range(k_tile_count):
    # ... normal compute ...

    # Debug: print accumulator and early-exit after first tile
    if k_tile == 0 and tidx == 0 and bidx == 0:
        cute.print_tensor(tCrC)
    if k_tile == 0:
        return   # Early exit for fast debugging
```

### Recipe 3: "Is my swizzle correct?" (Steps 4-5)

```python
# In @cute.jit host function:
print("sA_layout:", sA_layout)
print("sA_layout type:", sA_layout.type)
cute.print_latex(sA_layout)   # Generate visual diagram
```

Look for: consecutive threads reading consecutive SMEM rows should hit different
banks. The LaTeX color diagram makes bank-conflict patterns visible at a glance.

### Recipe 4: "Which registers does ldmatrix load into?" (Steps 4-5)

```python
# After S2R copy via ldmatrix:
if tidx == 0 and bidx == 0:
    cute.printf("=== tCrA_copy after ldmatrix ===\n")
    cute.print_tensor(tCrA_copy)
    cute.printf("=== tCrA (MMA view) ===\n")
    cute.print_tensor(tCrA)
```

Comparing these two views shows how `retile()` maps the ldmatrix register
layout back to the MMA fragment layout.

### Recipe 5: "Verify SMEM epilogue: RMEM -> SMEM -> GMEM" (Step 5)

```python
# Epilogue section:
if tidx == 0 and bidx == 0:
    cute.printf("=== FP32 accumulator ===\n")
    cute.print_tensor(tCrC)
    cute.printf("=== After FP16 convert ===\n")
    cute.print_tensor(tCrD)

cute.autovec_copy(tCrD, tCsC)    # RMEM -> SMEM
cute.arch.sync_threads()

if tidx == 0 and bidx == 0:
    cute.printf("=== SMEM after R2S ===\n")
    cute.print_tensor(sC)         # Verify SMEM contents match
```

### Recipe 6: "Inspect generated PTX for correct instruction selection" (Steps 4-5)

```bash
CUTE_DSL_KEEP_PTX=1 python step4_tensorop_gemm.py --mnk 128,128,128

# Check for expected instructions:
grep "mma.sync"   *.ptx   # Should see mma.sync.aligned.m16n8k16
grep "ldmatrix"   *.ptx   # Should see ldmatrix.sync.aligned.m8n8
grep "cp.async"   *.ptx   # Should see cp.async.ca.shared.global

# For step 5, also check vectorization width:
grep "cp.async.*16"  *.ptx   # 128-bit = 16 bytes
```

### Recipe 7: "Understand the copy thread layout" (Steps 2-5)

```python
# In @cute.jit, after constructing tiled_copy_A:
print("tiled_copy_A type:", tiled_copy_A.type)
cute.print_latex_tv(tiled_copy_A)   # Visual thread-value decomposition
```

This shows exactly how the 128 (or 256) threads cooperate to load a
(BM, BK) tile — which thread loads which elements.

---

## Quick Reference Table

| Tool | Where | What It Shows | Best For |
|------|-------|---------------|----------|
| `print(layout)` | `@cute.jit` | Layout structure | Verify shapes & strides |
| `.type` property | `@cute.jit` | Compile-time type | Check MMA/copy config |
| `cute.print_latex()` | `@cute.jit` | TiKZ color grid | Visualize data layout |
| `cute.print_latex_tv()` | `@cute.jit` | Thread-value map | Understand thread ownership |
| `cute.printf()` | `@cute.kernel` | Scalar values | Runtime value inspection |
| `cute.print_tensor()` | `@cute.kernel` | Tensor data + layout | Verify copy correctness |
| `make_identity_tensor()` | `@cute.kernel` | Coordinate mapping | Thread-to-element mapping |
| `assert_()` | `@cute.kernel` | Runtime invariant check | Catch logic errors early |
| `CUTE_DSL_PRINT_IR=1` | Shell env | MLIR passes | Compiler debugging |
| `CUTE_DSL_KEEP_PTX=1` | Shell env | PTX assembly | Instruction verification |
| `CUTE_DSL_DRYRUN=1` | Shell env | Compile-only | Fast syntax/type checking |
| `CUTE_DSL_GENERATE_LINE_INFO=1` | Shell env | Source correlation | NCU profiling |
| `benchmark()` | Python | Kernel timing (ms) | Performance measurement |
