# Step 1 Study Notes: `make_tiled_mma` and `permutation_mnk`

## The Three Inputs to `make_tiled_mma`

```python
op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)                    # 1. atom
atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))    # 2. atom tiling
permutation_mnk = (perm_M, perm_N, None)                           # 3. data reordering
tiled_mma = cute.make_tiled_mma(op, atoms_layout, permutation_mnk)
```

### 1. `op` — The MMA Atom

The smallest indivisible hardware operation. Wraps one PTX instruction and encodes its thread↔data contract as CuTe layouts.

| Level | Example | Instruction | Shape | Threads |
|-------|---------|-------------|-------|---------|
| Scalar SIMT | `MmaUniversalOp(Float32)` | `fma.rn.f32` | 1×1×1 | 1 |
| Tensor Core | `MmaF16BF16Op(FP16,FP32,(16,8,16))` | `mma.sync.m16n8k16` | 16×8×16 | 32 |

Step 1 uses the scalar atom: 1 thread, 1×1×1 shape, trivial TV layouts `(1,1):(0,0)`.

### 2. `atoms_layout` — How Atoms Tile the CTA

```
(16, 16, 1):(16, 1, 0)  →  16×16 = 256 atoms in M×N grid
thread_id = m_pos × 16 + n_pos
```

Each atom (=thread for scalar FMA) is assigned a position in the M×N output tile. With BM=128, BN=128 and 16×16 atoms, each thread initially covers `(128/16) × (128/16) = 8×8 = 64` output elements.

### 3. `permutation_mnk` — Data Reordering for Spatial Locality

```python
perm_M = cute.make_layout((F, R), stride=(R, 1))  # (16, 4):(4, 1)
perm_N = cute.make_layout((F, R), stride=(R, 1))  # (16, 4):(4, 1)
```

**What it does**: Reorganizes each dimension so that each thread's elements are grouped into R consecutive elements instead of being scattered.

**Key parameters**:
- **R = 4**: Consecutive group size. Chosen to match 128-bit vector load width (4 × float32 = 128 bits → `ld.global.v4.f32`)
- **F = 16**: Distribution factor. Must equal atoms_layout dimension (16) for R consecutive elements to actually be contiguous in memory
- **F × R = 64**: Coverage per permutation tile

**Without permutation** (`(16,1):(1,1)`): each thread's 8 M-elements are individually scattered → 8 scalar loads per dimension.

**With permutation** (`(16,4):(4,1)`): each thread's 8 M-elements form 2 blocks of 4 consecutive → 2 vector loads per dimension.

**PTX evidence**:
| Config | Load instructions | Type |
|--------|-------------------|------|
| `(16,4):(4,1)` | 32× `ld.global.v4.f32` | 128-bit vector |
| `(16,1):(1,1)` | 128× `ld.global.f32` | 32-bit scalar |

Both produce identical 64× `fma.rn.f32` — the permutation only affects memory access patterns, not compute.

---

## How `permutation_mnk` Works Internally: `thrfrg_C`

The function `thrfrg_C` in `mma_atom.hpp` (line 252) transforms a C tile from `(M, N)` to `((ThrV,(ThrM,ThrN)), (FrgV,(RestM,RestN)))` in 4 steps.

**Concrete trace for step1** (C tile = 128×128, perm = (16,4):(4,1), atom = 1×1, threads = 16×16):

```
Step 1: logical_divide by permutation
  (128, 128)  →  (((16,4),2), ((16,4),2))
  Each flat dim split: 16 groups × 4 consecutive × 2 outer blocks

Step 2: zipped_divide by atom shape (1,1)
  →  ((1,1), ((16,4),2, (16,4),2))
  Trivial for scalar atom — just separates atom-interior from rest

Step 3: compose with AtomLayoutC_TV (1,1):(0,0)
  →  ((ThrV=1, FrgV=1), (RestM, RestN))
  Trivial for scalar atom — maps atom (M,N) to (Thread, Value)

Step 4: zipped_divide by thread layout (16, 16)
  →  ((1, (16,16)), (1, (8,8)))
      ↑ threads      ↑ per-thread data
      256 total       64 each

  RestM = 8 = 4 consecutive × 2 blocks  ← from R=4
  RestN = 8 = 4 consecutive × 2 blocks  ← from R=4
```

**Final shape**: `((ThrV=1, ThrM=16, ThrN=16), (FrgV=1, RestM=8, RestN=8))`
- 256 threads × 64 elements = 16384 = 128×128 ✓
- Each thread's 64 elements are in **2×2 blocks of 4×4 consecutive** positions

### Key insight

The permutation's R parameter shapes `RestM` and `RestN` — the per-thread data layout. With R=4, each rest dimension becomes `(4_consecutive, 2_blocks)` instead of `(1, 8_scattered)`, enabling vector memory access.

---

## `local_tile` — CTA-Level Tensor Partitioning

### Why it exists

Each CTA must extract its piece of A, B, C from global memory. The three matrices share one tiling decision `(BM, BN, BK)` but each lives in a different subspace of M×N×K. `local_tile` solves this with a single, layout-independent abstraction.

### Signature

```python
gX = cute.local_tile(tensor, tiler, coord, proj)
```

### The 3 operations (in order)

**1. `proj` — project 3D problem onto this matrix's dimensions** via `dice`:
- `1` → keep that dimension
- `None` → drop it

```
cta_tiler = (BM=128, BN=128, BK=8)
cta_coord = (bidx,   bidy,   None)

A (M,K): proj=(1,None,1) → tiler=(128,8),   coord=(bidx,None)
B (N,K): proj=(None,1,1) → tiler=(128,8),   coord=(bidy,None)
C (M,N): proj=(1,1,None) → tiler=(128,128), coord=(bidx,bidy)
```

**2. `zipped_divide` — split each dimension into (tile, num_tiles)**:

```
mA = (256, 32):(1, 256)  with tiler (128, 8)
→ ((128, 8), (2, 4))
    ↑one tile  ↑how many tiles (M-blocks × K-tiles)
```

**3. `coord` slicing — select this CTA's block, keep loop dims**:
- Integer coord → fix that "which tile" dimension (select block)
- `None` coord → keep as a loop dimension (extra mode in result)

### Concrete results (block 0,0; M=256, N=128, K=32)

```
gA = (128, 8, 4):(1, 256, 2048)     — BM × BK × num_k_tiles
gB = (128, 8, 4):(1, 128, 1024)     — BN × BK × num_k_tiles
gC = (128, 128):(128, 1)            — BM × BN (fully sliced, no loop dim)
```

The 3rd mode of gA/gB (num_k_tiles=4) is the K-loop dimension:
```python
for k_tile in range(num_k_tiles):        # 0,1,2,3
    tCgA = thr_mma.partition_A(gA[None, None, k_tile])  # (128,8) slice
    tCgB = thr_mma.partition_B(gB[None, None, k_tile])  # (128,8) slice
```

### Key design property

Tile shape is defined **once** as `(BM,BN,BK)`. Changing it (e.g. to `(64,256,16)`) automatically updates all three matrices' views with zero kernel code changes — strides and offsets are derived from the tensor's layout, not hardcoded.

---

## Thread-Level Partitioning: `get_slice`, `partition_C/A/B`, `make_fragment_C`

### `get_slice(tidx)` — Thread ID → Per-Thread Cursor

Converts flat thread index to a 4D `(V,M,N,K)` coordinate via `thr_layout_vmnk_.get_flat_coord(thr_idx)`. Returns a `ThrMMA` — a per-thread cursor that knows which data elements belong to this thread.

### `partition_C/A/B` — Extract This Thread's Data

Each calls the corresponding `thrfrg_*` function (same 4-step pipeline, different MNK dimensions), then slices by this thread's coordinate:

| Function | Dims | `thrfrg_*` uses | Output modes |
|----------|------|-----------------|--------------|
| `partition_C(gC)` | (M,N) | perm_M, perm_N, ThrM, ThrN | **(V, M, N)** |
| `partition_A(gA)` | (M,K) | perm_M, perm_K, ThrM, ThrK | **(V, M, K)** |
| `partition_B(gB)` | (N,K) | perm_N, perm_K, ThrN, ThrK | **(V, N, K)** |

Mode order is always **(V, then matrix's two dimensions in order)**. K is always mode 2 for A and B — enabling uniform `[None, None, k_block]` indexing.

### Concrete partition results (thread 0, block 0,0)

```
tCgC: (1, (4,2), (4,2)):(0, (128,8192), (1,64))   — 64 C-elements in GMEM
tCgA: (1, (4,2), 8):(0, (1,64), 256)               — 64 A-elements in GMEM
tCgB: (1, (4,2), 8):(0, (1,64), 256)               — 64 B-elements in GMEM
```

The `(4,2)` decomposition in M/N modes = permutation's R=4 effect: 2 blocks of 4 consecutive elements.

### `make_fragment_C(tCgC)` — Register Accumulator

Static method on `TiledMMA` (not per-thread). Allocates a fresh register tensor with **same shape** as input but **compact strides** in RMEM:

```
tCrC: (1, (4,2), (4,2)):(0, (1,4), (8,32))   — same shape, packed strides
```

### CuTe Naming Convention

```
t  C  g  C
↑  ↑  ↑  ↑
|  |  |  └── tensor identity: C (or A, B)
|  |  └───── memory space: g=global, r=register, s=shared
|  └──────── partitioned by: C-partition layout (MMA's mapping)
└─────────── "t" = thread-partitioned
```

---

## GMEM → RMEM Copy: `make_fragment_like` and `autovec_copy`

### `tCrA = cute.make_fragment_like(tCgA)`

Allocates register tensor with **same shape, compact strides**:

```
tCgA: (1,(4,2),8):(0,(1,64),256)    — GMEM strides (from A's memory layout)
tCrA: (1,(4,2),8):(0,(1,4), 8)      — RMEM strides (packed contiguous)
```

### `cute.autovec_copy(tCgA, tCrA)` — Auto-Vectorizing Copy

Automatically determines the widest safe vector instruction by computing GCD of:
1. **Common contiguous elements** between src and dst layouts
2. **Pointer alignment** (align<16> = 128 bits)
3. **Hardware cap** (256 bits max)

For step1: 4 contiguous f32 × 32 bits = 128 bits → `ld.global.v4.f32`

Per thread per K-tile: 2 vector loads for M (8 elements in (4,2) groups) × 8 K-columns = **16 vector loads**.

---

## Inner K-Block Loop and `cute.gemm`

### Two-Level K-Tiling

```
Total K ──── Outer loop (k_tile): BK-sized tiles ──── Inner loop (k_block): atom-K-sized blocks
  32            4 iterations (BK=8)                      8 iterations (atom_K=1)
```

Outer loop manages **data movement** (GMEM→RMEM), inner loop manages **compute granularity**.

### `num_k_blocks = cute.size(tCrA, mode=[2])`

Mode 2 = K-dimension of `(V, M, K)`. For step1: K=8 (8 elements per BK tile, atom_K=1).

### `cute.gemm(tiled_mma, tCrC, tCrA[...,k], tCrB[...,k], tCrC)`

Semantics: **D = A × B + C** (in-place, D and C are both `tCrC`)

```python
tCrA[None, None, k_block]   # (V, M, K) → (V, M) = (1, (4,2)) → 8 values
tCrB[None, None, k_block]   # (V, N, K) → (V, N) = (1, (4,2)) → 8 values
```

For scalar 1×1×1 atom: computes outer product → **8 × 8 = 64 FMAs** per call.

```
Total FMAs per thread: 4 k_tiles × 8 k_blocks × 64 FMAs = 2,048
All 256 threads: 2,048 × 256 = 524,288 = 2×M×N×K/2 ✓
```

---

## Epilogue: `cute.copy` with `CopyUniversalOp`

### `atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)`

`CopyUniversalOp` = scalar copy atom (`st.global.f32`). The simplest copy instruction, analogous to `MmaUniversalOp` for compute.

### `cute.copy(atom, tCrC, tCgC)`

Copies 64 accumulator elements from registers to global memory. Uses scalar stores:

```
tCrC: (1,(4,2),(4,2)):(0,(1,4),(8,32))       — register (packed)
tCgC: (1,(4,2),(4,2)):(0,(128,8192),(1,64))   — global (strided back to C's layout)
```

Same shape ensures element-to-element correspondence; different strides handle address translation.

### `autovec_copy` vs `cute.copy`

| | `autovec_copy` | `cute.copy(atom, ...)` |
|---|---|---|
| Atom | Auto-constructed from layout analysis | User provides explicitly |
| In step1 | Loads: GMEM→RMEM (vectorized) | Epilogue: RMEM→GMEM (scalar) |
| PTX | `ld.global.v4.f32` | `st.global.f32` |

The scalar epilogue stores are the first optimization target for later steps (step 5 adds SMEM staging for vectorized stores).

---

## Step 1 Complete Data Path

```
GMEM ──autovec_copy──→ RMEM ──cute.gemm──→ RMEM ──cute.copy──→ GMEM
 (A,B)  (ld.v4.f32)   (tCrA,B) (fma.rn)  (tCrC) (st.f32)     (C)
         16 vec loads            64 FMAs/call      64 scalar stores
         per k-tile              per k-block       per CTA tile
```
