# Example 03: TV Layout Elementwise Add — Study Notes

## Problem: Manual Index Math Doesn't Scale

Example 2 (zipped_divide) hardcodes thread-to-data mapping:

```python
ni = thread_idx % n      # manual math
mi = thread_idx // n     # tightly coupled to layout
a_val = gA[(None, (mi, ni))].load()
```

Change vectorization width or thread arrangement → rewrite the kernel.

## Solution: TV Layout — Decouple Thread Mapping from Data Layout

TV Layout = rank-2 layout: `(thread_id, value_id) → tile coordinate`

- **Host** constructs the mapping (what goes where)
- **Kernel** just indexes `(tidx, None)` — no manual math

---

## Two-Level Tiling Architecture

```
Full Tensor (M, N)
    │
    │ Level 1: zipped_divide(mA, tiler_mn)        ← HOST
    │ Split into block-sized tiles
    ▼
  gA: ((TileM, TileN), (RestM, RestN))
       ═══════════════  ══════════════
       one block's tile  grid of blocks
    │
    │ Level 2: composition(blkA, tv_layout)       ← KERNEL
    │ Map threads to elements within tile
    ▼
  tidfrgA: (num_threads, values_per_thread) → physical address
           ═══════════  ═════════════════
           which thread  which elements
```

---

## TV Layout Construction (Small Config)

### Step 1: Thread Layout — how threads are arranged

```python
thr_layout = cute.make_ordered_layout((2, 4), order=(1, 0))
# Result: (2, 4) : (4, 1)
```

`order=(1, 0)`: dimension with order 0 gets stride 1 (fastest).

```
         stride 1 (fast) ──►
         c0   c1   c2   c3
  r0:    T0   T1   T2   T3      stride 4 (slow)
  r1:    T4   T5   T6   T7         │
                                    ▼
```

T0-T3 are contiguous columns → **coalesced warp access**.

### Step 2: Value Layout — how many elements per thread

```python
coalesced_ldst_bytes = 8                                          # LD.64 = 8 bytes
val_layout = cute.make_ordered_layout((2, coalesced_ldst_bytes), order=(1, 0))  # (2, 8) bytes
val_layout = cute.recast_layout(dtype.width, 8, val_layout)       # → (2, 4) FP16
# Result: (2, 4) : (4, 1)
```

`recast_layout(dst_bits=16, src_bits=8, layout)`: converts byte units → element units.
Same code works for any dtype:

| dtype | bytes/elem | 8 bytes → elements |
|-------|-----------|-------------------|
| FP8   | 1         | 8                 |
| FP16  | 2         | 4                 |
| FP32  | 4         | 2                 |

Each thread's data footprint:

```
  v0  v1  v2  v3    ← 4 contiguous FP16 = one LD.64
  v4  v5  v6  v7    ← 4 contiguous FP16 = one LD.64
                     = 8 values, 2 loads per thread
```

### Step 3: Combine — `make_layout_tv`

```python
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
# tiler_mn  = (4, 16)                              ← tile size per block
# tv_layout = ((4,2),(4,2)):((16,2),(4,1))          ← (tid, vid) → tile coord
```

**tiler_mn** = `(thr_rows × val_rows, thr_cols × val_cols)` = `(2×2, 4×4)` = **(4, 16)**

**tv_layout** shape `((4,2),(4,2))`: dims are reordered **fastest first**:

```
Input:  thr = (rows=2, cols=4):(4, 1)
                        ^^^^     ^
                        fast     stride 1

TV output: thread mode = (cols=4, rows=2) = (4, 2)
                          ^^^^
                          fast first (stride-1 dim first)
```

This is NOT (4 rows, 2 cols). It's (4=fast_dim, 2=slow_dim), ordered by speed.

### Combined Tile: Which Thread Owns What

```
Tile (4 × 16):

     c0  c1  c2  c3 │ c4  c5  c6  c7 │ c8  c9 c10 c11 │ c12 c13 c14 c15
r0:  T0  T0  T0  T0 │ T1  T1  T1  T1 │ T2  T2  T2  T2 │ T3  T3  T3  T3
r1:  T0  T0  T0  T0 │ T1  T1  T1  T1 │ T2  T2  T2  T2 │ T3  T3  T3  T3
r2:  T4  T4  T4  T4 │ T5  T5  T5  T5 │ T6  T6  T6  T6 │ T7  T7  T7  T7
r3:  T4  T4  T4  T4 │ T5  T5  T5  T5 │ T6  T6  T6  T6 │ T7  T7  T7  T7

8 threads × 8 values = 64 elements = 4 × 16 ✓
```

---

## Kernel Logic — 3 Steps

```python
# Step 1: Get this block's tile
blkA = gA[((None, None), bidx)]              # (4,16):(32,1)

# Step 2: Compose with TV layout
tidfrgA = cute.composition(blkA, tv_layout)  # (tid,vid) → address

# Step 3: Slice this thread's values
thrA = tidfrgA[(tidx, None)]                 # vid → address
thrC[None] = thrA.load() + thrB.load()       # vectorized load/add/store
```

No manual index math. No knowledge of tile size in the kernel.

---

## Composition Traced (Small Config, M=4, N=32)

### Input

```
blkA:      (4, 16) : (32, 1)               — tile coord → memory offset
tv_layout: ((4,2),(4,2)) : ((16,2),(4,1))   — (tid, vid) → tile coord
```

### Output

```
tidfrgA:   ((4,2),(4,2)) : ((4,64),(1,32))  — (tid, vid) → memory offset
```

Shape stays same. Strides change to physical memory offsets:

```
offset = t0*4 + t1*64 + v0*1 + v1*32

t0*4:  next thread-col → jumps 4 elements (= val_cols)
t1*64: next thread-row → jumps 2 rows (= val_rows × N = 2 × 32)
v0*1:  next element in row (contiguous → enables LD.64)
v1*32: next row for same thread (= N)
```

### Concrete Trace

```
T0 (t0=0,t1=0):  offsets 0,1,2,3,32,33,34,35        → row 0-1, col 0-3   ✓
T1 (t0=1,t1=0):  offsets 4,5,6,7,36,37,38,39        → row 0-1, col 4-7   ✓
T4 (t0=0,t1=1):  offsets 64,65,66,67,96,97,98,99    → row 2-3, col 0-3   ✓
```

Visually in memory:

```
       c0  c1  c2  c3 │ c4  c5  c6  c7 │...│ c12 c13 c14 c15
       ────────────────┼────────────────┼   ┼────────────────
r0:    T0  T0  T0  T0 │ T1  T1  T1  T1 │   │ T3  T3  T3  T3
       0   1   2   3  │ 4   5   6   7  │   │ 12  13  14  15
       ── v0 stride=1►  ─ t0 stride=4 ►

r1:    T0  T0  T0  T0 │ T1  T1  T1  T1 │   │ T3  T3  T3  T3
       32  33  34  35  │ 36  37  38  39 │   │ 44  45  46  47
       ▲ v1 stride=32 (next row for same thread)

r2:    T4  T4  T4  T4 │ T5  T5  T5  T5 │   │ T7  T7  T7  T7
       64  65  66  67  │ 68  ...        │   │
       ▲ t1 stride=64 (next thread-row = 2 × 32)

r3:    T4  T4  T4  T4 │ T5  ...
       96  97  98  99  │
```

---

## After Thread Slicing: `thrA`

```
thrA = tidfrgA[(tidx, None)]

thrA: (4, 2) : (1, 32)
       ↑  ↑     ↑   ↑
       │  │     │   └─ stride 32 = next row (= N)
       │  │     └───── stride 1 = contiguous
       │  └─────────── 2 rows
       └────────────── 4 elements per row

IMPORTANT: (4, 2) is NOT (4 rows, 2 cols).
It's (fast_dim=4, slow_dim=2), ordered by stride:
  sub-dim 0: size=4, stride=1  → 4 contiguous elements (one LD.64)
  sub-dim 1: size=2, stride=32 → 2 groups (2 rows apart)
```

`thrA.load()` emits:

```
Load 1: LD.64 → v0, v1, v2, v3  (4 contiguous FP16)
Load 2: LD.64 → v4, v5, v6, v7  (4 contiguous FP16, +32 elements away)
```

### Alignment: `align<16>` → `align<8>`

```
T0: base + 0  bytes → 16-byte aligned
T1: base + 8  bytes → 8-byte aligned only  ← worst case
T2: base + 16 bytes → 16-byte aligned
T3: base + 24 bytes → 8-byte aligned only

Conservative: align<8> → LD.64 (not LD.128)
```

---

## CuTe Type Annotations Explained

```
                ┌─── Element type: FP16 (2 bytes)
                │     ┌─── Memory space: GPU DRAM
                │     │     ┌─── Pointer alignment
                │     │     │
                ▼     ▼     ▼
!cute.memref<   f16,  gmem, align<16>,  "((4,16),(1,2)):((32,1),(0,16))" >
                                          │        │      │        │
                                          │        │      │        └─ tile grid stride
                                          │        │      └─ tile interior stride
                                          │        └─ tile grid shape (mode 1)
                                          └─ tile interior shape (mode 0)
```

---

## Grid Launch

```python
elementwise_add_kernel(...).launch(
    grid=[cute.size(gC, mode=[1]), 1, 1],         # number of block tiles
    block=[cute.size(tv_layout, mode=[0]), 1, 1],  # threads per block
)
```

Small config: `grid=2, block=8`
Large config: `grid=4096, block=256`

---

## Comparison: Example 2 vs Example 3

| Aspect               | Ex2 (zipped_divide)      | Ex3 (TV Layout)              |
|-----------------------|--------------------------|------------------------------|
| Thread mapping        | Manual `%` and `//`      | `composition` + slice        |
| Vectorization width   | Hardcoded `(1, 4)`       | Configurable via `val_layout`|
| Thread arrangement    | Implicit (linear)        | Explicit via `thr_layout`    |
| Kernel reusability    | Rewrite for new tiling   | Same kernel, new TV layout   |
| Coalescing control    | Implicit                 | Explicit via `order=(1, 0)`  |

TV Layout is the **foundation for all serious CuTe kernels** (GEMM, attention, convolution).
