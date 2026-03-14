"""
Example 3: TV Layout Elementwise Add Kernel with CuTe DSL
=========================================================

Key Concepts (NEW in this example):
  - TV Layout: A rank-2 layout mapping (thread_index, value_index) -> logical coords
  - cute.make_ordered_layout: Create layouts with specified dimension ordering
  - cute.recast_layout: Convert byte-based layouts to element-based layouts
  - cute.make_layout_tv: Combine thread + value layouts into a TV layout + tiler
  - cute.composition: Compose two layouts (e.g., tensor layout ∘ TV layout)
  - Two-level tiling: thread-block level + thread level

Architecture (Two Levels of Tiling):

  Level 1 - Thread Block Tiling (host side):
    mA: (M, N)
    gA = zipped_divide(mA, (TileM, TileN))   # ((TileM,TileN), (RestM,RestN))
    Each thread block gets one (TileM, TileN) tile via gA[((None,None), bidx)]

  Level 2 - Thread Level Tiling (kernel side, via TV Layout):
    blkA: (TileM, TileN) -> physical address
    tv_layout: (tid, vid) -> (TileM, TileN)  logical coords
    tidfrgA = composition(blkA, tv_layout)    (tid, vid) -> physical address
    thrA = tidfrgA[(tidx, None)]              vid -> physical address

Small Config (for visualization):
  thr_layout = (2, 4):(4, 1)     — 8 threads, 4 cols coalesced
  val_layout = (2, 4):(4, 1)     — 8 values per thread (after recast: (2,4))
  Tiler: (4, 16), TV: (8 threads, 8 values)

  Tile (4 x 16) thread assignment:
       c0  c1  c2  c3 | c4  c5  c6  c7 | c8  c9 c10 c11 | c12 c13 c14 c15
  r0: [T0  v0..v3   ] | [T1  v0..v3   ] | [T2  v0..v3   ] | [T3  v0..v3   ]
  r1: [T0  v4..v7   ] | [T1  v4..v7   ] | [T2  v4..v7   ] | [T3  v4..v7   ]
  r2: [T4  v0..v3   ] | [T5  v0..v3   ] | [T6  v0..v3   ] | [T7  v0..v3   ]
  r3: [T4  v4..v7   ] | [T5  v4..v7   ] | [T6  v4..v7   ] | [T7  v4..v7   ]

  Each thread: 4 contiguous FP16 per row = 64-bit LD.64, across 2 rows = 8 values

Run:
  python 03_tv_layout_elementwise_add.py [--large]
"""

import argparse
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# GPU Kernel: uses TV layout for thread-to-data mapping
# (Same kernel works for ANY TV layout — that's the point!)
# =============================================================================
@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,       # Tiled: ((TileM,TileN), num_blocks)
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,  # (tid, vid) -> logical coord in (TileM, TileN)
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # ---- Level 1: Thread-block tiling ----
    # Slice out this block's tile: (TileM, TileN) -> physical address
    # gA = !cute.memref< f16, gmem, align<16>, "((4,16),(1,2)):((32,1),(0,16))" >
    blkA = gA[((None, None), bidx)]
    blkB = gB[((None, None), bidx)]
    blkC = gC[((None, None), bidx)]
    print(f"[DSL INFO] Block {bidx}: blkA = {blkA}")


    # ---- Level 2: Thread-level tiling via TV layout ----
    # Compose: (tid, vid) -> logical coord -> physical address
    #   blkA:      logical coord -> physical address
    #   tv_layout: (tid, vid)    -> logical coord
    #   result:    (tid, vid)    -> physical address
    # blkA = tensor<ptr<f16, gmem, align<16>> o (4,16):(32,1)>
    """
      blkA: (4, 16) : (32, 1)
         │   │     │   │
         │   │     │   └─ col stride = 1 (contiguous)
         │   │     └───── row stride = 32 (= N, full tensor width)
         │   └─────────── 16 cols
         └─────────────── 4 rows
    """
    print(f"[DSL INFO] TV Layout for block {bidx}: {tv_layout}")
    # blkA: tensor<ptr<f16, gmem, align<16>> o (4,16):(32,1)>
    # tv_layout: ((4,2),(4,2)):((16,2),(4,1))
    tidfrgA = cute.composition(blkA, tv_layout)
    # tidfrgA: !cute.memref<f16, gmem, align<16>, "((4,2),(4,2)):((4,64),(1,32))">
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print("Composed with TV layout:")
    print(f"  tidfrgA: {tidfrgA.type}")

    # Slice out this thread's values: vid -> physical address
    thrA = tidfrgA[(tidx, None)]  # (V) -> physical address
    #  thrA = tensor<ptr<f16, gmem, align<8>> o ((4,2)):((1,32))>
    """
                    ┌─── FP16 elements                                                                
                    │     ┌─── GPU global memory                                                      
                    │     │     ┌─── alignment dropped to 8 bytes (not guaranteed                     
                    │     │     │    16-byte aligned after thread offset)
                    ▼     ▼     ▼     
      thrA: <       f16,  gmem, align<8>,  "((4,2)):((1,32))" >
                                            ─────   ──────
                                            shape   stride
                                            8 vals  memory offsets
        
    """
    print(f"[DSL INFO] Thread {tidx}: typethrA = {thrA.type}")
    print(f"[DSL INFO] Thread {tidx}: thrA = {thrA}")
    thrB = tidfrgB[(tidx, None)]
    thrC = tidfrgC[(tidx, None)]

    # Vectorized load, compute, store
    thrC[None] = thrA.load() + thrB.load()


# =============================================================================
# Host JIT Function: constructs TV layout and tiles tensors
# =============================================================================

# --- Small config: 8 threads, 8 values/thread, tile (4, 16) ---
# Good for understanding. Each thread does LD.64 (4 FP16 = 8 bytes).
#
#   thr_layout = (2, 4):(4, 1)    — 2 rows x 4 cols = 8 threads
#   val_layout = (2, 4):(4, 1)    — 2 rows x 4 cols = 8 values/thread
#                                    (recast from (2, 8) bytes)
#   tiler = (2*2, 4*4) = (4, 16)
#   tv_layout: (8, 8) -> (4, 16)
#
#   Tensor: M=4, N=32 → 2 blocks of (4, 16)
@cute.jit
def elementwise_add_small(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # coalesced load/store bytes = 8 (64-bit LD.64) → 4 FP16 per load
    coalesced_ldst_bytes = 8  # 64-bit = 8 bytes = 4 FP16 per load

    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    # --- Thread Layout: (2, 4) with order (1,0) → stride (4, 1) ---
    # 4 contiguous threads on columns (coalesced), 2 thread-rows
    # Total: 8 threads per block
    thr_layout = cute.make_ordered_layout((2, 4), order=(1, 0))

    # --- Value Layout: (2, 8) bytes → recast to (2, 4) FP16 elements ---
    # Each thread: 4 FP16 per row (= 8 bytes = LD.64), across 2 rows
    # Total: 8 values per thread
    val_layout = cute.make_ordered_layout((2, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    print(f"[SMALL] val_layout layout: {val_layout}")
    # --- Combine: tiler = (4, 16), tv = (8 threads, 8 values) ---
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    # tiler_mn: Shape, (4, 16)
    # tv_layout: Layout: ((4,2),(4,2)):((16,2),(4,1))
    from cute_viz import display_tv_layout, display_layout
    display_layout(thr_layout)

    print(f"[SMALL] Thread layout: {thr_layout}")
    print(f"[SMALL] Value layout:  {val_layout}")
    print(f"[SMALL] Tiler:         {tiler_mn}")
    print(f"[SMALL] TV Layout:     {tv_layout}")

    # --- Tile tensors at block level ---
    gA = cute.zipped_divide(mA, tiler_mn)
    """
                     ┌─── Element type: FP16 (2 bytes)                                                                                           
                     │    ┌─── Memory space: GPU global memory (DRAM)                                                         
                     │    │    ┌─── Pointer alignment: 16 bytes → enables LD.128                                              
                     │    │    │         
                     ▼    ▼    ▼                                                                                              
  gA = !cute.memref< f16, gmem, align<16>, "((4,16),(1,2)):((32,1),(0,16))" >
                                              │       │      │       │
                                              │       │      │       └─ tile grid stride
                                              │       │      │          0  → 1 row-tile (no step)
                                              │       │      │          16 → col-tile step (=TileN)
                                              │       │      │
                                              │       │      └─ tile interior stride
                                              │       │         32 → row step (= N)
                                              │       │         1  → col step (contiguous)
                                              │       │
                                              │       └─ tile grid shape (mode 1)
                                              │          1 row-tile  (4/4 = 1)
                                              │          2 col-tiles (32/16 = 2)
                                              │
                                              └─ tile interior shape (mode 0)
                                                 4 rows × 16 cols = one block's tile
    
    """                                            
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    print(f"[SMALL] gA = {gA.type}")

    num_blocks = cute.size(gC, mode=[1])
    num_threads = cute.size(tv_layout, mode=[0])
    print(f"[SMALL] Grid: {num_blocks} blocks x {num_threads} threads")

    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[num_blocks, 1, 1],
        block=[num_threads, 1, 1],
    )


# --- Large config: 256 threads, 128 values/thread, tile (64, 512) ---
# Production-quality. Each thread does LD.128 (8 FP16 = 16 bytes).
@cute.jit
def elementwise_add_large(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # coalesced load/store bytes
    coalesced_ldst_bytes = 16  # 128-bit = 16 bytes = 8 FP16 per load

    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    # --- Thread Layout: (4, 64) with order (1,0) → stride (64, 1) ---
    # 64 contiguous threads on columns, 4 thread-rows
    # Total: 256 threads per block
    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))

    # --- Value Layout: (16, 16) bytes → recast to (16, 8) FP16 ---
    # Each thread: 8 FP16 per row (= 16 bytes = LD.128), across 16 rows
    # Total: 128 values per thread
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)

    # --- Combine: tiler = (64, 512), tv = (256 threads, 128 values) ---
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)


    print(f"[LARGE] Thread layout: {thr_layout}")
    print(f"[LARGE] Value layout:  {val_layout}")
    print(f"[LARGE] Tiler:         {tiler_mn}")
    print(f"[LARGE] TV Layout:     {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    print(f"[LARGE] gA = {gA.type}")

    num_blocks = cute.size(gC, mode=[1])
    num_threads = cute.size(tv_layout, mode=[0])
    print(f"[LARGE] Grid: {num_blocks} blocks x {num_threads} threads")

    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[num_blocks, 1, 1],
        block=[num_threads, 1, 1],
    )


# =============================================================================
# Layout Visualizer (pure Python, no CuTe needed)
# =============================================================================
def visualize_layout(shape, stride, name="layout"):
    """Evaluate a CuTe-style layout: coord → index, and print as 2D grid."""
    rows, cols = shape
    sr, sc = stride
    print(f"\n  {name}: ({rows},{cols}):({sr},{sc})")
    print(f"  Mapping: (r, c) → r*{sr} + c*{sc}")
    print()

    # Header
    header = "      " + "".join(f"c{c:<4d}" for c in range(cols))
    print(header)

    for r in range(rows):
        vals = [r * sr + c * sc for c in range(cols)]
        row_str = f"  r{r}:  " + "".join(f"{v:<5d}" for v in vals)
        print(row_str)


def visualize_tv_mapping(thr_shape, thr_stride, val_shape, val_stride):
    """
    Show which thread owns which elements in the tile.

    thr_layout maps thread_id → (thr_row, thr_col) in thread grid
    val_layout maps value_id  → (val_row, val_col) offset within thread's region

    Tile element at (row, col):
      row = thr_row * val_rows + val_row
      col = thr_col * val_cols + val_col
      → owned by thread (thr_row * thr_stride_r + thr_col * thr_stride_c)
    """
    thr_rows, thr_cols = thr_shape
    thr_sr, thr_sc = thr_stride
    val_rows, val_cols = val_shape

    tile_rows = thr_rows * val_rows
    tile_cols = thr_cols * val_cols

    print(f"\n  Tile ({tile_rows} x {tile_cols}): which thread owns each element")
    print(f"  thr=({thr_rows},{thr_cols}):({thr_sr},{thr_sc}), val=({val_rows},{val_cols})")
    print()

    # Build the tile grid: tile[r][c] = thread_id
    tile = [[0]*tile_cols for _ in range(tile_rows)]

    for tr in range(thr_rows):
        for tc in range(thr_cols):
            tid = tr * thr_sr + tc * thr_sc
            for vr in range(val_rows):
                for vc in range(val_cols):
                    r = tr * val_rows + vr
                    c = tc * val_cols + vc
                    tile[r][c] = tid

    # Print
    header = "       " + "".join(f"c{c:<4d}" for c in range(tile_cols))
    print(header)
    for r in range(tile_rows):
        row_str = f"  r{r}:   " + "".join(f"T{tile[r][c]:<4d}" for c in range(tile_cols))
        print(row_str)

    # Also show per-thread value assignments
    print(f"\n  Per-thread value assignment (value_id → tile coordinate):")
    for tr in range(thr_rows):
        for tc in range(thr_cols):
            tid = tr * thr_sr + tc * thr_sc
            coords = []
            for vr in range(val_rows):
                for vc in range(val_cols):
                    r = tr * val_rows + vr
                    c = tc * val_cols + vc
                    coords.append(f"({r},{c})")
            print(f"    T{tid}: {', '.join(coords)}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TV Layout Elementwise Add")
    parser.add_argument("--large", action="store_true",
                        help="Use large config (256 threads, M=16384 N=8192)")
    args = parser.parse_args()

    if args.large:
        # Large: production config
        M, N = 16384, 8192
        jit_fn = elementwise_add_large
        print(f"=== Large Config: M={M}, N={N} ===")
        print(f"  thr=(4,64)=256 threads, val=(16,8)=128 vals/thr, tile=(64,512)")
    else:
        # Small: visualization config
        M, N = 4, 32
        jit_fn = elementwise_add_small

        print(f"=== Small Config: M={M}, N={N} ({M*N} elements) ===")
        print(f"  thr=(2,4)=8 threads, val=(2,4)=8 vals/thr, tile=(4,16)")

        # --- Visualize thr_layout ---
        print(f"\n{'='*60}")
        print(f"  THREAD LAYOUT: make_ordered_layout((2,4), order=(1,0))")
        print(f"{'='*60}")
        print(f"  order=(1,0): dim 1 (cols) varies fastest → stride 1")
        print(f"               dim 0 (rows) varies slower  → stride 4")
        visualize_layout((2, 4), (4, 1), "thr_layout")
        print(f"\n  → T0,T1,T2,T3 are contiguous columns = coalesced warp access!")

        # --- Visualize val_layout ---
        print(f"\n{'='*60}")
        print(f"  VALUE LAYOUT: (2,8) bytes → recast → (2,4) FP16 elements")
        print(f"{'='*60}")
        print(f"  Each thread reads 4 contiguous FP16 per row (LD.64), 2 rows")
        visualize_layout((2, 4), (4, 1), "val_layout")
        print(f"\n  → v0,v1,v2,v3 contiguous in memory = one LD.64 instruction")
        print(f"  → v4,v5,v6,v7 on next row = second LD.64 instruction")

        # --- Visualize combined TV mapping in tile ---
        print(f"\n{'='*60}")
        print(f"  COMBINED: TV Layout in one tile (4 x 16)")
        print(f"{'='*60}")
        visualize_tv_mapping((2, 4), (4, 1), (2, 4), (4, 1))

        # --- Full tensor ---
        print(f"\n{'='*60}")
        print(f"  FULL TENSOR ({M} x {N}) = 2 block tiles")
        print(f"{'='*60}")
        print(f"  Block 0: cols 0-15     Block 1: cols 16-31")

    print()

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled_fn = cute.compile(jit_fn, a_, b_, c_)
    compiled_fn(a_, b_, c_)

    # Verify
    torch.testing.assert_close(c, a + b)
    print("\nCorrectness: PASSED")
