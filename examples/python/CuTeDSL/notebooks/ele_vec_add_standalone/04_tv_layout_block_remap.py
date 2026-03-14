"""
Example 4: TV Layout with Thread-Block Index Remapping
======================================================

Key Concepts (NEW in this example):
  - Thread-block index remapping for better memory locality
  - cute.make_ordered_layout on tile grid shape to transpose block traversal
  - cute.composition(tiled_tensor, (None, remap_block)) to remap only block indices

Problem:
  When tensors are row-major (contiguous on N dimension), the default block
  ordering visits tiles column-first:

    Block 0 -> tile (0,0)    Block 1 -> tile (1,0)    Block 2 -> tile (2,0) ...
    (jumps across rows, non-contiguous memory access between consecutive blocks)

  With remapping, consecutive blocks visit tiles row-first:

    Block 0 -> tile (0,0)    Block 1 -> tile (0,1)    Block 2 -> tile (0,2) ...
    (stays in same row, better L2 cache utilization)

How It Works:
  remap_block = make_ordered_layout(
      (RestM_shape, RestN_shape),  # grid of tiles
      order=(1, 0)                 # transpose: N varies fastest
  )
  gA = composition(gA, (None, remap_block))
    - None: don't touch the tile-internal layout (1st mode)
    - remap_block: remap block index -> (RestN, RestM) order (2nd mode)

Run:
  python 04_tv_layout_block_remap.py
"""

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# GPU Kernel (same as Example 3 -- the optimization is in the HOST code)
# =============================================================================
@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Thread-block level slice
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    # Compose with TV layout: (tid, vid) -> physical address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    # Thread level slice
    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    # Vectorized load, compute, store
    thrC[None] = thrA.load() + thrB.load()


# =============================================================================
# Host JIT Function: adds block remapping on top of Example 3
# =============================================================================
@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    coalesced_ldst_bytes = 16

    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    # TV layout construction (same as Example 3)
    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Tiler: {tiler_mn}")
    print(f"[DSL INFO] TV Layout: {tv_layout}")

    # Tile at thread-block level
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    # =====================================================================
    # NEW: Thread-block index remapping
    # =====================================================================
    # The 2nd mode of gA has shape (RestM, RestN) = number of tiles in each dim.
    # By default, linear block index maps to (RestM, RestN) in column-major order.
    #
    # We create a remapping layout that transposes this:
    #   remap_block: linear_index -> (RestN, RestM) with order=(1,0)
    # This makes consecutive block indices traverse columns first (row-major tile order),
    # which is better for row-major tensors (contiguous on N dim).
    remap_block = cute.make_ordered_layout(
        cute.select(gA.shape[1], mode=[1, 0]),  # (RestN, RestM)
        order=(1, 0),
    )
    gA = cute.composition(gA, (None, remap_block))
    gB = cute.composition(gB, (None, remap_block))
    gC = cute.composition(gC, (None, remap_block))

    print("[DSL INFO] After block remapping:")
    print(f"[DSL INFO]   gA = {gA.type}")

    # Launch
    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


# =============================================================================
# Benchmark Utility
# =============================================================================
def benchmark(compiled_fn, a_, b_, c_, num_elements):
    avg_time_us = cute.testing.benchmark(
        compiled_fn,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )
    dtype = a_.element_type
    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s

    print(f"Performance Metrics:")
    print(f"  Kernel execution time: {avg_time_us:.4f} us")
    print(f"  Memory throughput:     {achieved_bandwidth:.2f} GB/s")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    M, N = 16384, 8192

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    num_elements = sum([a.numel(), b.numel(), c.numel()])

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled_fn = cute.compile(elementwise_add, a_, b_, c_)
    compiled_fn(a_, b_, c_)

    # Verify
    torch.testing.assert_close(c, a + b)
    print("Correctness: PASSED")

    # Benchmark
    benchmark(compiled_fn, a_, b_, c_, num_elements)
