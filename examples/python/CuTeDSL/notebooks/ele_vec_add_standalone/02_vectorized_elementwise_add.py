"""
Example 2: Vectorized Elementwise Add Kernel with CuTe DSL
==========================================================

Key Concepts (NEW in this example):
  - cute.zipped_divide(tensor, tiler): Partition tensor into tiles
  - Vectorized load/store: Load 4 elements at once (128-bit) per thread
  - Slicing with (None, (mi, ni)): Extract a sub-tensor for vectorized access

What Changed from Example 1:
  Instead of 1 element per thread, each thread now handles 4 contiguous elements
  in a single 128-bit load/store operation. This is key for saturating memory
  bandwidth (Little's Law: more bytes in-flight per thread).

How zipped_divide Works:
  mA:  (M, N)  : (N, 1)          -- original 2D tensor
  gA = zipped_divide(mA, (1, 4)) -- partition into groups of (1, 4)
  gA:  ((1,4), (M, N/4)) : ((0,1), (N, 4))
        ~~~~   ~~~~~~~~
        tiler   num tiles
        (per-   (thread
        thread)  domain)

  To access thread's data:  gA[(None, (mi, ni))].load()
    - None: take all 4 values in the tile
    - (mi, ni): select which tile (which thread)

Run:
  python 02_vectorized_elementwise_add.py
"""

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# GPU Kernel: each thread loads/stores 4 elements via vectorized access
# =============================================================================
@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,  # Already tiled: ((1,4), (M, N/4))
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Navigate in the TILE domain (2nd mode), not the original tensor domain
    m, n = gA.shape[1]  # shape of the thread-domain (num tiles)
    ni = thread_idx % n
    mi = thread_idx // n

    # Vectorized load: (None, (mi, ni)) extracts the (1,4) sub-tensor
    # .load() generates a single 128-bit load instruction
    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()

    # This is equivalent to loading 4 elements one by one:
    #   v0 = gA[(0, (mi, ni))]  # => mA[mi, ni*4 + 0]
    #   v1 = gA[(1, (mi, ni))]  # => mA[mi, ni*4 + 1]
    #   v2 = gA[(2, (mi, ni))]  # => mA[mi, ni*4 + 2]
    #   v3 = gA[(3, (mi, ni))]  # => mA[mi, ni*4 + 3]
    # But .load() does it in one 128-bit transaction!

    # Vectorized store
    gC[(None, (mi, ni))] = a_val + b_val


# =============================================================================
# Host JIT Function: tiles the tensors, then launches kernel
# =============================================================================
@cute.jit
def vectorized_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    threads_per_block = 256

    # Tile into groups of (1, 4) -- each thread handles 4 contiguous column elements
    gA = cute.zipped_divide(mA, (1, 4))
    gB = cute.zipped_divide(mB, (1, 4))
    gC = cute.zipped_divide(mC, (1, 4))

    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")

    # Grid size = total number of tiles / threads per block
    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
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

    # assumed_align=16 is CRITICAL for vectorized loads (128-bit = 16 bytes)
    # Without this, the compiler cannot prove alignment and won't vectorize
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled_fn = cute.compile(vectorized_elementwise_add, a_, b_, c_)
    compiled_fn(a_, b_, c_)

    # Verify correctness
    torch.testing.assert_close(c, a + b)
    print("Correctness: PASSED")

    # Benchmark
    benchmark(compiled_fn, a_, b_, c_, num_elements)
