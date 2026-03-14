"""
Example 1: Naive Elementwise Add Kernel with CuTe DSL
=====================================================

Key Concepts:
  - @cute.kernel: Defines a GPU kernel function
  - @cute.jit: Defines a host-side JIT function that launches kernels
  - cute.arch.thread_idx/block_idx/block_dim: CUDA thread indexing
  - from_dlpack: Convert PyTorch tensors to CuTe tensors
  - cute.compile: JIT compile kernel for specific input types
  - 1:1 thread-to-element mapping (simplest approach)

How It Works:
  Each CUDA thread processes exactly ONE element:
    1. Compute global thread ID: thread_idx = bidx * bdim + tidx
    2. Map linear ID to 2D coordinates: mi = thread_idx // N, ni = thread_idx % N
    3. Load A[mi,ni] and B[mi,ni], add them, store to C[mi,ni]

  This is memory-bound and NOT optimized -- serves as baseline.

Run:
  python 01_naive_elementwise_add.py
"""

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# GPU Kernel: runs on device, one thread per element
# =============================================================================
@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,  # Input tensor A
    gB: cute.Tensor,  # Input tensor B
    gC: cute.Tensor,  # Output tensor C = A + B
):
    # Get CUDA thread coordinates
    tidx, _, _ = cute.arch.thread_idx()  # Thread index within block (0 to bdim-1)
    bidx, _, _ = cute.arch.block_idx()   # Block index in grid
    bdim, _, _ = cute.arch.block_dim()   # Number of threads per block

    # Global thread index: unique ID across all blocks
    thread_idx = bidx * bdim + tidx

    # Map linear thread index -> 2D tensor coordinates
    m, n = gA.shape                  # (M, N)
    ni = thread_idx % n              # Column (fast-varying, coalesced access)
    mi = thread_idx // n             # Row (slow-varying)

    # Load, compute, store
    a_val = gA[mi, ni]
    b_val = gB[mi, ni]
    gC[mi, ni] = a_val + b_val


# =============================================================================
# Host JIT Function: configures and launches the kernel
# =============================================================================
@cute.jit
def naive_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    num_threads_per_block = 256  # Multiple of 32 (warp size)
    m, n = mA.shape

    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1),
        block=(num_threads_per_block, 1, 1),
    )


# =============================================================================
# Benchmark Utility
# =============================================================================
def benchmark(compiled_fn, a_, b_, c_):
    num_elements = sum([a_.size, b_.size, c_.size])
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
# Main: Setup, Run, Verify, Benchmark
# =============================================================================
if __name__ == "__main__":
    M, N = 16384, 8192

    # Create test data on GPU (float16)
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    # Convert PyTorch tensors -> CuTe tensors
    # assumed_align=16 tells compiler it can use 128-bit (16 byte) aligned loads
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    # JIT compile for these specific input types/shapes
    compiled_fn = cute.compile(naive_elementwise_add, a_, b_, c_)

    # Run the kernel
    compiled_fn(a_, b_, c_)

    # Verify correctness against PyTorch
    torch.testing.assert_close(c, a + b)
    print("Correctness: PASSED")

    # Benchmark
    benchmark(compiled_fn, a_, b_, c_)
