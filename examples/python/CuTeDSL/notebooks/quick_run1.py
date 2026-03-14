import torch
from functools import partial
from typing import List

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# Define test dimensions
# M, N = 16384, 8192  # Using large matrices to measure performance
M, N = 128, 64  # Using large matrices to measure performance

# Create test data on GPU
# ----------------------
# Using float16 (half precision) for:
# - Reduced memory bandwidth requirements
# - Better performance on modern GPUs
a = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input A
b = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input B
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)  # Output buffer

# Calculate total elements for bandwidth calculations
num_elements = sum([a.numel(), b.numel(), c.numel()])

# Convert PyTorch tensors to CuTe tensors
# -------------------------------------
# from_dlpack creates CuTe tensor views of PyTorch tensors
# assumed_align=16 ensures proper memory alignment for vectorized access
a_ = from_dlpack(a, assumed_align=16)  # CuTe tensor A
b_ = from_dlpack(b, assumed_align=16)  # CuTe tensor B
c_ = from_dlpack(c, assumed_align=16)  # CuTe tensor C


@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor in unit of vector
    m, n = gA.shape[1]  # thread-domain
    ni = thread_idx % n
    mi = thread_idx // n

    # Map logical index to physical address via tensor layout
    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()
    print(f"[DSL INFO] sliced gA = {gA[(None, (mi, ni))]}")
    print(f"[DSL INFO] sliced gB = {gB[(None, (mi, ni))]}")

    # Perform element-wise addition
    gC[(None, (mi, ni))] = a_val + b_val




# %%
@cute.jit
def vectorized_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block = 256

    gA = cute.zipped_divide(mA, (1, 4))
    gB = cute.zipped_divide(mB, (1, 4))
    gC = cute.zipped_divide(mC, (1, 4))

    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")
    # print gA shape
    print(f"[DSL INFO]   gA shape = {gA.shape}")
    print(f"[DSL INFO]   gA shape = {gA.shape[1]}")

    grid = (cute.size(gC, mode=[1]) // threads_per_block, 1, 1)
    block = (threads_per_block, 1, 1)
    print(f"[DSL INFO] Launching kernel with grid={grid} and block={block}")
    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        # grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        # block=(threads_per_block, 1, 1),
            grid=grid,
            block=block,
    )


a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

compiled_func = cute.compile(vectorized_elementwise_add, a_, b_, c_)
compiled_func(a_, b_, c_)

# verify correctness
torch.testing.assert_close(c, a + b)