# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Step 5: Vectorized G2S + ldmatrix S2R + Register Pipeline + SMEM Epilogue
=========================================================================

The final step builds on Step 4 by adding all the remaining optimizations
that make a production-quality GEMM kernel:

NEW CONCEPTS introduced here (vs Step 4):
  - 128-bit vectorized cp.async G2S copies (8 x FP16 per thread per atom)
  - Register double-buffering: prefetch tCrA/B[k+1] while computing tCrA/B[k]
  - SMEM epilogue: RMEM -> SMEM -> GMEM for coalesced global writes
  - Vectorized epilogue copy with tiled_copy_C

DATA PATH:
  Mainloop:  GMEM --(128b cp.async)--> SMEM[swizzled] --(ldmatrix)--> RMEM --> HMMA
  Epilogue:  RMEM --> SMEM --> RMEM (retiled) --(vectorized)--> GMEM

Why these optimizations matter:
  - 128-bit G2S: saturates GMEM bandwidth (4x fewer copy instructions)
  - Register pipeline: hides SMEM->register latency and breaks register
    dependencies between consecutive k-blocks
  - SMEM epilogue: the MMA's register layout doesn't match the global memory
    layout. Staging through SMEM allows re-tiling for coalesced 128-bit stores.

Fixed configuration:
  - A: (M, K) M-major FP16, B: (N, K) N-major FP16, C: (M, N) N-major FP16
  - Tile: 128x128x32, Threads: 128 (4 warps), Stages: 3
  - MMA: HMMA 16x8x16, atom layout (2,2,1)

Run:
    python step5_vectorized_gemm.py --mnk 512,512,512
"""

import argparse
import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

# ===========================================================================
# Constants
# ===========================================================================
BM, BN, BK = 128, 128, 32
NUM_STAGES = 3
MMA_M, MMA_N, MMA_K = 16, 8, 16
ATOM_M, ATOM_N, ATOM_K = 2, 2, 1
NUM_THREADS = ATOM_M * ATOM_N * ATOM_K * 32  # 128
COPY_BITS = 128  # NEW: 128-bit vectorized copies


# ===========================================================================
# Kernel
# ===========================================================================
@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    sC_layout: cute.ComposedLayout,      # NEW: epilogue SMEM layout
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_copy_C: cute.TiledCopy,        # NEW: epilogue copy
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    cta_tiler = (BM, BN, BK)

    # ---- Tile global tensors ----
    gA = cute.local_tile(mA, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, 1, None))

    # ---- Allocate SMEM ----
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
    # Reuse sA's memory for epilogue C (they don't overlap in time)
    sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=cutlass.Float16), sC_layout)

    # ---- G2S copy partitions ----
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    # ---- Epilogue copy partitions ----
    thr_copy_C = tiled_copy_C.get_slice(tidx)
    tCsC_epilogue = thr_copy_C.partition_S(sC)
    tCgC_epilogue = thr_copy_C.partition_D(gC)

    # ---- MMA partitions ----
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCsC = thr_mma.partition_C(sC)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    # ---- S2R copy via ldmatrix ----
    # For M-major A (COL_MAJOR): needs transpose=True for ldmatrix
    # For N-major B (COL_MAJOR): needs transpose=True for ldmatrix
    atom_s2r_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), mA.element_type
    )
    atom_s2r_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), mB.element_type
    )
    tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
    tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

    thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
    thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsA_copy = thr_s2r_A.partition_S(sA)
    tCrA_copy = thr_s2r_A.retile(tCrA)
    tCsB_copy = thr_s2r_B.partition_S(sB)
    tCrB_copy = thr_s2r_B.retile(tCrB)

    k_tile_count = cute.size(tAgA, mode=[3])
    num_k_blocks = cute.size(tCrA, mode=[2])

    # ---- Clear SMEM ----
    tAsA.fill(0)
    tBsB.fill(0)
    cute.arch.sync_threads()

    # ===========================================================================
    # Prologue
    # ===========================================================================
    k_tile_index = cutlass.Int32(0)
    for stage in range(NUM_STAGES - 1):
        if stage < k_tile_count:
            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, stage])
            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, stage])
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

    # ===========================================================================
    # Prefetch first register k-block
    # ===========================================================================
    smem_pipe_read = 0
    smem_pipe_write = NUM_STAGES - 1

    tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
    tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]

    if num_k_blocks > 1:
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
        cute.arch.sync_threads()
        cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy[None, None, 0])
        cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy[None, None, 0])

    # ===========================================================================
    # Mainloop with register pipeline
    # ===========================================================================
    for k_tile in range(k_tile_count):
        for k_block in cutlass.range(num_k_blocks, unroll_full=True):
            if k_block == num_k_blocks - 1:
                # Switch to next SMEM stage
                tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
                tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
                cute.arch.cp_async_wait_group(NUM_STAGES - 2)
                cute.arch.sync_threads()

            # NEW: Prefetch NEXT k_block from SMEM -> registers
            # This is the register pipeline: while we compute k_block,
            # we load k_block+1 into a different set of registers
            k_block_next = (k_block + 1) % num_k_blocks
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy[None, None, k_block_next])
            cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy[None, None, k_block_next])

            # Issue next G2S copy (interleaved: copy A, then gemm, then copy B)
            if k_block == 0:
                if k_tile + NUM_STAGES - 1 < k_tile_count:
                    cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                              tAsA[None, None, None, smem_pipe_write])

            # Compute current k_block
            cute.gemm(tiled_mma, tCrC,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block],
                      tCrC)

            if k_block == 0:
                if k_tile + NUM_STAGES - 1 < k_tile_count:
                    cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index],
                              tBsB[None, None, None, smem_pipe_write])
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()
                smem_pipe_write = smem_pipe_read
                smem_pipe_read = smem_pipe_read + 1
                if smem_pipe_read == NUM_STAGES:
                    smem_pipe_read = 0

    # ===========================================================================
    # NEW: SMEM Epilogue — RMEM -> SMEM -> GMEM for coalesced writes
    # ===========================================================================
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # Convert FP32 accumulator to FP16
    tCrD = cute.make_fragment_like(tCrC, cutlass.Float16)
    tCrD[None] = tCrC.load().to(cutlass.Float16)

    # RMEM -> SMEM: write results into shared memory (reusing A/B buffer)
    cute.autovec_copy(tCrD, tCsC)

    cute.arch.sync_threads()

    # SMEM -> RMEM (retiled for vectorized copy)
    tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
    cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

    # RMEM -> GMEM: vectorized 128-bit store
    cute.copy(tiled_copy_C, tCrC_epilogue, tCgC_epilogue)
    return


# ===========================================================================
# Host function
# ===========================================================================
@cute.jit
def host_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # ---- Swizzled SMEM layouts for A/B (same as Step 4) ----
    major_mode_size = 64
    swizzle_bits = int(math.log2(major_mode_size * cutlass.Float16.width // COPY_BITS))
    swizzle_bits = min(swizzle_bits, 3)
    layout_atom_A = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size)),
    )
    sA_layout = cute.tile_to_shape(layout_atom_A, (BM, BK, NUM_STAGES), (0, 1, 2))

    layout_atom_B = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size)),
    )
    sB_layout = cute.tile_to_shape(layout_atom_B, (BN, BK, NUM_STAGES), (0, 1, 2))

    # ---- NEW: SMEM layout for epilogue C ----
    # C is N-major (row-major), so layout_atom has N contiguous
    # Note: for C, major_mode_size is NOT capped at 64 (unlike A/B)
    c_major_mode_size = BN  # 128 (full N-dim of tile)
    c_swizzle_bits = int(math.log2(c_major_mode_size * cutlass.Float16.width // COPY_BITS))
    c_swizzle_bits = min(c_swizzle_bits, 3)
    layout_atom_C = cute.make_composed_layout(
        cute.make_swizzle(c_swizzle_bits, 3, 4),
        0,
        cute.make_layout((8, c_major_mode_size), stride=(c_major_mode_size, 1)),
    )
    sC_layout = cute.tile_to_shape(layout_atom_C, (BM, BN), (0, 1))

    # ---- NEW: 128-bit vectorized G2S copy ----
    copy_elems = COPY_BITS // cutlass.Float16.width  # 128/16 = 8 elements
    atom_async_copy = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=COPY_BITS,  # 128 bits = 8 x FP16
    )

    # A is M-major: vectorize along M-dim
    # (BM/copy_elems, NUM_THREADS/(BM/copy_elems)) threads
    shape_dim_0_A = BM // copy_elems  # 128/8 = 16
    thr_layout_A = cute.make_layout(
        (shape_dim_0_A, NUM_THREADS // shape_dim_0_A),
        stride=(1, shape_dim_0_A),
    )
    val_layout_A = cute.make_layout((copy_elems, 1))
    tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy, thr_layout_A, val_layout_A)

    # B is N-major: vectorize along N-dim
    shape_dim_0_B = BN // copy_elems  # 128/8 = 16
    thr_layout_B = cute.make_layout(
        (shape_dim_0_B, NUM_THREADS // shape_dim_0_B),
        stride=(1, shape_dim_0_B),
    )
    val_layout_B = cute.make_layout((copy_elems, 1))
    tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy, thr_layout_B, val_layout_B)

    # ---- NEW: Vectorized epilogue copy for SMEM -> GMEM ----
    c_copy_bits = 128
    c_copy_elems = c_copy_bits // cutlass.Float16.width  # 8
    atom_sync_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mC.element_type,
        num_bits_per_copy=c_copy_bits,
    )
    # C is N-major (row-major): vectorize along N-dim
    shape_dim_1_C = BN // c_copy_elems  # 128/8 = 16
    thr_layout_C = cute.make_layout(
        (NUM_THREADS // shape_dim_1_C, shape_dim_1_C),
        stride=(shape_dim_1_C, 1),
    )
    val_layout_C = cute.make_layout((1, c_copy_elems))
    tiled_copy_C = cute.make_tiled_copy_tv(atom_sync_copy, thr_layout_C, val_layout_C)

    # ---- Tensor core MMA (same as Step 4) ----
    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (MMA_M, MMA_N, MMA_K))
    tC = cute.make_layout((ATOM_M, ATOM_N, ATOM_K))
    tiled_mma = cute.make_tiled_mma(
        op, tC,
        permutation_mnk=(
            ATOM_M * MMA_M,
            ATOM_N * MMA_N * 2,
            ATOM_K * MMA_K,
        ),
    )

    # ---- Calculate SMEM size (max of AB stages vs C epilogue) ----
    smem_size = max(
        cute.size_in_bytes(cutlass.Float16, sC_layout),
        cute.size_in_bytes(cutlass.Float16, sA_layout)
        + cute.size_in_bytes(cutlass.Float16, sB_layout),
    )

    grid = (*cute.ceil_div(mC.shape, (BM, BN)), 1)

    gemm_kernel(
        mA, mB, mC,
        sA_layout, sB_layout, sC_layout,
        tiled_copy_A, tiled_copy_B, tiled_copy_C,
        tiled_mma,
    ).launch(
        grid=grid,
        block=[NUM_THREADS, 1, 1],
        smem=smem_size,
    )


# ===========================================================================
# Run
# ===========================================================================
def run_gemm(mnk: Tuple[int, int, int]):
    import torch

    M, N, K = mnk
    assert M % BM == 0 and N % BN == 0 and K % BK == 0, (
        f"Problem size ({M}, {N}, {K}) must be divisible by tile ({BM}, {BN}, {BK})"
    )

    torch.manual_seed(42)
    a = torch.randn(K, M, dtype=torch.float16, device="cuda").permute(1, 0)
    b = torch.randn(K, N, dtype=torch.float16, device="cuda").permute(1, 0)
    c = torch.zeros(M, N, dtype=torch.float16, device="cuda")

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print(f"Step 5: Fully Optimized GEMM (vec G2S + ldmatrix + reg pipeline + SMEM epilogue)")
    print(f"  Problem: M={M}, N={N}, K={K}")
    print(f"  Tile: {BM}x{BN}x{BK}, Threads: {NUM_THREADS}, Stages: {NUM_STAGES}")
    print(f"  MMA: FP16 HMMA 16x8x16, atom layout ({ATOM_M},{ATOM_N},{ATOM_K})")
    print(f"  G2S: 128-bit vectorized cp.async")
    print(f"  S2R: ldmatrix 8x8x16b")
    print(f"  Epilogue: RMEM -> SMEM -> GMEM (vectorized 128-bit)")

    print("  Compiling...")
    compiled = cute.compile(host_gemm, mA, mB, mC)

    print("  Running...")
    compiled(mA, mB, mC)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-1, rtol=1e-1)
    print("  PASS — results match reference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Fully Optimized GEMM")
    parser.add_argument("--mnk", type=lambda s: tuple(int(x) for x in s.split(",")),
                        default=(512, 512, 512))
    args = parser.parse_args()
    run_gemm(args.mnk)
