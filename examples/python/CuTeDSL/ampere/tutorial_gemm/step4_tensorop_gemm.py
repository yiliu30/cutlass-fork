# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Step 4: FP16 Tensor Cores + Swizzle — HMMA + Bank-Conflict-Free SMEM
=====================================================================

Builds on Step 3 by upgrading from scalar FP32 FMA to FP16 tensor core
instructions (HMMA 16x8x16) and adding swizzled SMEM layouts to prevent
bank conflicts.

NEW CONCEPTS introduced here (vs Step 3):
  - MmaF16BF16Op — warp-level tensor core MMA (16x8x16 HMMA instruction)
  - FP16 input / FP32 accumulator — mixed-precision computation
  - atom_layout (2,2,1) — tile 4 MMA atoms across 2M x 2N = 128 threads
  - make_composed_layout + make_swizzle — bank-conflict-free SMEM layout
  - make_tiled_copy_A/B — derive S2R copy layout from the MMA pattern
  - LdMatrix8x8x16bOp — warp-level SMEM->register load (ldmatrix PTX)
  - Larger K tile: 32 (vs 8 in Steps 1-3) to match tensor core efficiency

DATA PATH:  GMEM --(cp.async)--> SMEM[swizzled] --(ldmatrix)--> RMEM --> HMMA --> GMEM

Why tensor cores + swizzle:
  - Tensor cores provide 8x-16x more FLOPS than scalar FMA
  - Without swizzle, threads in the same warp loading consecutive SMEM rows
    hit the same bank (32 banks, 32-bit each). Swizzle XORs address bits so
    consecutive rows map to different banks.

Fixed configuration:
  - A: (M, K) M-major FP16, B: (N, K) N-major FP16, C: (M, N) N-major FP16
  - Tile: 128x128x32, Threads: 128 (4 warps), Stages: 3
  - MMA: HMMA 16x8x16, atom layout (2,2,1)

Run:
    python step4_tensorop_gemm.py --mnk 512,512,512
"""

import argparse
import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

# ===========================================================================
# Constants — note the changes from Steps 1-3
# ===========================================================================
BM, BN, BK = 128, 128, 32       # Larger K tile for tensor cores
NUM_STAGES = 3
MMA_M, MMA_N, MMA_K = 16, 8, 16  # HMMA instruction shape
ATOM_M, ATOM_N, ATOM_K = 2, 2, 1  # Atom layout
NUM_THREADS = ATOM_M * ATOM_N * ATOM_K * 32  # = 128 threads (4 warps)


# ===========================================================================
# Kernel
# ===========================================================================
@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.ComposedLayout,   # NEW: swizzled layout type
    sB_layout: cute.ComposedLayout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    cta_tiler = (BM, BN, BK)

    # ---- Tile global tensors ----
    gA = cute.local_tile(mA, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, 1, None))

    # ---- Allocate SMEM (swizzled layout) ----
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

    # ---- G2S copy partitions (same pattern as Step 3) ----
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    # ---- MMA partitions ----
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    # ---- NEW: S2R copy via ldmatrix (warp-level optimized) ----
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
    tCsA_copy = thr_s2r_A.partition_S(sA)    # SMEM source for ldmatrix
    tCrA_copy = thr_s2r_A.retile(tCrA)       # retile register to ldmatrix layout
    tCsB_copy = thr_s2r_B.partition_S(sB)
    tCrB_copy = thr_s2r_B.retile(tCrB)

    k_tile_count = cute.size(tAgA, mode=[3])
    num_k_blocks = cute.size(tCrA, mode=[2])

    # ---- Clear SMEM for safety, then sync ----
    tAsA.fill(0)
    tBsB.fill(0)
    cute.arch.sync_threads()

    # ===========================================================================
    # Prologue: prefetch first (NUM_STAGES - 1) tiles
    # ===========================================================================
    k_tile_index = cutlass.Int32(0)
    for stage in range(NUM_STAGES - 1):
        if stage < k_tile_count:
            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, stage])
            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, stage])
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

    # ===========================================================================
    # Prefetch first k-block from SMEM -> registers
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
    # Mainloop
    # ===========================================================================
    for k_tile in range(k_tile_count):
        for k_block in cutlass.range(num_k_blocks, unroll_full=True):
            if k_block == num_k_blocks - 1:
                tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
                tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
                cute.arch.cp_async_wait_group(NUM_STAGES - 2)
                cute.arch.sync_threads()

            # Prefetch NEXT k_block from SMEM -> registers
            k_block_next = (k_block + 1) % num_k_blocks
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy[None, None, k_block_next])
            cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy[None, None, k_block_next])

            # Issue next G2S copy (interleaved with compute)
            if k_block == 0:
                if k_tile + NUM_STAGES - 1 < k_tile_count:
                    cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, smem_pipe_write])

            # Compute current k_block
            cute.gemm(tiled_mma, tCrC,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block],
                      tCrC)

            if k_block == 0:
                if k_tile + NUM_STAGES - 1 < k_tile_count:
                    cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, smem_pipe_write])
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()
                smem_pipe_write = smem_pipe_read
                smem_pipe_read = smem_pipe_read + 1
                if smem_pipe_read == NUM_STAGES:
                    smem_pipe_read = 0

    # ---- Epilogue ----
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # Convert FP32 accum to FP16 and store
    tCrD = cute.make_fragment_like(tCrC, cutlass.Float16)
    tCrD[None] = tCrC.load().to(cutlass.Float16)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
    cute.copy(atom, tCrD, tCgC)
    return


# ===========================================================================
# Host function
# ===========================================================================
@cute.jit
def host_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # ---- NEW: Swizzled SMEM layout ----
    # For M-major A (stride=(1, BM)): layout_atom is (BM_atom, 8) with M contiguous
    # major_mode_size = BM = 128, capped at 64 for swizzle
    major_mode_size_A = 64
    swizzle_bits_A = int(math.log2(major_mode_size_A * cutlass.Float16.width // 128))
    swizzle_bits_A = min(swizzle_bits_A, 3)
    layout_atom_A = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits_A, 3, 3),
        0,
        cute.make_layout((major_mode_size_A, 8), stride=(1, major_mode_size_A)),
    )
    sA_layout = cute.tile_to_shape(layout_atom_A, (BM, BK, NUM_STAGES), (0, 1, 2))

    # For N-major B (stride=(1, BN)): same structure
    major_mode_size_B = 64
    swizzle_bits_B = int(math.log2(major_mode_size_B * cutlass.Float16.width // 128))
    swizzle_bits_B = min(swizzle_bits_B, 3)
    layout_atom_B = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits_B, 3, 3),
        0,
        cute.make_layout((major_mode_size_B, 8), stride=(1, major_mode_size_B)),
    )
    sB_layout = cute.tile_to_shape(layout_atom_B, (BN, BK, NUM_STAGES), (0, 1, 2))

    # ---- G2S copy: cp.async with 32-bit copies ----
    # cp.async minimum transfer is 32 bits (4 bytes = 2 x FP16 elements)
    g2s_copy_bits = 32
    g2s_copy_elems = g2s_copy_bits // cutlass.Float16.width  # 2 elements
    atom_async_copy = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        mA.element_type,
        num_bits_per_copy=g2s_copy_bits,
    )

    # Thread layout for G2S: M-major A
    # A is M-major: contiguous in M-dim, so thread layout has M as fast dim
    # With 2 elements per copy atom, threads cover (BM/2, BK) positions
    shape_dim_0_A = BM // g2s_copy_elems  # 128/2 = 64
    thr_layout_A = cute.make_layout(
        (shape_dim_0_A, NUM_THREADS // shape_dim_0_A),
        stride=(1, shape_dim_0_A),
    )
    val_layout_A = cute.make_layout((g2s_copy_elems, 1))
    tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy, thr_layout_A, val_layout_A)

    # B is N-major: same structure
    shape_dim_0_B = BN // g2s_copy_elems  # 128/2 = 64
    thr_layout_B = cute.make_layout(
        (shape_dim_0_B, NUM_THREADS // shape_dim_0_B),
        stride=(1, shape_dim_0_B),
    )
    val_layout_B = cute.make_layout((g2s_copy_elems, 1))
    tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy, thr_layout_B, val_layout_B)

    # ---- NEW: Tensor core MMA ----
    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (MMA_M, MMA_N, MMA_K))
    tC = cute.make_layout((ATOM_M, ATOM_N, ATOM_K))
    tiled_mma = cute.make_tiled_mma(
        op, tC,
        permutation_mnk=(
            ATOM_M * MMA_M,      # 2*16 = 32
            ATOM_N * MMA_N * 2,  # 2*8*2 = 32
            ATOM_K * MMA_K,      # 1*16 = 16
        ),
    )

    grid = (*cute.ceil_div(mC.shape, (BM, BN)), 1)

    gemm_kernel(
        mA, mB, mC,
        sA_layout, sB_layout,
        tiled_copy_A, tiled_copy_B,
        tiled_mma,
    ).launch(
        grid=grid,
        block=[NUM_THREADS, 1, 1],
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
    # FP16 tensors
    # A: (M, K) M-major => stride (1, M)
    a = torch.randn(K, M, dtype=torch.float16, device="cuda").permute(1, 0)
    # B: (N, K) N-major => stride (1, N)
    b = torch.randn(K, N, dtype=torch.float16, device="cuda").permute(1, 0)
    # C: (M, N) N-major => stride (N, 1)
    c = torch.zeros(M, N, dtype=torch.float16, device="cuda")

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print(f"Step 4: Tensor Core GEMM (HMMA 16x8x16 + swizzled SMEM)")
    print(f"  Problem: M={M}, N={N}, K={K}")
    print(f"  Tile: {BM}x{BN}x{BK}, Threads: {NUM_THREADS}, Stages: {NUM_STAGES}")
    print(f"  MMA: FP16 HMMA 16x8x16, atom layout ({ATOM_M},{ATOM_N},{ATOM_K})")
    print(f"  FP16 in / FP32 accum / FP16 out")

    print("  Compiling...")
    compiled = cute.compile(host_gemm, mA, mB, mC)

    print("  Running...")
    compiled(mA, mB, mC)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-1, rtol=1e-1)
    print("  PASS — results match reference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Tensor Core GEMM")
    parser.add_argument("--mnk", type=lambda s: tuple(int(x) for x in s.split(",")),
                        default=(512, 512, 512))
    args = parser.parse_args()
    run_gemm(args.mnk)
