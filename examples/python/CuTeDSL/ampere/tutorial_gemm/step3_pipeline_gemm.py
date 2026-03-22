# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Step 3: Multi-Stage Async Pipeline — cp.async + 3-Stage Buffering
=================================================================

Builds on Step 2 by replacing synchronous copies with Ampere's cp.async
hardware instruction and adding multi-stage SMEM buffering.

NEW CONCEPTS introduced here (vs Step 2):
  - CopyG2SOp (cp.async) — hardware async DMA from GMEM to SMEM
  - cp_async_commit_group / cp_async_wait_group — async copy fencing
  - Multi-stage SMEM: sA/sB gain a 3rd dimension for pipeline stages
  - Prologue: prefetch first (num_stages-1) tiles before entering mainloop
  - Mainloop: overlap copy of stage N+2 with compute on stage N

DATA PATH:  GMEM --(cp.async)--> SMEM[stage] --> sync --> RMEM --> FMA --> GMEM

Why pipelining helps:
  In Step 2, we wait for each GMEM->SMEM copy to finish before computing.
  With 3-stage pipelining, we hide GMEM latency: while computing on stage 0,
  we're already loading stage 2. This overlaps memory access with computation.

Pipeline structure (3 stages, labeled 0/1/2):
  Prologue: load stage 0, load stage 1
  Mainloop iteration i:
    - wait for stage (i % 3) to finish loading
    - compute on stage (i % 3)
    - start loading stage ((i+2) % 3) for tile i+2

Fixed configuration:
  - A: (M, K) M-major, B: (N, K) N-major, C: (M, N) N-major
  - Tile: 128x128x8, Threads: 256, Stages: 3

Run:
    python step3_pipeline_gemm.py --mnk 512,512,512
"""

import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# ===========================================================================
# Constants
# ===========================================================================
BM, BN, BK = 128, 128, 8
NUM_THREADS = 256
NUM_STAGES = 3


# ===========================================================================
# Kernel
# ===========================================================================
@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
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

    # ---- NEW: Allocate multi-stage SMEM buffers ----
    # sA: (BM, BK, NUM_STAGES) — 3 copies of the tile for pipelining
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

    # ---- G2S copy partitions (now 4D: includes pipeline stage dim) ----
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)   # (CPY, CPY_M, CPY_K, k_tiles)
    tAsA = thr_copy_A.partition_D(sA)   # (CPY, CPY_M, CPY_K, stages)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    # ---- MMA partitions (now include stage dim on SMEM side) ----
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)  # (MMA, MMA_M, MMA_K, stages)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    k_tile_count = cute.size(tAgA, mode=[3])
    num_k_blocks = cute.size(tCrA, mode=[2])

    # ===========================================================================
    # NEW: Prologue — prefetch first (NUM_STAGES - 1) tiles using cp.async
    # ===========================================================================
    for stage in range(NUM_STAGES - 1):
        if stage < k_tile_count:
            cute.copy(tiled_copy_A, tAgA[None, None, None, stage], tAsA[None, None, None, stage])
            cute.copy(tiled_copy_B, tBgB[None, None, None, stage], tBsB[None, None, None, stage])
        cute.arch.cp_async_commit_group()

    # ===========================================================================
    # NEW: Mainloop — overlapped copy and compute
    # ===========================================================================
    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(NUM_STAGES - 1)

    for k_tile in range(k_tile_count):
        # Wait for the read stage's async copy to complete
        # We want at most (NUM_STAGES - 2) groups still in flight
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
        cute.arch.sync_threads()

        # ---- Compute on the completed stage ----
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]

        for k_block in range(num_k_blocks):
            # S2R copy
            cute.autovec_copy(tCsA_p[None, None, k_block], tCrA[None, None, k_block])
            cute.autovec_copy(tCsB_p[None, None, k_block], tCrB[None, None, k_block])
            # MMA
            cute.gemm(tiled_mma, tCrC,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block],
                      tCrC)

        # ---- Issue next async copy (if there are more tiles) ----
        next_k = k_tile + NUM_STAGES - 1
        if next_k < k_tile_count:
            cute.copy(tiled_copy_A, tAgA[None, None, None, next_k], tAsA[None, None, None, smem_pipe_write])
            cute.copy(tiled_copy_B, tBgB[None, None, None, next_k], tBsB[None, None, None, smem_pipe_write])
        cute.arch.cp_async_commit_group()

        # ---- Advance pipeline pointers ----
        smem_pipe_write = smem_pipe_read
        smem_pipe_read = smem_pipe_read + 1
        if smem_pipe_read == NUM_STAGES:
            smem_pipe_read = cutlass.Int32(0)

    # ---- Epilogue ----
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
    cute.copy(atom, tCrC, tCgC)
    return


# ===========================================================================
# Host function
# ===========================================================================
@cute.jit
def host_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # ---- SMEM layouts with pipeline stage dimension ----
    sA_layout = cute.make_layout(
        (BM, BK, NUM_STAGES),
        stride=(1, BM, BK * BM),
    )
    sB_layout = cute.make_layout(
        (BN, BK, NUM_STAGES),
        stride=(1, BN, BK * BN),
    )

    # ---- NEW: Async copy atom using cp.async ----
    atom_async_copy = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        mA.element_type,
        num_bits_per_copy=32,  # 1 float = 32 bits (not vectorized yet)
    )

    # Thread layout for G2S: same as Step 2
    thr_layout_A = cute.make_layout((NUM_THREADS // BK, BK), stride=(BK, 1))
    val_layout_A = cute.make_layout((1, 1))
    tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy, thr_layout_A, val_layout_A)

    thr_layout_B = cute.make_layout((NUM_THREADS // BK, BK), stride=(BK, 1))
    val_layout_B = cute.make_layout((1, 1))
    tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy, thr_layout_B, val_layout_B)

    # ---- Tiled MMA (same as Steps 1-2) ----
    op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    permutation_tiler_M = cute.make_layout((16, 4), stride=(4, 1))
    permutation_tiler_N = cute.make_layout((16, 4), stride=(4, 1))
    tiled_mma = cute.make_tiled_mma(
        op, atoms_layout,
        permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
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
    a = torch.randn(K, M, dtype=torch.float32, device="cuda").permute(1, 0)
    b = torch.randn(K, N, dtype=torch.float32, device="cuda").permute(1, 0)
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print(f"Step 3: Pipelined GEMM (cp.async + 3-stage SMEM)")
    print(f"  Problem: M={M}, N={N}, K={K}")
    print(f"  Tile: {BM}x{BN}x{BK}, Threads: {NUM_THREADS}, Stages: {NUM_STAGES}")
    print(f"  G2S: cp.async (CopyG2SOp), 32-bit per copy")

    print("  Compiling...")
    compiled = cute.compile(host_gemm, mA, mB, mC)

    print("  Running...")
    compiled(mA, mB, mC)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)
    print("  PASS — results match reference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Pipelined GEMM")
    parser.add_argument("--mnk", type=lambda s: tuple(int(x) for x in s.split(",")),
                        default=(512, 512, 512))
    args = parser.parse_args()
    run_gemm(args.mnk)
