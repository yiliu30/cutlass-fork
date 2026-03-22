# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Step 2: Shared Memory Tiling — GMEM -> SMEM -> RMEM -> GMEM
============================================================

Builds on Step 1 by adding shared memory as a staging buffer between
global memory and registers. This is the fundamental GPU optimization:
threads cooperatively load a tile into fast shared memory, synchronize,
then each thread reads its partition from shared memory.

NEW CONCEPTS introduced here (vs Step 1):
  - SmemAllocator + allocate_tensor — allocate shared memory buffers
  - cute.arch.sync_threads() — __syncthreads() barrier
  - make_tiled_copy_tv for G2S copy — thread cooperation to load SMEM
  - Two-phase loop: copy G->S, sync, compute from S

DATA PATH:  GMEM -> SMEM (cooperative load) -> sync -> RMEM -> FMA -> RMEM -> GMEM

Why shared memory helps:
  In Step 1, every thread independently loads from GMEM. With SMEM, threads
  cooperatively load a tile once, then ALL threads can read from it — massive
  reduction in GMEM bandwidth. SMEM is ~100x faster than GMEM.

Fixed configuration (same as Step 1):
  - A: (M, K) M-major, B: (N, K) N-major, C: (M, N) N-major
  - Tile: 128x128x8, Threads: 256
  - 1 SMEM stage (no pipelining yet)

Run:
    python step2_smem_gemm.py --mnk 512,512,512
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

    # ---- Tile global tensors for this CTA ----
    gA = cute.local_tile(mA, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, 1, None))

    # ---- NEW: Allocate shared memory buffers ----
    # sA: (BM, BK) = (128, 8), sB: (BN, BK) = (128, 8)
    # These are shared across all 256 threads in the block
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

    # ---- Partition copy tiles for this thread ----
    # tiled_copy distributes the (BM, BK) copy across NUM_THREADS threads
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)  # source: global memory
    tAsA = thr_copy_A.partition_D(sA)  # dest: shared memory
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    # ---- Partition MMA tiles for this thread ----
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)  # thread's A slice from SMEM
    tCsB = thr_mma.partition_B(sB)  # thread's B slice from SMEM
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    num_k_tiles = cute.size(gA, mode=[2])
    num_k_blocks = cute.size(tCrA, mode=[2])

    # ---- K-tile mainloop ----
    for k_tile in range(num_k_tiles):
        # Phase 1: Cooperative GMEM -> SMEM copy
        # All 256 threads participate to load the (128, 8) tile
        cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile], tAsA)
        cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile], tBsB)

        # Phase 2: Synchronize — wait for all threads to finish SMEM writes
        cute.arch.sync_threads()

        # Phase 3: SMEM -> RMEM -> MMA
        for k_block in range(num_k_blocks):
            cute.autovec_copy(tCsA[None, None, k_block], tCrA[None, None, k_block])
            cute.autovec_copy(tCsB[None, None, k_block], tCrB[None, None, k_block])
            cute.gemm(tiled_mma, tCrC,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block],
                      tCrC)

        # Sync before overwriting SMEM in the next iteration
        cute.arch.sync_threads()

    # ---- Epilogue: RMEM -> GMEM ----
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
    cute.copy(atom, tCrC, tCgC)
    return


# ===========================================================================
# Host function
# ===========================================================================
@cute.jit
def host_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # ---- SMEM layouts (1 stage, no pipeline dimension) ----
    # sA: (BM, BK) M-major = stride (1, BM)
    sA_layout = cute.make_layout((BM, BK), stride=(1, BM))
    # sB: (BN, BK) N-major = stride (1, BN)
    sB_layout = cute.make_layout((BN, BK), stride=(1, BN))

    # ---- Copy atom: synchronous CopyUniversalOp (1 element at a time) ----
    atom_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), mA.element_type
    )

    # Thread layout for G2S copy of A: (BM, BK) with 256 threads
    # 256 threads / 8 cols = 32 threads per column, each handles 128/32 = 4 rows
    # Layout: (32, 8) with stride (8, 1) — threads are k-major
    thr_layout_A = cute.make_layout((NUM_THREADS // BK, BK), stride=(BK, 1))
    val_layout_A = cute.make_layout((1, 1))
    tiled_copy_A = cute.make_tiled_copy_tv(atom_copy, thr_layout_A, val_layout_A)

    # Thread layout for G2S copy of B: same structure
    thr_layout_B = cute.make_layout((NUM_THREADS // BK, BK), stride=(BK, 1))
    val_layout_B = cute.make_layout((1, 1))
    tiled_copy_B = cute.make_tiled_copy_tv(atom_copy, thr_layout_B, val_layout_B)

    # ---- Tiled MMA (same as Step 1) ----
    op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    permutation_tiler_M = cute.make_layout((16, 4), stride=(4, 1))
    permutation_tiler_N = cute.make_layout((16, 4), stride=(4, 1))
    tiled_mma = cute.make_tiled_mma(
        op, atoms_layout,
        permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
    )

    grid = (*cute.ceil_div(mC.shape, (BM, BN)), 1)

    gemm_kernel(mA, mB, mC, sA_layout, sB_layout,
                tiled_copy_A, tiled_copy_B, tiled_mma).launch(
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

    print(f"Step 2: SMEM Tiling GEMM (GMEM -> SMEM -> RMEM -> GMEM)")
    print(f"  Problem: M={M}, N={N}, K={K}")
    print(f"  Tile: {BM}x{BN}x{BK}, Threads: {NUM_THREADS}")
    print(f"  SMEM: 1 stage, sync copy (CopyUniversalOp)")

    print("  Compiling...")
    compiled = cute.compile(host_gemm, mA, mB, mC)

    print("  Running...")
    compiled(mA, mB, mC)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)
    print("  PASS — results match reference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: SMEM Tiling GEMM")
    parser.add_argument("--mnk", type=lambda s: tuple(int(x) for x in s.split(",")),
                        default=(512, 512, 512))
    args = parser.parse_args()
    run_gemm(args.mnk)
