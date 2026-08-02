# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Step 1 ANNOTATED: Every operation explained with types and memory effects
=========================================================================

Legend:
  [VIEW]    — no data movement, just reinterprets pointer+layout (free, compile-time)
  [COMPUTE] — real ALU work (FMA, add, mul)
  [LOAD]    — real memory read  (global→register, shared→register)
  [STORE]   — real memory write (register→global, register→shared)
  [ALLOC]   — reserves register space (register file allocation)

Run:
    python step1_annotated.py --mnk 256,128,32
"""

import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


# ===========================================================================
# Constants
# ===========================================================================
BM, BN, BK = 128, 128, 8  # CTA tile shape
NUM_THREADS = 256


# ===========================================================================
# Kernel
# ===========================================================================
@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,          # INPUT:  Tensor<ptr<f32, gmem>, Layout<(M,K), (1,M)>>
    mB: cute.Tensor,          # INPUT:  Tensor<ptr<f32, gmem>, Layout<(N,K), (1,N)>>
    mC: cute.Tensor,          # OUTPUT: Tensor<ptr<f32, gmem>, Layout<(M,N), (N,1)>>
    tiled_mma: cute.TiledMma, # INPUT:  compile-time MMA descriptor (no data)
):
    """
    Each CTA (thread block) computes one (BM=128, BN=128) tile of C.
    256 threads collaborate; each thread computes 64 output elements.
    """

    # =========================================================================
    # [VIEW] Get thread/block indices — reads hardware registers, essentially free
    # =========================================================================
    tidx, _, _ = cute.arch.thread_idx()   # → Int32 (threadIdx.x, 0..255)
    bidx, bidy, _ = cute.arch.block_idx() # → Int32, Int32 (blockIdx.x, blockIdx.y)

    cta_tiler = (BM, BN, BK)  # compile-time constant tuple, not a tensor

    # =========================================================================
    # STEP A: local_tile — Slice global tensors for this CTA
    # =========================================================================
    # [VIEW] No data movement. Computes new pointer offset + new layout.
    #
    # What it does conceptually:
    #   1. Apply `proj` to select relevant dimensions from the 3D tiler
    #   2. Divide the tensor's dimensions by the projected tiler sizes
    #   3. Offset to `coord`-th tile in each divided dimension
    #   Result: a smaller tensor pointing into the SAME global memory
    #
    # -------------------------------------------------------------------------
    # local_tile for A:
    #   Input:  mA = Tensor<ptr<f32,gmem>, (256, 32):(1, 256)>
    #   tiler after proj=(1,None,1): (128, 8) — keeps M and K, drops N
    #   coord used: (bidx, _, None) — bidx selects M-tile, None means "keep all K-tiles"
    #   Output: gA = Tensor<ptr<f32,gmem>, (128, 8, 4):(1, 256, 2048)>
    #                                       ^^^  ^^  ^
    #                                       BM   BK  num_k_tiles = K/BK = 32/8 = 4
    # -------------------------------------------------------------------------
    gA = cute.local_tile(mA, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, None, 1))

    # -------------------------------------------------------------------------
    # local_tile for B:
    #   Input:  mB = Tensor<ptr<f32,gmem>, (128, 32):(1, 128)>
    #   tiler after proj=(None,1,1): (128, 8) — keeps N and K, drops M
    #   Output: gB = Tensor<ptr<f32,gmem>, (128, 8, 4):(1, 128, 1024)>
    #                                       ^^^  ^^  ^
    #                                       BN   BK  num_k_tiles
    # -------------------------------------------------------------------------
    gB = cute.local_tile(mB, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(None, 1, 1))

    # -------------------------------------------------------------------------
    # local_tile for C:
    #   Input:  mC = Tensor<ptr<f32,gmem>, (256, 128):(128, 1)>
    #   tiler after proj=(1,1,None): (128, 128) — keeps M and N, drops K
    #   Output: gC = Tensor<ptr<f32,gmem>, (128, 128):(128, 1)>
    #   (Only one tile since problem_M/BM = 2, this picks tile `bidx`)
    # -------------------------------------------------------------------------
    gC = cute.local_tile(mC, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, 1, None))

    # =========================================================================
    # STEP B: Partition — assign elements to this thread
    # =========================================================================

    # -------------------------------------------------------------------------
    # [VIEW] get_slice: select this thread's partition descriptor from tiled_mma
    #   Input:  tiled_mma (all-threads descriptor) + tidx (scalar)
    #   Output: thr_mma — a "cursor" encoding which elements belong to thread tidx
    #   No data, no memory. Pure index computation.
    # -------------------------------------------------------------------------
    thr_mma = tiled_mma.get_slice(tidx)

    # -------------------------------------------------------------------------
    # [VIEW] partition_C: reinterpret gC through this thread's ownership lens
    #   Input:  gC = Tensor<ptr<f32,gmem>, (128, 128):(128, 1)>
    #   Output: tCgC = Tensor<ptr<f32,gmem>, (1, (4,2), (4,2)):(0, (128,8192), (1,64))>
    #                                         ^   ^^^^   ^^^^
    #                                         V   M_rest N_rest
    #
    #   V=1: scalar atom → 1 value per "atom invocation"
    #   M_rest=(4,2): 8 positions along M
    #     stride (128, 8192) means: 4 consecutive rows (stride 128 = one row),
    #                               then jump 8192 = 64 rows for next group
    #   N_rest=(4,2): 8 positions along N
    #     stride (1, 64) means: 4 consecutive columns,
    #                           then jump 64 columns for next group
    #
    #   Still points to GLOBAL MEMORY. No copy happened.
    # -------------------------------------------------------------------------
    tCgC = thr_mma.partition_C(gC)

    # -------------------------------------------------------------------------
    # [ALLOC] make_fragment_C: allocate register-file accumulator
    #   Input:  tCgC (used only for shape/type, not data)
    #   Output: tCrC = Tensor<ptr<f32,RMEM>, (1,(4,2),(4,2)):(0,(1,4),(8,32))>
    #
    #   This ACTUALLY allocates 64 × 4 bytes = 256 bytes of registers.
    #   The strides are dense/packed for register storage (different from gmem strides).
    # -------------------------------------------------------------------------
    tCrC = tiled_mma.make_fragment_C(tCgC)

    # -------------------------------------------------------------------------
    # [COMPUTE] fill: write 0.0 to all 64 register slots
    #   Real work: 64 register writes (MOV immediate)
    # -------------------------------------------------------------------------
    tCrC.fill(0.0)

    # =========================================================================
    # STEP C: K-tile loop — the actual computation
    # =========================================================================
    num_k_tiles = cute.size(gA, mode=[2])  # [VIEW] → 4 (compile-time constant)

    for k_tile in range(num_k_tiles):  # 4 iterations

        # ---------------------------------------------------------------------
        # [VIEW] Slice one K-tile: gA[:, :, k_tile]
        #   Input:  gA = Tensor<..., (128, 8, 4):(1, 256, 2048)>
        #   Output: gA_k = Tensor<..., (128, 8):(1, 256)>
        #   Just advances pointer by k_tile * 2048 * sizeof(f32), new layout drops dim 2
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # [VIEW] partition_A: thread's view of A for this k-tile
        #   Input:  gA[:,:,k_tile] = Tensor<ptr<f32,gmem>, (128, 8):(1, 256)>
        #   Output: tCgA = Tensor<ptr<f32,gmem>, (1, (4,2), 8):(0, (1,64), 256)>
        #                                         ^   ^^^^   ^
        #                                         V   M_rest K (full BK=8)
        #
        #   (4,2) matches partition_C's M pattern — same 8 M positions this thread owns
        #   K=8 means this thread needs all 8 K values (for its M positions)
        #   Total elements: 1 × 8 × 8 = 64 per thread per k_tile
        # ---------------------------------------------------------------------
        tCgA = thr_mma.partition_A(gA[None, None, k_tile])

        # ---------------------------------------------------------------------
        # [VIEW] partition_B: thread's view of B for this k-tile
        #   Input:  gB[:,:,k_tile] = Tensor<ptr<f32,gmem>, (128, 8):(1, 128)>
        #   Output: tCgB = Tensor<ptr<f32,gmem>, (1, (4,2), 8):(0, (1,64), 128)>
        #                                         ^   ^^^^   ^
        #                                         V   N_rest K
        # ---------------------------------------------------------------------
        tCgB = thr_mma.partition_B(gB[None, None, k_tile])

        # ---------------------------------------------------------------------
        # [ALLOC] make_fragment_like: allocate register space for A tile
        #   Output: tCrA = Tensor<ptr<f32,RMEM>, (1,(4,2),8)>
        #   64 floats = 256 bytes of registers (could be optimized to 8 if streamed)
        # ---------------------------------------------------------------------
        tCrA = cute.make_fragment_like(tCgA)

        # ---------------------------------------------------------------------
        # [ALLOC] make_fragment_like: allocate register space for B tile
        #   Output: tCrB = Tensor<ptr<f32,RMEM>, (1,(4,2),8)>
        #   64 floats = 256 bytes of registers
        # ---------------------------------------------------------------------
        tCrB = cute.make_fragment_like(tCgB)

        # ---------------------------------------------------------------------
        # ██████████████████████████████████████████████████████████████████████
        # [LOAD] autovec_copy: GLOBAL MEMORY → REGISTERS (real data transfer!)
        # ██████████████████████████████████████████████████████████████████████
        #   Input:  tCgA in GMEM (64 elements scattered with strides (1,64,256))
        #   Output: tCrA in RMEM (64 elements packed)
        #
        #   Hardware: generates LDG (load global) instructions.
        #   With stride (1,64,256), elements are NOT contiguous → poor coalescing.
        #   This is the #1 perf bottleneck: 64 scattered global loads per thread.
        #
        #   Bandwidth cost: 64 × 4B = 256B per thread × 256 threads = 64KB per k_tile
        #   (for A alone; B doubles it to 128KB)
        # ██████████████████████████████████████████████████████████████████████
        cute.autovec_copy(tCgA, tCrA)
        cute.autovec_copy(tCgB, tCrB)

        # ---------------------------------------------------------------------
        # Inner K loop: within this BK=8 tile, do element-wise FMA
        # ---------------------------------------------------------------------
        num_k_blocks = cute.size(tCrA, mode=[2])  # [VIEW] → 8

        for k_block in range(num_k_blocks):  # 8 iterations

            # -----------------------------------------------------------------
            # [VIEW] Slice k_block from fragments
            #   tCrA[:,:,k_block] → Tensor<f32,RMEM, (1,(4,2))> — 8 A values
            #   tCrB[:,:,k_block] → Tensor<f32,RMEM, (1,(4,2))> — 8 B values
            # -----------------------------------------------------------------

            # -----------------------------------------------------------------
            # █████████████████████████████████████████████████████████████████
            # [COMPUTE] cute.gemm: THE ACTUAL MATH
            # █████████████████████████████████████████████████████████████████
            #   For 1×1×1 scalar atom, this expands to:
            #     for each (m, n) in thread's 8×8 output positions:
            #       tCrC[m,n] += tCrA[m,k_block] * tCrB[n,k_block]
            #
            #   = 8 × 8 = 64 FMA instructions per call
            #   × 8 k_blocks × 4 k_tiles = 2048 FMAs per thread total
            #   × 256 threads = 524,288 FMAs = matches M×N×K = 256×128×32/2 CTAs
            #
            #   Hardware: FFMA (fused float multiply-add) instructions
            #   All operands are in registers — this part is fast.
            # █████████████████████████████████████████████████████████████████
            cute.gemm(tiled_mma, tCrC,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block],
                      tCrC)

    # =========================================================================
    # STEP D: Epilogue — write results to global memory
    # =========================================================================

    # -------------------------------------------------------------------------
    # [VIEW] make_copy_atom: creates a copy descriptor (no data)
    #   CopyUniversalOp = simplest possible copy (one element at a time, STG)
    # -------------------------------------------------------------------------
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)

    # -------------------------------------------------------------------------
    # ██████████████████████████████████████████████████████████████████████████
    # [STORE] cute.copy: REGISTERS → GLOBAL MEMORY (real data transfer!)
    # ██████████████████████████████████████████████████████████████████████████
    #   Input:  tCrC in RMEM, shape (1,(4,2),(4,2)) — 64 accumulated values
    #   Output: tCgC in GMEM, shape (1,(4,2),(4,2)) strides (0,(128,8192),(1,64))
    #
    #   Hardware: STG (store global) instructions.
    #   Same scattering problem as loads — 64 stores with non-contiguous addresses.
    #   Cost: 64 × 4B = 256B per thread × 256 threads = 64KB written back.
    # ██████████████████████████████████████████████████████████████████████████
    cute.copy(atom, tCrC, tCgC)


# ===========================================================================
# Host function
# ===========================================================================
@cute.jit
def host_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    """
    Host-side setup: constructs TiledMMA, computes grid, launches kernel.
    This runs on CPU at compile time — NO GPU execution here.
    """

    # -------------------------------------------------------------------------
    # MMA atom: 1×1×1 scalar FMA (Float32)
    # One thread, one element, one multiply-add. Simplest possible.
    # -------------------------------------------------------------------------
    op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)

    # -------------------------------------------------------------------------
    # atoms_layout: (16, 16, 1) stride (16, 1, 0)
    # Arranges 256 atoms (= 256 threads) in a 16×16 grid over M×N
    # Each atom "natively" covers 1×1, so 16×16 atoms cover 16×16 elements
    # -------------------------------------------------------------------------
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))

    # -------------------------------------------------------------------------
    # PERMUTATION — THE ONE DESIGN CHOICE
    # -------------------------------------------------------------------------
    # permutation = (Groups, Chunk):(group_stride, chunk_stride)
    #
    #   Groups       = how many groups to divide positions into (one per thread)
    #   Chunk        = how many consecutive positions per group
    #   group_stride = distance between the START of adjacent groups
    #   chunk_stride = distance between elements WITHIN a group
    #
    # Thread k, element r → position = k × group_stride + r × chunk_stride
    #
    # The permutation covers Groups×Chunk positions total.
    # If tile_dim > Groups×Chunk, a "rest" factor repeats the pattern:
    #   rest = tile_dim / (Groups × Chunk)
    # Total positions per thread = Chunk × rest
    #
    # -------------------------------------------------------------------------
    # (16, 4):(4, 1) means:
    #   16 groups, 4 elements per group (chunk), groups start 4 apart, elements stride 1
    #   Group k starts at position k×4, contains {k*4, k*4+1, k*4+2, k*4+3}
    #   Covers 16×4 = 64 positions. Tile=128, rest=2 → second copy at +64.
    #   Thread 0 → {0,1,2,3} ∪ {64,65,66,67} = 8 positions
    #
    # -------------------------------------------------------------------------
    # ALTERNATIVES (same thread count, same element count, different patterns):
    #
    # (16,4):(4,1)  Thread 0 → {0,1,2,3, 64,65,66,67}
    #               chunk=4 consecutive, chunk_stride=1 (good for 128-bit vectorized loads)
    #
    # (16,4):(1,16) Thread 0 → {0,16,32,48, 64,80,96,112}
    #               chunk=4, chunk_stride=16 (scattered, bad for vectorization)
    #
    # (16,8):(8,1)  Thread 0 → {0,1,2,3,4,5,6,7}
    #               chunk=8 consecutive, no rest (simplest, but less A/B data reuse)
    #
    # -------------------------------------------------------------------------
    # logical_divide result (M-dim, base stride=128 because C is row-major (M,N):(N,1)):
    #
    # Permutation        | Result shape:stride             | Thread 0 rows
    # (16,4):(4,1)       | (16,4,2):(512,128,8192)        | {0,1,2,3,64,65,66,67}
    # (16,4):(1,16)      | (16,4,2):(128,2048,8192)       | {0,16,32,48,64,80,96,112}
    # (16,8):(8,1)       | (16,8):(1024,128)              | {0,1,2,3,4,5,6,7}
    #
    # Formula: output_stride = perm_stride × input_base_stride
    #   e.g., group_stride=4 × base=128 = 512 (group spacing in memory)
    #
    # -------------------------------------------------------------------------
    # logical_divide result (N-dim, base stride=1 because C is row-major (M,N):(N,1)):
    #
    # Permutation        | Result shape:stride             | Thread 0 cols
    # (16,4):(4,1)       | (16,4,2):(4,1,64)              | {0,1,2,3,64,65,66,67}
    # (16,4):(1,16)      | (16,4,2):(1,16,64)             | {0,16,32,48,64,80,96,112}
    # (16,8):(8,1)       | (16,8):(8,1)                   | {0,1,2,3,4,5,6,7}
    #
    # Formula: output_stride = perm_stride × input_base_stride
    #   e.g., group_stride=4 × base=1 = 4 (group spacing in columns)
    #
    # NOTE: With SAME permutation (16,4):(4,1) for both M and N:
    #   Thread 0 owns rows {0,1,2,3,64,65,66,67} × cols {0,1,2,3,64,65,66,67}
    #   = 8×8 = 64 elements of C (the 4×4 blocks pattern in the visual)
    # -------------------------------------------------------------------------
    permutation_M = cute.make_layout((16, 4), stride=(4, 1))
    permutation_N = cute.make_layout((16, 4), stride=(4, 1))

    tiled_mma = cute.make_tiled_mma(
        op,
        atoms_layout,
        permutation_mnk=(permutation_M, permutation_N, None),
    )

    # -------------------------------------------------------------------------
    # Grid/Block dimensions
    #   grid = (M/BM, N/BN, 1) = (256/128, 128/128, 1) = (2, 1, 1)
    #   block = (256, 1, 1)
    # -------------------------------------------------------------------------
    grid = (*cute.ceil_div(mC.shape, (BM, BN)), 1)
    block = (NUM_THREADS, 1, 1)

    gemm_kernel(mA, mB, mC, tiled_mma).launch(grid=grid, block=block)


# ===========================================================================
# Summary of real work per thread per CTA tile:
# ===========================================================================
#
#   Operation          | Count        | Type    | Notes
#   -------------------|--------------|---------|-------------------------------
#   LDG (load A)       | 64 × 4 = 256| [LOAD]  | scattered, poor coalescing
#   LDG (load B)       | 64 × 4 = 256| [LOAD]  | scattered, poor coalescing
#   FFMA (compute)     | 64 × 32=2048| [COMPUTE]| all in registers, fast
#   STG (store C)      | 64          | [STORE] | scattered, poor coalescing
#   -------------------|--------------|---------|-------------------------------
#   Total loads:  512 × 4B = 2KB from GMEM
#   Total stores: 64 × 4B = 256B to GMEM
#   Total FLOPs:  2048 FMAs = 4096 flops
#   Arithmetic intensity: 4096 / (2KB+256B) = ~1.8 flops/byte (very low!)
#
#   Why it's slow:
#   1. No shared memory → same A/B values loaded by multiple threads (no reuse)
#   2. Scattered access patterns → can't coalesce into 128B transactions
#   3. Scalar FMA → no tensor core acceleration
#
# ===========================================================================


def run_gemm(mnk: Tuple[int, int, int]):
    M, N, K = mnk
    assert M % BM == 0 and N % BN == 0 and K % BK == 0

    torch.manual_seed(42)
    a = torch.arange(M * K, dtype=torch.float32, device="cuda").reshape(K, M).permute(1, 0)
    b = torch.arange(N * K, dtype=torch.float32, device="cuda").reshape(K, N).permute(1, 0)
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print(f"Step 1 Annotated: Naive GEMM")
    print(f"  Problem: M={M}, N={N}, K={K}")
    print(f"  Tile: {BM}x{BN}x{BK}, Threads: {NUM_THREADS}")

    compiled = cute.compile(host_gemm, mA, mB, mC)
    compiled(mA, mB, mC)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)
    print("  PASS!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=lambda s: tuple(int(x) for x in s.split(",")),
                        default=(256, 128, 32))
    args = parser.parse_args()
    run_gemm(args.mnk)
