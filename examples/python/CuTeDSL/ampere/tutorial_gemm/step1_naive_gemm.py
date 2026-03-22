# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Step 1: Naive GEMM — Direct GMEM -> RMEM -> GMEM (no shared memory)
====================================================================

This is the simplest possible GEMM in CuTe DSL. It demonstrates the core
abstractions without any optimization:

NEW CONCEPTS introduced here:
  - @cute.kernel / @cute.jit — GPU kernel vs host-side setup
  - cute.local_tile — partition a tensor into CTA-level tiles
  - cute.make_tiled_mma + MmaUniversalOp — SIMT FMA (scalar 1x1x1 MMA)
  - thr_mma.partition_A/B/C — thread-level data partitioning for MMA
  - tiled_mma.make_fragment_C — allocate accumulator registers
  - cute.gemm — execute the tiled MMA
  - cute.copy with CopyUniversalOp — register-to-global-memory epilogue

DATA PATH:  GMEM -> RMEM (load) -> FMA -> RMEM (accum) -> GMEM (store)

Fixed configuration:
  - A: (M, K) M-major (column-major in BLAS terms, stride = (1, M))
  - B: (N, K) N-major (stride = (1, N))
  - C: (M, N) N-major (row-major, stride = (N, 1))
  - Tile: 128x128x8, Threads: 256
  - Problem size must be divisible by tile size (no predication)

Run:
    python step1_naive_gemm.py --mnk 512,512,512
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
# Kernel: runs on the GPU
# ===========================================================================
@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,         # (M, K) M-major
    mB: cute.Tensor,         # (N, K) N-major
    mC: cute.Tensor,         # (M, N) N-major
    tiled_mma: cute.TiledMma,
):
    """Each thread block computes a (BM, BN) tile of C by looping over K."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    cta_tiler = (BM, BN, BK)

    # ----- Step A: Tile the global tensors for this CTA -----
    # gA: (BM, BK, num_k_tiles) — A's tile for this block row
    # gB: (BN, BK, num_k_tiles) — B's tile for this block col
    # gC: (BM, BN)              — C's output tile
    print(f"Thread ({tidx}) - block ({bidx}, {bidy}), cta_tiler: {cta_tiler}")
    gA = cute.local_tile(mA, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, None, 1))
                                                                        # ^ keep M, drop N, keep K 
    gB = cute.local_tile(mB, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(None, 1, 1))
                                                                        # ^ drop M, keep N, keep K
    gC = cute.local_tile(mC, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, 1, None))
                                                                        # ^ keep M, keep N, drop K
    """
    proj for A: (1, None, 1)  → keep M, drop N, keep K  → tiler becomes (128, 8)                         
    proj for B: (None, 1, 1)  → drop M, keep N, keep K  → tiler becomes (128, 8)                         
    proj for C: (1, 1, None)  → keep M, keep N, drop K  → tiler becomes (128, 128)  
    """
    print(f"Thread ({tidx}) - gA: {gA}")
    # Thread (?) - gA: tensor<ptr<f32, gmem, align<16>> o (128,8,4):(1,256,2048)>
    print(f"Thread ({tidx}) - gB: {gB}")
    print(f"Thread ({tidx}) - gC: {gC}")
    if tidx == 0 and bidx == 0 and bidy == 0:
        cute.print_tensor(gA)

    # ----- Step B: Get this thread's MMA partition -----
    thr_mma = tiled_mma.get_slice(tidx)
    print(f"Thread ({tidx}) - thr_mma: {thr_mma}")
    """
    Thread (?) - thr_mma: Tiled MMA
    Thr Layout VMNK: (1,16,16,1):(0,16,1,0)
    Permutation MNK: ((16,4):(4,1),(16,4):(4,1),_)
    MMA Atom
    ThrID:           1:0
    Shape MNK:       (1,1,1)
    TV Layout A:     (1,1):(0,0)
    TV Layout B:     (1,1):(0,0)
    TV Layout C:     (1,1):(0,0)

    """
    # Partition C for this thread and allocate accumulator
    # partition_C(gC) — Which Elements Are Mine?
    # Thread (?) - gC: tensor<ptr<f32, gmem, align<16>> o (128,128):(128,1)>
    tCgC = thr_mma.partition_C(gC)        # thread's view of global C
    tCrC = tiled_mma.make_fragment_C(tCgC) # register accumulator
    print(f"Thread ({tidx}) - tCgC: {tCgC}")
    print(f"Thread ({tidx}) - tCrC: {tCrC}")
    # Thread (?) - tCgC: tensor<ptr<f32, gmem, align<16>> o (1,(4,2),(4,2)):(0,(128,8192),(1,64))>
    # Thread (?) - tCrC: tensor<ptr<f32, rmem> o (1,(4,2),(4,2)):(0,(1,4),(8,32))>
    tCrC.fill(0.0)
    """
      tiled_mma           (all-threads object, passed from host)                                           
      │                                                                                                
      ├── get_slice(tidx)     → ThrMMA  (one-thread cursor)                                            
      │                            │                                                                   
      │                            ├── partition_C(gC) → this thread's C elements                      
      │                            ├── partition_A(gA) → this thread's A elements                      
      │                            └── partition_B(gB) → this thread's B elements                      
      │                                                                                                
      ├── make_fragment_C(tCgC) → register accumulator (same shape, in RMEM)                           
      └── (shared across all threads — compile-time layout info)   


        tCgC, tCrC — these follow a strict CuTe naming pattern:                                              
                                                                                                            
        t  C  g  C                                                                                           
        ↑  ↑  ↑  ↑                                                                                           
        |  |  |  └── which tensor: C (could be A, B)                                                         
        |  |  └───── memory space: g=global, r=register, s=shared                                            
        |  └──────── partitioned for: C (the MMA's C-partition layout)                                       
        └─────────── "t" = thread-partitioned tensor        
    """
    num_k_tiles = cute.size(gA, mode=[2])
    print(f"Thread ({tidx}) - num_k_tiles: {num_k_tiles}")

    # ----- Step C: K-tile loop — load from GMEM, compute -----
    for k_tile in range(num_k_tiles):
        # Partition A and B for this k-tile
        # gA: tensor<ptr<f32, gmem, align<16>> o (128,8,4):(1,256,2048)>
        gA_ktile= gA[None, None, k_tile]  # add MMA-partition dims back for partitioning
        print(f"Thread ({tidx}) - gA_ktile: {gA_ktile}")
        # Thread (?) - gA_ktile: tensor<ptr<f32, gmem, align<16>> o (128,8):(1,256)>
        tCgA = thr_mma.partition_A(gA[None, None, k_tile])  # thread's A slice
        # Thread (?) tCgA: tensor<ptr<f32, gmem, align<16>> o (1,(4,2),8):(0,(1,64),256)>
        # (FrgV, RestM, RestK)  =  (V, M, K) = (1, (4,2), 8)
        tCgB = thr_mma.partition_B(gB[None, None, k_tile])  # thread's B slice
        # print first thread's view of the tiles being loaded from GMEM
        # if tidx == 0 and bidx == 0 and bidy == 0:
        #     print(f"Thread ({tidx}) tCgA: {tCgA}")
        #     cute.print_tensor(tCgA)
            # cute.print_tensor(tCgB)
        # Copy GMEM -> RMEM (registers)
        tCrA = cute.make_fragment_like(tCgA)
        # Thread (?) tCrA: tensor<ptr<f32, rmem, align<32>> o (1,(4,2),8):(0,(1,4),8)>
        tCrB = cute.make_fragment_like(tCgB)
        cute.autovec_copy(tCgA, tCrA)
        cute.autovec_copy(tCgB, tCrB)
        if tidx == 0 and bidx == 0 and bidy == 0 and k_tile == 0:
            print(f"Thread ({tidx}) tCgA: {tCgA }\n===============")
            print(f"Thread ({tidx}) tCrA: {tCrA }\n===============")
            cute.print_tensor(tCrA)
            print(f"\n===============")
            # print(f"Thread ({tidx}) tCrB: {tCrB}")
            # cute.print_tensor(tCrB)

        # Inner k-block loop (within the BK tile, driven by MMA partitioning)
        num_k_blocks = cute.size(tCrA, mode=[2])
        # num_k_blocks 8
        #  tCrA: tensor<ptr<f32, rmem, align<32>> o (1,(4,2),8):(0,(1,4),8)>
        for k_block in range(num_k_blocks):
            cute.gemm(tiled_mma, tCrC,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block],
                      tCrC)

    # ----- Step D: Epilogue — write accumulator to GMEM -----
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
    print(f"Thread ({tidx}) - copy atom: {atom}")
    print(f"Thread ({tidx}) - tCrC before copy: {tCrC}")
    print(f"Thread ({tidx}) - tCgC before copy: {tCgC}")
    cute.copy(atom, tCrC, tCgC)
    return


# ===========================================================================
# Host function: sets up MMA, computes grid, launches kernel
# ===========================================================================
@cute.jit
def host_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # ---- Create tiled MMA ----
    # MmaUniversalOp(Float32) is a 1x1x1 scalar FMA — the simplest MMA atom.
    # atoms_layout tiles 256 atoms in a (16, 16, 1) grid = 256 threads.
    # Each thread thus owns (128/16)*(128/16) = 8*8 = 64 output elements.
    # The permutation makes each thread handle 4 consecutive elements for
    # better memory access patterns.
    op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    # 16
    permutation_tiler_M = cute.make_layout((16, 4), stride=(4, 1))
    """
      permutation_tiler_M = cute.make_layout((16, 4), stride=(4, 1))                                                              
                                          ^^  ^                                                                               
                                          F   R                                                                               
                                          │   │                                                                               
                                          │   └── R=4: consecutive group size                                                 
                                          │       = 4 floats = 128 bits = one vector load                                     
                                          │                                                                                   
                                          └── F=16: just coverage/R = 64/4                                                    
                                              (follows from R, not independently chosen)  
    """
    permutation_tiler_N = cute.make_layout((16, 4), stride=(4, 1))
    tiled_mma = cute.make_tiled_mma(
        op,
        atoms_layout,
        permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
    )
    print(f"op: {op}")
    print(f"atoms_layout: {atoms_layout}")
    print(f"permutation_tiler_M: {permutation_tiler_M}")
    print(f"permutation_tiler_N: {permutation_tiler_N}")
    print(f"tiled_mma: {tiled_mma}")
    """
    tiled_mma: Tiled MMA
    Thr Layout VMNK: (1,16,16,1):(0,16,1,0)
    Permutation MNK: ((16,4):(4,1),(16,4):(4,1),_)
    MMA Atom
    ThrID:           1:0
    Shape MNK:       (1,1,1)
    TV Layout A:     (1,1):(0,0)
    TV Layout B:     (1,1):(0,0)
    TV Layout C:     (1,1):(0,0)
    """
    # ---- Grid dimensions ----
    grid = (*cute.ceil_div(mC.shape, (BM, BN)), 1)
    block = (NUM_THREADS, 1, 1)
    print(f"grid: {grid}")
    print(f"block: ({NUM_THREADS}, 1, 1)")
    gemm_kernel(mA, mB, mC, tiled_mma).launch(
        grid=grid,
        block=block,
    )


def inspect_torch_tensor(t: torch.Tensor, name: str):
    print(f"{name}: shape={t.shape}, stride={t.stride()}, dtype={t.dtype}, device={t.device}")

# ===========================================================================
# Run function: creates tensors, compiles, executes, verifies
# ===========================================================================
def run_gemm(mnk: Tuple[int, int, int]):
    import torch

    M, N, K = mnk
    assert M % BM == 0 and N % BN == 0 and K % BK == 0, (
        f"Problem size ({M}, {N}, {K}) must be divisible by tile ({BM}, {BN}, {BK})"
    )

    torch.manual_seed(42)
    # # A: (M, K) M-major => stride (1, M). Create (K, M) then permute.
    # a = torch.randn(K, M, dtype=torch.float32, device="cuda").permute(1, 0)
    # # B: (N, K) N-major => stride (1, N). Create (K, N) then permute.
    # b = torch.randn(K, N, dtype=torch.float32, device="cuda").permute(1, 0)
    # Init A and B as arange for easier debugging and verification. A[m,k] = m*K+k, B[n,k] = n*K+k.
    a = torch.arange(M * K, dtype=torch.float32, device="cuda").reshape(K, M).permute(1, 0)
    b = torch.arange(N * K, dtype=torch.float32, device="cuda").reshape(K, N).permute(1, 0)
    # a: shape=(256, 32), stride=(1, 256)
    # b: shape=(128, 32), stride=(1, 128)
    inspect_torch_tensor(a, "a")
    inspect_torch_tensor(b, "b")
    # C: (M, N) N-major => stride (N, 1). Standard row-major.
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    inspect_torch_tensor(c, "c")

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print(f"Step 1: Naive GEMM (GMEM -> RMEM -> GMEM)")
    print(f"  Problem: M={M}, N={N}, K={K}")
    print(f"  Tile: {BM}x{BN}x{BK}, Threads: {NUM_THREADS}")
    print(f"  MMA: FP32 SIMT FMA (1x1x1)")
    print(f"  A stride: {a.stride()}, B stride: {b.stride()}, C stride: {c.stride()}")

    print("  Compiling...")
    compiled = cute.compile(host_gemm, mA, mB, mC)

    print("  Running...")
    compiled(mA, mB, mC)
    torch.cuda.synchronize()

    # Verify: C = A @ B^T in standard math, but in our layout:
    # A is (M,K) M-major, B is (N,K) N-major => C[m,n] = sum_k A[m,k]*B[n,k]
    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)
    print("  PASS — results match reference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Naive GEMM")
    parser.add_argument("--mnk", type=lambda s: tuple(int(x) for x in s.split(",")),
                        # default=(512,256, 128)
                        default=(256, 128, 32)
                        
                        )
    args = parser.parse_args()
    run_gemm(args.mnk)
