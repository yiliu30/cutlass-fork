"""
Step 1 CUDA Equivalent: Line-by-line mapping from CuTe DSL to pure CUDA
========================================================================

This file shows what the CuTe kernel compiles down to in raw CUDA terms.
Each section maps directly to the annotated step1.

Problem: C[M,N] = A[M,K] * B[N,K]^T  (A is col-major, B is col-major, C is row-major)
Tile: 128×128×8, Threads: 256, arranged as 16×16 grid over output tile.
Each thread computes 8×8 = 64 output elements.

Permutation (16,4):(4,1) gives each thread:
  M positions: [m_base, m_base+1, m_base+2, m_base+3, m_base+64, m_base+65, m_base+66, m_base+67]
  N positions: [n_base, n_base+1, n_base+2, n_base+3, n_base+64, n_base+65, n_base+66, n_base+67]
where m_base = (tidx/16)*4, n_base = (tidx%16)*4
"""

# ==============================================================================
# CUDA PSEUDOCODE (what the GPU actually executes)
# ==============================================================================

CUDA_KERNEL = """
// Compile-time constants (same as CuTe BM, BN, BK, NUM_THREADS)
#define BM 128
#define BN 128
#define BK 8
#define NUM_THREADS 256

// Thread ownership: 16×16 grid, each thread owns 8×8 output elements
// grouped as (4 consecutive + 4 at offset 64) in both M and N
#define THREAD_M 8    // positions per thread in M
#define THREAD_N 8    // positions per thread in N
#define GROUP_SIZE 4  // R from permutation — consecutive elements per group
#define GROUP_GAP 64  // distance between groups (F×R/num_threads_in_dim... = 64)

__global__ void gemm_naive(
    const float* A,  // (M, K) col-major: A[m,k] at A[m + k*M]
    const float* B,  // (N, K) col-major: B[n,k] at B[n + k*N]
    float* C,        // (M, N) row-major: C[m,n] at C[m*N + n]
    int M, int N, int K
) {
    // =========================================================================
    // CuTe equivalent: tidx, bidx, bidy = cute.arch.thread_idx/block_idx()
    // =========================================================================
    int tidx = threadIdx.x;           // 0..255
    int bidx = blockIdx.x;            // which M-tile
    int bidy = blockIdx.y;            // which N-tile

    // =========================================================================
    // CuTe equivalent: thr_mma = tiled_mma.get_slice(tidx)
    // Decode thread position in the 16×16 atom grid
    // =========================================================================
    int thr_m = tidx / 16;            // 0..15 — which row in thread grid
    int thr_n = tidx % 16;            // 0..15 — which col in thread grid

    // =========================================================================
    // CuTe equivalent: partition_C(gC) → computes which M,N positions are mine
    //
    // Permutation (16,4):(4,1) means:
    //   m_base = thr_m * GROUP_SIZE = thr_m * 4
    //   Two groups: [m_base..m_base+3] and [m_base+64..m_base+67]
    //   Same for N with thr_n
    // =========================================================================
    int m_positions[THREAD_M];  // this thread's 8 M-indices within tile
    int n_positions[THREAD_N];  // this thread's 8 N-indices within tile

    // Group 0: 4 consecutive starting at thr_m*4
    // Group 1: 4 consecutive starting at thr_m*4 + 64
    for (int g = 0; g < 2; g++) {
        for (int i = 0; i < GROUP_SIZE; i++) {
            m_positions[g * GROUP_SIZE + i] = thr_m * GROUP_SIZE + g * GROUP_GAP + i;
            n_positions[g * GROUP_SIZE + i] = thr_n * GROUP_SIZE + g * GROUP_GAP + i;
        }
    }
    // Thread 0 (thr_m=0, thr_n=0): m_pos = [0,1,2,3,64,65,66,67]
    //                               n_pos = [0,1,2,3,64,65,66,67]
    // Thread 1 (thr_m=0, thr_n=1): m_pos = [0,1,2,3,64,65,66,67]
    //                               n_pos = [4,5,6,7,68,69,70,71]

    // =========================================================================
    // CuTe equivalent: tCrC = tiled_mma.make_fragment_C(tCgC); tCrC.fill(0)
    // [ALLOC] 64 registers for accumulator
    // =========================================================================
    float accum[THREAD_M][THREAD_N];  // 8×8 = 64 floats in registers
    for (int m = 0; m < THREAD_M; m++)
        for (int n = 0; n < THREAD_N; n++)
            accum[m][n] = 0.0f;

    // =========================================================================
    // CuTe equivalent: local_tile(mA, ...) then the k_tile loop
    //
    // gA base pointer for this CTA: A + bidx * BM  (col-major, M-stride=1)
    // gB base pointer for this CTA: B + bidy * BN  (col-major, N-stride=1)
    // =========================================================================
    const float* A_cta = A + bidx * BM;        // start of this CTA's M-rows
    const float* B_cta = B + bidy * BN;        // start of this CTA's N-rows

    int num_k_tiles = K / BK;  // = 32/8 = 4

    // =========================================================================
    // CuTe equivalent: for k_tile in range(num_k_tiles):
    // =========================================================================
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {

        // =====================================================================
        // CuTe equivalent: tCgA = thr_mma.partition_A(gA[:,:,k_tile])
        //                  tCrA = make_fragment_like(tCgA)
        //                  autovec_copy(tCgA, tCrA)     ← [LOAD] from GMEM!
        //
        // Load this thread's A values: 8 M-positions × 8 K-values = 64 loads
        // A[m, k] lives at A_cta[m + k * M]
        // =====================================================================
        float regA[THREAD_M][BK];  // 8 × 8 = 64 floats
        for (int m = 0; m < THREAD_M; m++) {
            int m_idx = m_positions[m];  // local M index within tile
            for (int k = 0; k < BK; k++) {
                int global_k = k_tile * BK + k;
                regA[m][k] = A_cta[m_idx + global_k * M];  // [LOAD] LDG
            }
        }

        // =====================================================================
        // CuTe equivalent: tCgB = thr_mma.partition_B(gB[:,:,k_tile])
        //                  tCrB = make_fragment_like(tCgB)
        //                  autovec_copy(tCgB, tCrB)     ← [LOAD] from GMEM!
        //
        // Load this thread's B values: 8 N-positions × 8 K-values = 64 loads
        // B[n, k] lives at B_cta[n + k * N]
        // =====================================================================
        float regB[THREAD_N][BK];  // 8 × 8 = 64 floats
        for (int n = 0; n < THREAD_N; n++) {
            int n_idx = n_positions[n];
            for (int k = 0; k < BK; k++) {
                int global_k = k_tile * BK + k;
                regB[n][k] = B_cta[n_idx + global_k * N];  // [LOAD] LDG
            }
        }

        // =====================================================================
        // CuTe equivalent: for k_block in range(8):
        //                      cute.gemm(tiled_mma, tCrC, tCrA[:,:,k], tCrB[:,:,k], tCrC)
        //
        // [COMPUTE] 64 FMAs per k_block × 8 k_blocks = 512 FMAs per k_tile
        // =====================================================================
        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < THREAD_M; m++) {
                for (int n = 0; n < THREAD_N; n++) {
                    accum[m][n] += regA[m][k] * regB[n][k];  // [COMPUTE] FFMA
                }
            }
        }
    }  // end k_tile loop

    // =========================================================================
    // CuTe equivalent: cute.copy(atom, tCrC, tCgC)
    //
    // [STORE] Write 64 accumulated values back to global C
    // C is row-major: C[m,n] at C[(bidx*BM + m_idx) * N + (bidy*BN + n_idx)]
    // =========================================================================
    for (int m = 0; m < THREAD_M; m++) {
        int global_m = bidx * BM + m_positions[m];
        for (int n = 0; n < THREAD_N; n++) {
            int global_n = bidy * BN + n_positions[n];
            C[global_m * N + global_n] = accum[m][n];  // [STORE] STG
        }
    }
}

// Launch config:
// dim3 grid(M/BM, N/BN, 1);   // = (2, 1, 1) for M=256, N=128
// dim3 block(256, 1, 1);
// gemm_naive<<<grid, block>>>(A, B, C, M, N, K);
""";


# ==============================================================================
# LINE-BY-LINE MAPPING TABLE
# ==============================================================================

MAPPING = """
┌─────────────────────────────────────────┬───────────────────────────────────────────────┬──────────┐
│ CuTe DSL                                │ Pure CUDA equivalent                          │ HW work? │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ tidx = cute.arch.thread_idx()           │ int tidx = threadIdx.x;                       │ free     │
│ bidx, bidy = cute.arch.block_idx()      │ int bidx = blockIdx.x/y;                      │ free     │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ gA = cute.local_tile(mA, tiler, coord,  │ const float* A_cta = A + bidx * BM;           │ [VIEW]   │
│                       proj)             │ (pointer arithmetic only)                     │ free     │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ thr_mma = tiled_mma.get_slice(tidx)     │ int thr_m = tidx/16; int thr_n = tidx%16;     │ [VIEW]   │
│                                         │ compute m_positions[], n_positions[]           │ free     │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ tCgC = thr_mma.partition_C(gC)          │ (positions already computed above)             │ [VIEW]   │
│                                         │ m_positions[8], n_positions[8]                 │ free     │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ tCrC = tiled_mma.make_fragment_C(tCgC)  │ float accum[8][8];                            │ [ALLOC]  │
│ tCrC.fill(0.0)                          │ memset(accum, 0, sizeof(accum));              │ regs     │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ gA[None, None, k_tile]                  │ (adjust global_k = k_tile * BK + k)           │ [VIEW]   │
│                                         │                                               │ free     │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ tCgA = thr_mma.partition_A(...)         │ (loop over m_positions × k)                   │ [VIEW]   │
│                                         │                                               │ free     │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ tCrA = cute.make_fragment_like(tCgA)    │ float regA[8][8];                             │ [ALLOC]  │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ cute.autovec_copy(tCgA, tCrA)           │ regA[m][k] = A_cta[m_idx + k*M];             │ [LOAD]   │
│                                         │ generates LDG.128 where possible              │ GMEM→REG │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ cute.gemm(tiled_mma, tCrC,             │ accum[m][n] += regA[m][k] * regB[n][k];       │[COMPUTE] │
│           tCrA[:,:,k], tCrB[:,:,k],    │ FFMA instruction                              │ ALU      │
│           tCrC)                          │                                               │          │
├─────────────────────────────────────────┼───────────────────────────────────────────────┼──────────┤
│ cute.copy(atom, tCrC, tCgC)             │ C[global_m*N + global_n] = accum[m][n];       │ [STORE]  │
│                                         │ generates STG instructions                    │ REG→GMEM │
└─────────────────────────────────────────┴───────────────────────────────────────────────┴──────────┘

Key insight: CuTe's "magic" is that local_tile + partition encode the index arithmetic
(thr_m*4, groups at +64, etc.) into Layout objects. The GPU never computes these at runtime
for compile-time-known layouts — the compiler folds them into immediate offsets in LDG/STG.

The CUDA version makes the index math EXPLICIT in source code.
CuTe makes it IMPLICIT via layout composition — but the generated PTX is near-identical.
""";


# ==============================================================================
# MEMORY ACCESS PATTERN COMPARISON
# ==============================================================================

ACCESS_PATTERN = """
Thread 0 (thr_m=0, thr_n=0) memory accesses for k_tile=0:

LOAD A — 64 addresses (m_positions × BK):
  A[0 + 0*256], A[0 + 1*256], ..., A[0 + 7*256]     ← row 0, 8 K values (stride 256)
  A[1 + 0*256], A[1 + 1*256], ..., A[1 + 7*256]     ← row 1
  A[2 + 0*256], A[2 + 1*256], ..., A[2 + 7*256]     ← row 2
  A[3 + 0*256], A[3 + 1*256], ..., A[3 + 7*256]     ← row 3
  A[64+ 0*256], A[64+ 1*256], ..., A[64+ 7*256]     ← row 64 (GROUP_GAP jump!)
  A[65+ 0*256], A[65+ 1*256], ..., A[65+ 7*256]     ← row 65
  A[66+ 0*256], A[66+ 1*256], ..., A[66+ 7*256]     ← row 66
  A[67+ 0*256], A[67+ 1*256], ..., A[67+ 7*256]     ← row 67

LOAD B — same pattern with N=128 stride:
  B[0 + 0*128], B[0 + 1*128], ..., B[0 + 7*128]
  B[1 + 0*128], ... etc.

STORE C — 64 addresses:
  C[0*128+0], C[0*128+1], C[0*128+2], C[0*128+3],   ← 4 consecutive (coalesced!)
  C[0*128+64], C[0*128+65], C[0*128+66], C[0*128+67] ← jump to +64
  C[1*128+0], ...                                      ← next M row
  ...

Coalescing analysis:
  - A loads: threads 0..15 share same thr_m=0, load A[0..3, 64..67]
    → 16 threads hit rows 0-3 and 64-67, stride-256 apart. BAD coalescing.
  - B loads: threads 0,16,32,...,240 share thr_n=0, load B[0..3, 64..67]
    → these threads are 16 apart in tidx, hitting same B rows. BAD.
  - C stores: threads with consecutive tidx have consecutive thr_n,
    so they write to adjacent N positions. PARTIALLY coalesced within groups of 4.
""";

print("This file is a reference document. See CUDA_KERNEL, MAPPING, and ACCESS_PATTERN strings.")
print("\n" + MAPPING)
