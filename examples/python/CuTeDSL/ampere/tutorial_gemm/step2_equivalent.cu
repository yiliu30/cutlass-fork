/*
 * step2_equivalent.cu
 * Pure CUDA equivalent of step2_smem_gemm.py (CuTe DSL)
 *
 * What step2 adds over step1: Shared memory tiling
 * - Cooperative load: all 256 threads together load a (128,8) tile into SMEM
 * - __syncthreads() ensures data is ready
 * - Each thread then reads its elements from SMEM (fast) instead of GMEM (slow)
 * - Data reuse: each A[m,k] loaded once from GMEM, read by multiple threads from SMEM
 *
 * C[M,N] = A[M,K] * B[N,K]^T
 * A: col-major (stride 1,M), B: col-major (stride 1,N), C: row-major (stride N,1)
 * Tile: 128×128×8, Threads: 256, MMA: scalar FMA
 *
 * Compile: nvcc -arch=sm_80 -O2 step2_equivalent.cu -o step2_equivalent && ./step2_equivalent
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BM 128
#define BN 128
#define BK 8
#define NUM_THREADS 256
#define GROUP_SIZE 4
#define GROUP_GAP 64

__global__ void gemm_smem(
    const float* __restrict__ A,  // (M,K) col-major
    const float* __restrict__ B,  // (N,K) col-major
    float* __restrict__ C,        // (M,N) row-major
    int M, int N, int K)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    // =========================================================================
    // CuTe: SmemAllocator → sA, sB
    // =========================================================================
    __shared__ float sA[BM * BK];  // (128, 8) flattened
    __shared__ float sB[BN * BK];  // (128, 8) flattened

    // =========================================================================
    // CuTe: tiled_mma.get_slice(tidx) + partition_C
    // Same ownership pattern as step1
    // =========================================================================
    const int thr_m = tidx / 16;
    const int thr_n = tidx % 16;

    int m_pos[8], n_pos[8];
    #pragma unroll
    for (int g = 0; g < 2; g++) {
        for (int i = 0; i < GROUP_SIZE; i++) {
            m_pos[g*GROUP_SIZE + i] = thr_m*GROUP_SIZE + g*GROUP_GAP + i;
            n_pos[g*GROUP_SIZE + i] = thr_n*GROUP_SIZE + g*GROUP_GAP + i;
        }
    }

    // =========================================================================
    // CuTe: make_fragment_C + fill(0)
    // =========================================================================
    float acc[8][8];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        for (int n = 0; n < 8; n++)
            acc[m][n] = 0.0f;

    // =========================================================================
    // CuTe: tiled_copy.get_slice(tidx) — cooperative copy thread assignment
    //
    // In CuTe step2: tiled_copy uses thread_layout (32,8):(8,1)
    //   → 32 threads along M (each handles 128/32=4 M-rows)
    //   → 8 threads along K (each handles 8/8=1 K-col)
    //   Mapping: copy_thr_m = tidx % 32, copy_thr_k = tidx / 32
    //   But 256 threads with (32,8) = 256 threads. Each loads 4 elements of A, 4 of B.
    //
    // For A (128×8): 256 threads, 1024 elements → 4 per thread
    //   Simple linear: tidx loads elements [tidx*4 .. tidx*4+3]?
    //   No — CuTe's (32,8):(8,1) means:
    //     thread (tm, tk) where tm=tidx%32, tk=tidx/32 loads A[tm*4+i, tk] for i=0..3
    //   → 4 consecutive M-rows, 1 K-column per thread
    // =========================================================================
    const int copy_m = (tidx % 32) * 4;  // starting M-row for this thread's copy
    const int copy_k = tidx / 32;         // which K-column this thread copies

    // CTA base pointers
    const float* A_cta = A + bidx * BM;
    const float* B_cta = B + bidy * BN;

    // =========================================================================
    // K-tile loop
    // =========================================================================
    const int num_k_tiles = K / BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {

        // =====================================================================
        // CuTe: cute.copy(tiled_copy_A, tAgA, tAsA)  — cooperative G2S
        // [LOAD] GMEM → SMEM: each thread loads 4 elements of A and 4 of B
        //
        // sA is (128, 8) stored M-major: sA[m + k*128]
        // A is (M, K) col-major: A[m + k*M]
        // =====================================================================
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int m_idx = copy_m + i;
            int k_idx = copy_k;
            int global_k = kt * BK + k_idx;
            sA[m_idx + k_idx * BM] = A_cta[m_idx + global_k * M];
        }

        // sB is (128, 8) stored N-major: sB[n + k*128]
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int n_idx = copy_m + i;  // reuse same pattern for B
            int k_idx = copy_k;
            int global_k = kt * BK + k_idx;
            sB[n_idx + k_idx * BN] = B_cta[n_idx + global_k * N];
        }

        // =====================================================================
        // CuTe: cute.arch.sync_threads()
        // Barrier: ensure all threads finished writing SMEM before reading
        // =====================================================================
        __syncthreads();

        // =====================================================================
        // CuTe: inner k_block loop with autovec_copy(S→R) + cute.gemm
        // [LOAD] SMEM → RMEM, then [COMPUTE] FMA
        //
        // Difference from step1: reading from sA/sB (shared, ~30 cycles)
        //                        instead of A/B (global, ~400 cycles)
        // Same values loaded once, shared among 16 threads with same M-row (for A)
        //                          or same N-col (for B)
        // =====================================================================
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float rA[8], rB[8];

            // S→R load: read this thread's 8 M-positions from sA
            #pragma unroll
            for (int m = 0; m < 8; m++)
                rA[m] = sA[m_pos[m] + k * BM];

            // S→R load: read this thread's 8 N-positions from sB
            #pragma unroll
            for (int n = 0; n < 8; n++)
                rB[n] = sB[n_pos[n] + k * BN];

            // FMA: 8×8 = 64 multiply-adds
            #pragma unroll
            for (int m = 0; m < 8; m++)
                for (int n = 0; n < 8; n++)
                    acc[m][n] += rA[m] * rB[n];
        }

        // Need sync before next iteration overwrites SMEM
        __syncthreads();
    }

    // =========================================================================
    // CuTe: cute.copy(atom, tCrC, tCgC) — epilogue R→G
    // [STORE] Write 64 accumulated values to global C
    // =========================================================================
    #pragma unroll
    for (int m = 0; m < 8; m++) {
        int gm = bidx * BM + m_pos[m];
        for (int n = 0; n < 8; n++) {
            int gn = bidy * BN + n_pos[n];
            C[gm * N + gn] = acc[m][n];
        }
    }
}

// =============================================================================
// Host: reference + verification
// =============================================================================
void ref_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // C[m,n] = sum_k A[m,k] * B[n,k]  (A col-major, B col-major, C row-major)
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m + k * M] * B[n + k * N];
            C[m * N + n] = sum;
        }
}

int main() {
    const int M = 256, N = 128, K = 32;
    size_t sA = M * K * sizeof(float);
    size_t sB = N * K * sizeof(float);
    size_t sC = M * N * sizeof(float);

    // Host alloc
    float *hA = (float*)malloc(sA);
    float *hB = (float*)malloc(sB);
    float *hC = (float*)malloc(sC);
    float *hRef = (float*)malloc(sC);

    // Init: A[m,k] = (m + k*M) % 13, B[n,k] = (n + k*N) % 7  (small values to avoid fp error)
    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++)
            hA[m + k * M] = (float)((m + k * M) % 13);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            hB[n + k * N] = (float)((n + k * N) % 7);
    memset(hC, 0, sC);

    // Device alloc
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sA);
    cudaMalloc(&dB, sB);
    cudaMalloc(&dC, sC);
    cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sC);

    // Launch
    dim3 grid(M / BM, N / BN, 1);
    dim3 block(NUM_THREADS, 1, 1);
    gemm_smem<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost);

    // Reference
    ref_gemm(hA, hB, hRef, M, N, K);

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(hC[i] - hRef[i]);
        if (err > max_err) max_err = err;
    }
    printf("Step 2 (SMEM): M=%d, N=%d, K=%d, max_error=%.6f — %s\n",
           M, N, K, max_err, max_err < 1e-3f ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hRef);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return max_err < 1e-3f ? 0 : 1;
}
