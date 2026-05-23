/*
 * step3_equivalent.cu
 * Pure CUDA equivalent of step3_pipeline_gemm.py (CuTe DSL)
 *
 * What step3 adds over step2: Multi-stage async pipeline
 * - cp.async hardware DMA: GPU copies GMEM→SMEM without thread involvement
 * - 3-stage circular buffer: overlap copy of next tile with compute of current
 * - Prologue prefetches first 2 stages before mainloop starts
 * - commit_group / wait_group: track in-flight async copies
 *
 * C[M,N] = A[M,K] * B[N,K]^T
 * Tile: 128×128×8, Threads: 256, Stages: 3, MMA: scalar FMA
 *
 * Compile: nvcc -arch=sm_80 -O2 step3_equivalent.cu -o step3_equivalent && ./step3_equivalent
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

#define BM 128
#define BN 128
#define BK 8
#define NUM_THREADS 256
#define NUM_STAGES 3
#define GROUP_SIZE 4
#define GROUP_GAP 64

// ============================================================================
// PTX helpers for cp.async
// CuTe equivalent: CopyG2SOp → cp.async.ca.shared.global
// ============================================================================
__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_ptr) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group(int N) {
    // wait until at most N groups are still in flight
    if (N == 0) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if (N == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else asm volatile("cp.async.wait_group 2;\n" ::: "memory");
}

__global__ void gemm_pipeline(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    // =========================================================================
    // CuTe: SmemAllocator → sA/sB with NUM_STAGES dimension
    // 3 stages: circular buffer for pipeline
    // =========================================================================
    __shared__ float sA[NUM_STAGES][BM * BK];  // [stage][128*8]
    __shared__ float sB[NUM_STAGES][BN * BK];

    // Output ownership (same as step1/step2)
    const int thr_m = tidx / 16;
    const int thr_n = tidx % 16;
    int m_pos[8], n_pos[8];
    #pragma unroll
    for (int g = 0; g < 2; g++)
        for (int i = 0; i < GROUP_SIZE; i++) {
            m_pos[g*GROUP_SIZE + i] = thr_m*GROUP_SIZE + g*GROUP_GAP + i;
            n_pos[g*GROUP_SIZE + i] = thr_n*GROUP_SIZE + g*GROUP_GAP + i;
        }

    float acc[8][8];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        for (int n = 0; n < 8; n++)
            acc[m][n] = 0.0f;

    // Copy thread assignment (same as step2)
    const int copy_m = (tidx % 32) * 4;
    const int copy_k = tidx / 32;

    const float* A_cta = A + bidx * BM;
    const float* B_cta = B + bidy * BN;
    const int num_k_tiles = K / BK;

    // =========================================================================
    // CuTe: Prologue — prefetch first (NUM_STAGES-1) = 2 stages
    // Issue cp.async for stages 0 and 1, commit each as a group
    // =========================================================================
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1 && s < num_k_tiles; s++) {
        // G2S copy for stage s
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int m_idx = copy_m + i;
            cp_async_4B(&sA[s][m_idx + copy_k * BM],
                        &A_cta[m_idx + (s * BK + copy_k) * M]);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int n_idx = copy_m + i;
            cp_async_4B(&sB[s][n_idx + copy_k * BN],
                        &B_cta[n_idx + (s * BK + copy_k) * N]);
        }
        cp_async_commit();
    }

    // =========================================================================
    // CuTe: Mainloop — compute on read_stage, issue copy for write_stage
    // =========================================================================
    int read_stage = 0;
    int write_stage = NUM_STAGES - 1;  // next stage to write into

    for (int kt = 0; kt < num_k_tiles; kt++) {
        // Wait for read_stage to be ready (at most NUM_STAGES-2 groups in flight)
        cp_async_wait_group(NUM_STAGES - 2);
        __syncthreads();

        // =====================================================================
        // [COMPUTE] from read_stage (same as step2's inner loop)
        // =====================================================================
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float rA[8], rB[8];
            #pragma unroll
            for (int m = 0; m < 8; m++)
                rA[m] = sA[read_stage][m_pos[m] + k * BM];
            #pragma unroll
            for (int n = 0; n < 8; n++)
                rB[n] = sB[read_stage][n_pos[n] + k * BN];
            #pragma unroll
            for (int m = 0; m < 8; m++)
                for (int n = 0; n < 8; n++)
                    acc[m][n] += rA[m] * rB[n];
        }

        // =====================================================================
        // [LOAD] Issue cp.async for NEXT tile (write_stage)
        // This overlaps with the compute above on next iteration
        // =====================================================================
        int next_kt = kt + NUM_STAGES - 1;
        if (next_kt < num_k_tiles) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m_idx = copy_m + i;
                cp_async_4B(&sA[write_stage][m_idx + copy_k * BM],
                            &A_cta[m_idx + (next_kt * BK + copy_k) * M]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int n_idx = copy_m + i;
                cp_async_4B(&sB[write_stage][n_idx + copy_k * BN],
                            &B_cta[n_idx + (next_kt * BK + copy_k) * N]);
            }
        }
        cp_async_commit();

        // Advance circular buffer pointers
        read_stage = (read_stage + 1) % NUM_STAGES;
        write_stage = (write_stage + 1) % NUM_STAGES;
    }

    // =========================================================================
    // Epilogue: R→G
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
// Host
// =============================================================================
void ref_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
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

    float *hA = (float*)malloc(sA), *hB = (float*)malloc(sB);
    float *hC = (float*)malloc(sC), *hRef = (float*)malloc(sC);

    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++)
            hA[m + k * M] = (float)((m + k * M) % 13);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            hB[n + k * N] = (float)((n + k * N) % 7);
    memset(hC, 0, sC);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sA); cudaMalloc(&dB, sB); cudaMalloc(&dC, sC);
    cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sC);

    dim3 grid(M / BM, N / BN, 1);
    dim3 block(NUM_THREADS, 1, 1);
    gemm_pipeline<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost);
    ref_gemm(hA, hB, hRef, M, N, K);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(hC[i] - hRef[i]);
        if (err > max_err) max_err = err;
    }
    printf("Step 3 (Pipeline): M=%d, N=%d, K=%d, max_error=%.6f — %s\n",
           M, N, K, max_err, max_err < 1e-3f ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hRef);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return max_err < 1e-3f ? 0 : 1;
}
