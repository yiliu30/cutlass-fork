/*
 * step4_equivalent.cu
 * Pure CUDA equivalent of step4_tensorop_gemm.py (CuTe DSL)
 *
 * What step4 adds over step3: Tensor cores + swizzled SMEM
 * - wmma (maps to same mma.sync hardware as CuTe's MmaF16BF16Op)
 * - Swizzled SMEM: XOR-based address remapping for bank-conflict-free access
 * - 128 threads (4 warps), BK=32, 3-stage cp.async pipeline
 *
 * CuTe uses raw mma.sync.m16n8k16 + ldmatrix; this uses wmma m16n16k16 (same HW).
 * Key insight: CuTe's make_composed_layout + swizzle + partition_A/B + ldmatrix does
 * exactly what "de-swizzle + wmma::load_matrix_sync" does here — maps thread lanes to
 * SMEM addresses matching the hardware's fragment layout, with XOR to avoid bank conflicts.
 *
 * C[M,N] = A[M,K] * B[N,K]^T
 * A: (M,K) col-major FP16, B: (N,K) col-major FP16, C: (M,N) row-major FP16
 * Tile: 128×128×32, Threads: 128, Stages: 3
 *
 * Compile: nvcc -arch=sm_80 -O2 step4_equivalent.cu -o step4_equivalent && ./step4_equivalent
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 32
#define NUM_THREADS 128
#define NUM_STAGES 3
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// Swizzled SMEM addressing
// CuTe: make_composed_layout(make_swizzle(3,3,3), 0, layout_atom(64,8):(1,64))
// ============================================================================
__device__ __forceinline__ int smem_A_idx(int m, int k, int stage) {
    int atom_row = m / 64;
    int atom_col = k / 8;
    int ml = m % 64;
    int kl = k % 8;
    int ks = kl ^ ((ml >> 3) & 7);  // XOR swizzle
    return stage * (BM * BK) + (atom_row * 4 + atom_col) * 512 + ml + ks * 64;
}

__device__ __forceinline__ int smem_B_idx(int n, int k, int stage) {
    int atom_row = n / 64;
    int atom_col = k / 8;
    int nl = n % 64;
    int kl = k % 8;
    int ks = kl ^ ((nl >> 3) & 7);
    return stage * (BN * BK) + (atom_row * 4 + atom_col) * 512 + nl + ks * 64;
}

// ============================================================================
// PTX cp.async
// ============================================================================
__device__ __forceinline__ void cp_async_4B(void* s, const void* g) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(a),"l"(g):"memory");
}
__device__ __forceinline__ void cp_async_commit(){asm volatile("cp.async.commit_group;\n":::"memory");}
__device__ __forceinline__ void cp_async_wait(int N){
    if(N==0)asm volatile("cp.async.wait_group 0;\n":::"memory");
    else if(N==1)asm volatile("cp.async.wait_group 1;\n":::"memory");
    else asm volatile("cp.async.wait_group 2;\n":::"memory");
}

__global__ void gemm_tensorop(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x, bidy = blockIdx.y;

    // Dynamic SMEM: sA + sB (swizzled) + tempA + tempB (linear, for wmma load)
    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = sA + BM * BK * NUM_STAGES;
    // Linear temp buffers for de-swizzled wmma tiles (4 warps × 16×16 each)
    half* tempA = sB + BN * BK * NUM_STAGES;
    half* tempB = tempA + 4 * WMMA_M * WMMA_K;

    const int warp_id = tidx / 32;
    const int lane_id = tidx % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    // Accumulators: 4×4 wmma tiles per warp (64/16 = 4 in each dim)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4][4];
    for (int m = 0; m < 4; m++)
        for (int n = 0; n < 4; n++)
            wmma::fill_fragment(acc[m][n], 0.0f);

    // G2S copy: 2 FP16 per cp.async, 16 rounds per stage
    const int copy_m = (tidx % 64) * 2;
    const int copy_k = tidx / 64;
    const half* A_cta = A + bidx * BM;
    const half* B_cta = B + bidy * BN;
    const int nkt = K / BK;

    // =========================================================================
    // Prologue: prefetch 2 stages with SWIZZLED writes
    // =========================================================================
    for (int s = 0; s < NUM_STAGES - 1 && s < nkt; s++) {
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            int kc = copy_k + r * 2;
            cp_async_4B(&sA[smem_A_idx(copy_m, kc, s)], &A_cta[copy_m + (s*BK+kc)*M]);
        }
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            int kc = copy_k + r * 2;
            cp_async_4B(&sB[smem_B_idx(copy_m, kc, s)], &B_cta[copy_m + (s*BK+kc)*N]);
        }
        cp_async_commit();
    }

    // =========================================================================
    // Mainloop
    // =========================================================================
    int rs = 0, ws = NUM_STAGES - 1, nk = NUM_STAGES - 1;

    for (int kt = 0; kt < nkt; kt++) {
        cp_async_wait(NUM_STAGES - 2);
        __syncthreads();

        for (int k16 = 0; k16 < 2; k16++) {
            int ko = k16 * WMMA_K;

            for (int mi = 0; mi < 4; mi++) {
                int mb = warp_m * 64 + mi * WMMA_M;
                // De-swizzle A (16×16) into tempA as col-major: tempA[m + k*16]
                for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
                    int row = i % WMMA_M;  // m
                    int col = i / WMMA_M;  // k
                    tempA[warp_id * 256 + i] = sA[smem_A_idx(mb + row, ko + col, rs)];
                }
                __syncwarp();

                // wmma load A as col_major: ptr[k * ldm + m], ldm = WMMA_M = 16
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fa;
                wmma::load_matrix_sync(fa, &tempA[warp_id * 256], WMMA_M);

                for (int ni = 0; ni < 4; ni++) {
                    int nb = warp_n * 64 + ni * WMMA_N;
                    // De-swizzle B (16×16) into tempB as col-major: tempB[n + k*16]
                    for (int i = lane_id; i < WMMA_N * WMMA_K; i += 32) {
                        int row = i % WMMA_N;  // n
                        int col = i / WMMA_N;  // k
                        tempB[warp_id * 256 + i] = sB[smem_B_idx(nb + row, ko + col, rs)];
                    }
                    __syncwarp();

                    // C = A * B^T → wmma load B as row_major: ptr[n * ldm + k] → B^T
                    // Our tempB[n + k*16] is col-major(n). Row-major read: ptr[row*ldm+col]
                    // = ptr[n*16 + k] → we need tempB stored as tempB[n*16 + k]
                    // But we stored tempB[n + k*16]... that's col-major(n) = row-major(k)
                    // wmma row_major B: ptr[k * ldm + n], ldm = WMMA_N = 16
                    // = tempB[k*16 + n]. We have tempB[n + k*16]. Same thing!
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, &tempB[warp_id * 256], WMMA_N);

                    wmma::mma_sync(acc[mi][ni], fa, fb, acc[mi][ni]);
                }
            }
        }

        // Issue next G2S
        if (nk < nkt) {
            #pragma unroll
            for (int r = 0; r < 16; r++) {
                int kc = copy_k + r * 2;
                cp_async_4B(&sA[smem_A_idx(copy_m, kc, ws)], &A_cta[copy_m + (nk*BK+kc)*M]);
            }
            #pragma unroll
            for (int r = 0; r < 16; r++) {
                int kc = copy_k + r * 2;
                cp_async_4B(&sB[smem_B_idx(copy_m, kc, ws)], &B_cta[copy_m + (nk*BK+kc)*N]);
            }
        }
        cp_async_commit();
        nk++;
        rs = (rs+1) % NUM_STAGES;
        ws = (ws+1) % NUM_STAGES;
    }

    // =========================================================================
    // Epilogue
    // =========================================================================
    cp_async_wait(0);
    __syncthreads();

    for (int mi = 0; mi < 4; mi++) {
        int mb = bidx*BM + warp_m*64 + mi*WMMA_M;
        for (int ni = 0; ni < 4; ni++) {
            int nb = bidy*BN + warp_n*64 + ni*WMMA_N;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out;
            for (int i = 0; i < acc[mi][ni].num_elements; i++)
                out.x[i] = __float2half(acc[mi][ni].x[i]);
            wmma::store_matrix_sync(&C[mb*N + nb], out, N, wmma::mem_row_major);
        }
    }
}

// =============================================================================
void ref_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += __half2float(A[m + k*M]) * __half2float(B[n + k*N]);
            C[m*N+n] = s;
        }
}

int main() {
    const int M = 256, N = 128, K = 64;
    size_t szA = M*K*2, szB = N*K*2, szC = M*N*2;

    half *hA = (half*)malloc(szA), *hB = (half*)malloc(szB), *hC = (half*)malloc(szC);
    float *hRef = (float*)malloc(M*N*4);

    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = __float2half((rand()%5-2)*0.1f);
    for (int i = 0; i < N*K; i++) hB[i] = __float2half((rand()%5-2)*0.1f);

    half *dA, *dB, *dC;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB); cudaMalloc(&dC, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, szC);

    size_t dyn_smem = (BM*BK + BN*BK) * NUM_STAGES * 2  // sA + sB
                    + 4 * 256 * 2   // tempA (4 warps × 256 half)
                    + 4 * 256 * 2;  // tempB
    dim3 grid(M/BM, N/BN), block(NUM_THREADS);
    cudaFuncSetAttribute(gemm_tensorop, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
    gemm_tensorop<<<grid, block, dyn_smem>>>(dA, dB, dC, M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost);
    ref_gemm(hA, hB, hRef, M, N, K);

    float max_err = 0;
    for (int i = 0; i < M*N; i++) {
        float e = fabsf(__half2float(hC[i]) - hRef[i]);
        if (e > max_err) max_err = e;
    }
    printf("Step 4 (TensorOp+Swizzle): M=%d, N=%d, K=%d, max_error=%.6f — %s\n",
           M, N, K, max_err, max_err < 0.1f ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hRef);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return max_err < 0.1f ? 0 : 1;
}
