/*
 * step5_equivalent.cu
 * Pure CUDA equivalent of step5_vectorized_gemm.py (CuTe DSL)
 *
 * What step5 adds over step4: Vectorized loads + SMEM epilogue
 * - 128-bit (16B) cp.async: loads 8 FP16 at once (vs 2 in step4) → 4× fewer instructions
 * - SMEM epilogue: write accumulators to SMEM, retile for coalesced 128-bit stores to GMEM
 *   Without this, epilogue stores are scattered (fragment layout ≠ coalesced layout)
 *   With SMEM epilogue: R → SMEM (fragment order) → R (retiled) → GMEM (128-bit coalesced)
 *
 * C[M,N] = A[M,K] * B[N,K]^T
 * A: (M,K) col-major FP16, B: (N,K) col-major FP16, C: (M,N) row-major FP16
 * Tile: 128×128×32, Threads: 128, Stages: 3
 *
 * Compile: nvcc -arch=sm_80 -O2 step5_equivalent.cu -o step5_equivalent && ./step5_equivalent
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

// Swizzle (same as step4)
__device__ __forceinline__ int smem_A_idx(int m, int k, int stage) {
    int atom_row = m / 64, atom_col = k / 8;
    int ml = m % 64, kl = k % 8;
    int ks = kl ^ ((ml >> 3) & 7);
    return stage * (BM * BK) + (atom_row * 4 + atom_col) * 512 + ml + ks * 64;
}
__device__ __forceinline__ int smem_B_idx(int n, int k, int stage) {
    int atom_row = n / 64, atom_col = k / 8;
    int nl = n % 64, kl = k % 8;
    int ks = kl ^ ((nl >> 3) & 7);
    return stage * (BN * BK) + (atom_row * 4 + atom_col) * 512 + nl + ks * 64;
}

// ============================================================================
// cp.async 16B (128 bits = 8 FP16) — the key step5 upgrade
// ============================================================================
__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(gmem):"memory");
}
__device__ __forceinline__ void cp_async_commit(){asm volatile("cp.async.commit_group;\n":::"memory");}
__device__ __forceinline__ void cp_async_wait(int N){
    if(N==0)asm volatile("cp.async.wait_group 0;\n":::"memory");
    else if(N==1)asm volatile("cp.async.wait_group 1;\n":::"memory");
    else asm volatile("cp.async.wait_group 2;\n":::"memory");
}

__global__ void gemm_vectorized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x, bidy = blockIdx.y;

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = sA + BM * BK * NUM_STAGES;
    half* tempA = sB + BN * BK * NUM_STAGES;
    half* tempB = tempA + 4 * 256;
    // SMEM epilogue buffer (reuses sA/sB space since mainloop is done)
    // Actually need separate space since we declare it here — but let's use sA region
    half* sC = sA;  // reuse sA space for epilogue (no temporal overlap)

    const int warp_id = tidx / 32, lane_id = tidx % 32;
    const int warp_m = warp_id / 2, warp_n = warp_id % 2;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4][4];
    for (int m = 0; m < 4; m++)
        for (int n = 0; n < 4; n++)
            wmma::fill_fragment(acc[m][n], 0.0f);

    // =========================================================================
    // G2S with 128-bit cp.async (8 FP16 per copy)
    // CuTe: CopyG2SOp with num_bits_per_copy=128, thread_layout (16,8), val_layout (8,1)
    //   → 16 threads in M (each handles 128/16=8 M-rows), 8 threads in K
    //   → Each cp.async loads 8 consecutive M-elements (128 bits)
    //   → Total per thread: (128/16)*(32/8) = 8*4 = 32 elements = 4 cp.async calls
    //      Wait: 128*32=4096 elements / 128 threads / 8 per copy = 4 copies per thread ✓
    // =========================================================================
    // Thread layout for vectorized copy: (16, 8) stride (8, 1)
    //   copy_m_group = tidx % 16 → handles M rows [group*8 .. group*8+7]
    //   copy_k_col = tidx / 16   → handles K column copy_k_col
    // Wait, 16×8 = 128 ✓. Each thread does 1 cp.async of 16B per "call"
    // For 4096 elements at 8 per copy: need 512 copies total / 128 threads = 4 per thread
    const int copy_group = tidx % 16;   // 0..15
    const int copy_kcol = tidx / 16;    // 0..7
    // M-positions: copy_group * 8 .. copy_group * 8 + 7  (8 consecutive)
    // K-positions: need to cover 32 / 8 = 4 K-columns per K-thread-group
    const int copy_m_start = copy_group * 8;  // 8 consecutive M-rows per copy

    const half* A_cta = A + bidx * BM;
    const half* B_cta = B + bidy * BN;
    const int nkt = K / BK;

    // However, cp.async 16B loads from GMEM which is contiguous in M (col-major).
    // 8 consecutive M elements = 16 bytes ✓ (perfectly aligned for 128-bit)
    // But SMEM is swizzled — we can't directly cp.async into arbitrary swizzled positions
    // with 16B granularity because the swizzle breaks byte-contiguity.
    //
    // Solution: for pedagogical equivalent, use 4B cp.async (same as step4) to correctly
    // handle swizzle. In the real CuTe kernel, the swizzle is designed so that 8 consecutive
    // M-elements map to consecutive SMEM bytes within one atom row — making 16B cp.async safe.
    //
    // For correctness: use non-swizzled SMEM with 16B cp.async for the G2S,
    // then de-swizzle during S2R (same perf model, simpler code).
    // Actually let's just use linear SMEM with 16B cp.async — the swizzle's perf benefit
    // is orthogonal to understanding the vectorized concept.

    // Prologue with 16B cp.async (no swizzle — linear col-major SMEM)
    // sA_lin[m + k*BM] per stage, stored at sA[stage*BM*BK + m + k*BM]
    for (int s = 0; s < NUM_STAGES - 1 && s < nkt; s++) {
        // Each thread does 4 cp.async of 16B (covers 128M × 32K = 4096 elements)
        // Thread layout: (16, 8), each 16B = 8 M-elements
        // Round 0: copy_m_start, k=copy_kcol → 8 elements
        // Need 32/8 = 4 K rounds per thread
        #pragma unroll
        for (int kr = 0; kr < 4; kr++) {
            int k_col = copy_kcol + kr * 8;
            int offset = s * BM * BK + copy_m_start + k_col * BM;
            int gk = s * BK + k_col;
            cp_async_16B(&sA[offset], &A_cta[copy_m_start + gk * M]);
        }
        #pragma unroll
        for (int kr = 0; kr < 4; kr++) {
            int k_col = copy_kcol + kr * 8;
            int offset = s * BN * BK + copy_m_start + k_col * BN;
            int gk = s * BK + k_col;
            cp_async_16B(&sB[offset], &B_cta[copy_m_start + gk * N]);
        }
        cp_async_commit();
    }

    // =========================================================================
    // Mainloop (same compute as step4, but with linear SMEM + vectorized G2S)
    // =========================================================================
    int rs = 0, ws = NUM_STAGES - 1, nk = NUM_STAGES - 1;

    for (int kt = 0; kt < nkt; kt++) {
        cp_async_wait(NUM_STAGES - 2);
        __syncthreads();

        for (int k16 = 0; k16 < 2; k16++) {
            int ko = k16 * WMMA_K;
            for (int mi = 0; mi < 4; mi++) {
                int mb = warp_m * 64 + mi * WMMA_M;
                // Load A from linear SMEM (col-major: stride 1, BM)
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fa;
                wmma::load_matrix_sync(fa, &sA[rs*BM*BK + mb + ko*BM], BM);

                for (int ni = 0; ni < 4; ni++) {
                    int nb = warp_n * 64 + ni * WMMA_N;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, &sB[rs*BN*BK + nb + ko*BN], BN);
                    wmma::mma_sync(acc[mi][ni], fa, fb, acc[mi][ni]);
                }
            }
        }

        if (nk < nkt) {
            #pragma unroll
            for (int kr = 0; kr < 4; kr++) {
                int k_col = copy_kcol + kr * 8;
                int offset = ws * BM * BK + copy_m_start + k_col * BM;
                cp_async_16B(&sA[offset], &A_cta[copy_m_start + (nk*BK+k_col)*M]);
            }
            #pragma unroll
            for (int kr = 0; kr < 4; kr++) {
                int k_col = copy_kcol + kr * 8;
                int offset = ws * BN * BK + copy_m_start + k_col * BN;
                cp_async_16B(&sB[offset], &B_cta[copy_m_start + (nk*BK+k_col)*N]);
            }
        }
        cp_async_commit();
        nk++;
        rs = (rs+1) % NUM_STAGES;
        ws = (ws+1) % NUM_STAGES;
    }

    cp_async_wait(0);
    __syncthreads();

    // =========================================================================
    // SMEM Epilogue: R → SMEM → retiled R → coalesced 128-bit GMEM stores
    //
    // CuTe: autovec_copy(tCrC → sC), sync, autovec_copy(sC → retiled), cute.copy(tiled_copy_C)
    //
    // Why: wmma fragment layout is NOT coalesced for global stores.
    // By writing to SMEM first, then re-reading in a coalesced pattern,
    // we can issue 128-bit STG (store 8 FP16 at once) for full bandwidth.
    //
    // Here: use wmma::store to SMEM (row-major), sync, then each thread reads
    // 8 consecutive elements and does a 128-bit store to GMEM.
    // =========================================================================

    // Process one (16×16) tile at a time through SMEM (limited buffer)
    // sC can hold one 16×16 = 256 half tile (512 bytes) — plenty of space
    // Actually sC points to sA which has BM*BK*NUM_STAGES = huge. Use first 16*128 elements.
    // Strategy: store one full row of wmma tiles (16 M-rows × 128 N-cols) at a time
    //           = 4 wmma N-tiles stored row-major, then coalesced copy to GMEM

    for (int mi = 0; mi < 4; mi++) {
        int m_base_local = warp_m * 64 + mi * WMMA_M;

        for (int ni = 0; ni < 4; ni++) {
            int n_base_local = warp_n * 64 + ni * WMMA_N;

            // Convert FP32 → FP16 and store to SMEM via wmma
            // Use sC region: sC[m_local * BN + n_local] for the (128×128) tile
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out;
            for (int i = 0; i < acc[mi][ni].num_elements; i++)
                out.x[i] = __float2half(acc[mi][ni].x[i]);

            // Store to SMEM (row-major within the 128×128 sC buffer)
            wmma::store_matrix_sync(&sC[m_base_local * BN + n_base_local],
                                    out, BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // =========================================================================
    // Coalesced 128-bit stores from SMEM to GMEM
    // sC is row-major (128×128). Each thread stores 128*128/128 = 128 elements
    // = 16 × 128-bit stores per thread
    // Thread assignment: 128 threads over 128 rows → 1 row per thread? No, 128 vals/row.
    // Better: flatten all 16384 elements, each thread does 128 elements = 16 STG.128
    // Linear: thread tidx handles elements [tidx*128 .. tidx*128+127]
    // But that's 128 elements = 16 × 8 FP16 per 128-bit ✓
    //
    // CuTe equivalent: tiled_copy_C with thread_layout (8,16) val (1,8)
    //   8 threads in M × 16 threads in N × 8 values per copy = covers 8M × 128N = 1024 per call
    //   Need 128*128/1024 = 16 rounds
    // =========================================================================
    // Simple: each thread writes 128 FP16 as 16 vectorized 128-bit stores
    int flat_start = tidx * 128;  // 128 FP16 per thread (16384 total / 128 threads)
    half* C_tile = C + (bidx * BM) * N + bidy * BN;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int flat_idx = flat_start + i * 8;
        int row = flat_idx / BN;
        int col = flat_idx % BN;
        // 128-bit store (8 FP16)
        *reinterpret_cast<int4*>(&C_tile[row * N + col]) =
            *reinterpret_cast<int4*>(&sC[flat_idx]);
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

    // Dynamic SMEM: sA + sB + tempA + tempB (tempA/B not used here since no swizzle)
    // But sC reuses sA space. Need max(mainloop, epilogue).
    // Mainloop: (128*32 + 128*32) * 3 * 2 = 49152 bytes
    // Epilogue sC: 128*128 * 2 = 32768 bytes (fits in sA space of 24576... NO)
    // sA = 128*32*3 = 12288 half = 24576 bytes. sC needs 128*128 = 16384 half = 32768 bytes.
    // Need more! Use sA+sB combined = 49152 bytes > 32768 ✓
    size_t dyn_smem = (BM*BK + BN*BK) * NUM_STAGES * 2;  // 49152

    dim3 grid(M/BM, N/BN), block(NUM_THREADS);
    cudaFuncSetAttribute(gemm_vectorized, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
    gemm_vectorized<<<grid, block, dyn_smem>>>(dA, dB, dC, M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost);
    ref_gemm(hA, hB, hRef, M, N, K);

    float max_err = 0;
    for (int i = 0; i < M*N; i++) {
        float e = fabsf(__half2float(hC[i]) - hRef[i]);
        if (e > max_err) max_err = e;
    }
    printf("Step 5 (Vectorized+SMEM Epilogue): M=%d, N=%d, K=%d, max_error=%.6f — %s\n",
           M, N, K, max_err, max_err < 0.1f ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hRef);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return max_err < 0.1f ? 0 : 1;
}
