/*
 * Pure CUDA implementation of the vectorized elementwise add kernel
 * from quick_run.py (CuTeDSL version).
 *
 * Reproduces the exact same logic:
 *   - zipped_divide(tensor, (1, 4)) partitions a (M, N) row-major FP16 tensor
 *     into tiles of 4 contiguous elements along the column dimension.
 *   - Each thread loads 4 FP16 elements via a single 64-bit vectorized load,
 *     adds them, and stores the result with a single 64-bit vectorized store.
 *
 * Layout equivalence:
 *   CuTeDSL:  gA = zipped_divide(mA, (1, 4))
 *             gA layout: ((1,4), (M, N/4)) : ((0,1), (N, 4))
 *
 *   CUDA:     thread_idx → (mi, ni) where mi = thread_idx / (N/4)
 *                                          ni = thread_idx % (N/4)
 *             each thread loads *(half4*)(A + mi * N + ni * 4)
 *
 * Build:
 *   nvcc -O2 -arch=sm_80 vec_add.cu -o vec_add
 *
 * Run:
 *   ./vec_add [M] [N]    (defaults: M=128, N=64)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// Error checking macro
// ============================================================================
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// Kernel 1: Naive — one element per thread (matches naive_elementwise_add_kernel)
// ============================================================================
__global__ void naive_elementwise_add_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N)
{
    // Global thread index
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Map to 2D coordinates (same as CuTeDSL: ni = thread_idx % n, mi = thread_idx // n)
    int ni = thread_idx % N;   // column index (faster varying)
    int mi = thread_idx / N;   // row index   (slower varying)

    // Bounds check
    if (mi < M && ni < N) {
        int offset = mi * N + ni;
        C[offset] = __hadd(A[offset], B[offset]);
    }
}

// ============================================================================
// Kernel 2: Vectorized — 4 FP16 elements per thread via 64-bit load/store
// (matches vectorized_elementwise_add_kernel with zipped_divide(tensor, (1,4)))
// ============================================================================

// We use uint2 (64 bits = 4 × FP16) as the vector type for load/store.
// This is equivalent to the CuTeDSL's .load() on a (1,4):(0,1) tile slice.

// Helper union for reinterpreting 4 half values as uint2 (64-bit)
union Half4 {
    uint2   vec;        // 64-bit for vectorized ld/st
    __half  elems[4];   // 4 × FP16
};

__global__ void vectorized_elementwise_add_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N)
{
    /*
     * CuTeDSL equivalent:
     *   gA = zipped_divide(mA, (1, 4))
     *   gA layout: ((1,4), (M, N/4)) : ((0,1), (N, 4))
     *                mode0    mode1
     *
     *   m, n = gA.shape[1]       # mode1 shape = (M, N/4)
     *   ni = thread_idx % n      # col-tile index
     *   mi = thread_idx // n     # row-tile index
     *   a_val = gA[(None, (mi, ni))].load()   # load 4 contiguous FP16
     */

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Tile grid dimensions (mode 1 of zipped_divide result)
    int n_tiles = N / 4;          // number of col-tiles
    // int m_tiles = M;           // number of row-tiles (1 row per tile since tiler mode-0 = 1)

    // Map thread index to tile coordinates
    int ni = thread_idx % n_tiles;   // which col-tile (0 .. N/4-1)
    int mi = thread_idx / n_tiles;   // which row-tile (0 .. M-1)

    // Bounds check
    if (mi >= M) return;

    // Compute base element offset: mi * N + ni * 4
    // This matches CuTeDSL stride: (N, 4) for mode1
    int base_offset = mi * N + ni * 4;

    // Vectorized load: 4 contiguous FP16 elements = 64 bits
    // Equivalent to gA[(None, (mi, ni))].load()
    const uint2* A_vec = reinterpret_cast<const uint2*>(A + base_offset);
    const uint2* B_vec = reinterpret_cast<const uint2*>(B + base_offset);
    uint2* C_vec = reinterpret_cast<uint2*>(C + base_offset);

    Half4 a_val, b_val, c_val;
    a_val.vec = *A_vec;   // single 64-bit load  (LD.64)
    b_val.vec = *B_vec;   // single 64-bit load  (LD.64)

    // Element-wise addition
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c_val.elems[i] = __hadd(a_val.elems[i], b_val.elems[i]);
    }

    // Vectorized store: single 64-bit store (ST.64)
    *C_vec = c_val.vec;
}

// ============================================================================
// Host code
// ============================================================================
void fill_random_half(float* h_float, __half* h_half, int n) {
    for (int i = 0; i < n; i++) {
        // Random values in [-1, 1]
        h_float[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_half[i] = __float2half(h_float[i]);
    }
}

bool verify_results(const __half* h_C, const float* h_A_float, const float* h_B_float,
                    int M, int N, const char* kernel_name) {
    bool pass = true;
    int errors = 0;
    const int max_print_errors = 10;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            float expected = h_A_float[idx] + h_B_float[idx];
            float actual = __half2float(h_C[idx]);
            float diff = fabsf(expected - actual);
            // FP16 has ~3 decimal digits of precision
            if (diff > 1e-2f) {
                if (errors < max_print_errors) {
                    printf("  [%s] MISMATCH at (%d,%d): expected %.6f, got %.6f (diff=%.6f)\n",
                           kernel_name, i, j, expected, actual, diff);
                }
                pass = false;
                errors++;
            }
        }
    }

    if (pass) {
        printf("  [%s] PASSED ✓ — all %d elements match\n", kernel_name, M * N);
    } else {
        printf("  [%s] FAILED ✗ — %d / %d mismatches\n", kernel_name, errors, M * N);
    }
    return pass;
}

int main(int argc, char** argv) {
    // Default dimensions matching quick_run.py
    int M = 128, N = 64;
    if (argc >= 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }

    // N must be divisible by 4 for vectorization
    if (N % 4 != 0) {
        fprintf(stderr, "Error: N (%d) must be divisible by 4\n", N);
        return 1;
    }

    int total_elements = M * N;
    size_t size_half = total_elements * sizeof(__half);
    size_t size_float = total_elements * sizeof(float);

    printf("=== Vectorized Elementwise Add (Pure CUDA) ===\n");
    printf("Dimensions: M=%d, N=%d (%d elements)\n\n", M, N, total_elements);

    // ---- Allocate host memory ----
    float* h_A_float = (float*)malloc(size_float);
    float* h_B_float = (float*)malloc(size_float);
    __half* h_A = (__half*)malloc(size_half);
    __half* h_B = (__half*)malloc(size_half);
    __half* h_C_naive = (__half*)malloc(size_half);
    __half* h_C_vec = (__half*)malloc(size_half);

    srand(42);
    fill_random_half(h_A_float, h_A, total_elements);
    fill_random_half(h_B_float, h_B, total_elements);

    // ---- Allocate device memory ----
    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_half));
    CUDA_CHECK(cudaMalloc(&d_B, size_half));
    CUDA_CHECK(cudaMalloc(&d_C, size_half));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_half, cudaMemcpyHostToDevice));

    int threads_per_block = 256;

    // ==================================================================
    // Test 1: Naive kernel
    // ==================================================================
    printf("[1] Naive kernel (1 element/thread):\n");
    {
        CUDA_CHECK(cudaMemset(d_C, 0, size_half));

        int grid_size = (total_elements + threads_per_block - 1) / threads_per_block;
        naive_elementwise_add_kernel<<<grid_size, threads_per_block>>>(
            d_A, d_B, d_C, M, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size_half, cudaMemcpyDeviceToHost));
        verify_results(h_C_naive, h_A_float, h_B_float, M, N, "Naive");
    }

    // ==================================================================
    // Test 2: Vectorized kernel (4 elements/thread via 64-bit load/store)
    // ==================================================================
    printf("\n[2] Vectorized kernel (4 FP16 elements/thread, 64-bit ld/st):\n");
    {
        CUDA_CHECK(cudaMemset(d_C, 0, size_half));

        // Total tiles = M * (N/4).  Each thread handles one tile.
        int total_tiles = M * (N / 4);
        int grid_size = (total_tiles + threads_per_block - 1) / threads_per_block;

        printf("  Grid: %d blocks × %d threads = %d threads for %d tiles\n",
               grid_size, threads_per_block, grid_size * threads_per_block, total_tiles);

        vectorized_elementwise_add_kernel<<<grid_size, threads_per_block>>>(
            d_A, d_B, d_C, M, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C_vec, d_C, size_half, cudaMemcpyDeviceToHost));
        verify_results(h_C_vec, h_A_float, h_B_float, M, N, "Vectorized");
    }

    // ==================================================================
    // Test 3: Cross-verify naive vs vectorized (must produce identical results)
    // ==================================================================
    printf("\n[3] Cross-verification (naive vs vectorized):\n");
    {
        bool match = true;
        for (int i = 0; i < total_elements; i++) {
            if (__half2float(h_C_naive[i]) != __half2float(h_C_vec[i])) {
                printf("  MISMATCH at linear index %d: naive=%.6f, vec=%.6f\n",
                       i, __half2float(h_C_naive[i]), __half2float(h_C_vec[i]));
                match = false;
            }
        }
        if (match) {
            printf("  Naive and Vectorized produce IDENTICAL results ✓\n");
        }
    }

    // ==================================================================
    // Print layout analysis (matching CuTeDSL output)
    // ==================================================================
    printf("\n=== Layout Analysis ===\n");
    printf("Original tensor:  (%d, %d) : (%d, 1)  [row-major FP16]\n", M, N, N);
    printf("After zipped_divide(tensor, (1, 4)):\n");
    printf("  Shape:  ((1, 4), (%d, %d))\n", M, N/4);
    printf("  Stride: ((0, 1), (%d, 4))\n", N);
    printf("\n");
    printf("  Mode 0 = tile interior: 4 contiguous FP16 = 8 bytes = 64-bit vector\n");
    printf("  Mode 1 = tile grid:     %d × %d = %d tiles (one per thread)\n",
           M, N/4, M * (N/4));
    printf("\n");
    printf("  Thread (mi, ni) loads from offset: mi × %d + ni × 4\n", N);
    printf("  Example: Thread 0 → (mi=0, ni=0) → elements [0,1,2,3]\n");
    printf("  Example: Thread 1 → (mi=0, ni=1) → elements [4,5,6,7]\n");
    printf("  Example: Thread %d → (mi=1, ni=0) → elements [%d,%d,%d,%d]\n",
           N/4, N, N+1, N+2, N+3);

    // Cleanup
    free(h_A_float); free(h_B_float);
    free(h_A); free(h_B);
    free(h_C_naive); free(h_C_vec);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("\nDone.\n");
    return 0;
}
