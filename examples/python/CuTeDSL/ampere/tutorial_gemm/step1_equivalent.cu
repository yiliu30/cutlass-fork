// step1_equivalent.cu
// Pure CUDA equivalent of step1_naive_gemm.py (CuTe DSL)
// C[M,N] = A[M,K] * B[N,K]^T
// A: col-major (stride 1,M), B: col-major (stride 1,N), C: row-major (stride N,1)
// Tile: 128×128×8, Threads: 256, MMA: scalar FMA

#define BM 128
#define BN 128
#define BK 8
#define NUM_THREADS 256
#define GROUP_SIZE 4   // R from permutation (16,4):(4,1)
#define GROUP_GAP 64   // spacing between groups

__global__ void gemm_naive(
    const float* __restrict__ A,  // (M,K) col-major
    const float* __restrict__ B,  // (N,K) col-major
    float* __restrict__ C,        // (M,N) row-major
    int M, int N, int K)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    // Thread's position in 16×16 grid (atoms_layout)
    const int thr_m = tidx / 16;  // 0..15
    const int thr_n = tidx % 16;  // 0..15

    // Compute owned positions (permutation: 2 groups of 4 consecutive)
    int m_pos[8], n_pos[8];
    #pragma unroll
    for (int g = 0; g < 2; g++) {
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i++) {
            m_pos[g*GROUP_SIZE + i] = thr_m*GROUP_SIZE + g*GROUP_GAP + i;
            n_pos[g*GROUP_SIZE + i] = thr_n*GROUP_SIZE + g*GROUP_GAP + i;
        }
    }

    // Accumulator in registers
    float acc[8][8];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        for (int n = 0; n < 8; n++)
            acc[m][n] = 0.0f;

    // CTA base pointers
    const float* A_cta = A + bidx * BM;   // offset in M
    const float* B_cta = B + bidy * BN;   // offset in N

    // K-tile loop
    for (int kt = 0; kt < K / BK; kt++) {

        // Load A: 8 M-positions × 8 K-values
        float rA[8][8];
        #pragma unroll
        for (int m = 0; m < 8; m++)
            for (int k = 0; k < BK; k++)
                rA[m][k] = A_cta[m_pos[m] + (kt*BK + k) * M];

        // Load B: 8 N-positions × 8 K-values
        float rB[8][8];
        #pragma unroll
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < BK; k++)
                rB[n][k] = B_cta[n_pos[n] + (kt*BK + k) * N];

        // Compute: 8×8×8 = 512 FMAs per k_tile
        #pragma unroll
        for (int k = 0; k < BK; k++)
            for (int m = 0; m < 8; m++)
                for (int n = 0; n < 8; n++)
                    acc[m][n] += rA[m][k] * rB[n][k];
    }

    // Store C
    #pragma unroll
    for (int m = 0; m < 8; m++) {
        int gm = bidx*BM + m_pos[m];
        for (int n = 0; n < 8; n++) {
            int gn = bidy*BN + n_pos[n];
            C[gm*N + gn] = acc[m][n];
        }
    }
}
