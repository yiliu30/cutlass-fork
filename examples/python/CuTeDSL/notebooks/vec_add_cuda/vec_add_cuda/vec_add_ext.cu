
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// --- Naive kernel: 1 element per thread ---
__global__ void naive_add_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ni = idx % N;
    int mi = idx / N;
    if (mi < M && ni < N) {
        int offset = mi * N + ni;
        C[offset] = __hadd(A[offset], B[offset]);
    }
}

// --- Vectorized kernel: 4 FP16 elements per thread via 64-bit load/store ---
//
// This is the pure CUDA equivalent of the CuTeDSL code:
//   gA = cute.zipped_divide(mA, (1, 4))
//   gA layout: ((1,4), (M, N/4)) : ((0,1), (N, 4))
//   thread_idx → (mi, ni) in mode-1 tile grid
//   a_val = gA[(None, (mi, ni))].load()   ← vectorized 64-bit load
//
union Half4 {
    uint2    vec;       // 64 bits for vectorized ld/st
    __half   elems[4];  // 4 × FP16
};

__global__ void vectorized_add_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Tile grid: M row-tiles × (N/4) col-tiles
    int n_tiles = N / 4;
    int ni = thread_idx % n_tiles;   // col-tile index
    int mi = thread_idx / n_tiles;   // row-tile index

    if (mi >= M) return;

    // Base offset = mi * N + ni * 4  (stride (N, 4) from mode-1)
    int base = mi * N + ni * 4;

    // Vectorized 64-bit load (equivalent to .load() on a (1,4):(0,1) tile)
    Half4 a_val, b_val, c_val;
    a_val.vec = reinterpret_cast<const uint2*>(A + base)[0];
    b_val.vec = reinterpret_cast<const uint2*>(B + base)[0];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c_val.elems[i] = __hadd(a_val.elems[i], b_val.elems[i]);
    }

    // Vectorized 64-bit store
    reinterpret_cast<uint2*>(C + base)[0] = c_val.vec;
}

// --- PyTorch C++ extension entry points ---

torch::Tensor naive_add(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Inputs must be float16");
    TORCH_CHECK(A.sizes() == B.sizes(), "Shape mismatch");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0), N = A.size(1);
    auto C = torch::zeros_like(A);

    int threads = 256;
    int total = M * N;
    int blocks = (total + threads - 1) / threads;

    naive_add_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N);

    return C;
}

torch::Tensor vectorized_add(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Inputs must be float16");
    TORCH_CHECK(A.sizes() == B.sizes(), "Shape mismatch");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0), N = A.size(1);
    TORCH_CHECK(N % 4 == 0, "N must be divisible by 4 for vectorization");

    auto C = torch::zeros_like(A);

    int threads = 256;
    int total_tiles = M * (N / 4);
    int blocks = (total_tiles + threads - 1) / threads;

    vectorized_add_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_add", &naive_add, "Naive elementwise add (1 elem/thread)");
    m.def("vectorized_add", &vectorized_add, "Vectorized elementwise add (4 FP16/thread, 64-bit ld/st)");
}
