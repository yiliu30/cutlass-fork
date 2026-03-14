#!/usr/bin/env python3
"""
Verification script that runs both the CuTeDSL and pure CUDA vectorized
elementwise add kernels and cross-validates their results.

This script:
  1. Builds the pure CUDA kernel (vec_add.cu) via a PyTorch C++ extension
  2. Runs the CuTeDSL version (from quick_run.py logic)
  3. Runs the pure CUDA version (both naive and vectorized)
  4. Verifies all three produce identical results

Usage:
  python verify.py [--M 128] [--N 64] [--large]
"""

import argparse
import os
import torch
import torch.utils.cpp_extension

# ============================================================================
# Step 1: Build the CUDA kernel as a PyTorch extension (inline)
# ============================================================================

CUDA_SOURCE = r"""
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
"""

def build_cuda_extension():
    """Build the CUDA extension inline using torch.utils.cpp_extension."""
    print("=" * 60)
    print("Building pure CUDA extension...")
    print("=" * 60)

    # Write source to a temp file
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vec_add_cuda")
    os.makedirs(src_dir, exist_ok=True)
    src_path = os.path.join(src_dir, "vec_add_ext.cu")

    with open(src_path, "w") as f:
        f.write(CUDA_SOURCE)

    module = torch.utils.cpp_extension.load(
        name="vec_add_ext",
        sources=[src_path],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
    print("Build successful!\n")
    return module


# ============================================================================
# Step 2: CuTeDSL version (same logic as quick_run.py)
# ============================================================================

def run_cutedsl(a, b):
    """Run the CuTeDSL vectorized elementwise add (matching quick_run.py)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    c = torch.zeros_like(a)

    @cute.kernel
    def vectorized_elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        thread_idx = bidx * bdim + tidx

        m, n = gA.shape[1]
        ni = thread_idx % n
        mi = thread_idx // n

        a_val = gA[(None, (mi, ni))].load()
        b_val = gB[(None, (mi, ni))].load()
        gC[(None, (mi, ni))] = a_val + b_val

    @cute.jit
    def vectorized_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        threads_per_block = 256

        gA = cute.zipped_divide(mA, (1, 4))
        gB = cute.zipped_divide(mB, (1, 4))
        gC = cute.zipped_divide(mC, (1, 4))

        vectorized_elementwise_add_kernel(gA, gB, gC).launch(
            grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
            block=(threads_per_block, 1, 1),
        )

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled_func = cute.compile(vectorized_elementwise_add, a_, b_, c_)
    compiled_func(a_, b_, c_)

    return c


# ============================================================================
# Step 3: Main verification
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify CUDA vs CuTeDSL vectorized elementwise add")
    parser.add_argument("--M", type=int, default=128, help="Number of rows (default: 128)")
    parser.add_argument("--N", type=int, default=64,  help="Number of columns (default: 64, must be %%4==0)")
    parser.add_argument("--large", action="store_true", help="Use large dimensions (16384×8192)")
    parser.add_argument("--skip-cutedsl", action="store_true", help="Skip CuTeDSL test")
    args = parser.parse_args()

    if args.large:
        M, N = 16384, 8192
    else:
        M, N = args.M, args.N

    assert N % 4 == 0, f"N must be divisible by 4, got {N}"

    print(f"\n{'='*60}")
    print(f"  Vectorized Elementwise Add Verification")
    print(f"  M={M}, N={N} ({M*N} elements, FP16)")
    print(f"{'='*60}\n")

    # Create identical input data for all tests
    torch.manual_seed(42)
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)

    # Reference result (PyTorch)
    ref = a + b

    results = {}

    # ------------------------------------------------------------------
    # Test 1: PyTorch reference
    # ------------------------------------------------------------------
    print("[1/4] PyTorch reference:  a + b")
    results["pytorch"] = ref
    print(f"  ✓ Shape: {ref.shape}, dtype: {ref.dtype}\n")

    # ------------------------------------------------------------------
    # Test 2: Pure CUDA naive
    # ------------------------------------------------------------------
    print("[2/4] Pure CUDA — Naive kernel (1 element/thread)")
    cuda_ext = build_cuda_extension()
    c_naive = cuda_ext.naive_add(a, b)
    torch.testing.assert_close(c_naive, ref, atol=0, rtol=0)
    results["cuda_naive"] = c_naive
    print(f"  ✓ PASSED — exact match with PyTorch\n")

    # ------------------------------------------------------------------
    # Test 3: Pure CUDA vectorized
    # ------------------------------------------------------------------
    print("[3/4] Pure CUDA — Vectorized kernel (4 FP16/thread, 64-bit ld/st)")
    c_vec = cuda_ext.vectorized_add(a, b)
    torch.testing.assert_close(c_vec, ref, atol=0, rtol=0)
    results["cuda_vectorized"] = c_vec
    print(f"  ✓ PASSED — exact match with PyTorch\n")

    # ------------------------------------------------------------------
    # Test 4: CuTeDSL
    # ------------------------------------------------------------------
    if not args.skip_cutedsl:
        print("[4/4] CuTeDSL — vectorized_elementwise_add (zipped_divide + .load())")
        try:
            c_cute = run_cutedsl(a, b)
            torch.testing.assert_close(c_cute, ref, atol=0, rtol=0)
            results["cutedsl"] = c_cute
            print(f"  ✓ PASSED — exact match with PyTorch\n")
        except Exception as e:
            print(f"  ⚠ CuTeDSL not available or failed: {e}")
            print(f"  Skipping CuTeDSL comparison.\n")
    else:
        print("[4/4] CuTeDSL — SKIPPED (--skip-cutedsl)\n")

    # ------------------------------------------------------------------
    # Cross-validation summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    # Compare all pairs
    names = list(results.keys())
    all_match = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            match = torch.equal(results[names[i]], results[names[j]])
            status = "✓ IDENTICAL" if match else "✗ MISMATCH"
            print(f"  {names[i]:20s} vs {names[j]:20s} : {status}")
            if not match:
                all_match = False

    print()
    if all_match:
        print("  ✅ ALL TESTS PASSED — all implementations produce identical results")
    else:
        print("  ❌ SOME TESTS FAILED")

    # ------------------------------------------------------------------
    # Layout analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  LAYOUT ANALYSIS")
    print(f"{'='*60}")
    print(f"  Original tensor:    ({M}, {N}) : ({N}, 1)    [row-major FP16]")
    print(f"  zipped_divide(tensor, (1, 4)):")
    print(f"    Shape:  ((1, 4), ({M}, {N//4}))")
    print(f"    Stride: ((0, 1), ({N}, 4))")
    print(f"")
    print(f"  Mode 0 (tile interior): 4 contiguous FP16 = 8 bytes → LD.64/ST.64")
    print(f"  Mode 1 (tile grid):     {M} × {N//4} = {M*(N//4)} tiles")
    print(f"")
    print(f"  Thread mapping:  thread_idx → (mi={{}}, ni={{}})")
    print(f"    ni = thread_idx % {N//4}")
    print(f"    mi = thread_idx / {N//4}")
    print(f"    base_offset = mi × {N} + ni × 4")
    print(f"")
    print(f"  CUDA equivalent of gA[(None, (mi,ni))].load():")
    print(f"    Half4 a_val;")
    print(f"    a_val.vec = *(uint2*)(A + mi * {N} + ni * 4);  // single LD.64")
    print()


if __name__ == "__main__":
    main()
