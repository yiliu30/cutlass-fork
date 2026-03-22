"""
Step 1 Deep Trace: M=256, N=256, K=512
  → 2x2 = 4 CTAs, 64 k-tiles per CTA
  → using arange data so we can verify exact values
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

BM, BN, BK = 128, 128, 8
NUM_THREADS = 256


@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    cta_tiler = (BM, BN, BK)

    gA = cute.local_tile(mA, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=cta_tiler, coord=(bidx, bidy, None), proj=(1, 1, None))

    print(f"gA: {gA}")
    print(f"gB: {gB}")
    print(f"gC: {gC}")

    thr_mma = tiled_mma.get_slice(tidx)
    tCgC = thr_mma.partition_C(gC)
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    print(f"tCgC: {tCgC}")
    print(f"tCrC: {tCrC}")

    num_k_tiles = cute.size(gA, mode=[2])

    for k_tile in range(num_k_tiles):
        tCgA = thr_mma.partition_A(gA[None, None, k_tile])
        tCgB = thr_mma.partition_B(gB[None, None, k_tile])

        if k_tile == 0:
            print(f"tCgA: {tCgA}")
            print(f"tCgB: {tCgB}")

        tCrA = cute.make_fragment_like(tCgA)
        tCrB = cute.make_fragment_like(tCgB)
        cute.autovec_copy(tCgA, tCrA)
        cute.autovec_copy(tCgB, tCrB)

        num_k_blocks = cute.size(tCrA, mode=[2])
        if k_tile == 0:
            print(f"num_k_blocks: {num_k_blocks}")

        for k_block in range(num_k_blocks):
            cute.gemm(tiled_mma, tCrC,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block],
                      tCrC)

    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
    cute.copy(atom, tCrC, tCgC)
    return


@cute.jit
def host_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    permutation_tiler_M = cute.make_layout((16, 4), stride=(4, 1))
    permutation_tiler_N = cute.make_layout((16, 4), stride=(4, 1))

    tiled_mma = cute.make_tiled_mma(
        op, atoms_layout,
        permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
    )

    print(f"===== TiledMma =====")
    print(f"{tiled_mma}")
    print(f"====================")

    grid = (*cute.ceil_div(mC.shape, (BM, BN)), 1)
    print(f"grid: {grid}")
    print(f"mA CuTe: shape={mA.shape}, stride={mA.stride}")
    print(f"mB CuTe: shape={mB.shape}, stride={mB.stride}")
    print(f"mC CuTe: shape={mC.shape}, stride={mC.stride}")

    gemm_kernel(mA, mB, mC, tiled_mma).launch(
        grid=grid, block=(NUM_THREADS, 1, 1),
    )


M, N, K = 256, 256, 512
torch.manual_seed(42)

# A: (M, K) M-major. torch.permute gives us shape (K, M) stride (1, K).
# CuTe sees: shape (K, M) = (512, 256), stride (1, 512).
# local_tile with proj=(1, None, 1) tiles dim-0 by BM=128, dim-1(→2) by BK=8
# BUT dim-0 is K=512, dim-1 is M=256...
# Wait: from_dlpack uses the DLPack shape directly.
# So mA is (K, M) in CuTe. Then proj=(1, None, 1) tiles:
#   dim-0 (K) by cta_tiler[0] = BM = 128
#   dim-1 (M) by cta_tiler[2] = BK = 8
# That's WRONG!
#
# The original step1 does the same permute but with M=N=K=512,
# so both dims are 512 and it happens to work because the shape is square.

# Let's just use the same random init as the original:
a_mk = torch.randn(M, K, dtype=torch.float32, device="cuda")  # logical (M,K) contiguous
# For M-major: want stride (1, M) → create (K, M) contiguous then permute → (M, K):(K, 1)
# No wait. "M-major" means stride-0 = 1, stride-1 = M. In torch that's column-major.
# torch.randn(K, M).permute(1,0) → shape (M,K), stride (1, K). stride = (1, K=512).
# That means A[m,k] = ptr + m*1 + k*K. Consecutive m's are adjacent. That IS M-major,
# but the stride-in-K is K, not M! Unless M == K.
#
# Original code: stride = (1, M). So: A[m,k] = ptr + m*1 + k*M.
# After permute: shape (M,K) stride (1, K). But from_dlpack sees shape (K,M) stride (1,K).
#
# Actually no. Let me re-read the original code:
# a = torch.arange(M*K,...).reshape(M,K).permute(1,0)
# reshape(M,K) → shape (M,K), stride (K, 1)
# permute(1,0) → shape (K,M), stride (1, K)
# from_dlpack(a) → CuTe tensor with shape (K, M), stride (1, K)
#
# So CuTe mA is (K, M):(1, K). Then local_tile with cta_tiler=(BM,BN,BK):
# proj=(1, None, 1): dim-0 gets tiler[0]=BM, dim-1 gets tiler[2]=BK
# So dim-0 (size K=512) is tiled by BM=128 → wait that doesn't match.
#
# Unless local_tile is smarter than I think...

a = torch.randn(K, M, dtype=torch.float32, device="cuda").permute(1, 0)
b = torch.randn(K, N, dtype=torch.float32, device="cuda").permute(1, 0)
c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

print(f"=== Torch tensors ===")
print(f"a: shape={a.shape}, stride={a.stride()}")
print(f"b: shape={b.shape}, stride={b.stride()}")
print(f"c: shape={c.shape}, stride={c.stride()}")

mA = from_dlpack(a, assumed_align=16)
mB = from_dlpack(b, assumed_align=16)
mC = from_dlpack(c, assumed_align=16)

print("\nCompiling...")
compiled = cute.compile(host_gemm, mA, mB, mC)
print("Running...")
compiled(mA, mB, mC)
torch.cuda.synchronize()

ref = torch.einsum("mk,nk->mn", a.float(), b.float())
torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)
print("PASS!")
