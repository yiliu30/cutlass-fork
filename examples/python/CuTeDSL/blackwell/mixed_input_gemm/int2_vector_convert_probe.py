"""
Int2 VECTOR-convert probe (closes task #20) -- v4, one-thread-per-row
====================================================================
Validates the one open unknown for the naive int2 path: an Int2 *vector*
`.to(BFloat16)` (the op `cvt_tensor_a` runs as `src.load().to(mma_dtype)`).

Rules learned (all consistent with how the int4 GEMM behaves):
  - Int2 elements cannot be scalar-indexed (read/write). Move them as vectors.
  - Recasting a single Int32 *register* to Int2 degenerates to lane 0; the Int2
    vector must come from a *memory* Int2 view via copy.

Construction (mirrors GEMM recast-smem -> rmem Int2 fragment -> load):
  a. recast packed Int32 gmem tensor -> Int2 gmem view (ptr<i2,gmem>, K*16 cols)
  b. one thread per row: slice that row's Int2 view (memory slice, legal)
  c. make an Int2 register fragment like the row, autovec_copy into it
  d. .load() -> Int2 vector ; .to(BFloat16) -> vector itofp under test
  e. autovec_copy the bf16 fragment back out to the row of C
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def int2_row_kernel(gA_i2: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    row = bidx * bdim + tidx                      # one thread per row

    gA_row = gA_i2[(row, None)]                    # (K,) Int2 memory slice
    gC_row = gC[(row, None)]                       # (K,) bf16 memory slice

    # (c) Int2 register fragment, filled by vector copy -- no sub-byte deref
    frag_i2 = cute.make_fragment_like(gA_row)
    cute.autovec_copy(gA_row, frag_i2)

    # (d) load Int2 vector and widen -- the op task #20 validates
    frag_bf16 = cute.make_fragment_like(gC_row)
    frag_bf16.store(frag_i2.load().to(cutlass.BFloat16))

    # (e) copy bf16 fragment back to gmem row
    cute.autovec_copy(frag_bf16, gC_row)


@cute.jit
def int2_row(mA_packed: cute.Tensor, mC: cute.Tensor):
    mA_i2 = cute.recast_tensor(mA_packed, cutlass.Int2)
    print(f"[DSL INFO] packed   = {mA_packed}")
    print(f"[DSL INFO] int2view = {mA_i2}")
    M, _ = mA_i2.shape
    threads_per_block = 128
    int2_row_kernel(mA_i2, mC).launch(
        grid=(M // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def pack_int2(a_logical: torch.Tensor) -> torch.Tensor:
    M, K = a_logical.shape
    a_2bit = a_logical.to(torch.int32) & 0x3
    a_grp = a_2bit.reshape(M, K // 16, 16)
    packed = torch.zeros(M, K // 16, dtype=torch.int32, device=a_logical.device)
    for j in range(16):
        packed |= a_grp[:, :, j] << (2 * j)
    return packed


def main():
    print("=== Int2 VECTOR convert probe v4 (task #20) ===")
    M, K = 256, 512
    torch.manual_seed(0)
    a_logical = torch.randint(-2, 2, (M, K), dtype=torch.int32, device="cuda")
    a_packed = pack_int2(a_logical)
    c = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")

    a_ = from_dlpack(a_packed, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    compiled = cute.compile(int2_row, a_, c_)
    compiled(a_, c_)

    ref = a_logical.to(torch.float32).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, rtol=0, atol=0)
    print(f"  codes seen: {sorted(set(a_logical.unique().tolist()))}")
    print("  PASSED -- Int2 *vector* .to(BFloat16) bit-exact via copy path")
    print(">>> task #20: width-parametric itofp handles int2 vectors; "
          "no new cvt_i2_bf16 primitive needed for the naive path.")


if __name__ == "__main__":
    main()
