"""
Int2 GO/NO-GO probe
===================
Two questions, answered in order:

  (A) TYPE REGISTRATION  — does `cutlass.Int2` exist and behave like Int4
      at the Python/Numeric level (width, signedness, string parse, ALL_DTYPES)?

  (B) ROUND-TRIP THROUGH THE DSL  — can a genuine 2-bit element survive the
      load/convert path? We DON'T yet have TMA/smem proven for i2, so we probe
      the *minimal* thing the GEMM needs: take int2 logical values, run them
      through an in-trace convert to bf16, and match a torch reference.

If (B) compiles and matches, the naive int2 convert primitive is viable and we
can proceed to wire the GEMM. If it rejects 2-bit elements, STOP and report.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import dtype, ALL_DTYPES


# -----------------------------------------------------------------------------
# (A) Type registration — pure Python, no GPU
# -----------------------------------------------------------------------------
def check_type_registration():
    print("=== (A) Int2 type registration ===")
    assert cutlass.Int2.width == 2, cutlass.Int2.width
    assert cutlass.Int2.signed is True, cutlass.Int2.signed
    assert dtype("Int2") is cutlass.Int2, dtype("Int2")
    assert cutlass.Int2 in ALL_DTYPES
    print(f"  cutlass.Int2 = {cutlass.Int2}  width={cutlass.Int2.width} "
          f"signed={cutlass.Int2.signed}")
    print(f"  dtype('Int2') -> {dtype('Int2')}   in ALL_DTYPES: "
          f"{cutlass.Int2 in ALL_DTYPES}")
    print("  (A) PASSED\n")


# -----------------------------------------------------------------------------
# (B) Round-trip: int2 logical -> bf16 via an in-trace convert
#
# We feed the SAME packed-int32 representation the standalone uses, but this
# time we ALSO reconstruct each value through cutlass.Int2 arithmetic so the
# 2-bit Numeric type is actually exercised in-trace (not just Int32 math).
# -----------------------------------------------------------------------------
INT2_PER_WORD = 16


@cute.kernel
def int2_typed_kernel(gA_packed: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    w = bidx * bdim + tidx

    _, words_per_row = gA_packed.shape
    row = w // words_per_row
    word_in_row = w % words_per_row
    packed = gA_packed[row, word_in_row]
    base_k = word_in_row * INT2_PER_WORD

    for j in cutlass.range_constexpr(INT2_PER_WORD):
        raw = (packed >> (2 * j)) & 0x3            # {0,1,2,3} in an Int32
        signed = (raw ^ 0x2) - 0x2                 # {-2,-1,0,1} in an Int32
        # Route through the Int2 Numeric type explicitly: build an Int2 from the
        # extracted value, then widen. This is what proves the 2-bit type works.
        as_i2 = cutlass.Int2(signed)               # <-- exercises Int2 ctor/cast
        val = as_i2.to(cutlass.Float32).to(cutlass.BFloat16)
        gC[row, base_k + j] = val


@cute.jit
def int2_typed(mA_packed: cute.Tensor, mC: cute.Tensor):
    threads_per_block = 256
    total_words = cute.size(mA_packed)
    int2_typed_kernel(mA_packed, mC).launch(
        grid=(total_words // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def pack_int2(a_logical: torch.Tensor) -> torch.Tensor:
    M, K = a_logical.shape
    a_2bit = a_logical.to(torch.int32) & 0x3
    a_grp = a_2bit.reshape(M, K // INT2_PER_WORD, INT2_PER_WORD)
    packed = torch.zeros(M, K // INT2_PER_WORD, dtype=torch.int32, device=a_logical.device)
    for j in range(INT2_PER_WORD):
        packed |= a_grp[:, :, j] << (2 * j)
    return packed


def check_roundtrip():
    print("=== (B) Int2 round-trip through DSL convert ===")
    from cutlass.cute.runtime import from_dlpack
    M, K = 256, 512
    torch.manual_seed(0)
    a_logical = torch.randint(-2, 2, (M, K), dtype=torch.int32, device="cuda")
    a_packed = pack_int2(a_logical)
    c = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")

    a_ = from_dlpack(a_packed, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    compiled = cute.compile(int2_typed, a_, c_)
    compiled(a_, c_)

    ref = a_logical.to(torch.float32).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, rtol=0, atol=0)
    print(f"  codes seen: {sorted(set(a_logical.unique().tolist()))}")
    print("  (B) PASSED — Int2 element survived the in-trace convert\n")


if __name__ == "__main__":
    check_type_registration()
    check_roundtrip()
    print(">>> GO: Int2 type is viable through the naive convert path.")
