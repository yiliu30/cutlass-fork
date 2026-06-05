"""
Standalone int2 -> bf16 Dequantization Kernel (CuTe DSL)
========================================================

WHAT THIS IS
  A self-contained, correctness-first example that takes 2-bit signed integer
  weights (values in {-2, -1, 0, 1}), converts them to bfloat16, and multiplies
  by a per-128-element "scale" factor -- exactly the dequantization step that a
  mixed-input int2 x bf16 GEMM would run on its A operand, but lifted out on its
  own so we can prove it against a PyTorch reference before touching the GEMM.

WHY IT LOOKS THE WAY IT DOES
  The CuTe DSL has NO 2-bit integer type (Int4 is the narrowest it offers). So
  instead of inventing one, we store 16 packed int2 values inside one ordinary
  32-bit integer -- a type that already exists -- and unpack them in the kernel
  with plain shift/mask/sign-extend arithmetic. This sidesteps the hard part
  (teaching TMA, shared memory, and the tensor core about a brand-new 2-bit type)
  entirely, which is deferred to the real GEMM integration.

PACKING SCHEME
  Logical A:  (M, K) int2 in {-2,-1,0,1}
  Packed  A:  (M, K//16) int32  -- field j of word w holds K-position (w*16 + j),
                                    stored in bits [2j, 2j+1]
  Scale:      (M, K//128) bf16  -- one scale per 128 contiguous K elements.
                                    Since 128/16 = 8, all 16 lanes of one word
                                    share a single scale (group = word // 8).
  Output  C:  (M, K) bf16

THE 2-BIT SIGN-EXTEND TRICK
  raw    = (word >> (2*j)) & 0x3      -> {0,1,2,3}  (unsigned 2-bit field)
  signed = (raw ^ 0x2) - 0x2          -> {0,1,-2,-1} (two's-complement of 2 bits)
    00 -> ( 0 ^ 2) - 2 =  0
    01 -> ( 1 ^ 2) - 2 =  1
    10 -> ( 2 ^ 2) - 2 = -2
    11 -> ( 3 ^ 2) - 2 = -1
  (The mask `& 0x3` makes the preceding shift's signedness irrelevant, so it is
   safe even though Int32 right-shift is arithmetic.)

RUN
  /home/yiliu7/workspace/venvs/cutlass-dsl/bin/python int2_convert_standalone.py
  (Requires a Blackwell GPU + CUDA 13.x. The sage-local interpreter in CLAUDE.md
   is stale; use the cutlass-dsl venv above.)
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# Compile-time layout constants.
INT2_PER_WORD = 16          # 32-bit word / 2 bits = 16 packed int2 values
WORDS_PER_GROUP = 8         # 128-element scale group / 16 ints-per-word = 8 words


# =============================================================================
# GPU Kernel: one int32 word per thread -> 16 bf16 outputs
# =============================================================================
@cute.kernel
def int2_dequant_kernel(
    gA_packed: cute.Tensor,  # (M, K//16) int32  -- 16 packed int2 per element
    gScale: cute.Tensor,     # (M, K//128) bf16  -- one scale per 128-K group
    gC: cute.Tensor,         # (M, K) bf16       -- dequantized output
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    w = bidx * bdim + tidx                 # global int32-word index

    # Map the linear word index back to (row, word-within-row).
    _, words_per_row = gA_packed.shape
    row = w // words_per_row
    word_in_row = w % words_per_row

    packed = gA_packed[row, word_in_row]   # Int32 holding 16 packed int2 values

    # One scale per 128 K-elements; 128/16 = 8 words share a scale group.
    # Widen bf16 -> fp32 once so the multiply happens in fp32 (see reference note).
    s = gScale[row, word_in_row // WORDS_PER_GROUP].to(cutlass.Float32)

    base_k = word_in_row * INT2_PER_WORD   # first K-column this word writes

    # range_constexpr fully unrolls at trace time, so `j` stays a static Python
    # int -- `2*j` becomes a shift-immediate and `base_k + j` a static offset.
    for j in cutlass.range_constexpr(INT2_PER_WORD):
        raw = (packed >> (2 * j)) & 0x3        # extract 2-bit field -> {0,1,2,3}
        signed = (raw ^ 0x2) - 0x2             # sign-extend -> {0,1,-2,-1}
        # convert -> fp32, scale in fp32, then round once to bf16
        val = (signed.to(cutlass.Float32) * s).to(cutlass.BFloat16)
        gC[row, base_k + j] = val


# =============================================================================
# Host JIT launcher
# =============================================================================
@cute.jit
def int2_dequant(
    mA_packed: cute.Tensor,
    mScale: cute.Tensor,
    mC: cute.Tensor,
):
    threads_per_block = 256
    total_words = cute.size(mA_packed)     # M * (K//16)

    print("[DSL INFO] int2 dequant tensors:")
    print(f"[DSL INFO]   mA_packed = {mA_packed}")
    print(f"[DSL INFO]   mScale    = {mScale}")
    print(f"[DSL INFO]   mC        = {mC}")

    int2_dequant_kernel(mA_packed, mScale, mC).launch(
        grid=(total_words // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


# =============================================================================
# Host helpers
# =============================================================================
def pack_int2(a_logical: torch.Tensor) -> torch.Tensor:
    """Pack (M, K) logical int2 values into (M, K//16) int32 words.

    Field j of word w (bits [2j, 2j+1]) holds K-position w*16 + j.
    Vectorized -- no Python loop over M*K, only a 16-step loop over bit-fields.
    """
    M, K = a_logical.shape
    assert K % INT2_PER_WORD == 0, "K must be a multiple of 16 to pack into int32"
    a_2bit = a_logical.to(torch.int32) & 0x3                  # low 2 bits only
    a_grp = a_2bit.reshape(M, K // INT2_PER_WORD, INT2_PER_WORD)
    packed = torch.zeros(M, K // INT2_PER_WORD, dtype=torch.int32, device=a_logical.device)
    for j in range(INT2_PER_WORD):
        packed |= a_grp[:, :, j] << (2 * j)
    return packed


def reference_dequant(a_logical: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Ground-truth dequant: fp32 multiply, round once to bf16.

    Matches the kernel's arithmetic exactly: signed int (exact in fp32) times
    scale (bf16 widened to fp32, exact), product in fp32, single round to bf16.
    """
    scale_full = scale.to(torch.float32).repeat_interleave(128, dim=1)  # (M, K)
    return (a_logical.to(torch.float32) * scale_full).to(torch.bfloat16)


def run_case(M: int, K: int, seed: int = 0, scale_mode: str = "rand", verbose: bool = True):
    """Build data for one (M, K) case, run the kernel, verify against torch."""
    assert K % 128 == 0, "this example requires K % 128 == 0 (one scale per 128-K group)"
    assert (M * (K // INT2_PER_WORD)) % 256 == 0, "total words must be a multiple of 256"
    torch.manual_seed(seed)

    # Logical int2 weights in {-2,-1,0,1} (randint high is exclusive).
    a_logical = torch.randint(-2, 2, (M, K), dtype=torch.int32, device="cuda")

    # Per-128-K-group bf16 scales.
    if scale_mode == "ones":
        scale = torch.ones(M, K // 128, dtype=torch.bfloat16, device="cuda")
    else:
        scale = torch.rand(M, K // 128, dtype=torch.float32, device="cuda").to(torch.bfloat16)

    a_packed = pack_int2(a_logical)
    c = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")

    a_ = from_dlpack(a_packed, assumed_align=16)
    s_ = from_dlpack(scale, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled = cute.compile(int2_dequant, a_, s_, c_)
    compiled(a_, s_, c_)

    ref = reference_dequant(a_logical, scale)
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=1e-2)

    if verbose:
        codes = set(a_logical.unique().tolist())
        print(f"  [M={M}, K={K}, scale={scale_mode}, seed={seed}] "
              f"codes={sorted(codes)} -> Correctness: PASSED")
    return c, ref, a_logical, scale


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=== int2 -> bf16 dequant: standalone correctness run ===")

    # Primary case: K a multiple of both 16 (packing) and 128 (scale group).
    run_case(M=256, K=512, seed=0, scale_mode="rand")

    # scale = ones isolates the convert path from the multiply.
    run_case(M=256, K=512, seed=1, scale_mode="ones")

    # A different K that is still a multiple of 128 (384 = 3 * 128).
    run_case(M=256, K=384, seed=2, scale_mode="rand")

    print("All cases passed.")
