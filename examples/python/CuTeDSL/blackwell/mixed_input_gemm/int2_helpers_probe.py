"""
Task #21 verification — is_valid_scale_granularity now accepts Int2.

Checks the single edit made to utils/mixed_input_helpers.py:
  - Int2 with valid ConvertScale granularity (m=1, k=128, k%k==0, k%tiler==0) -> True
  - Int2 mirrors Int4 exactly (same inputs -> same verdict)
  - Int2 with bad granularity still rejected (regression guard)
  - is_shuffle_a still excludes Int2 (must stay False -> naive generic convert path)
"""

import cutlass
from cutlass.utils.mixed_input_helpers import (
    is_valid_scale_granularity,
    is_shuffle_a,
)

# Representative GEMM params (mirror the int4 example defaults).
K = 4096
MMA_TILER_K = 64
SCALE_M = 1
SCALE_K = 128  # ConvertScale: one scale per 128-K group


def check_scale_granularity():
    print("=== is_valid_scale_granularity: Int2 vs Int4 ===")
    for dt in (cutlass.Int4, cutlass.Int2):
        ok = is_valid_scale_granularity(SCALE_M, SCALE_K, dt, K, MMA_TILER_K)
        print(f"  {dt.__name__:5s} valid(m={SCALE_M},k={SCALE_K}) -> {ok}")
        assert ok is True, f"{dt} should accept the ConvertScale granularity"

    # Int2 must match Int4 across a sweep of (m,k) settings.
    for sm in (0, 1, 2):
        for sk in (0, 32, 64, 128, 100):  # 100 is not a multiple of 64 -> invalid
            v4 = is_valid_scale_granularity(sm, sk, cutlass.Int4, K, MMA_TILER_K)
            v2 = is_valid_scale_granularity(sm, sk, cutlass.Int2, K, MMA_TILER_K)
            assert v2 == v4, f"Int2/Int4 disagree at (m={sm},k={sk}): {v2} vs {v4}"
    print("  Int2 verdict matches Int4 across all sampled (m,k)")

    # Explicit regression guard: a known-bad granularity is rejected.
    bad = is_valid_scale_granularity(1, 100, cutlass.Int2, K, MMA_TILER_K)
    assert bad is False, "scale_k=100 (not %64) must be rejected for Int2"
    print("  bad granularity (k=100) correctly rejected for Int2")
    print("  PASSED\n")


def check_shuffle_excludes_int2():
    print("=== is_shuffle_a: Int2 must stay False (naive path) ===")
    # Same args that make Int4 shuffle True.
    s4 = is_shuffle_a("k", K, cutlass.Int4, cutlass.BFloat16, 128)
    s2 = is_shuffle_a("k", K, cutlass.Int2, cutlass.BFloat16, 128)
    print(f"  Int4 shuffle -> {s4}   Int2 shuffle -> {s2}")
    assert s4 is True, "sanity: Int4 should shuffle under these args"
    assert s2 is False, "Int2 must NOT shuffle (no int2 shuffle intrinsic)"
    print("  PASSED — Int2 takes generic .to(bf16) convert\n")


if __name__ == "__main__":
    check_scale_granularity()
    check_shuffle_excludes_int2()
    print(">>> task #21: mixed_input_helpers.py accepts Int2; shuffle excluded.")
