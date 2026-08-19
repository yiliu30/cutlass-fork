# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the production-facing one-CTA MXFP4 prefill API."""

import pytest
import torch

import cutlass
import cutlass.torch as cutlass_torch
from blackwell.kernel.blockscaled_gemm.dense_blockscaled_gemm_production import (
    Mxfp4PrefillGemm,
    pack_mxfp4_scales,
    pack_mxfp4_values,
)


pytestmark = [pytest.mark.arch(["100a"])]


@pytest.mark.L0
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
def test_mxfp4_prefill_production_correctness(output_dtype):
    m, n, k = 128, 256, 256
    a_ref = torch.randint(-2, 2, (m, k), device="cuda", dtype=torch.float32)
    b_ref = torch.randint(-2, 2, (n, k), device="cuda", dtype=torch.float32)
    a = pack_mxfp4_values(a_ref)
    b = pack_mxfp4_values(b_ref)

    sf_dtype = cutlass_torch.dtype(cutlass.Float8E8M0FNU)
    a_logical_scales = torch.full(
        (m, k // 32), 2.0, device="cuda", dtype=sf_dtype
    )
    b_logical_scales = torch.ones((n, k // 32), device="cuda", dtype=sf_dtype)
    a_scales = pack_mxfp4_scales(a_logical_scales, m, k)
    b_scales = pack_mxfp4_scales(b_logical_scales, n, k)

    gemm = Mxfp4PrefillGemm(output_dtype)
    out = torch.empty((m, n), device="cuda", dtype=output_dtype)
    result = gemm(a, b, a_scales, b_scales, (m, n, k), out=out)
    torch.cuda.synchronize()

    expected = ((a_ref * 2.0) @ b_ref.transpose(0, 1)).to(output_dtype)
    torch.testing.assert_close(result, expected, atol=0.1, rtol=0.01)
    assert result.data_ptr() == out.data_ptr()


@pytest.mark.L0
def test_mxfp4_prefill_production_rejects_unpadded_shape():
    gemm = Mxfp4PrefillGemm()
    with pytest.raises(ValueError, match="N must be divisible by 256"):
        gemm(
            torch.empty((128, 128), device="cuda", dtype=torch.uint8),
            torch.empty((128, 128), device="cuda", dtype=torch.uint8),
            torch.empty((128, 8), device="cuda", dtype=cutlass_torch.dtype(cutlass.Float8E8M0FNU)),
            torch.empty((128, 8), device="cuda", dtype=cutlass_torch.dtype(cutlass.Float8E8M0FNU)),
            (128, 128, 256),
        )
