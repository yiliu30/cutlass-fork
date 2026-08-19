# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Production-facing MXFP4 prefill GEMM API for Blackwell.

This module keeps the device implementation in
``dense_blockscaled_gemm_persistent.py`` and adds the contract needed by a
real model integration:

* fixed MXFP4 (E2M1 x E2M1) with UE8M0 scales, vector size 32;
* K-major, already-packed operands;
* one-CTA clusters with the warp-specialized persistent kernel;
* compiled-kernel caching by shape and output dtype;
* explicit validation instead of an implicit PyTorch fallback.

The first version is intentionally prefill-oriented. Inputs must be padded to
the supported tile and K-block sizes; callers decide how to handle shapes that
do not satisfy the contract.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.runtime import make_ptr


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(current_dir, "..", "..", "..", "..")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

from cute.blackwell.kernel.blockscaled_gemm.dense_blockscaled_gemm_persistent import (
    Sm100BlockScaledPersistentDenseGemmKernel,
    create_and_reorder_scale_factor_tensor,
    scaled_mm,
)


MMA_TILE_MN = (128, 256)
CLUSTER_SHAPE_MN = (1, 1)
SF_VEC_SIZE = 32
AB_DTYPE = cutlass.Float4E2M1FN
SF_DTYPE = cutlass.Float8E8M0FNU
SUPPORTED_OUTPUT_DTYPES = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


@dataclass(frozen=True)
class Mxfp4PrefillShape:
    """Logical shape of a single prefill GEMM."""

    m: int
    n: int
    k: int

    @property
    def mnkl(self) -> Tuple[int, int, int, int]:
        return (self.m, self.n, self.k, 1)


def _torch_stream(stream: Optional[torch.cuda.Stream]) -> cuda.CUstream:
    torch_stream = stream if stream is not None else torch.cuda.current_stream()
    return cuda.CUstream(torch_stream.cuda_stream)


def _is_fp4_storage(tensor: torch.Tensor) -> bool:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return tensor.dtype in {torch.int8, torch.uint8} or (
        fp4_dtype is not None and tensor.dtype == fp4_dtype
    )


def _require_cuda_tensor(name: str, tensor: torch.Tensor, device: torch.device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _validate_shape(shape: Mxfp4PrefillShape) -> None:
    if min(shape.m, shape.n, shape.k) <= 0:
        raise ValueError(f"M, N, and K must be positive, got {shape}")
    if shape.m % MMA_TILE_MN[0] != 0:
        raise ValueError(f"M must be divisible by 128, got {shape.m}")
    if shape.n % MMA_TILE_MN[1] != 0:
        raise ValueError(f"N must be divisible by 256, got {shape.n}")
    if shape.k % 256 != 0:
        raise ValueError(f"K must be divisible by 256, got {shape.k}")


def _validate_inputs(
    shape: Mxfp4PrefillShape,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out: Optional[torch.Tensor],
    output_dtype: torch.dtype,
) -> torch.device:
    _validate_shape(shape)
    if not torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    device = a.device
    if device.type != "cuda":
        raise ValueError("A must be a CUDA tensor")
    if device.index != torch.cuda.current_device():
        raise ValueError(
            "the input device must be the current CUDA device; "
            f"current={torch.cuda.current_device()} input={device.index}"
        )
    for name, tensor in (
        ("B", b),
        ("a_scales", a_scales),
        ("b_scales", b_scales),
    ):
        _require_cuda_tensor(name, tensor, device)

    if not _is_fp4_storage(a) or not _is_fp4_storage(b):
        raise TypeError("A and B must contain prepacked Float4E2M1FN storage")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be two-dimensional K-major tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous packed storage")
    # A packed byte contains two FP4 values. Some CuTe conversion paths retain
    # one byte per logical value, so accept either representation as long as the
    # backing allocation is large enough for the logical K dimension.
    minimum_packed_bytes = shape.m * shape.k // 2
    if a.numel() < minimum_packed_bytes or b.numel() < shape.n * shape.k // 2:
        raise ValueError("A/B packed storage is smaller than the logical FP4 matrix")
    if a.data_ptr() % 16 or b.data_ptr() % 16:
        raise ValueError("A and B must be 16-byte aligned")

    expected_scale_elements_a = shape.m * (shape.k // SF_VEC_SIZE)
    expected_scale_elements_b = shape.n * (shape.k // SF_VEC_SIZE)
    expected_sf_dtype = cutlass_torch.dtype(SF_DTYPE)
    for name, tensor, expected in (
        ("a_scales", a_scales, expected_scale_elements_a),
        ("b_scales", b_scales, expected_scale_elements_b),
    ):
        if tensor.dtype != expected_sf_dtype:
            raise TypeError(f"{name} must have dtype {expected_sf_dtype}")
        if tensor.numel() != expected:
            raise ValueError(
                f"{name} must contain {expected} packed UE8M0 values, got {tensor.numel()}"
            )
        if tensor.data_ptr() % 32:
            raise ValueError(f"{name} must be 32-byte aligned")

    if output_dtype not in SUPPORTED_OUTPUT_DTYPES:
        raise TypeError("output_dtype must be torch.float16 or torch.bfloat16")
    if out is not None:
        _require_cuda_tensor("out", out, device)
        if out.dtype != output_dtype or out.shape != (shape.m, shape.n):
            raise ValueError(
                f"out must have shape {(shape.m, shape.n)} and dtype {output_dtype}"
            )
        if not out.is_contiguous() or out.data_ptr() % 16:
            raise ValueError("out must be contiguous and 16-byte aligned")
    return device


def pack_mxfp4_scales(scales: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Pack logical ``(rows, ceil(K/32))`` UE8M0 scales for the MMA kernel.

    This helper is intended for one-time model setup, not the GEMM hot path.
    """

    if not isinstance(scales, torch.Tensor) or scales.ndim != 2:
        raise TypeError("scales must be a two-dimensional torch.Tensor")
    expected_shape = (rows, (k + SF_VEC_SIZE - 1) // SF_VEC_SIZE)
    if scales.shape != expected_shape:
        raise ValueError(f"scales must have shape {expected_shape}, got {scales.shape}")
    if scales.dtype != cutlass_torch.dtype(SF_DTYPE):
        raise TypeError(f"scales must have dtype {cutlass_torch.dtype(SF_DTYPE)}")
    # The layout-conversion helper performs a CuTe JIT copy from its source
    # pointer; keep that source in host memory, matching the repository's
    # existing scale-preparation path.
    logical = scales.detach().to(device="cpu").contiguous().unsqueeze(-1)
    return create_and_reorder_scale_factor_tensor(
        1, rows, k, SF_VEC_SIZE, SF_DTYPE, logical
    )


def pack_mxfp4_values(values: torch.Tensor) -> torch.Tensor:
    """Convert a logical FP32/FP16 matrix to prepacked Float4E2M1FN storage."""

    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise TypeError("values must be a two-dimensional torch.Tensor")
    if values.device.type != "cuda":
        raise ValueError("values must be a CUDA tensor")
    if values.shape[1] % 32 != 0:
        raise ValueError("the K dimension must be divisible by 32")
    cute_tensor, packed = cutlass_torch.cute_tensor_like(
        values.detach().cpu(),
        AB_DTYPE,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    # Keep the CuTe tensor alive while conversion runs; the returned Torch
    # tensor owns the device allocation used by the production API.
    del cute_tensor
    return packed


class Mxfp4PrefillGemm:
    """Cached production MXFP4 GEMM with a one-CTA cluster."""

    def __init__(
        self,
        output_dtype: torch.dtype = torch.float16,
        max_cached_variants: int = 16,
    ):
        if output_dtype not in SUPPORTED_OUTPUT_DTYPES:
            raise TypeError("output_dtype must be torch.float16 or torch.bfloat16")
        if max_cached_variants <= 0:
            raise ValueError("max_cached_variants must be positive")
        self.output_dtype = output_dtype
        self.max_cached_variants = max_cached_variants
        self._compiled = OrderedDict()

    def _get_compiled(
        self, shape: Mxfp4PrefillShape, stream: cuda.CUstream
    ):
        key = (torch.cuda.current_device(), shape.m, shape.n, shape.k, self.output_dtype)
        compiled = self._compiled.get(key)
        if compiled is not None:
            self._compiled.move_to_end(key)
            return compiled

        gemm = Sm100BlockScaledPersistentDenseGemmKernel(
            SF_VEC_SIZE, MMA_TILE_MN, CLUSTER_SHAPE_MN
        )
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)
        compiled = scaled_mm(
            gemm,
            AB_DTYPE,
            AB_DTYPE,
            SUPPORTED_OUTPUT_DTYPES[self.output_dtype],
            SF_DTYPE,
            "k",
            "k",
            "n",
            max_active_clusters,
            stream,
        )
        self._compiled[key] = compiled
        self._compiled.move_to_end(key)
        while len(self._compiled) > self.max_cached_variants:
            self._compiled.popitem(last=False)
        return compiled

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        shape: Tuple[int, int, int],
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        problem = Mxfp4PrefillShape(*shape)
        device = _validate_inputs(
            problem, a, b, a_scales, b_scales, out, self.output_dtype
        )
        if stream is not None and stream.device != device:
            raise ValueError(f"stream must be on {device}, got {stream.device}")
        output = out if out is not None else torch.empty(
            (problem.m, problem.n), dtype=self.output_dtype, device=device
        )
        cu_stream = _torch_stream(stream)
        compiled = self._get_compiled(problem, cu_stream)
        a_ptr = make_ptr(AB_DTYPE, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        b_ptr = make_ptr(AB_DTYPE, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        sfa_ptr = make_ptr(
            SF_DTYPE, a_scales.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
        )
        sfb_ptr = make_ptr(
            SF_DTYPE, b_scales.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
        )
        c_ptr = make_ptr(
            SUPPORTED_OUTPUT_DTYPES[self.output_dtype],
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        compiled(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            c_ptr,
            problem.mnkl,
            cu_stream,
        )
        return output


def _parse_shape(value: str) -> Tuple[int, int, int]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected M,N,K") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected exactly M,N,K")
    return result


def _time_us(fn, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnk", type=_parse_shape, default=(512, 512, 256))
    parser.add_argument("--output_dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup_iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--parity_tolerance",
        type=float,
        default=0.05,
        help="Allowed slowdown versus BF16 torch.mm (default: 5%%)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shape = Mxfp4PrefillShape(*args.mnk)
    _validate_shape(shape)
    m, n, k = args.mnk
    values_a = torch.randint(-2, 2, (m, k), device="cuda", dtype=torch.float32)
    values_b = torch.randint(-2, 2, (n, k), device="cuda", dtype=torch.float32)
    a = pack_mxfp4_values(values_a)
    b = pack_mxfp4_values(values_b)
    sf_dtype = cutlass_torch.dtype(SF_DTYPE)
    logical_a_scales = torch.ones((m, k // SF_VEC_SIZE), device="cuda", dtype=sf_dtype)
    logical_b_scales = torch.ones((n, k // SF_VEC_SIZE), device="cuda", dtype=sf_dtype)
    a_scales = pack_mxfp4_scales(logical_a_scales, m, k)
    b_scales = pack_mxfp4_scales(logical_b_scales, n, k)
    output_dtype = torch.bfloat16 if args.output_dtype == "bf16" else torch.float16
    gemm = Mxfp4PrefillGemm(output_dtype)
    output = gemm(a, b, a_scales, b_scales, args.mnk)
    torch.cuda.synchronize()
    print(f"production MXFP4 prefill: PASS shape={args.mnk} output={output.dtype}")
    if args.benchmark:
        output = torch.empty_like(output)
        bf16_a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        bf16_b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        bf16_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        mxfp4_us = _time_us(
            lambda: gemm(a, b, a_scales, b_scales, args.mnk, out=output),
            args.warmup_iterations,
            args.iterations,
        )
        bf16_us = _time_us(
            lambda: torch.mm(bf16_a, bf16_b, out=bf16_output),
            args.warmup_iterations,
            args.iterations,
        )
        flops = 2.0 * m * n * k
        print(
            f"production MXFP4: {mxfp4_us:.2f} us, "
            f"{flops / mxfp4_us / 1.0e6:.1f} TFLOP/s"
        )
        print(
            f"BF16 torch.mm: {bf16_us:.2f} us, "
            f"{flops / bf16_us / 1.0e6:.1f} TFLOP/s"
        )
        speedup = bf16_us / mxfp4_us
        print(f"production speedup vs BF16: {speedup:.3f}x")
        if mxfp4_us > bf16_us * (1.0 + args.parity_tolerance):
            raise RuntimeError(
                "production MXFP4 exceeded the BF16 torch.mm parity margin "
                f"({args.parity_tolerance:.1%})"
            )
        print(f"production parity: PASS (margin {args.parity_tolerance:.1%})")


if __name__ == "__main__":
    main()
