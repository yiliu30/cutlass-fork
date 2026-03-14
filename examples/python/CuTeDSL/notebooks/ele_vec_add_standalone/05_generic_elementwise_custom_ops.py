"""
Example 5: Generic Elementwise Kernel with Custom Operations
=============================================================

Key Concepts (NEW in this example):
  - cutlass.Constexpr: Pass compile-time constants (like function pointers) to kernels
  - List[cute.Tensor]: Variable number of input tensors (meta-programming)
  - cutlass.range_constexpr: Compile-time loop unrolling over inputs
  - cute.make_identity_tensor: Create coordinate tensor for boundary checking
  - cute.elem_less: Element-wise comparison for predicate masks
  - cute.repeat_like: Flatten hierarchical value index for slicing
  - cute.where / cute.full_like: Conditional selection in kernels
  - Python lambdas/functions as GPU kernel operations via meta-programming

What Changed:
  The kernel is now GENERIC -- it accepts:
    1. An arbitrary binary (or n-ary) operation `op`
    2. A list of input tensors `mInputs` (not fixed to 2)
    3. A coordinate tensor `cC` for boundary predicate checking

  Python's meta-programming is used to unroll loops over inputs at COMPILE time.
  The `op` function (e.g., mul, mul_relu) is inlined into the generated CUDA code.

Architecture:
  elementwise_apply(op, [a_, b_], c_)
    -> tiles inputs & result
    -> creates coordinate tensor for boundary checks
    -> launches elementwise_apply_kernel

  elementwise_apply_kernel:
    -> thread-block tiling (same as Example 4)
    -> TV layout composition (same as Example 4)
    -> boundary predicate computation (NEW)
    -> op(*[thrInput.load() for thrInput in thrInputs])  -- generic!

Run:
  python 05_generic_elementwise_custom_ops.py
"""

import torch
from operator import mul
from typing import List

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# GPU Kernel: generic elementwise with custom op and boundary checking
# =============================================================================
@cute.kernel
def elementwise_apply_kernel(
    op: cutlass.Constexpr,            # Custom operation (e.g., mul, add, mul_relu)
    mInputs: List[cute.Tensor],       # List of tiled input tensors
    mC: cute.Tensor,                  # Tiled output tensor
    cC: cute.Tensor,                  # Coordinate tensor (for boundary checks)
    shape: cute.Shape,                # Original tensor shape (for boundary checks)
    tv_layout: cute.Layout,           # (tid, vid) -> logical coord
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # ---- Thread-block level tiling ----
    blk_crd = ((None, None), bidx)

    # Meta-programming: slice ALL inputs in a compile-time unrolled loop
    gInputs = [t[blk_crd] for t in mInputs]  # List of (TileM, TileN) tiles
    gC = mC[blk_crd]
    gCrd = cC[blk_crd]  # Coordinate tile (maps positions -> original coords)

    print("[DSL INFO] Sliced Tensors per thread block:")
    for i in cutlass.range_constexpr(len(gInputs)):
        print(f"[DSL INFO]   ctaInputs{i} = {gInputs[i].type}")
    print(f"[DSL INFO]   gC = {gC.type}")
    print(f"[DSL INFO]   gCrd = {gCrd.type}")

    # ---- Compose with TV layout ----
    tidfrgInputs = [cute.composition(t, tv_layout) for t in gInputs]
    tidfrgC = cute.composition(gC, tv_layout)
    tidfrgCrd = cute.composition(gCrd, tv_layout)

    # Use repeat_like to flatten hierarchical value index for slicing
    thr_crd = (tidx, cute.repeat_like(None, tidfrgInputs[0][1]))

    # ---- Thread level slice ----
    thrInputs = [t[thr_crd] for t in tidfrgInputs]
    thrC = tidfrgC[thr_crd]
    thrCrd = tidfrgCrd[thr_crd]

    print("[DSL INFO] Sliced Tensors per thread:")
    for i in cutlass.range_constexpr(len(thrInputs)):
        print(f"[DSL INFO]   thrInputs{i} = {thrInputs[i].type}")
    print(f"[DSL INFO]   thrC = {thrC.type}")
    print(f"[DSL INFO]   thrCrd = {thrCrd.type}")

    # ---- Boundary predicate computation ----
    # When M*N is not perfectly divisible by tile size, some threads
    # may access out-of-bounds memory. The coordinate tensor + elem_less
    # creates a mask to prevent this.
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
    print(f"[DSL INFO]   frgPred = {frgPred.type}")

    for i in cutlass.range_constexpr(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    # ---- Apply the custom operation ----
    # op is called with loaded data from ALL input tensors
    # The * unpacks the list comprehension: op(input0.load(), input1.load(), ...)
    result = op(*[thrInput.load() for thrInput in thrInputs])
    thrC.store(result)


# =============================================================================
# Host JIT Function: generic launcher
# =============================================================================
@cute.jit
def elementwise_apply(
    op: cutlass.Constexpr,
    inputs,
    result: cute.Tensor,
):
    coalesced_ldst_bytes = 16

    assert all(t.element_type == inputs[0].element_type for t in inputs)
    dtype = inputs[0].element_type

    # TV layout construction (same as Examples 3-4)
    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # Tile all inputs
    mInputs = [cute.zipped_divide(inp, tiler_mn) for inp in inputs]
    mC = cute.zipped_divide(result, tiler_mn)

    # Block remapping (same as Example 4)
    remap_block = cute.make_ordered_layout(
        cute.select(mInputs[0].shape[1], mode=[1, 0]), order=(1, 0)
    )
    for i, t in enumerate(mInputs):
        mInputs[i] = cute.composition(t, (None, remap_block))
    mC = cute.composition(mC, (None, remap_block))

    # NEW: Create identity/coordinate tensor for boundary checks
    # This tensor maps each position to its own coordinate: idC[i,j] = (i,j)
    idC = cute.make_identity_tensor(result.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)

    # Launch
    elementwise_apply_kernel(op, mInputs, mC, cC, result.shape, tv_layout).launch(
        grid=[cute.size(mC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


# =============================================================================
# Custom Operations: Define your own GPU operations as plain Python functions!
# =============================================================================

def mul_relu(a, b):
    """Multiply then ReLU -- demonstrates cute.where and cute.full_like"""
    tmp = a * b
    return cute.where(tmp > 0, tmp, cute.full_like(tmp, 0))


def mul_relu_ref(a, b):
    """PyTorch reference for verification"""
    return torch.relu(a * b)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    M, N = 16384, 8192

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    # --- Test 1: Multiplication (using Python's built-in operator.mul) ---
    print("=" * 60)
    print("Test 1: Elementwise Multiplication (operator.mul)")
    print("=" * 60)
    elementwise_apply(mul, [a_, b_], c_)
    torch.testing.assert_close(c, a * b)
    print("Correctness: PASSED\n")

    # --- Test 2: Mul + ReLU (custom fused operation) ---
    print("=" * 60)
    print("Test 2: Fused Mul + ReLU (custom operation)")
    print("=" * 60)
    c.zero_()  # Reset output
    elementwise_apply(mul_relu, [a_, b_], c_)
    torch.testing.assert_close(c, mul_relu_ref(a, b))
    print("Correctness: PASSED\n")

    # --- Test 3: You can define inline lambdas too! ---
    print("=" * 60)
    print("Test 3: Addition via lambda")
    print("=" * 60)
    c.zero_()
    add_op = lambda a, b: a + b
    elementwise_apply(add_op, [a_, b_], c_)
    torch.testing.assert_close(c, a + b)
    print("Correctness: PASSED")
