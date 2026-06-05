# int2 × bf16 mixed-input GEMM — root-cause of the compile segfault

**Date:** 2026-06-05
**Status:** Int2-typed-A-through-TMA approach is **blocked by a hardware/driver limitation.**
**Where it breaks:** MLIR→PTX lowering pass (`base_dsl/compiler.py:148 pm.run()`), *after* a
clean trace and full IR generation. Segfault (exit 139), no Python-level diagnostic.

---

## TL;DR

The fused warp-specialized GEMM moves operand A from global → shared memory with a **TMA
(Tensor Memory Accelerator) descriptor**. TMA descriptors are built from a fixed enum of
data formats baked into the CUDA driver. **The smallest sub-byte format that enum supports is
4-bit. There is no 2-bit TMA format anywhere — not in the DSL, not in the driver, not in C++
CUTLASS.** So int2's A operand silently falls back to a 32-bit (`U32`) TMA format, which then
contradicts the 2-bit smem layout / swizzle / copy atoms the rest of the kernel built. A
downstream lowering pass trips over that contradiction and dereferences null.

int4 works because 4-bit *does* have a native TMA format (`U4`).

---

## Evidence chain

### 1. The two operands that differ between int4 (works) and int2 (crashes)

Captured the full generated IR for both via `CUTE_DSL_KEEP_IR=1` (int4 → exit 0, 662 KB;
int2 → segfault, but IR file fully written, 630 KB). Only the A operand differs:

| A-operand construct | int4 (works)                     | int2 (crashes)                   |
|---------------------|----------------------------------|----------------------------------|
| TMA load atom       | `tma_format = U4`, copy_bits 65536 | `tma_format = U8`, copy_bits 32768 |
| A smem swizzle      | `S<2,4,3>` (64-byte), 8 stages   | `S<1,4,3>` (32-byte), 10 stages  |

The int2 IR is otherwise a clean i4→i2 substitution — it *generates* fine. The break is purely
that the TMA descriptor format is wrong for 2-bit.

### 2. The DSL has no 2-bit TMA format

Enumerated `cute._mlir.dialects.cute_nvgpu.TmaDataFormat`:

```
U4, U4_UNPACK_U8, U6_UNPACK_U8, U8, U16, U32, U64, S32, S64,
BF16_RN, F16_RN, F32_RN, F64_RN, TF32_RN, ...
has any U2* member: False
has U4* member:     True
```

`get_default_tma_format()` for each width:

```
i2 unpack=False -> U32        <-- no 2-bit format; falls back to 32-bit word
i2 unpack=True  -> U32
i4 unpack=False -> U4         <-- native 4-bit
i4 unpack=True  -> U4_UNPACK_U8
i8 unpack=False -> U8
```

### 3. The CUDA driver enum itself bottoms out at 4-bit

`include/cute/arch/copy_sm90_desc.hpp:206 to_CUtensorMapDataType<T>()` — the full set of
sub-byte tensor-map formats the driver exposes:

```cpp
float_e2m1_t (fp4)  -> CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B    // 4-bit
float_e2m3_t (fp6)  -> CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B   // 6-bit
type_erased_float4  -> CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B    // 4-bit
```

There is **no `*_2_*` / 2-bit `CUtensorMapDataType`**. This is the CUDA driver's tensor-map
API, i.e. a hardware/driver boundary — not something the DSL can patch around.

### 4. C++ CUTLASS confirms the same gap

`int2b_t` exists as a *compute* type (`include/cutlass/integer_subbyte.h:210
using int2b_t = integer_subbyte<2, true>`), but it has **no entry** in
`to_CUtensorMapDataType()`. So even hand-written C++ CUTLASS cannot build a TMA descriptor
for int2 — confirming this is a known boundary, not a Python-layer omission.

---

## Why it's a segfault and not a clean error

`compiler.py:148` runs the lowering pipeline with `enable_verifier(False)`. The malformed
(format-vs-layout-mismatched) IR is never verified, so instead of a "format X incompatible
with layout Y" diagnostic, a C++ pass walks a structure it assumes is well-formed and hits a
null pointer. Turning the verifier on would convert it to a named error, but the underlying
gap is the same.

---

## What this means for the goal (int2 A × bf16 B GEMM)

The MMA itself is **bf16 × bf16** — A is up-converted to bf16 *before* the math. Nothing in the
compute path needs a native int2 MMA. The *only* blocked link is moving the packed int2 bytes
gmem→smem via TMA. So the goal is still reachable; we just can't carry a literally-typed `Int2`
tensor through the TMA-based collective.

### Option 1 — Two kernels: dequant (int2→bf16) then a normal bf16×bf16 GEMM  [RECOMMENDED]
- Reuses the **already-built and verified** int2→bf16 convert kernel (Phase A standalone) plus
  the fully-working bf16 GEMM. No hardware blocker.
- Cost: one extra global-memory round-trip for the dequantized A (A is materialized as bf16 in
  gmem between the two kernels), so the weight-compression benefit is only on storage, not on
  the GEMM's own bandwidth.
- Effort: small — wire two existing, trusted pieces together.

### Option 2 — Fused, with int2 transported as *bytes*
- Declare A to the TMA/smem path as a **byte type** (4 packed int2 per byte → TMA uses the
  existing `U8` format), then unpack 4 int2→bf16 inside the convert step (extend the proven
  `cvt_i2_bf16` from a 2×-style to a 4×-style unpack, mirroring how int4's `use_unpack` does 2×).
- Keeps weights compressed all the way into smem (best bandwidth), single kernel.
- Effort: substantial — transport-type ≠ logical-type plumbing through smem-layout sizing, the
  gmem→smem tile accounting, and the K-dimension bookkeeping. Real re-architecture, with risk.

### Option 3 — Stop at the root-cause finding
- Document that fused int2 mixed-input GEMM is blocked by the missing 2-bit TMA format; ship the
  standalone int2→bf16 convert as the int2 deliverable.

---

## Reproduce

```bash
cd examples/python/CuTeDSL/blackwell/mixed_input_gemm
# int4 baseline — PASSES:
python mixed_input_gemm.py --a_dtype Int4 --b_dtype BFloat16 \
  --scale_granularity_m 1 --scale_granularity_k 128 --c_dtype BFloat16 \
  --acc_dtype Float32 --mma_tiler_mnk 256,128,128 --cluster_shape_mn 2,1 \
  --use_2cta_instrs --use_tma_store --a_major k --mnkl 1024,8192,6144,1 --tolerance 0.1
# int2 — only --a_dtype changes — SEGFAULTS (exit 139):
python mixed_input_gemm.py --a_dtype Int2 ... (same flags)
# Capture the IR that proves it:
CUTE_DSL_KEEP_IR=1 CUTE_DSL_DUMP_DIR=/tmp/int2_keepir python mixed_input_gemm.py --a_dtype Int2 ...
```

Probe scripts: `/tmp/int2_tma_format_probe.py`, `/tmp/int2_tma_enum.py`.
