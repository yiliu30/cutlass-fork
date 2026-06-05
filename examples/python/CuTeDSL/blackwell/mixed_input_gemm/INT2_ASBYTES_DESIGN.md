# int2-as-bytes fused GEMM — implementation design

**Date:** 2026-06-05
**Status:** design locked, implementing.
**Goal:** int2 (2-bit signed) A × bf16 B mixed-input GEMM, single fused kernel, weights stay
compressed all the way into smem. Chosen over the two-kernel dequant path (Option 1) by the user.

---

## Why the naive int2 path segfaults (recap, now refined to TWO contradictions)

Running `--a_dtype Int2` segfaults (exit 139) inside MLIR→PTX lowering. The IR (captured via
`CUTE_DSL_KEEP_IR=1`) shows the A operand carries two mutually-inconsistent facts because there is
**no 2-bit TMA descriptor format** (the CUDA driver enum bottoms out at 4-bit):

- **Contradiction #1 — swizzle (FIXED).** `get_smem_layout_atom_ab` computes the A smem swizzle in
  *element* units. int2 (256-bit major) selected `K_SW32` (`S<1,4,3>`), a 2-bit-granular byte
  reorder that contradicts byte transport. Patched: force `K_INTER` (non-swizzled) for width-2.
  Confirmed in IR — raw A smem lost its `S<>` prefix, `copy_bits` 32768→16384. **Insufficient
  alone** — still segfaults.
- **Contradiction #2 — TMA descriptor type (THIS DESIGN).** The A TMA atom is still
  `<i2, tma_gbasis=(64,128,1) [i2 units], tma_format=U8>`: a byte-format transport against an
  i2-dimensioned address basis. The descriptor's stride units are tied to the element type, so the
  only fix is to present A to the TMA path as a genuine **byte type**.

## The MMA never sees raw A — this localizes the fix

`cute.gemm` (line 1470) consumes `tCrA`, the *transformed* bf16 fragment (TMEM/smem). Raw int2
only lives on the transport path: gmem → TMA → smem → reg fragment → `cvt_tensor_a`. So the entire
re-architecture is confined to A's transport; B, scale, accumulator, MMA loop are untouched.

---

## The storage/logical split

| world   | dtype | per-tile K            | drives                                                          |
|---------|-------|-----------------------|----------------------------------------------------------------|
| storage | Int8  | `mma_tiler_K // 4` =32 | A gmem recast, TMA atom, raw smem layout, raw copy, `tArA` alloc |
| logical | Int2  | `mma_tiler_K` =128    | recast target, `cvt_tensor_a`, scale, host reference            |

They meet at **one recast** `Int8(32) → Int2(128)` between the raw copy and the convert.

Because 1 byte = 4 int2, A's per-tile extent shrinks 4× in lockstep with its element count, so
`k_tile_cnt = cute.size(gA_mkl, mode=[3])` comes out identical to B's `K_total/128`. The pipeline
(load2trans / trans2mma stages, all the `< k_tile_cnt` guards) needs **no change**.

---

## Seams (file : line — change)

All edits in the **installed wheel** first (run+verify), then mirror validated edits to the repo
example files. Wheel root:
`/home/yiliu7/workspace/venvs/cutlass-dsl/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/`

### A. host — present A as Int8 bytes with storage-K
- `mixed_input_host_utils.py::create_i2_tensor_and_scale` (~243): today sets
  `cute_a_quant_tensor.element_type = cutlass.Int2`. Keep an Int2 logical tensor for the reference,
  but hand the kernel an **Int8** view with shape `(M, K//4)` (4 packed int2 per byte). The kernel's
  `a.element_type` then becomes `Int8` → all width-parametric transport plumbing follows for free.

### B. kernel — a storage/logical dtype pair
- `mixed_input_gemm.py:444` `self.a_dtype = a.element_type` → now `Int8` (storage).
  Add `self.a_logical_dtype = cutlass.Int2` (drives convert + scale).
- `self.mma_tiler_a` = `(M, N, K//4)` — a **byte tiler** used ONLY on the A transport seams below.

### C. TMA atom + smem (contradiction #2 fix — byte-typed gbasis)
- `:496` `make_tiled_tma_atom_A(a_op, a, smem_layout_a_per_stage, self.mma_tiler, ...)` — pass the
  Int8 `a` and `self.mma_tiler_a` so the gbasis is byte-dimensioned, format U8. Consistent.
- `:564` `compute_smem_layout(..., self.a_dtype=Int8, ...)` — raw smem becomes Int8 K=32. The
  width-2 swizzle patch in `blackwell_helpers.py` becomes a no-op (Int8 swizzles normally) — keep
  it anyway as a correct guard, or revert; decide after it runs.

### D. A gmem tiling + MMA partition (the risky seam)
- `:877` `gA_mkl = cute.local_tile(mA_mkl, slice_(self.mma_tiler_a, ...))` — byte tiler → gA K=32.
- `:901` `tCgA = thr_mma.partition_A(gA_mkl)` — `thr_mma` is the bf16 MMA (K=128); partitioning a
  K=32 byte tensor against it mismatches. Options, in order of preference:
  1. Build a **byte tiled_mma_a** (same atom, A operand Int8) purely for `partition_A`/coordinate
     generation, leaving the real `tiled_mma` for the math. Lowest semantic risk.
  2. Manually construct the A TMA partition without `partition_A` (replicate what `partition_A`
     does at byte granularity).

### E. convert seam — recast + 4× shape expand
- `:1188` `tArA = make_rmem_tensor(tAsA_input[...].shape, Int8)` — raw fragment, 32 Int8.
- `:1191` `tArA_transform = make_rmem_tensor(<logical 128 shape>, mma_dtype)` — **decouple** from
  `tAsA_input` shape; must be 128 bf16, not 32.
- `:1333` after `autovec_copy` into `tArA_load`: insert `tArA_i2 = cute.recast_tensor(tArA_load, cutlass.Int2)`
  → 4× expand 32→128.
- `:1356` `cvt_tensor_a(tArA_i2[(None,idx)], self.mma_dtype, self.shuffle_a)` — feed the Int2 view.
  (`cvt_tensor_a` + `is_shuffle_a` already wired for Int2 in prior session.)
- Scale (`tSrS_load`, `:1362`) stays at logical-K=128 — already correct, it was built from logical A.

### F. validation guards already wired (prior session)
- `mixed_input_helpers.py::is_valid_scale_granularity` width==2 branch (663). OK.
- `is_shuffle_a` returns False for int2 (no shuffle). OK.

---

## Incremental milestones (each commit self-contained + testable)

1. **Compiles (IR lowers, no segfault).** Get the byte-typed TMA atom + byte smem + byte tiler so
   the A operand IR is self-consistent (`<i8, gbasis [byte units], U8>`). Success = exit 0 on
   `cute.compile` even if numerics are wrong / the recast not yet wired. Verifies contradiction #2
   is actually resolved.
2. **Correct.** Wire the recast + tArA_transform shape decouple; run end-to-end vs torch reference
   (`assert_close`, bf16 tolerance). Edge cases: all four int2 codes present, scale≠1, K%128==0.
3. **Mirror + commit.** Copy validated wheel edits into repo example files; commit + push to fork
   `yi/study/cute`. (Wheel-only files like blackwell_helpers.py have no repo mirror — note in commit.)

## Tradeoffs / risk
- Seam **D** (byte tiled_mma for partition_A) is the single biggest unknown — if `partition_A`
  can't be cleanly reused at byte granularity, fall to manual partition (D.2), more code.
- Benefit vs Option 1 (two-kernel dequant): A stays 4× compressed through smem (bandwidth win), one
  kernel. Cost: this plumbing + risk. User accepted.
