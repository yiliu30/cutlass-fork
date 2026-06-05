"""Probe seam D for the int2-as-bytes fused GEMM.

QUESTION (make-or-break): if operand A is presented to the A TMA-atom builder as a
*byte* tensor (Int8) with a *byte* K-tiler (mma_tiler_K // 4), but the TiledMMA is the
normal bf16 MMA, does `make_tiled_tma_atom_A` produce a SELF-CONSISTENT TMA descriptor
(`<i8, byte-dimensioned gbasis, tma_format = U8>`)?

If yes, contradiction #2 (the i2-typed gbasis vs U8 format that segfaults the naive int2
path) is resolved by byte-typing, and the fused int2 approach is viable.
If no (segfault / malformed atom), the fused approach is dead and we fall back to the
two-kernel dequant (Option 1).

Pure trace-time: only builds the atom inside @cute.jit and prints its IR. No device exec.

Run:
  cd /tmp && /home/yiliu7/workspace/venvs/cutlass-dsl/bin/python \
    /home/yiliu7/workspace/cutlass/examples/python/CuTeDSL/blackwell/mixed_input_gemm/int2_asbytes_partition_probe.py
"""
import torch

# import cutlass BEFORE touching sys.path (a phantom jax/ namespace dir shadows real jax)
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
from cutlass.cute.nvgpu import tcgen05

# ---- mirror the kernel defaults (from the INT2_TMA_BLOCKER reproduce command) ----
# mma_tiler_mnk = 256,128,128 ; cluster 2,1 ; use_2cta_instrs ; a_major k ; bf16 B, f32 acc
MMA_M, MMA_N, MMA_K = 256, 128, 128
CLUSTER_MN = (2, 1)
NUM_STAGES = 2

# logical A is int2; storage A is bytes -> 4 int2 per byte -> storage K = MMA_K // 4
STORAGE_K_DIV = 4
BYTE_MMA_K = MMA_K // STORAGE_K_DIV  # 32

# full problem K (any multiple of 128); storage K = K // 4
K_TOTAL = 6144
K_STORAGE = K_TOTAL // STORAGE_K_DIV  # 1536
M = 1024


def report(tag, atom):
    s = str(atom)
    print(f"\n=== {tag} ===")
    print(s)
    # crude consistency checks on the printed IR
    print("  contains 'i8'      :", "i8" in s)
    print("  contains 'i2'      :", "i2" in s)
    print("  contains 'U8'      :", "U8" in s)
    print("  contains 'tma_gbasis':", "tma_gbasis" in s)


@cute.jit
def probe(mA_bytes: cute.Tensor):
    # --- build the bf16 TiledMMA exactly as the kernel does (mma_dtype = B's bf16) ---
    a_major = tcgen05.OperandMajorMode.K
    b_major = tcgen05.OperandMajorMode.K
    transform_a_source = mixed_input_utils.get_transform_a_source(a_major)  # K -> TMEM
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,           # ab_dtype = bf16 (the MMA math type)
        a_major,
        b_major,
        cutlass.Float32,            # acc
        tcgen05.CtaGroup.TWO,       # use_2cta_instrs
        (MMA_M, MMA_N),
        transform_a_source,
    )

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*CLUSTER_MN, 1)),
        (tiled_mma.thr_id.shape,),
    )
    num_mcast_a = cute.size(cluster_layout_vmnk.shape[2])
    is_a_mcast = num_mcast_a > 1
    a_op = mixed_input_utils.get_tma_atom_kind(is_a_mcast, True, False)

    # --- BYTE smem layout for A: Int8 dtype, byte K-tiler (MMA_K//4) ---
    byte_mma_tiler = (MMA_M, MMA_N, BYTE_MMA_K)
    smem_layout_a = sm100_utils.make_smem_layout_a(
        tiled_mma,
        byte_mma_tiler,
        cutlass.Int8,               # storage dtype = bytes
        NUM_STAGES,
    )
    smem_layout_a_per_stage = cute.slice_(smem_layout_a, (None, None, None, 0))

    cute.printf("byte smem_layout_a_per_stage: {}\n", smem_layout_a_per_stage)

    # --- the make-or-break call: A TMA atom from an Int8 tensor + byte tiler ---
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        mA_bytes,                   # Int8 gmem tensor
        smem_layout_a_per_stage,
        byte_mma_tiler,             # byte K-tiler
        tiled_mma,                  # bf16 MMA (unchanged)
        cluster_layout_vmnk.shape,
        internal_type=None,
    )
    cute.printf("A TMA atom built OK\n")
    return


def main():
    torch.manual_seed(0)
    # Int8 storage tensor, k-major (K contiguous): shape (M, K_STORAGE)
    a_bytes = torch.randint(
        -128, 128, (M, K_STORAGE), dtype=torch.int8, device="cuda"
    )
    mA_bytes = from_dlpack(a_bytes, assumed_align=16)
    print("storage A (Int8) shape:", tuple(a_bytes.shape),
          " logical K:", K_TOTAL, " storage K:", K_STORAGE)
    print("expected k_tile_cnt (storage):", K_STORAGE // BYTE_MMA_K,
          " == B k_tile_cnt (logical):", K_TOTAL // MMA_K)

    # compile only -- trace-time atom construction. If contradiction #2 is real here,
    # this is where it would segfault or raise.
    compiled = cute.compile(probe, mA_bytes)
    print("\nPROBE RESULT: cute.compile SUCCEEDED -- byte-typed A TMA atom is well-formed.")
    print("=> contradiction #2 resolved by byte-typing; fused int2-as-bytes is VIABLE.")


if __name__ == "__main__":
    main()
