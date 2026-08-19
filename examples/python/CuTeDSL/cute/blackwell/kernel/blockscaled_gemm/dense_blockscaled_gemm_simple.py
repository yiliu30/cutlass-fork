# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""A fixed, educational MXFP4 block-scaled GEMM for Blackwell.

This example deliberately keeps one layout, one MMA tile, one scale format,
and one pipeline. It is a small learning companion to
dense_blockscaled_gemm_persistent.py.
"""

import argparse
import math
from typing import Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import make_ptr


# Fixed educational configuration: MXFP4 = E2M1 x E2M1 with UE8M0 scales.
mma_tiler_mn = (128, 256)
mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E8M0FNU
c_dtype = cutlass.Float16
sf_vec_size = 32
num_ab_stage = 4
num_acc_stage = 1


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@cute.jit
def cvt_scale_layout(
    sf_ref_ptr: cute.Pointer,
    sf_mma_ptr: cute.Pointer,
    mn: int,
    sf_k: int,
    l: int,
    mma_shape: tuple,
):
    """Convert (M, ceil(K/32), L) scales to the Blackwell MMA layout."""
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    permuted_shape = tuple(mma_shape[i] for i in mma_permute_order)
    cute_layout = cute.make_ordered_layout(permuted_shape, order=(2, 1, 4, 0, 3, 5))
    sf_ref = cute.make_tensor(
        sf_ref_ptr,
        cute.make_layout((mn, sf_k, l), stride=(sf_k, 1, mn * sf_k)),
    )
    sf_mma = cute.make_tensor(sf_mma_ptr, cute_layout)
    sf_mma = cute.group_modes(sf_mma, 0, 3)
    sf_mma = cute.group_modes(sf_mma, 1, 3)
    for i in cutlass.range(cute.size(sf_ref)):
        sf_mma[sf_ref.layout.get_hier_coord(i)] = sf_ref[sf_ref.layout.get_hier_coord(i)]


def make_scales(mnkl: Tuple[int, int, int, int]):
    m, n, k, l = mnkl
    sf_k = ceil_div(k, sf_vec_size)
    sf_torch_dtype = cutlass_torch.dtype(sf_dtype)
    # E8M0 is stored as an exponent byte.  In particular, the byte value 127
    # encodes 1.0; a literal uint8 value of 1 would encode 2**-126.
    ref_a = torch.ones((l, m, sf_k), dtype=sf_torch_dtype).permute(1, 2, 0)
    ref_b = torch.ones((l, n, sf_k), dtype=sf_torch_dtype).permute(1, 2, 0)
    mma_shape_a = (l, ceil_div(m, 128), ceil_div(sf_k, 4), 32, 4, 4)
    mma_shape_b = (l, ceil_div(n, 128), ceil_div(sf_k, 4), 32, 4, 4)
    mma_a = torch.ones(mma_shape_a, dtype=sf_torch_dtype).permute(3, 4, 1, 5, 2, 0)
    mma_b = torch.ones(mma_shape_b, dtype=sf_torch_dtype).permute(3, 4, 1, 5, 2, 0)
    cvt_scale_layout(
        make_ptr(sf_dtype, ref_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(sf_dtype, mma_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        m,
        sf_k,
        l,
        mma_shape_a,
    )
    cvt_scale_layout(
        make_ptr(sf_dtype, ref_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(sf_dtype, mma_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        n,
        sf_k,
        l,
        mma_shape_b,
    )
    return (
        ref_a,
        ref_b,
        mma_a.cuda(),
        mma_b.cuda(),
    )


def make_inputs(mnkl: Tuple[int, int, int, int]):
    m, n, k, l = mnkl
    if m % mma_tiler_mn[0] or n % mma_tiler_mn[1] or k % 256:
        raise ValueError(
            f"mnkl must satisfy M%{mma_tiler_mn[0]}=0, N%{mma_tiler_mn[1]}=0, K%256=0"
        )
    # Use values exactly representable by FP4, then convert them to CuTe FP4.
    # Keeping the FP32 tensors gives us a simple and reliable reference path.
    a_ref = torch.randint(-2, 2, (l, m, k), dtype=torch.float32, device="cuda").permute(
        1, 2, 0
    )
    b_ref = torch.randint(-2, 2, (l, n, k), dtype=torch.float32, device="cuda").permute(
        1, 2, 0
    )
    a, _ = cutlass_torch.cute_tensor_like(
        a_ref.cpu(), ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b, _ = cutlass_torch.cute_tensor_like(
        b_ref.cpu(), ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    a = cutlass_torch.convert_cute_tensor(a_ref, a, ab_dtype, is_dynamic_layout=True)
    b = cutlass_torch.convert_cute_tensor(b_ref, b, ab_dtype, is_dynamic_layout=True)
    c = torch.empty((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)
    sfa_ref, sfb_ref, sfa, sfb = make_scales(mnkl)
    return a_ref, b_ref, a, b, c, sfa_ref, sfb_ref, sfa, sfb


def make_pointers(a, b, c, sfa, sfb):
    return (
        a.iterator,
        b.iterator,
        make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(sf_dtype, sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(sf_dtype, sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
    )


def reference(mnkl, a_ref, b_ref, sfa_ref, sfb_ref):
    m, n, k, l = mnkl
    output = torch.empty((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)
    for batch in range(l):
        scale_a = torch.repeat_interleave(
            sfa_ref[:, :, batch].float(), sf_vec_size, dim=1
        )[:, :k].cuda()
        scale_b = torch.repeat_interleave(
            sfb_ref[:, :, batch].float(), sf_vec_size, dim=1
        )[:, :k].cuda()
        output[:, :, batch] = (
            (a_ref[:, :, batch] * scale_a)
            @ (b_ref[:, :, batch] * scale_b).transpose(0, 1)
        ).to(torch.float16)
    return output


def timed(fn, warmups: int, iterations: int):
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
    return {
        "median_us": samples[len(samples) // 2],
        "min_us": samples[0],
        "p95_us": samples[max(0, math.ceil(len(samples) * 0.95) - 1)],
    }



class Sm100BlockScaledDenseGemmKernel:
    def __init__(self):
        self.threads_per_cta = 128
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512

        # set stages for ab_pipeline and acc_pipeline
        self.num_acc_stage = 1
        self.num_ab_stage = num_ab_stage

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: tuple,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # setup static attributes before smem/grid/tma computation
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        m, n, k, l = problem_size

        # Setup attributes that depend on gemm inputs
        mma_inst_tile_k = 4
        self.mma_tiler = (
            mma_tiler_mn[0],
            mma_tiler_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        # make address offset of c_tensor 256bit aligned,
        # so that epilogue could use vectorized store with larger vector size.
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout(
                (cute.assume(m, 32), cute.assume(n, 16), l),
                stride=(cute.assume(n, 16), 1, cute.assume(m * n, 512)),
            ),
        )
        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            sf_dtype,
            sf_vec_size,
            tcgen05.CtaGroup.ONE,
            mma_tiler_mn,
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            self.num_ab_stage,
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA load for A
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for B
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for SFA
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # TMA load for SFB
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Compute TMA load bytes
        a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Compute grid size
        grid = (
            cute.ceil_div(c_tensor.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(c_tensor.shape[1], self.cta_tile_shape_mnk[1]),
            c_tensor.shape[2],
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c_tensor,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()

        # Coords outside cluster
        cta_coord = (bidx, bidy, bidz)
        mma_tile_coord_mnl = (
            cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cta_coord[1],
            cta_coord[2],
        )

        #
        # Define shared storage for kernel
        #
        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        #
        # Initialize mainloop ab_pipeline, acc_pipeline and their states
        #
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_cta,
            ),
        ).make_participants()

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/SFA/SFB/C
        #
        # (MMA, MMA_M, MMA_K, RestK)
        thr_mma = tiled_mma.get_slice(0)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B/SFA/SFB
        #
        # TMA load A partition_S/D
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load partition for SFA tensor
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load partition for SFB tensor
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        #
        # Alloc tensor memory buffer
        #
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        #
        # Make SFA/SFB tmem tensor
        #
        # Get SFA tmem ptr
        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
            dtype=sf_dtype,
        )
        # (MMA, MMA_M, MMA_K)
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        # Get SFB tmem ptr
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=sf_dtype,
        )
        # (MMA, MMA_N, MMA_K)
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        #
        # Partition for S2T copy of SFA/SFB
        #
        # Make S2T CopyAtom
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
            sf_dtype,
        )
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSFA_compact = cute.filter_zeros(sSFA)
        # (MMA, MMA_MN, MMA_K)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSFB_compact = cute.filter_zeros(sSFB)
        # (MMA, MMA_MN, MMA_K)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        #
        # Slice to per mma tile index
        #
        # ((atom_v, rest_v), RestK)
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK)
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK)
        tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK)
        tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        #
        # Execute Data copy and Math computation in the k_tile loop
        #
        if warp_idx == 0:
            # Wait for accumulator buffer empty
            acc_empty = acc_producer.acquire_and_advance()
            # Set ACCUMULATE field to False for the first k_tile iteration
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            # Execute k_tile loop
            for k_tile in cutlass.range(
                k_tile_cnt, prefetch_stages=self.num_ab_stage - 2
            ):
                # Wait for AB buffer empty
                ab_empty = ab_producer.acquire_and_advance()

                # TMA load for A/B/SFA/SFB
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, ab_empty.count)],
                    tAsSFA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[(None, ab_empty.count)],
                    tBsSFB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )

                # Wait for AB buffer full
                ab_full = ab_consumer.wait_and_advance()

                #  Copy SFA/SFB to tmem
                s2t_stage_coord = (None, None, None, None, ab_full.index)
                tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t_staged,
                    tCtSFA_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t_staged,
                    tCtSFB_compact_s2t,
                )

                # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                num_kblocks = cute.size(tCrA, mode=[2])
                for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                    kblock_coord = (
                        None,
                        None,
                        kblock_idx,
                        ab_full.index,
                    )

                    sf_kblock_coord = (None, None, kblock_idx)
                    tiled_mma.set(
                        tcgen05.Field.SFA,
                        tCtSFA[sf_kblock_coord].iterator,
                    )
                    tiled_mma.set(
                        tcgen05.Field.SFB,
                        tCtSFB[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[kblock_coord],
                        tCrB[kblock_coord],
                        tCtAcc,
                    )
                    # Enable accumulate on tCtAcc after first kblock
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # Async arrive AB buffer empty
                ab_full.release()
            acc_empty.commit()

        #
        # Epilogue
        # Partition for epilogue
        #
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R_M, T2R_N, EPI_M, EPI_M)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        # (T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(tCgC)
        # (T2R_M, T2R_N, EPI_M, EPI_N）
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
        )
        # (T2R_M, T2R_N, EPI_M, EPI_N）
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, c_dtype
        )
        # STG Atom
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)
        tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

        # Release TMEM allocation lock
        tmem.relinquish_alloc_permit()

        # Wait for accumulator buffer full
        acc_full = acc_consumer.wait_and_advance()

        # Copy accumulator to register
        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        acc_vec = epilogue_op(tTR_rAcc.load().to(c_dtype))
        tTR_rC.store(acc_vec)
        # Store C to global memory
        cute.copy(simt_atom, tTR_rC, tTR_gC)

        acc_full.release()

        # Deallocate TMEM
        cute.arch.barrier()
        tmem.free(acc_tmem_ptr)

        return




# The benchmark/runner is intentionally kept below the kernel so the device code is easy to read.

def run(mnkl, warmups, iterations, skip_ref_check, benchmark, parity_tolerance):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    a_ref, b_ref, a, b, c, sfa_ref, sfb_ref, sfa, sfb = make_inputs(mnkl)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    gemm = Sm100BlockScaledDenseGemmKernel()
    a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr = make_pointers(a, b, c, sfa, sfb)
    compiled = cute.compile(
        gemm,
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        mnkl,
        stream,
    )
    compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, mnkl, stream)
    torch.cuda.synchronize()
    if not skip_ref_check:
        expected = reference(mnkl, a_ref, b_ref, sfa_ref, sfb_ref)
        torch.testing.assert_close(c, expected, atol=0.1, rtol=0.01)
        print("correctness: PASS")
    if not benchmark:
        return
    bf16_a = torch.randn((mnkl[0], mnkl[2]), device="cuda", dtype=torch.bfloat16)
    bf16_b = torch.randn((mnkl[2], mnkl[1]), device="cuda", dtype=torch.bfloat16)
    mxfp4 = timed(
        lambda: compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, mnkl, stream),
        warmups,
        iterations,
    )
    bf16 = timed(lambda: torch.mm(bf16_a, bf16_b), warmups, iterations)
    flops = 2.0 * mnkl[0] * mnkl[1] * mnkl[2]
    mxfp4["tflops"] = flops / mxfp4["median_us"] / 1.0e6
    bf16["tflops"] = flops / bf16["median_us"] / 1.0e6
    ratio = bf16["median_us"] / mxfp4["median_us"]
    print(f"MXFP4: {mxfp4}")
    print(f"BF16 torch.mm: {bf16}")
    print(f"MXFP4 speedup vs BF16: {ratio:.3f}x")
    if mxfp4["median_us"] > bf16["median_us"] * (1.0 + parity_tolerance):
        raise RuntimeError(
            "MXFP4 educational GEMM exceeded the allowed BF16 torch.mm parity "
            f"margin ({parity_tolerance:.1%})"
        )
    print(f"parity check: PASS (allowed margin {parity_tolerance:.1%})")


def parse_mnkl(value: str):
    try:
        values = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected M,N,K,L") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected exactly four integers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnkl", type=parse_mnkl, default=(8192, 8192, 1024, 1))
    parser.add_argument("--warmup_iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--parity_tolerance",
        type=float,
        default=0.10,
        help="Allowed slowdown versus BF16 torch.mm (default: 10%%)",
    )
    args = parser.parse_args()
    run(
        args.mnkl,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.benchmark,
        args.parity_tolerance,
    )
    print("PASS")


if __name__ == "__main__":
    main()
