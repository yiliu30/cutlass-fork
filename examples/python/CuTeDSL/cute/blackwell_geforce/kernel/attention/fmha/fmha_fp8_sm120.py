# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""FP8 (e4m3) flash-attention forward pass for SM120 (Blackwell GeForce).

Both GEMMs run in FP8 on the SM120 tensor cores:

* ``S = Q @ K^T``  -- e4m3 x e4m3 -> f32 (``mma.sync.aligned.m16n8k32``)
* ``O = P @ V``    -- e4m3 x e4m3 -> f32, with P (the softmax result)
                      quantized to e4m3

SM120 has FP8 tensor cores but no tcgen05/TMEM, so this kernel uses the
warp-level MMA path (``cute.nvgpu.warp.MmaFP8Op``) rather than the SM100 UMMA
path used by ``blackwell/kernel/attention/fmha``.  No non-block-scaled SM120
atom is exposed to CuTeDSL, so the SM89 atom is used; it lowers to the same
``mma.sync`` fp8 instruction the SM120 hardware executes.

The softmax result P is redistributed through shared memory.  The QK
accumulator's C fragment gives a lane columns ``{2j, 2j+1}`` of every
8-column group (lane ``j = tid % 4``), while the PV GEMM's A fragment wants
columns ``{4j .. 4j+3, 4j+16 .. 4j+19}``; for 8-bit operands a plain register
relayout cannot express that exchange, so a shared-memory round trip is used.
That round trip is warp-local -- the PV A operand only ever reads the rows the
same warp's QK accumulator wrote -- so it is fenced with ``sync_warp``.

V is stored in shared memory *transposed*, as a ``(head_dim, n_block)`` tile.
The PV B operand needs ``V^T``, and each of its 32-bit registers holds four
consecutive KV columns of one head-dim row.  Read through a transposed *view*
of an ``(n_block, head_dim)`` tile those four bytes are strided by the row
pitch, which forces four 8-bit ``ld.shared`` plus three ``prmt`` per register
(and SM120 has no byte-granular ``ldmatrix`` to do it in hardware).  Writing
the tile out transposed instead makes every operand register a single aligned
32-bit load; the byte transpose is folded into the gmem->smem staging, where
it costs eight ``prmt`` per 16 bytes staged.

Every K-major tile is padded so that the operand loads land on distinct
shared-memory banks.  A fragment register is read by the four lanes
``tid % 4`` and the eight lanes ``tid / 4``; without padding those reduce to
two bank offsets and every operand load serializes four-ways.  The pitches
below are chosen so the per-lane bank ``pitch * (tid/4) + tid%4`` is injective.

The per-lane softmax sums stay lane-local across KV tiles and are quad-reduced
once, in the epilogue -- reducing inside the tile loop would re-count the rows
already accumulated by the other lanes of the quad on every iteration.

A sequence the tile does not divide is covered by *sliding* the last tile back
into range instead of predicating every load: each tile then reads only rows
that exist, at the cost of re-covering some the previous tile already did.  On
the Q side the overlap is harmless, since those rows are recomputed from the
same inputs in the same order and rewritten with the same values; only the rows
past the end have to be suppressed at the store.  On the K/V side the overlap
is not harmless, because those columns would be counted a second time, so the
slide records where the tile *should* have started and the softmax masks the
columns it reaches back over to ``-inf``.

Note that fp8 attention loses relative precision as the context grows: with
``P`` prescaled by ``2**8`` and softmax probabilities of order ``1/seqlen_k``,
a long sequence pushes most of ``P`` into e4m3's subnormal range (~15% relative
error on those entries, versus ~6% in the normal range).  The effect is uniform
across the sequence, not localized to any tile.

Example:

.. code-block:: bash

    python examples/python/CuTeDSL/cute/blackwell_geforce/kernel/attention/fmha/fmha_fp8_sm120.py \\
      --head_dim 64 --seqlen_q 1024 --seqlen_k 1024 --num_head 8 --num_kv_head 8
"""

import argparse
import math
import sys

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack

LOG2_E = 1.4426950408889634

# P is scaled by 2**8 before the E4M3 cast; log2(448) is the E4M3 ceiling, so
# P in (0, 1] maps to (0, 256] and the smallest representable probability drops
# from 2**-9 to 2**-17.  The factor cancels in O = (P @ V) / row_sum.
P_PRESCALE_LOG2 = 8.0

# m16n8k32 is the only fp8 warp-MMA shape whose K matches the SM120 atom.
MMA_M, MMA_N, MMA_K = 16, 8, 32

# Shared-memory row pitches, in 32-bit words.  A: the K-major tiles Q and P,
# read as the A operand; K: the QK B operand; V: the transposed PV B operand.
# Q/K/P rows hold `d` or `n_block` bytes (16 words) and V rows `n_block` bytes.
PITCH_A = 20  # 80 B/row: 20*(tid/4) + tid%4 is injective over 32 lanes
PITCH_K = 24  # 96 B/row
PITCH_V = 17  # 68 B/row: keeps both the V load and the transposing store 2-way


def _make_g2s(rows, cols, num_threads, elems=16):
    """Tiled gmem->smem copy with 128-bit vectorization."""
    vec = cute.make_layout((1, elems))
    vpr = (rows * cols) // (num_threads * elems)
    thr = cute.make_layout((num_threads, vpr), stride=(vpr, 1))
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=elems * 8
    )
    return cute.make_tiled_copy_tv(atom, thr, vec)


def _make_g2s_i32(rows, cols, num_threads, elems=1):
    """32-bit gmem<->smem copy used by the register-staged K/V pipeline.

    e4m3 elements report only 8-bit alignment, which blocks every vectorized
    copy; viewing the same bytes as Int32 makes 32-bit alignment provable.
    The thread layout must be shaped like the (rows, cols/4) tensor -- a flat
    one silently covers only part of the tile.
    """
    c4 = cols // 4
    d1 = min(num_threads, c4)
    vec = cute.make_layout((1, elems))
    thr = cute.make_layout((num_threads // d1, d1), stride=(d1, 1))
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=elems * 32
    )
    return cute.make_tiled_copy_tv(atom, thr, vec)


@cute.jit
def _tile_offset(block, block_size, extent):
    """Start of ``block``'s tile, slid back so the whole tile stays in range.

    Equivalent to ``block * block_size`` whenever the tile divides ``extent``,
    so tile-aligned sequences take this path unchanged.
    """
    start = block * block_size
    if start + block_size > extent:
        start = extent - block_size
    return start


@cute.jit
def _acc_to_mn_view(acc):
    a = acc.layout
    sh = cute.make_layout(a.shape)
    mn = cute.make_layout(
        ((sh.shape[0][1], sh.shape[1]), (sh.shape[0][0], sh.shape[2])),
        stride=((sh.stride[0][1], sh.stride[1]), (sh.stride[0][0], sh.stride[2])),
    )
    return cute.make_tensor(acc.iterator, cute.composition(a, mn))


@cute.jit
def _quad_max(v):
    v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, 2, -1, 31))
    v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, 1, -1, 31))
    return v


@cute.jit
def _quad_sum(v):
    v = v + cute.arch.shuffle_sync_bfly(v, 2, -1, 31)
    v = v + cute.arch.shuffle_sync_bfly(v, 1, -1, 31)
    return v




@cute.kernel
def _fmha_fp8_sm120_kernel(mQ, mK, mV, mO, softmax_scale_log2, m_block_size: cutlass.Constexpr, n_block_size: cutlass.Constexpr, head_dim: cutlass.Constexpr,
           qk_mma: cute.TiledMma, pv_mma: cute.TiledMma, g2s: cute.TiledCopy,
           sc_Q: cute.TiledCopy, sc_K: cute.TiledCopy,
           sc_P: cute.TiledCopy, sc_V: cute.TiledCopy, r2s_P: cute.TiledCopy,
           sQ_l: cute.Layout, sK_l: cute.Layout, sV_l: cute.Layout, sP_l: cute.Layout,
           num_threads: cutlass.Constexpr,
           SharedStorage: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    m_block, batch, head = cute.arch.block_idx()
    d = head_dim

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.sQ.get_tensor(sQ_l, dtype=cutlass.Float8E4M3FN)
    sK = storage.sK.get_tensor(
        cute.make_layout((n_block_size, d), stride=(PITCH_K * 4, 1)),
        dtype=cutlass.Float8E4M3FN,
    )
    # V is stored transposed: an (head_dim, n_block) tile with the PV B
    # operand's four-byte register groups laid out contiguously, four rows of
    # the natural tile per word.
    sV = storage.sV.get_tensor(
        cute.make_layout((d, n_block_size), stride=(PITCH_V * 4, 1)),
        dtype=cutlass.Float8E4M3FN,
    )
    sK_i32 = storage.sK.get_tensor(
        cute.make_layout((n_block_size, d // 4), stride=(PITCH_K, 1))
    )
    sV_i32 = storage.sV.get_tensor(
        cute.make_layout((d, n_block_size // 4), stride=(PITCH_V, 1))
    )
    sP = storage.sP.get_tensor(sP_l, dtype=cutlass.Float8E4M3FN)

    # A sequence the tile does not divide is covered by *sliding* the last tile
    # back into range rather than by predicating every load: each row read is
    # then in bounds, and the rows the slide re-covers are recomputed from the
    # same inputs in the same order, so rewriting them is a no-op.  The store
    # still needs a bound, since the rows past the end must not be written.
    seqlen_q = mQ.shape[1]
    seqlen_k = mK.shape[1]
    m_off = _tile_offset(m_block, m_block_size, seqlen_q)
    gQ = cute.local_tile(
        cute.domain_offset((m_off, 0), mQ[batch, None, head, None]),
        (m_block_size, d),
        (0, 0),
    )
    gO = cute.local_tile(
        cute.domain_offset((m_off, 0), mO[batch, None, head, None]),
        (m_block_size, d),
        (0, 0),
    )

    tg = g2s.get_slice(tidx)
    cute.copy(g2s, tg.partition_S(gQ), tg.partition_D(sQ))
    cute.arch.sync_threads()

    thr = qk_mma.get_slice(tidx)
    thr_pv = pv_mma.get_slice(tidx)
    rQ = thr.make_fragment_A(thr.partition_A(sQ))
    scQ_t = sc_Q.get_slice(tidx)
    scK_t = sc_K.get_slice(tidx)
    scP_t = sc_P.get_slice(tidx)
    scV_t = sc_V.get_slice(tidx)
    r2s_t = r2s_P.get_slice(tidx)
    sQ_cv = scQ_t.partition_S(sQ)
    rQ_cv = scQ_t.retile(rQ)

    acc_S = cute.make_rmem_tensor(thr.partition_shape_C((m_block_size, n_block_size)), cutlass.Float32)
    acc_O = cute.make_rmem_tensor(thr_pv.partition_shape_C((m_block_size, d)), cutlass.Float32)
    acc_O.fill(0.0)

    mn_S = _acc_to_mn_view(acc_S)
    mn_O = _acc_to_mn_view(acc_O)
    n_row = cute.size(mn_S, mode=[0])
    n_col_S = cute.size(mn_S, mode=[1])
    n_col_O = cute.size(mn_O, mode=[1])
    id_O = _acc_to_mn_view(thr_pv.partition_C(cute.make_identity_tensor((m_block_size, d))))
    # Column index within the KV tile, for the slide's softmax mask.
    id_S = _acc_to_mn_view(thr.partition_C(cute.make_identity_tensor((m_block_size, n_block_size))))

    row_max = cute.make_rmem_tensor((n_row,), cutlass.Float32)
    row_sum = cute.make_rmem_tensor((n_row,), cutlass.Float32)
    for r in cutlass.range_constexpr(n_row):
        row_max[r] = -cutlass.Float32.inf
        row_sum[r] = 0.0

    rP = cute.make_fragment_like(acc_S, cutlass.Float8E4M3FN)
    rP_mn = _acc_to_mn_view(rP)
    rP_cv = r2s_t.retile(rP)
    sP_cv_out = r2s_t.partition_D(sP)

    # Q's register fragments are tile-invariant, so load them once here
    # instead of re-reading them from smem on every KV tile.
    for k in cutlass.range_constexpr(cute.size(rQ, mode=[2])):
        cute.copy(sc_Q, sQ_cv[None, None, k], rQ_cv[None, None, k])

    n_block_max = cute.ceil_div(mK.shape[1], n_block_size)
    g2s32 = _make_g2s_i32(n_block_size, d, num_threads)
    tg32 = g2s32.get_slice(tidx)

    # Register staging so the global-load latency overlaps the previous tile's
    # compute.  A plain gmem->smem copy compiles to serialized LDG->STS pairs,
    # putting the full memory latency on the critical path once per tile.
    # Per-thread staging sized to this thread's slice of the tiled copy.
    rK_stage = cute.make_fragment_like(tg32.partition_D(sK_i32), cutlass.Int32)

    # V's staging is a 4x4 byte transpose: each thread owns `n_vblk` blocks of
    # four KV rows by one four-byte head-dim group, loads the four source
    # words, and writes back the four transposed ones.
    n_wcol = d // 4
    n_vblk = (n_block_size // 4) * n_wcol // num_threads
    rV_words = cute.make_rmem_tensor((n_vblk * 4,), cutlass.Int32)

    nb0 = n_block_max - 1
    n_off0 = _tile_offset(nb0, n_block_size, seqlen_k)
    gK0 = cute.recast_tensor(
        cute.local_tile(cute.domain_offset((n_off0, 0), mK[batch, None, head, None]),
                        (n_block_size, d), (0, 0)),
        cutlass.Int32,
    )
    gV0 = cute.recast_tensor(
        cute.local_tile(cute.domain_offset((n_off0, 0), mV[batch, None, head, None]),
                        (n_block_size, d), (0, 0)),
        cutlass.Int32,
    )
    cute.copy(g2s32, tg32.partition_S(gK0), tg32.retile(rK_stage))
    for b in cutlass.range_constexpr(n_vblk):
        vg = (b * num_threads + tidx) // n_wcol
        vc = (b * num_threads + tidx) % n_wcol
        for i in cutlass.range_constexpr(4):
            rV_words[b * 4 + i] = gV0[4 * vg + i, vc]

    for n_tile in range(n_block_max):
        n_block = n_block_max - n_tile - 1
        n_start = n_block * n_block_size
        n_off = _tile_offset(n_block, n_block_size, seqlen_k)

        cute.copy(g2s32, tg32.retile(rK_stage), tg32.partition_D(sK_i32))
        for b in cutlass.range_constexpr(n_vblk):
            vg = (b * num_threads + tidx) // n_wcol
            vc = (b * num_threads + tidx) % n_wcol
            w0 = rV_words[b * 4 + 0]
            w1 = rV_words[b * 4 + 1]
            w2 = rV_words[b * 4 + 2]
            w3 = rV_words[b * 4 + 3]
            p0 = cute.arch.prmt(w0, w1, 0x5140)
            p1 = cute.arch.prmt(w0, w1, 0x7362)
            p2 = cute.arch.prmt(w2, w3, 0x5140)
            p3 = cute.arch.prmt(w2, w3, 0x7362)
            sV_i32[4 * vc + 0, vg] = cute.arch.prmt(p0, p2, 0x5410)
            sV_i32[4 * vc + 1, vg] = cute.arch.prmt(p0, p2, 0x7632)
            sV_i32[4 * vc + 2, vg] = cute.arch.prmt(p1, p3, 0x5410)
            sV_i32[4 * vc + 3, vg] = cute.arch.prmt(p1, p3, 0x7632)
        cute.arch.sync_threads()

        if n_block > 0:
            n_off_n = _tile_offset(n_block - 1, n_block_size, seqlen_k)
            gKn = cute.recast_tensor(
                cute.local_tile(cute.domain_offset((n_off_n, 0), mK[batch, None, head, None]),
                                (n_block_size, d), (0, 0)),
                cutlass.Int32,
            )
            gVn = cute.recast_tensor(
                cute.local_tile(cute.domain_offset((n_off_n, 0), mV[batch, None, head, None]),
                                (n_block_size, d), (0, 0)),
                cutlass.Int32,
            )
            cute.copy(g2s32, tg32.partition_S(gKn), tg32.retile(rK_stage))
            for b in cutlass.range_constexpr(n_vblk):
                vg = (b * num_threads + tidx) // n_wcol
                vc = (b * num_threads + tidx) % n_wcol
                for i in cutlass.range_constexpr(4):
                    rV_words[b * 4 + i] = gVn[4 * vg + i, vc]

        rK = thr.make_fragment_B(thr.partition_B(sK))
        sK_cv = scK_t.partition_S(sK)
        rK_cv = scK_t.retile(rK)

        acc_S.fill(0.0)
        for k in cutlass.range_constexpr(cute.size(rQ, mode=[2])):
            cute.copy(sc_K, sK_cv[None, None, k], rK_cv[None, None, k])
            cute.gemm(qk_mma, acc_S, rQ[None, None, k], rK[None, None, k], acc_S)

        # A slid last tile reaches back over KV columns an earlier iteration
        # already accumulated; drop them rather than count them twice.  Uniform
        # across the CTA, and false for every tile-aligned sequence.
        if n_off < n_start:
            for r in cutlass.range_constexpr(n_row):
                for c in cutlass.range_constexpr(n_col_S):
                    if n_off + id_S[r, c][1] < n_start:
                        mn_S[r, c] = -cutlass.Float32.inf

        for r in cutlass.range_constexpr(n_row):
            m = -cutlass.Float32.inf
            for c in cutlass.range_constexpr(n_col_S):
                m = cute.arch.fmax(m, mn_S[r, c])
            new_max = cute.arch.fmax(row_max[r], _quad_max(m))
            scale = cute.math.exp2((row_max[r] - new_max) * softmax_scale_log2, fastmath=True)
            for c in cutlass.range_constexpr(n_col_O):
                mn_O[r, c] = mn_O[r, c] * scale
            row_sum[r] = row_sum[r] * scale
            row_max[r] = new_max
            for c in cutlass.range_constexpr(n_col_S):
                p = cute.math.exp2((mn_S[r, c] - new_max) * softmax_scale_log2, fastmath=True)
                row_sum[r] = row_sum[r] + p
                rP_mn[r, c] = p.to(cutlass.Float8E4M3FN)
            # Row sums stay lane-local across tiles; the quad reduction is
            # applied once in the epilogue.

        # Only this warp reads back the P rows it just wrote: the PV A operand
        # is partitioned by M exactly as the QK accumulator is.
        cute.copy(r2s_P, rP_cv, sP_cv_out)
        cute.arch.sync_warp()

        rV = thr_pv.make_fragment_B(thr_pv.partition_B(sV))
        sV_cv = scV_t.partition_S(sV)
        rV_cv = scV_t.retile(rV)
        rPv = thr_pv.make_fragment_A(thr_pv.partition_A(sP))
        sP_cv = scP_t.partition_S(sP)
        rPv_cv = scP_t.retile(rPv)
        for k in cutlass.range_constexpr(cute.size(rPv, mode=[2])):
            cute.copy(sc_P, sP_cv[None, None, k], rPv_cv[None, None, k])
            cute.copy(sc_V, sV_cv[None, None, k], rV_cv[None, None, k])
            cute.gemm(pv_mma, acc_O, rPv[None, None, k], rV[None, None, k], acc_O)

        cute.arch.sync_threads()

    for r in cutlass.range_constexpr(n_row):
        denom = _quad_sum(row_sum[r])
        if denom == 0.0 or denom != denom:
            denom = cutlass.Float32(1.0)
        for c in cutlass.range_constexpr(n_col_O):
            co = id_O[r, c]
            # The slid tile recomputes some rows an earlier CTA already wrote
            # (identical values); only the rows past the end must be suppressed.
            if m_off + co[0] < seqlen_q:
                gO[co[0], co[1]] = mn_O[r, c] / denom

class FmhaFp8Sm120:
    """FP8 flash-attention forward for SM120, warp-MMA implementation."""

    def __init__(self, head_dim, m_block_size=64, n_block_size=64, num_warps=4):
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._num_warps = num_warps
        self._num_threads = num_warps * 32

    def can_implement(self, head_dim, seqlen_q, seqlen_k):
        # Sequence residues are covered by sliding the last tile back into
        # range, which needs the sequence to hold at least one full tile.
        if seqlen_q < self._m_block_size or seqlen_k < self._n_block_size:
            return False
        # The PV B operand is a transposed view of the smem V tile; that view
        # is only a valid permutation when the head dim matches the KV tile.
        if head_dim != self._n_block_size:
            return False
        if head_dim % MMA_K != 0:
            return False
        if self._n_block_size % MMA_K != 0:
            return False
        # Each warp must own exactly one 16-row M tile.  The operand copies
        # index the per-warp fragment's K mode positionally, and with more
        # than one M tile per warp that mode's position shifts -- indexing it
        # as [None, None, k] then silently loads the wrong registers.
        if self._m_block_size != self._num_warps * MMA_M:
            return False
        if head_dim % MMA_N != 0:
            return False
        # SM120 has (100 - 1) KiB of shared memory per CTA.
        smem = 4 * (
            self._m_block_size * PITCH_A
            + self._n_block_size * PITCH_K
            + head_dim * PITCH_V
            + self._m_block_size * PITCH_A
        )
        return smem <= (100 - 1) * 1024

    @cute.jit
    def __call__(self, mQ, mK, mV, mO, softmax_scale_log2, stream: cuda.CUstream):
        m_block_size = self._m_block_size
        n_block_size = self._n_block_size
        d = self._head_dim
        num_threads = self._num_threads

        sQ_l = cute.make_layout((m_block_size, d), stride=(PITCH_A * 4, 1))
        sK_l = cute.make_layout((n_block_size, d), stride=(PITCH_K * 4, 1))
        sV_l = cute.make_layout((d, n_block_size), stride=(PITCH_V * 4, 1))
        sP_l = cute.make_layout((m_block_size, n_block_size), stride=(PITCH_A * 4, 1))

        g2s = _make_g2s(m_block_size, d, num_threads)

        qk_op = warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32, (MMA_M, MMA_N, MMA_K))
        qk_mma = cute.make_tiled_mma(qk_op, cute.make_layout((self._num_warps, 1, 1)))
        pv_op = warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32, (MMA_M, MMA_N, MMA_K))
        pv_mma = cute.make_tiled_mma(pv_op, cute.make_layout((self._num_warps, 1, 1)))

        c8 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=8
        )
        sc_Q = cute.make_tiled_copy_A(c8, qk_mma)
        sc_K = cute.make_tiled_copy_B(c8, qk_mma)
        sc_P = cute.make_tiled_copy_A(c8, pv_mma)
        sc_V = cute.make_tiled_copy_B(c8, pv_mma)
        r2s_P = cute.make_tiled_copy_C(c8, qk_mma)

        @cute.struct
        class SharedStorage:
            # The K-major tiles are declared in words, since their pitches are
            # padded to spread banks; the MMAs read the same bytes through an
            # e4m3 view.  Int32 is also what makes 32-bit alignment provable to
            # the copy verifier, which e4m3's 8-bit alignment is not.
            sQ: cute.struct.MemRange[cutlass.Int32, m_block_size * PITCH_A]
            sK: cute.struct.MemRange[cutlass.Int32, n_block_size * PITCH_K]
            sV: cute.struct.MemRange[cutlass.Int32, d * PITCH_V]
            sP: cute.struct.MemRange[cutlass.Int32, m_block_size * PITCH_A]

        grid = (
            cute.ceil_div(mQ.shape[1], m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        _fmha_fp8_sm120_kernel(
            mQ, mK, mV, mO, softmax_scale_log2,
            m_block_size, n_block_size, d,
            qk_mma, pv_mma, g2s, sc_Q, sc_K, sc_P, sc_V, r2s_P,
            sQ_l, sK_l, sV_l, sP_l, num_threads, SharedStorage,
        ).launch(grid=grid, block=[num_threads, 1, 1], stream=stream)


def run(args):
    head_dim = args.head_dim
    torch.manual_seed(0)

    dtype = torch.float8_e4m3fn
    q = (torch.randn(args.batch_size, args.seqlen_q, args.num_head, head_dim, device="cuda") * 2).to(dtype)
    k = (torch.randn(args.batch_size, args.seqlen_k, args.num_kv_head, head_dim, device="cuda") * 2).to(dtype)
    v = (torch.randn(args.batch_size, args.seqlen_k, args.num_kv_head, head_dim, device="cuda") * 2).to(dtype)
    o = torch.zeros(args.batch_size, args.seqlen_q, args.num_head, head_dim, device="cuda", dtype=torch.float32)

    if args.num_head != args.num_kv_head:
        raise NotImplementedError("only num_head == num_kv_head is supported")

    fmha = FmhaFp8Sm120(
        head_dim=head_dim,
        m_block_size=args.m_block_size,
        n_block_size=args.n_block_size,
        num_warps=args.num_warps,
    )
    if not fmha.can_implement(head_dim, args.seqlen_q, args.seqlen_k):
        raise RuntimeError("problem not implementable")

    softmax_scale = args.softmax_scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    softmax_scale_log2 = softmax_scale * LOG2_E

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def tensors():
        return (
            from_dlpack(q, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(o, assumed_align=16),
        )

    compiled = cute.compile(
        fmha, *tensors(), softmax_scale_log2, stream, options="--opt-level=2"
    )

    def launch():
        compiled(*tensors(), softmax_scale_log2, stream)

    if args.skip_ref_check:
        for _ in range(max(1, args.iterations)):
            launch()
        torch.cuda.synchronize()
        print("SKIP_REF_CHECK")
        return

    launch()
    torch.cuda.synchronize()

    qf, kf, vf = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    # (batch, seqlen_q, seqlen_k, num_head) -- softmax runs over the KV axis.
    s = torch.einsum("bqhd,bkhd->bqkh", qf, kf) * softmax_scale
    p = torch.softmax(s, dim=2)
    # The kernel runs PV with P quantized to e4m3 (after the 2**8 prescale);
    # model that so the comparison reflects the kernel's arithmetic rather than
    # the inherent error of an e4m3 probability matrix.
    p_q = (p * (2.0 ** P_PRESCALE_LOG2)).to(torch.float8_e4m3fn).to(torch.float32)
    p_q = p_q / (2.0 ** P_PRESCALE_LOG2)
    ref = torch.einsum("bqkh,bkhd->bqhd", p_q, vf)
    ref = ref.to(torch.float8_e4m3fn).to(torch.float32)

    err = (o - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0), dim=1
    ).item()
    print(f"max_abs_err = {err:.6e}   cosine = {cos:.6f}")
    ok = err < args.tolerance and cos > args.cosine_threshold
    print("PASS" if ok else "FAIL")


def main():
    p = argparse.ArgumentParser(description="FP8 FMHA forward pass for SM120")
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seqlen_q", type=int, default=256)
    p.add_argument("--seqlen_k", type=int, default=256)
    p.add_argument("--num_head", type=int, default=4)
    p.add_argument("--num_kv_head", type=int, default=4)
    p.add_argument("--m_block_size", type=int, default=64)
    p.add_argument("--n_block_size", type=int, default=64)
    p.add_argument("--num_warps", type=int, default=4)
    p.add_argument("--softmax_scale", type=float, default=None)
    # e4m3 keeps only 3 mantissa bits, so the P matrix inside PV carries a
    # ~6% relative error per weight; the resulting bound on O is ~0.5 here.
    p.add_argument("--tolerance", type=float, default=0.75)
    p.add_argument("--cosine_threshold", type=float, default=0.98)
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--iterations", type=int, default=1)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    sys.exit(main())
