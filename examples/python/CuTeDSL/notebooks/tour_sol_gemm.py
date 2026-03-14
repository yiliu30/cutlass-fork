# %%
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

# %% [markdown]
# # Tour of SOL GEMM
# 
# This notebook demonstrates how to reach SOL (Speed Of Light) GEMM (GEneral Matrix Multiplication) based on Blackwell (tcgen05) step by step.
# 
# Before going through it, you need to get familiar with:
# 
# - tensor.ipynb
# - tensorssa.ipynb
# - cute_layout_algebra.ipynb
# - composed_layout.ipynb
# - elementwise_add.ipynb
# - async_pipeline.ipynb
# 
# These ipynb files will give you a basic knowledge on how to write a kernel by using CuTeDSL.
# 
# ## Learning Objectives
# 
# In this tutorial, you will learn writing an efficient gemm step by step:
# - How to implement basic GEMM kernel using CuTeDSL
# - How to subtile the acc
# - How to apply multi-stage by using software pipelining
# - How to vectorize the instructions for storing out
# 
# ## Understanding GEMM
# 
# GEMM is one of the most important operations in linear algebra and deep learning. Given two 2D tensors A with shape $(M, K)$ and B with shape $(N, K)$, the GEMM operation $C = A \times B$ is defined as:
# 
# $
#     C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} * B_{j,k}
# $
# 
# The result is a 2D tensor C with shape $(M, N)$.
# 
# where:
# - $i \in [0, M)$ represents the row index of $A$ and $C$
# - $j \in [0, N)$ represents the column index of $C$ and the row index of $B$
# - $k \in [0, K)$ repersents the column index of $A$ and $B$
# - $A_{i,k}$, $B_{j,k}$, and $C_{i,j}$ are the elements at position $(i,k)$, $(j,k)$ and $(i,j)$ in tensors $A$, $B$, and $C$ respectively
# 
# This operation has several important characteristics:
# 
# 1. **Parallelizable**: Each element can be computed independently. It helps take fully use of SMs in a GPU.
# 2. **Data Reusable**: $C_{i,:}$-s (The row $i$ of $C$) need the same data from $A_{i,:}$ while $C_{:,j}$-s (The column $j$ of $C$) need the same data from $B_{:,j}$. This data reuse pattern can help reduce the IO pressure
# 3. **Block-friendly**: A block of elements can be processed together. Each block is a sub-problem of the whole GEMM. It helps reduce the IO pressure for each SM. It gives possibility to accelerate the computation using MMA instructions.
# 4. **Bottleneck-flexible**: Unlike the elementwise_add, the bottleneck for GEMM is varied for different problem sizes. Let's calculate the compute/memory ratio for GEMM roughly: $ratio = \frac{M * N * K}{M*K + N*K + M*N} = \frac{1}{\frac{1}{N} + \frac{1}{M} + \frac{1}{K}}$.
# It's related to all M, N and K. To reach good enough perf, we need different strategies for different problem sizes accordingly.
# 
# 
# ## Naive GEMM
# 
# Let's start with a naive implementation to establish a baseline before exploring optimizations.
# 
# ![figure1](./images/blocked_gemm.svg "figure1: blocked gemm")
# 
# First of all, we need to set basic configurations.
# 
# - io_dtype: The datatype for tensors $A$, $B$, and $C$. For the most cases, it's also the input datatype of mma instructions (there're some exceptions, e.g. TF32 datatype, input transformation, etc.).
# 
# - acc_dtype: The datatype for the accumulation. Normally, set it as FP32 to avoid overflow. As C's datatype could be different from acc_dtype, the acc data needs to be converted to io_dtype before storing out.
# 
# - mma_inst_shape_mnk: The shape of one tcgen05 mma instruction can deal with. See more details in [PTX Document 9.7.16.2.1. Matrix Shape](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape).
# From beginning, we choose the biggest one as it's easy to reach SOL.
# 
# - mma_tiler_mnk: The GEMM kernel is normally implemented as blocked GEMM (see figure 1). Mma tiler is the block shape that one CTA or two CTAs will process. Whether one or two is determined by the issue granularity of tcgen05. See more details in [PTX Document 9.7.16.5. Issue Granularity](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-issue-granularity).
# From beginning, we choose one CTA to issue tcgen05 for simplicity.
# 
# - threads_per_cta: The number of threads we need to use in one CTA. To take fully use of a SM (streaming multiprocessor), it's at least 128.
# 
# - ab_stages: The number demonstrates how many blocks that TMA can load before each block's computation. It's usually limited by the smem capacity. For mma_tiler_mnk (128, 256, 64), we can set it as 4 at most.
# 
# - acc_stage: As each CTA only computes one block of acc and stores out, the number is 1.
# 
# 

# %%
io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
threads_per_cta = 128

# Pipeline stage configuration
ab_stages = 4
acc_stage = 1

# %% [markdown]
# Then, let's define the problem sizes and initialize the input & output tensors, i.e. $A$, $B$, and $C$.
# 
# We start with a typical computation bound case. i.e. 8kx8kx8k. It's also large enough for each dimension to avoid tile quantization issue.

# %%
m, n, k = 8192, 8192, 8192

# Make K-major tensors (torch tensors are row-major)
def make_tensors(mn, k, dtype):
    shape = (mn, k)
    return (
        torch.empty(*shape, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=dtype, device="cuda")
    )

a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
b = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))
a_tensor = (
    from_dlpack(a)
    .mark_layout_dynamic(leading_dim=1)
)
b_tensor = (
    from_dlpack(b)
    .mark_layout_dynamic(leading_dim=1)
)
c_tensor = (
    from_dlpack(c)
    .mark_layout_dynamic(leading_dim=1)
)

# %% [markdown]
# Before writing kernel, we need to configurate basic components in a GEMM operation.
# 
# 1. Tiled MMA. The tiled MMA helps calculate GEMM for one mma tile. We configurate it as tcgen05 MMA.
# 
# 2. Smem layous for A and B. They must match the post-partitioned (CTA-local) shapes expected by the MMA instructions.
# sm100_utils provides functions to determine the post-partitioned shape.
# These functions take the tiled MMA, and the mma tiler as inputs and returns a shape that is at least rank-3
# where the first mode has the same shape as the MMA instruction, 2nd and 3rd mode expresses the number of time
# MMA instr is repeated in M/N mode and K mode of MMA tile, respectively.
# These SMEM layouts are swizzled to improve MMA performance.
# 
# 3. TMA descriptors for A & B. We've already know A, B tensors' info in both GMEM (global memory) & SMEM (shared memory). We take those to generate TMA descriptors & tme tensors. They helps load a block of A & B from GMEM to SMEM.
# 
# ## Host Code
# 
# Host code constructs the components introduced above. Besides, we calculate the grid shape & launch the kernel with these as parameters.

# %%
@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    kernel: cutlass.Constexpr,
):
    # Construct tiled MMA
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    # Construct SMEM layouts for A and B
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    # Construct TMA load atoms
    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )

    # Launch the kernel
    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
    )

# %% [markdown]
# ## Structure of the Kernel
# 
# Let's breakdown how a GEMM kernel organize:
# 
# 1. **Prologue**: The phase before the first MMA instructions. It usually defines, fetches, allocates, partitions or calculates necessary components (listed below). What else, load multiple stages of data ahead of the first MMA to help hide GMEM latency.
#    - Indexing
#      * `block_idx` (bidx, bidy): Block index in the grid
#      * `mma_coord_mnk`: The location of which block the current MMA unit will calculate (see details in figure 1)
#      * `thread_idx` (tidx): Thread index within a block (0 to threads_per_cta - 1). We need this to slice the partition of tensor memory for each thread in a block (see details in [PTX Document 9.7.16.2.3.1 Memory Layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-layout))
#      * `warp_idx`: As TMA & tcgen05.mma only needs one thread to issue, some code only needs to execute by warp 0
#    - Allocation
#      * `smem` (storage, sA, sB): Allocate necessary smem usage for pipelines, A/B smem tensors as input of tcgen05.mma
#      * `tmem`: Allocate necessary tmem usage for Acc
#    - Pipeline (see more details in async_pipeline.ipynb)
#      * `PipelineTmaUmma`: Tma & tcgen05.mma units are async. PipelineTmaUmma helps notify: 1. tcgen05.mma when TMA fills A/B buffers to full; 2. TMA when tcgen05.mma consumes A/B buffer to empty
#      * `PipelineUmmaAsync`: It helps threads when tcgen05.mma finish the accumulation and Acc is ready
#      * `Barrier initialization`: barrier initialization work is done inside the pipeline create functions
#    - Partition
#      * `local_tile`: Get the block of A/B/C GMEM tensors for current MMA unit acoording to `mma_coord_mnk`.
#      * `TMA`: Get the tensor view from each TMA instruction
#      * `MMA`: Get the tensor view from each tcgen05.mma instruction
#    - TMA descriptor prefetch
#      * `cpasync.prefetch_descriptor`: helps shorten the latency of access tma descrptor, i.e. tma_atom_a, tma_atom_b
# 
# 2. **Mainloop**: The phase that carries out the main computation of GEMM. It's usually organized as a loop to iterate blocks in K dim for accumulation. The loop body contains:
#     - `Data prefetch` with a fixed stride (ab_stage - 1) ahead of current K block
#     - `MMA computation` for current K block
# 
# 3. **Epilogue**: The phase after the MMA instructions finish the accumulation. It usually contains:
#     - `Partition`: Get the tensor views from epi tiler (acc subtile) & each tcgen05.ld instruction
#     - `Acc fetch`: Load data from tensor memory to register
#     - `Fusion & datatype conversion`: Fuse some operations on C (optional); Datatype conversion if output type is different from acc type
#     - `Relinquish tmem alloc permit`: Give permit for following launched kernels
#     - `Storing`: TMA or st.global to store out
#     - `TMEM deallocation`: Deallocate tmem for Acc buffer
#     
#     Usually, we subtile the acc buffer to save resources of registers & smem (if using TMA to store C). For our mma_tiler (128, 256), each thread needs 256 registers if no subtiling. Besides, better instruction-level parallelism for interleavely issuing tcgen05.ld, data conversion & st.global.
# 
# ``` python
#         for i in cutlass.range(cute.size(tDtC, mode=[2])):
#             cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
#             tCrC.store(tCrAcc.load().to(io_dtype))
#             cute.autovec_copy(tCrC, tDgC[None, None, i])
# ```

# %%
@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    #
    # 1. Prepare args
    #

    # Current thread/warp/block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    # Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    # Allocate all TMEM columns
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    # Prefetch tma descriptor
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # Pipeline configuration
    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, threads_per_cta
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # Partition tensors for MMA and make fragments
    # (bM, bK, RestK)
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_K)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)
    # (MMA, MMA_M, MMA_K)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    # Partition tensors for TMA; This requires the tensors partitioned for MMA
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
    # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # Swap the pointer in tCtAcc
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    # (EpiTile)
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )
    # (EpiTile, NumTiles)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    # Every thread loads 32x128 bits
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # (TmemCpy,NumTmemCpy,NumTiles)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    # (TmemCpy,NumTmemCpy,NumTiles)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    # (TmemCpy,NumTmemCpy)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    # (TmemCpy,NumTmemCpy)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    #
    # 2. Main loop
    #
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        # Wait for a empty accumulator buffer
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile_idx in cutlass.range(num_k_tiles):
            # Issue TMA loads
            ab_empty = ab_producer.acquire_and_advance()
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

            # Execute one K-block worth of MMA instructions
            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Signal that the A/B buffers have been consumed and are ready for the next load
            ab_full.release()

        # Signal that the accumulator is fully computed
        acc_empty.commit()

    #
    # 3. Epilogue
    #

    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()

    # Wait for the accumulator buffer to be full
    acc_full = acc_consumer.wait_and_advance()

    # TMEM -> RMEM -> GEMM
    # Sub-tiling for better instruction-level parallelism
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    # Deallocate TMEM
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)

# %% [markdown]
# ## Performance Analysis and Benchmarking
# 
# To understand and improve our kernel's performance, we need to measure its execution time and computation throughput. Let's analyze several key metrics:
# 
# * **Execution Time**
#    - Measures raw kernel performance in microseconds
#    - Lower is better
#    - Affected by GPU clock speed, memory bandwidth, and kernel efficiency
# * **Computation Throughput**
#    - Measures how fast we compute (TFlops)
#    - Higher is better
#    - Theoretical peak varies by GPU model
#    - For GEMM:
#      * M * N * K FMAs to finish GEMM
#      * 2 Float operations for each FMA
#      * Total = M * N * K * 2
# 
# Below is our benchmarking utility that measures these metrics:

# %%
def benchmark(callable, a_tensor, b_tensor, c_tensor):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_tensor, b_tensor, c_tensor),
        warmup_iterations=1,
        iterations=2,
    )

    # Calculate metrics

    # Calculate total float ops calculated:
    # - M * N * K * 2 (FMA)
    total_float_ops = m * n * k * 2

    # Calculate achieved TFlops
    achieved_tflops = total_float_ops / (avg_time_us * 1000000)  # TFlops

    # Print results
    # ------------
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_tflops:.2f} tflops")

# %% [markdown]
# ## Test the first version of GEMM
# 
# You can run the following code to get the Tflops and verify the function is ok by using `torch.einsum` as a reference.
# 
# You should be able to reach about **450** TFlops.

# %%
# Compile the kernel for the specific input types
naive_kernel = cute.compile(host_function, a_tensor, b_tensor, c_tensor, kernel)

# Run the kernel
benchmark(naive_kernel, a_tensor, b_tensor, c_tensor)

# Verify Results
# -------------
# Compare our kernel output with PyTorch's native implementation
# Compute reference result and verify
ref = (torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32))).cpu()
torch.testing.assert_close(
    c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=1e-1, rtol=1e-05
)
print("Verification passed!")

# %% [markdown]
# 
# ## Enable software pipelining
# 
# Like what we said before, usually we prefetch multiple stages (ab_stages - 2) of A/B tensors to hide latency of GMEM (see figure 2). The dark area demonstrates the issue of TMA/tcgen05.mma while the light area demonstrates the latency correspondingly.
# It can use (ab\_stages - 1) * time of one stage mma to hide GMEM latency.
# 
# ![figure2](./images/software_pipelining_ab_stages_minus_2.svg "figure2: software_pipelining_with_prefetch_ab_stages_minus_2")
# 
# 
# 
# To enable this strategy, we:
# 1. write a loop to prefetch before the mainloop
# 2. A fixed stride ahead copy inside the mainloop.
# 
# 
# ``` python
# # Prefetch ab_stages - 2 blocks of A/B
# for stage in cutlass.range(ab_stages - 2):
#     ab_empty = ab_producer.acquire_and_advance()
#     cute.copy(...)
# 
# for k_tile_idx in cutlass.range(num_k_tiles):
#     # Issue TMA loads
#     if k_tile_idx + ab_stages - 2 < num_k_tiles:
#         ab_empty = ab_producer.acquire_and_advance()
#         cute.copy(...)
#     # Execute one K-block worth of MMA instructions
#     ab_full = ab_consumer.wait_and_advance()
#     cute.gemm(...)
#     # Signal that the A/B buffers have been consumed and are ready for the next load
#     ab_full.release()
# ```
# 
# For CuTeDSL, we have an attribute `prefetch_stages` for cutlass.range. It helps us write code like the general pattern but prefetch data like we write above.
# 
# ``` python
# for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
#     # Issue TMA loads
#     ab_empty = ab_producer.acquire_and_advance()
#     cute.copy(...)
#     # Execute one K-block worth of MMA instructions
#     ab_full = ab_consumer.wait_and_advance()
#     cute.gemm(...)
#     # Signal that the A/B buffers have been consumed and are ready for the next load
#     ab_full.release()
# ```
# 
# Figure 3 explains why we prefetch ab_stages - 2 instead of ab_stages - 1. For ab_stages - 1, Each TMA copy inside mainloop will be issued after the previous MMA finished. It will delay the issue of next MMA and cause bubbles between 2 blocks.
# 
# ![figure3](./images/software_pipelining_ab_stages_minus_1.svg "figure3: software_pipelining_with_prefetch_ab_stages_minus_1")
# 
# Let's test the perf with prefetch enabled. You can reach about **880** TFlops.

# %%
@cute.kernel
def kernel_with_prefetch(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    #
    # 1. Prepare args
    #

    # Current thread/warp/block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    # Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    # Allocate all TMEM columns
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    # Prefetch tma descriptor
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # Pipeline configuration
    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, threads_per_cta
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # Partition tensors for MMA and make fragments
    # (bM, bK, RestK)
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_K)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)
    # (MMA, MMA_M, MMA_K)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    # Partition tensors for TMA; This requires the tensors partitioned for MMA
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
    # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # Swap the pointer in tCtAcc
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    # (EpiTile)
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )
    # (EpiTile, NumTiles)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    # Every thread loads 32x128 bits
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # (TmemCpy,NumTmemCpy,NumTiles)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    # (TmemCpy,NumTmemCpy,NumTiles)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    # (TmemCpy,NumTmemCpy)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    # (TmemCpy,NumTmemCpy)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    #
    # 2. Main loop
    #
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        # Wait for a empty accumulator buffer
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
            # Issue TMA loads
            ab_empty = ab_producer.acquire_and_advance()
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

            # Execute one K-block worth of MMA instructions
            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Signal that the A/B buffers have been consumed and are ready for the next load
            ab_full.release()

        # Signal that the accumulator is fully computed
        acc_empty.commit()

    #
    # 3. Epilogue
    #

    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()

    # Wait for the accumulator buffer to be full
    acc_full = acc_consumer.wait_and_advance()

    # TMEM -> RMEM -> GEMM
    # Sub-tiling for better instruction-level parallelism
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    # Deallocate TMEM
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)

# %%
# Compile the kernel for the specific input types
prefetch_kernel = cute.compile(host_function, a_tensor, b_tensor, c_tensor, kernel_with_prefetch)

# Run the kernel
benchmark(prefetch_kernel, a_tensor, b_tensor, c_tensor)

# Verify Results
# -------------
# Compare our kernel output with PyTorch's native implementation
# Compute reference result and verify
ref = (torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32))).cpu()
torch.testing.assert_close(
    c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=1e-1, rtol=1e-05
)
print("Verification passed!")

# %% [markdown]
# ## Vectorized instructions for storing out
# 
# If we use NCU to profile this kernel, a sharply drop of TensorCore utilizition for each wave switching. That's because of lots of st.global.b16.
# CuTeDSL needs alignment & divisibility to choose vectorized instructions for cute.copy.
# We need to set these attributes correctly from cute tensors.
# 
# You can reach about **1400** Tflops after using vectorized instructions.

# %%
a_tensor_ = (
    from_dlpack(a, assumed_align=32)
    .mark_layout_dynamic(leading_dim=1)
    .mark_compact_shape_dynamic(mode=1, divisibility=k)
)
b_tensor_ = (
    from_dlpack(b, assumed_align=32)
    .mark_layout_dynamic(leading_dim=1)
    .mark_compact_shape_dynamic(mode=1, divisibility=k)
)
c_tensor_ = (
    from_dlpack(c, assumed_align=32)
    .mark_layout_dynamic(leading_dim=1)
    .mark_compact_shape_dynamic(mode=1, divisibility=n)
)

# %%
# Compile the kernel for the specific input types
vectorized_kernel = cute.compile(host_function, a_tensor_, b_tensor_, c_tensor_, kernel_with_prefetch)

# Run the kernel
benchmark(vectorized_kernel, a_tensor_, b_tensor_, c_tensor_)

# Verify Results
# -------------
# Compare our kernel output with PyTorch's native implementation
# Compute reference result and verify
ref = (torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32))).cpu()
torch.testing.assert_close(
    c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=1e-1, rtol=1e-05
)
print("Verification passed!")


