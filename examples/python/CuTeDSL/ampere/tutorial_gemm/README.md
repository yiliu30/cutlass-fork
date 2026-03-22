### Note
- What's the atom:

So the mental model is:                                                                          
                                                                                                
atom       = smallest hardware operation (wraps one PTX instruction)                             
atoms_layout = how many atoms, arranged how in MNK space                                         
tiled_mma  = atom × atoms_layout × permutation → full CTA-level compute pattern      

atom in example: `fma.rn.f32`

The Atom Solution: Encode the Contract as a Layout                                               
                                    
**CuTe's insight: every hardware MMA instruction is fully described by a set of layouts**:           
                                        
MMA Atom = {                                                                                     
    shape:      (M, N, K)         ← how big is one instruction                                   
    thr_id:     Layout            ← how many threads, numbered how                               
    TV_layout_A: Layout(Thr, Val) ← which thread holds which A-element                           
    TV_layout_B: Layout(Thr, Val) ← which thread holds which B-element                           
    TV_layout_C: Layout(Thr, Val) ← which thread holds which C-element                           
}                                                                                                
    
  ┌────────────────────────────────────────────────────────┐
  │  With atoms:                                           │
  │                                                        │
  │    atom = describe_hardware_instruction_as_layouts()   │
  │    tiled_mma = make_tiled_mma(atom, atoms_layout, perm)│
  │    thr_mma.partition_A(tensor)  ← SAME code for ALL   │
  │    thr_mma.partition_B(tensor)  ← architectures       │
  │    cute.gemm(tiled_mma, D, A, B, C)                   │
  │                                                        │
  │  Swap the atom → everything else adapts automatically  │
  └────────────────────────────────────────────────────────┘

  The Layered Abstraction

  Level 0: Hardware    fma.rn.f32 │ mma.sync.m16n8k16 │ wgmma.m64n256k16 │ umma
                           ↓                ↓                  ↓               ↓
  Level 1: Atom        MmaUniversal │ MmaF16BF16Op     │ SM90 MmaAtom     │ SM100 MmaAtom
           (encode       Op         │  (16,8,16)        │  (64,N,16)       │  (varies)
            contract     (1,1,1)    │  32 threads       │  128 threads     │  128 threads
            as layouts)  1 thread   │                    │                  │
                           ↓                ↓                  ↓               ↓
  Level 2: TiledMma    atom + atoms_layout + permutation → CTA-level pattern
           (tile atoms                ↓
            across CTA)  partition_A, partition_B, partition_C
                                      ↓
  Level 3: Kernel      cute.gemm(tiled_mma, D, A, B, C) → correct result
           (generic)


permutation_tiler_M = cute.make_layout((F, R), stride=(R, 1))

# CuTe DSL GEMM Tutorial Series (Ampere SM80)

A progressive 5-step tutorial that builds a high-performance GEMM kernel from scratch, adding exactly one new concept per step.

## Overview

| Step | File | New Concept | MMA Type | SMEM | Pipeline |
|------|------|-------------|----------|------|----------|
| 1 | `step1_naive_gemm.py` | Core abstractions | FP32 SIMT FMA | No | No |
| 2 | `step2_smem_gemm.py` | Shared memory tiling | FP32 SIMT FMA | Yes (1 stage) | No (sync) |
| 3 | `step3_pipeline_gemm.py` | Async pipeline | FP32 SIMT FMA | Yes (3 stages) | cp.async |
| 4 | `step4_tensorop_gemm.py` | Tensor cores + swizzle | FP16 HMMA 16x8x16 | Yes (3 stages) | cp.async |
| 5 | `step5_vectorized_gemm.py` | Vectorized copy + reg pipeline | FP16 HMMA 16x8x16 | Yes (3 stages) | cp.async + reg |

## Data Path Progression

```
Step 1:  GMEM ───────────────────────────────────> RMEM ──> FMA ──> RMEM ──> GMEM
Step 2:  GMEM ──(sync copy)──> SMEM ──> sync ──> RMEM ──> FMA ──> RMEM ──> GMEM
Step 3:  GMEM ──(cp.async)───> SMEM[3] ────────> RMEM ──> FMA ──> RMEM ──> GMEM
Step 4:  GMEM ──(cp.async)───> SMEM[3,swz] ──(ldmatrix)──> RMEM ──> HMMA ──> GMEM
Step 5:  GMEM ──(128b async)─> SMEM[3,swz] ──(ldmatrix)──> RMEM ──> HMMA ──> SMEM ──(128b)──> GMEM
```

## Fixed Configuration

All steps use the same matrix layout to eliminate layout complexity:
- **A:** (M, K) M-major — contiguous in M, stride = (1, M)
- **B:** (N, K) N-major — contiguous in N, stride = (1, N)
- **C:** (M, N) N-major — row-major, stride = (N, 1)
- Problem size must be divisible by tile size (no predication)

Steps 1–3: Tile 128×128×8, 256 threads, FP32
Steps 4–5: Tile 128×128×32, 128 threads (4 warps), FP16 in / FP32 accum / FP16 out

## Running

```bash
# Each step is self-contained and independently runnable
python step1_naive_gemm.py --mnk 512,512,512
python step2_smem_gemm.py --mnk 512,512,512
python step3_pipeline_gemm.py --mnk 512,512,512
python step4_tensorop_gemm.py --mnk 512,512,512
python step5_vectorized_gemm.py --mnk 512,512,512
```

Each verifies against `torch.einsum("mk,nk->mn", A.float(), B.float())`.

## Concept Reference

### Step 1: Core CuTe DSL Abstractions
- `@cute.kernel` — GPU kernel function
- `@cute.jit` — host-side JIT-compiled function
- `cute.local_tile(tensor, tiler, coord, proj)` — partition tensor into CTA tiles
- `cute.make_tiled_mma(op, layout, permutation)` — create thread-level MMA
- `thr_mma.partition_A/B/C()` — thread's view of the data
- `cute.gemm(tiled_mma, D, A, B, C)` — execute MMA: D = A * B + C

### Step 2: Shared Memory
- `SmemAllocator()` + `allocate_tensor()` — SMEM allocation
- `make_tiled_copy_tv(atom, thr_layout, val_layout)` — cooperative copy
- `cute.arch.sync_threads()` — `__syncthreads()`

### Step 3: Async Pipeline
- `CopyG2SOp()` — cp.async hardware DMA
- `cp_async_commit_group()` / `cp_async_wait_group(n)` — async fence
- Multi-stage SMEM: 3rd dimension for pipeline depth
- Prologue/mainloop pattern

### Step 4: Tensor Cores
- `MmaF16BF16Op(dtype, acc_dtype, shape)` — HMMA instruction
- `make_composed_layout(swizzle, offset, layout)` — bank-conflict-free SMEM
- `LdMatrix8x8x16bOp` — warp-level SMEM→register load
- `make_tiled_copy_A/B(atom, mma)` — derive S2R copy from MMA layout

### Step 5: Full Optimization
- 128-bit vectorized cp.async (8 × FP16 per copy)
- Register double-buffering (prefetch k+1 while computing k)
- SMEM epilogue staging for coalesced global writes
- `retile()` — match ldmatrix layout to MMA fragment layout
