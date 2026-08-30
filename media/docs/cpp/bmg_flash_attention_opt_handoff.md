# BMG Flash Attention Optimization Handoff

This log tracks up to 20 measured optimization attempts for the isolated
`examples/06_bmg_flash_attention_opt` fork.

## Success criterion

- GPU: Intel Arc Pro B70 (`intel_gpu_bmg_g31`)
- Workload: batch 1, 40 Q heads, 40 KV heads, Q/KV sequence length 32768,
  BF16 input, head dimension 128, non-causal prefill
- Timing: 5 warm-up iterations and 20 measured iterations
- Target: at least 135 TFLOP/s

The original `examples/06_bmg_flash_attention` implementation is not modified.

## Benchmark command

```bash
./build/examples/06_bmg_flash_attention_opt/06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_opt \
  --batch=1 --num_heads_q=40 --num_heads_kv=40 \
  --seq_len_qo=32768 --seq_len_kv=32768 \
  --head_size_qk=128 --head_size_vo=128 \
  --scheduler=Individual --warmup=5 --iterations=20 --verify=0
```

## Attempts

| Try | Change | Time (ms) | TFLOP/s | Disposition |
|---:|---|---:|---:|---|
| 1 | Unmodified fork baseline: QK `256x32x32`, PV `256x32x32`, 16 subgroups, 2 stages, FP32 output | 173.3021 | 126.890 | Baseline |
| 2 | Increase Q/K prefetch distance from 2 to 3 stages | 175.0731 | 125.606 | Rejected: 1.0% slower |
| 3 | QK `128x64x32`, PV `128x32x64`, 8 subgroups | 356.7016 | 61.649 | Rejected: 51.4% slower |
| 4 | QK depth tile 64 (`256x32x64`) | 276.9626 | 79.398 | Rejected: 37.4% slower |
| 5 | BF16 output with baseline tile | 172.9694 | 127.134 | Rejected: negligible change |
| 6 | Default compiler GRF allocation instead of forced 256 GRFs | 173.3744 | 126.837 | Rejected: negligible change |
| 7 | Wider PV tile (`256x64x32`, 2 V tiles) | 185.3244 | 118.658 | Rejected: 6.5% slower |
| 8 | Narrower PV tile (`256x16x32`, 8 V tiles) | 167.1224 | 131.582 | Kept provisionally: 3.7% faster |
| 9 | PV tile width 8 (`256x8x32`, 16 V tiles) | 911.1355 | 24.135 | Rejected: pathological lowering |
| 10 | PV width 16 with 3 pipeline stages | 167.7078 | 131.122 | Rejected: slower than 2 stages |
| 11 | PV width 16 with 1 pipeline stage | 176.5613 | 124.547 | Rejected: insufficient prefetch distance |
| 12 | 32 QK subgroups with PV width 16 | 287.9546 | 76.367 | Rejected: excessive workgroup size |
| 13 | 8 QK subgroups with PV width 16 | 1558.1653 | 14.113 | Rejected: severe register/work imbalance |
| 14 | Q tile 128, K tile 32, 8 subgroups, PV width 16 | 245.3634 | 89.623 | Rejected: workgroup decomposition loses throughput |
| 15 | Cache all four reordered Q D-fragments outside the KV loop | 241.6650 | 90.995 | Rejected: added live ranges overwhelm the scheduler |
| 16 | Compile best PV-width-16 variant with `-ffast-math` | 176.3975 | 124.663 | Rejected: compiler scheduling regressed |
| 17 | Remove vector-alias threshold and implicit-local-ID IGC settings | 178.0734 | 123.490 | Rejected: 6.1% below best |
| 18 | PV width 16 with K depth 64 (`256x16x64`) | 383.8490 | 57.289 | Rejected: incompatible PV decomposition |
| 19 | PV width 16 with 4 pipeline stages | 176.5300 | 124.569 | Rejected: added prefetch depth increases pressure |
| 20 | Final PV-width-16 verification: three 5-warmup/20-iteration runs | 176.4521 avg | 124.624 avg | Final retained configuration; target not reached |

## Final status

The 135 TFLOP/s target was not reached within the 20-attempt limit. The best
single measured result was **131.582 TFLOP/s** in try 8, a 3.7% improvement
over the 126.890 TFLOP/s fork baseline. Repeating the retained configuration in
try 20 produced 124.581, 124.498, and 124.794 TFLOP/s (124.624 TFLOP/s
average), indicating the earlier gain was not stable under the later thermal or
clock state.

The retained source change is PV tile width 16:

```cpp
using ShapeQK = Shape<_256, _32, _32>;
using ShapePV = Shape<_256, _16, _32>;
using ShapeOut = Shape<_256, _128>;
using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
constexpr int PipelineStages = 2;
```

The fork builds as the independent target
`06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_opt`. A batch-1, 40-head,
sequence-512, D128 non-causal correctness run completed successfully. No
change was made to the original `06_bmg_flash_attention` source.

The experiments show that the remaining gap is not solved by tile geometry,
pipeline depth, output type, GRF mode, or retaining Q fragments. Further work
requires restructuring the QK instruction schedule around the measured
four-way dependency wait at `0x20b0`, while controlling register live ranges.

## Standalone XMX utilization profile

During follow-up profiling, the retained source was found to have its
decode/prefill pipeline constants inverted: prefill instantiated
`XeDefault<1>` rather than the two-stage configuration used by try 8. This
explains why the three-run "restoration" in try 20 reproduced the one-stage
result from try 11. The fork now correctly uses one stage for decode and two
stages for prefill.

The corrected PV-width-16, two-stage configuration was profiled independently
with:

```bash
unitrace --metric-query --group ComputeBasic \
  --output fmha-opt-stage2-compute-basic.metrics \
  ./examples/06_bmg_flash_attention_opt/06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_opt \
  --batch=1 --num_heads_q=40 --num_heads_kv=40 \
  --seq_len_qo=32768 --seq_len_kv=32768 \
  --head_size_qk=128 --head_size_vo=128 \
  --scheduler=Individual --warmup=5 --iterations=20 --verify=0
```

The profiled run achieved 129.831 TFLOP/s in 169.3753 ms. Time-weighted
ComputeBasic metrics across its 25 FMHA kernel instances were:

| Metric | Optimized PV16 | Historical PV32 profile | Difference |
|---|---:|---:|---:|
| ALU2/XMX utilization | **82.703%** | 77.29% | +5.413 points |
| XVE active | 68.709% | 66.35% | +2.359 points |
| XVE stall | 30.615% | 32.62% | -2.005 points |
| XVE thread occupancy | 99.271% | 98.92% | +0.351 points |
| Multiple-pipe active | 21.180% | 17.56% | +3.620 points |
| ALU0 utilization | 22.787% | 21.30% | +1.487 points |
| ALU1 utilization | 27.011% | 24.64% | +2.371 points |
| ALU0+ALU2 overlap | 10.334% | Not recorded | -- |
| ALU0+ALU1 overlap | 4.342% | Not recorded | -- |
| L3 stall | 0.054% | 0.037% | +0.017 points |
| GPU memory read | 10.837 GB/s | 10.16 GB/s | +0.677 GB/s |
| GPU memory write | 4.097 GB/s | 4.20 GB/s | -0.103 GB/s |
| Average GPU clock | 2411.606 MHz | 2643 MHz | -231.394 MHz |

The standalone measurement confirms that PV width 16 improves execution-pipe
use rather than memory behavior. XMX utilization and multi-pipe overlap both
increase, while occupancy remains effectively full and L3 stalls remain
negligible. The remaining limit is still dependency latency and incomplete
ALU/XMX overlap, not external bandwidth.

### Precise BF16 XMX profile

Because ComputeBasic names the aggregate counter `ALU2`, a second standalone
run used BMG's `VectorEngineProfile`, which exposes typed XMX events:

```bash
unitrace --metric-query --group VectorEngineProfile \
  --output fmha-opt-vector-engine.metrics \
  ./examples/06_bmg_flash_attention_opt/06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_opt \
  --batch=1 --num_heads_q=40 --num_heads_kv=40 \
  --seq_len_qo=32768 --seq_len_kv=32768 \
  --head_size_qk=128 --head_size_vo=128 \
  --scheduler=Individual --warmup=5 --iterations=20 --verify=0
```

The run achieved 129.782 TFLOP/s in 169.4401 ms. Across 25 FMHA instances:

| Counter | Total events |
|---|---:|
| `XVE_INST_EXECUTED_ALU2_ALL` | 1,073,741,824,000 |
| `XVE_INST_EXECUTED_XMX_BF16` | 1,073,741,824,000 |
| `XVE_INST_EXECUTED_XMX_FP16` | 0 |
| `XVE_INST_EXECUTED_XMX_INT2` | 0 |
| `XVE_INST_EXECUTED_XMX_INT4` | 0 |
| `XVE_INST_EXECUTED_XMX_INT8` | 0 |
| GPU core clocks | 10,150,649,249 |

For this kernel, every counted ALU2 execution slot is therefore a BF16 XMX
instruction. This is measured equality, not a general claim that ALU2 always
means XMX.

ComputeBasic's reported ALU2 utilization follows:

```text
utilization = 100 * ALU2 events / (128 * GPU core clocks)
```

The factor 128 was verified independently for each ComputeBasic instance by
solving its reported ALU2 event, clock, and utilization counters. Applying the
same execution-slot normalization to the typed BF16 XMX events gives:

```text
BF16 XMX utilization
  = 100 * 1,073,741,824,000 / (128 * 10,150,649,249)
  = 82.641098%
```

Thus the precise standalone result is **82.6411% BF16 XMX utilization**. It is
consistent with the separate ComputeBasic run's 82.703% ALU2 result.

Raw metrics:

```text
/root/.copilot/session-state/e6e22079-1357-41fd-a116-dec9b41ea899/files/
  fmha-opt-stage2-compute-basic.metrics.3418004.metrics
  fmha-opt-vector-engine.metrics.3438226.metrics
```

## Follow-up experiments

| Try | Change | Time (ms) | TFLOP/s | Disposition |
|---:|---|---:|---:|---|
| 21 | Load Q once into four packed BF16 copy fragments; reorder one D slice inside each K iteration | 220.8612 | 99.566 | Rejected: correct, but persistent packed Q live ranges reduce throughput by 21.5% versus baseline |

Try 21 passed the sequence-512 correctness check. It performed better than
caching four reordered DPAS-A fragments (try 15: 90.995 TFLOP/s), confirming
that packed storage lowers pressure, but it remains substantially slower than
reloading Q. The compiler benefits more from ending each Q fragment's live
range within one D iteration than from eliminating repeated Q loads.

## Current consolidated status

Optimization continued in two additional isolated forks:

- `examples/06_bmg_flash_attention_pipeline`: QK D-loop software-pipeline
  experiments.
- `examples/06_bmg_flash_attention_schedule_trials`: independently buildable
  tries 22 through 39.

The best valid single measurement remains **try 27 at 131.636 TFLOP/s**
(167.0534 ms). Three repeat runs averaged **131.324 TFLOP/s**. Try 27 reverses
the QK head-dimension traversal, but the repeat result is within baseline
noise, so it is retained as the best measured implementation rather than a
proven scheduling improvement. It passes sequence-512 correctness.

No experiment reached the 135 TFLOP/s target. The remaining gap from the best
single result is 2.6%.

### Latest valid profile

Try 27 was collected with a true 5 ms ComputeBasic sampling interval. This
unitrace build interprets `-i` in microseconds, so the correct argument is
`-i 5000`, not `-i 5`.

| Metric | Try 27 |
|---|---:|
| Profiled performance | 130.741 TFLOP/s |
| XVE active | 70.599% |
| XVE stall | 29.245% |
| XVE occupancy | 99.793% |
| Multiple-pipe active | 21.542% |
| LSC access hold | 0.443% |
| DRAM read / write | 10.8 / 4.1 GB/s |
| ALU0 / ALU1 / ALU2 issue | 0.2320 / 0.2829 / 0.4181 inst/XVE/clock |
| Precise typed BF16 XMX utilization | 82.641% |

The kernel is compute/issue-bound rather than memory-bandwidth-bound.
Occupancy is effectively full and the activity profile is uniform until the
small final tail. Earlier instruction-level sampling attributed most remaining
opportunity to scoreboard and instruction-distance dependencies around QK
DPAS, not synchronization or external-memory stalls.

Fresh trace:

```text
/root/.copilot/session-state/e6e22079-1357-41fd-a116-dec9b41ea899/files/\
fmha-try27-compute-basic-5ms.metrics.3806642.csv
```

### Tries 22-39 summary

| Range | Experiments | Best/result |
|---|---|---|
| 22-31 | Q/K load, reorder, D traversal, and V-prefetch ordering | Try 27: 131.636 TFLOP/s |
| 32 | Load all V register fragments before softmax | 60.544 TFLOP/s; correct |
| 33-35 | Move accumulator rescaling before P conversion and vary PV order | 109.022-109.447 TFLOP/s; correct |
| 36 | Rescale each output tile before loading its V fragment | 123.931 TFLOP/s; correct |
| 37 | PV width 32 comparison | 127.073 TFLOP/s; correct |
| 38-39 | Fuse or manually reproduce the softmax P conversion | Rejected; incorrect |

PV16 and PV32 both compile with 256 GRFs, eight accumulator registers, 56
DPAS instructions, and zero spills/fills. PV32 emits fewer static instructions
(1,819 versus 1,857) but adds one synchronization instruction and uses a
different register-allocation schedule; it is 3.5% slower. Instruction count
and spilling therefore do not explain PV16's advantage.

Moving rescaling ahead of P conversion removes useful ALU/load/DPAS overlap,
while loading V early extends register live ranges. Both strategies sharply
regress. The compiler schedules the existing per-V-tile sequence better than
the manually serialized alternatives.

### Corrected `tSrS` to `tArP` mechanism

Compile-time dispatch probing shows:

```text
cute::Xe_Reorder<cute::ReorderKind::UU, float, cutlass::bfloat16_t>
```

For this PV16 kernel, `tSrS` and `tArP` have identical nominal subgroup TV and
per-thread fragment layouts. The operation is a unit-to-unit FP32-to-BF16
register conversion, not a cross-lane subgroup shuffle. CuTe's raw fragment
packing still cannot be reproduced safely with ordinary tensor indexing:
tries 38 and 39 both fail numerical verification.

This conversion touches only 32 values per work-item and is not supported by
the profile as the dominant bottleneck. Further mainloop-level attempts to
stream or duplicate it are low priority.

### Recommended continuation

1. Preserve the valid try 27 PV16 structure and immediate V-load-to-PV-DPAS
   consumption.
2. Work from generated ISA rather than C++ statement order. Adjust or guide
   SWSB dependency assignment around the QK and PV DPAS groups, then confirm
   reduced instruction-distance/scoreboard stalls.
3. Reduce FP32 ALU0/ALU1 work in softmax, especially repeated broadcasts,
   scaling, and reductions, without changing the verified CuTe P conversion.
4. Reject any change that adds persistent Q, K, V, P, or accumulator
   fragments, even if it appears to improve source-level overlap.
5. Require sequence-512 correctness first, then the exact 40-head 32K
   benchmark, followed by ComputeBasic and typed BF16 XMX profiling only for a
   reproducible winner.

Detailed implementations, commands, ISA artifacts, and per-try results are in
`media/docs/cpp/bmg_flash_attention_schedule_trials.md`.
