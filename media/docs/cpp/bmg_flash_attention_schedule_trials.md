# BMG Flash Attention Fine-Grained Scheduling Trials

This handoff records ten independently buildable scheduling implementations,
tries 22 through 31. Every implementation is preserved through the
`QK_SCHEDULE` compile definition in
`examples/06_bmg_flash_attention_schedule_trials/xe_fmha_fwd_mainloop_trials.hpp`.

## Success criterion

At least 135 TFLOP/s for batch 1, 40 Q/KV heads, sequence length 32768, BF16
D128 non-causal prefill, measured with 5 warm-up and 20 timed iterations.

## Results

| Try | Preserved implementation | Time (ms) | TFLOP/s | Disposition |
|---:|---|---:|---:|---|
| 32 | After masking, load all eight raw V fragments before softmax; reorder each immediately before PV DPAS | 363.2120 | 60.544 | Rejected; sequence-512 correctness passed |

Try 32 directly tests moving V register loads before softmax. It retains raw
BF16 V fragments rather than reordered DPAS-B fragments to minimize added
state, but their lifetimes still span softmax, P reorder, and prior PV tiles.
Performance falls 54.0% versus try 22. The current design—prefetch V before
softmax but load one V register fragment immediately before its PV use—is
strongly preferred.
| 22 | Baseline order: load Q, load K, reorder Q, reorder K, DPAS | 167.3047 | 131.438 | Reference |
| 23 | Load/reorder Q before loading/reordering K | 169.1761 | 129.984 | Rejected |
| 24 | Load/reorder K before loading/reordering Q | 167.6264 | 131.186 | Rejected |
| 25 | Load K, load Q, reorder K, reorder Q | 167.9410 | 130.940 | Rejected |
| 26 | Load Q, load K, reorder K, reorder Q | 168.9041 | 130.194 | Rejected |
| 27 | Traverse QK D slices in reverse order | **167.0534** | **131.636** | Best initial result; retained |
| 28 | Reverse V-prefetch order after QK | 167.6406 | 131.175 | Rejected |
| 29 | Interleave first four V prefetches with QK D iterations | 180.4823 | 121.842 | Rejected |
| 30 | Load K, load Q, reorder Q, reorder K | 167.4125 | 131.354 | Within baseline noise |
| 31 | Move all V prefetches before QK | 170.4174 | 129.037 | Rejected |

## Winner verification

Try 27 was repeated three times with the exact workload:

| Run | Time (ms) | TFLOP/s |
|---:|---:|---:|
| 1 | 167.4938 | 131.290 |
| 2 | 167.2393 | 131.490 |
| 3 | 167.6181 | 131.192 |
| **Average** | **167.4504** | **131.324** |

The sequence-512 correctness test passed. The repeated average is slightly
below try 22's single baseline result, so reverse traversal is not a confirmed
performance improvement despite producing the best initial measurement.

## Build and run

Every implementation remains independently buildable:

```bash
ninja -C build \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try22 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try23 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try24 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try25 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try26 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try27 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try28 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try29 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try30 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try31
```

The exact implementation is selected by each target's `QK_SCHEDULE` value in
`examples/06_bmg_flash_attention_schedule_trials/CMakeLists.txt`.

## Follow-up trials

| Try | Preserved implementation | Time (ms) | TFLOP/s | Disposition |
|---:|---|---:|---:|---|

## Conclusions

The 135 TFLOP/s objective was not reached. Simple source-order changes are
mostly absorbed or rescheduled by IGC, producing a narrow 129.984-131.636
TFLOP/s band. Moving V prefetches into the QK loop is actively harmful:
try 29 loses 7.3% because it adds SEND dependencies to the critical region.
Moving all V prefetches ahead of QK also loses 1.8%.

The results reinforce the instruction-level profile:

1. QK is sensitive to added SEND operations and live payloads.
2. Reordering existing Q/K copy and layout-conversion calls does not reliably
   change the generated critical path enough to improve throughput.
3. Further progress requires reducing an existing live range or modifying
   compiler/VISA scheduling—not adding source-level overlap.

Raw sweep output is retained at:

```text
/root/.copilot/session-state/e6e22079-1357-41fd-a116-dec9b41ea899/files/fmha-schedule-trials.txt
```

## Try 27 ComputeBasic profile

The best initial scheduling result was reprofiled with PTI `unitrace` on the
exact 40-head, 32K BF16 D128 non-causal workload. The profiled benchmark
reported 168.1974 ms and 130.741 TFLOP/s.

This `unitrace` build specifies `-i` in microseconds, so a 5 ms sampling period
requires `-i 5000`, not `-i 5`:

```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
sudo sysctl -w dev.xe.observation_paranoid=0

unitrace -k -g ComputeBasic -i 5000 -o fmha-try27-compute-basic-5ms.csv \
  build/examples/06_bmg_flash_attention_schedule_trials/\
06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try27 \
  --batch=1 --num_heads_q=40 --num_heads_kv=40 \
  --seq_len_qo=32768 --seq_len_kv=32768 \
  --head_size_qk=128 --head_size_vo=128 \
  --scheduler=Individual --warmup=5 --iterations=20 --verify=0
```

The trace contains all 25 launches and 1,177 samples. Truncated-launch
filtering leaves the complete set.

| Metric | Try 27 |
|---|---:|
| XVE active | 70.599% |
| XVE stall | 29.245% |
| XVE occupancy | 99.793% |
| Multiple-pipe active | 21.542% |
| GPU busy | 100.000% |
| LSC access hold | 0.443% |
| DRAM read / write | 10.8 / 4.1 GB/s |
| ALU0 issue rate | 0.2320 inst/XVE/clock |
| ALU1 issue rate | 0.2829 inst/XVE/clock |
| ALU2 issue rate | 0.4181 inst/XVE/clock |
| SEND issue rate | 0.0336 inst/XVE/clock |

ALU2 is the DPAS/XMX pipe for this workload. The four reconstructed rates sum
to 0.9665 issue slots per XVE clock. Interpreting that as 48.3% of a two-slot
ceiling assumes two issue slots; the trace itself only proves a width of at
least two. The separately collected typed BF16 XMX profile remains the precise
XMX utilization measurement: 82.641%.

The profile classifies this prefill kernel as compute/issue-bound:

- DRAM bandwidth and LSC request-port hold are negligible relative to hardware
  limits, so changing global-memory prefetch timing is not the primary lever.
- Occupancy is effectively full for this launch, so adding fragments to expose
  more parallelism is more likely to extend live ranges than improve residency.
- Activity is uniform for the first nine deciles. Only the final decile shows
  a small occupancy tail, so workgroup-tail optimization has limited upside.
- The 29.245% aggregate stall remains meaningful, but ComputeBasic cannot
  separate scoreboard, pipe, and instruction-distance stalls. The earlier
  instruction-level profile attributes most of this opportunity to dependency
  and instruction-distance waits around QK DPAS.

The metric schema on this driver exposes `GPU_MEMORY_L3_READ` rather than
`GPU_MEMORY_READ`; its DRAM events average about 252 bytes each. Therefore the
analysis script's generic 64-byte DRAM identity and comparisons with old traces
collected at a different sampling cadence are not used for conclusions here.

The full trace is retained at:

```text
/root/.copilot/session-state/e6e22079-1357-41fd-a116-dec9b41ea899/files/\
fmha-try27-compute-basic-5ms.metrics.3806642.csv
```

## Profile-driven next steps

1. **Stream the P conversion into PV.** The current code materializes all of
   `tArP` with `reorder(tSrS, tArP)` before any PV DPAS. Convert only the P
   fragment needed by the next PV operation, consume it immediately, and avoid
   retaining both complete `tSrS` and `tArP` representations across GEMM2.
2. **Shorten PV accumulator live ranges.** Split or drain `tArA` by V tile so
   fewer accumulator registers coexist with softmax and P fragments. This must
   preserve immediate V load-to-DPAS consumption; try 32 proved that moving V
   register loads earlier is harmful.
3. **Reduce scalar/FP32 issue work.** ALU0 plus ALU1 contributes 0.5149
   inst/XVE/clock, more than ALU2's 0.4181. Inspect generated ISA for the
   softmax reductions, exponentiation, rescaling broadcasts, and P-layout
   conversion. Prefer eliminating redundant moves or broadcasts over another
   source-order schedule.
4. **Compare PV16 and PV32 ISA and GRF allocation.** Record spill/fill counts,
   accumulator register ranges, DPAS-to-DPAS distance, and the instruction
   sequence generated for `tSrS` to `tArP`. This can identify why PV16 wins and
   whether a narrower streamed P fragment can retain that advantage.
5. **Reprofile structural variants.** Require a reduction in ALU0/1 issue work
   or XVE stall while keeping BF16 XMX activity and occupancy stable. Reject
   variants that merely move SENDs earlier or increase fragment lifetime.

## Structural follow-up: tries 33-37

Five independently buildable variants implement the profile-driven follow-up.
All five pass the sequence-512 correctness check.

| Try | Preserved implementation | Time (ms) | TFLOP/s | Delta vs try 27 |
|---:|---|---:|---:|---:|
| 33 | Rescale all eight PV accumulator tiles before creating `tArP` | 200.9213 | 109.447 | -16.9% |
| 34 | Try 33 with reverse PV tile traversal | 201.2117 | 109.289 | -17.0% |
| 35 | Try 33 plus a fresh P reorder immediately before every PV tile | 201.7055 | 109.022 | -17.2% |
| 36 | Keep full `tArP`, but rescale each accumulator tile before loading its V fragment | 177.4398 | 123.931 | -5.9% |
| 37 | Baseline schedule with PV tile width 32 instead of 16 | 173.0526 | 127.073 | -3.5% |

The baseline comparison is try 27's initial 131.636 TFLOP/s measurement. None
of the structural variants beats try 27, so no new `unitrace` profile was
collected.

Build the variants with:

```bash
ninja -C build \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try33 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try34 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try35 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try36 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try37
```

### Interpretation

Moving all accumulator rescaling ahead of P reorder is substantially worse,
even though it narrows the source-level overlap between `tArP` and rescaling.
This means the original per-V-tile sequence gives IGC more useful independent
work to interleave with V load/reorder and PV DPAS. Source-level live-range
separation does not translate to a better machine schedule.

Recomputing `tArP` for every PV tile does not shorten the allocated fragment
and adds redundant FP32-to-BF16 conversions. Delaying the V register load until after
rescaling also removes useful load/ALU overlap. These results reinforce the
earlier finding that manual serialization around PV is counterproductive.

### PV16 versus PV32 ISA

Matching non-causal, non-cache shader dumps were generated with
`IGC_VISAOptions=-perfmodel`.

| Property | PV16 / try 27 | PV32 / try 37 |
|---|---:|---:|
| GRFs | 256 | 256 |
| Accumulator registers | 8 | 8 |
| Spills / fills | 0 / 0 | 0 / 0 |
| Static instructions | 1,857 | 1,819 |
| DPAS instructions | 56 | 56 |
| `sync` instructions | 57 | 58 |
| Register allocator | graph-coloring RR/BC | graph-coloring FF/BC |

PV32 emits 2.0% fewer static instructions but runs 3.5% slower. Its regression
is therefore not caused by spills or instruction count. Changing the PV width
changes fragment layout, register assignment, and dependency scheduling; the
extra synchronization and less favorable DPAS dependency spacing outweigh the
smaller instruction stream.

The dumps and compiler reports are retained under:

```text
/root/.copilot/session-state/e6e22079-1357-41fd-a116-dec9b41ea899/files/\
igc-build-try27
/root/.copilot/session-state/e6e22079-1357-41fd-a116-dec9b41ea899/files/\
igc-build-try37
/root/.copilot/session-state/e6e22079-1357-41fd-a116-dec9b41ea899/files/\
fmha-structural-trials.txt
```

The retained best implementation remains try 27. Further work should preserve
the original per-V-tile overlap and operate below the C++ statement-order
level: tune SWSB dependencies around the existing P conversion/PV DPAS
sequence or reduce the FP32 softmax instruction stream. Merely moving the
existing conversion or rescaling loops does not improve the generated
schedule.

## Fused softmax/conversion experiments

Tries 38 and 39 investigated fusing the post-softmax P conversion into the
exponentiation loop. Both implementations are preserved, but neither is
numerically valid and neither replaces try 27.

Compiler type probing establishes that the exact production dispatch is:

```text
cute::Xe_Reorder<cute::ReorderKind::UU, float, cutlass::bfloat16_t>
```

This corrects the earlier interpretation that the PV16 operation performs a
cross-lane subgroup relayout. For this configuration, the nominal `tSrS` and
`tArP` subgroup TV layouts and per-thread fragment layouts are identical. The
operation is a unit-to-unit FP32-to-BF16 conversion implemented through
`Universal_Reorder_UU`.

CuTe still applies its conversion through raw register-fragment traversal.
Writing `tArP(i)` directly during softmax, writing through raw storage, and
reconstructing the projected-layout offset all fail the sequence-512
correctness test. Ordinary tensor indexing is therefore not a safe substitute
for the reorder primitive's register packing semantics.

| Try | Experiment | Correctness | 32K result |
|---:|---|---|---:|
| 38 | Write BF16 PV probabilities during FP32 exponentiation, using the projected layout mapping | Failed | 120.818 TFLOP/s |
| 39 | Replace `reorder()` with an explicit post-softmax `Universal_Reorder_UU` traversal | Failed | Not benchmarked |

Try 38's performance number is diagnostic only because its output is invalid.
The valid try 27 path remains:

```text
FP32 softmax and FP32 row-sum reduction
-> CuTe UU FP32-to-BF16 conversion
-> PV DPAS
```

Eliminating this pass safely requires a new CuTe conversion primitive whose
destination register packing exactly matches `partition_sg_fragment_A`, rather
than a mainloop-level indexing change. Since the conversion is only 32 values
per work-item and the valid profile is dominated by broader ALU/dependency
work, this is no longer the highest-confidence route to 135 TFLOP/s. The next
optimization should target the generated SWSB dependency distances around QK
and PV DPAS or reduce the FP32 softmax/broadcast instruction sequence while
retaining CuTe's conversion primitive.
