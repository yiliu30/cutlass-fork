# BMG Flash Attention QK D-Loop Pipeline Handoff

## Objective

Test an isolated QK D-loop software pipeline based on the `unitrace` hotspot
at `0x20b0`. The implementation starts the D+1 Q/K block loads before issuing
the current D DPAS operations, while retaining only one reordered Q/K operand
pair.

## Configuration

- Example: `examples/06_bmg_flash_attention_pipeline`
- Target: `06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_pipeline`
- QK tile: `256x32x32`
- PV tile: `256x16x32`
- Pipeline stages: 2
- Workload: batch 1, 40 heads, sequence 32768, BF16 D128, non-causal
- Timing: 5 warm-up and 20 measured iterations

## Results

| Variant | Time (ms) | TFLOP/s | Correctness | Disposition |
|---|---:|---:|---|---|
| Generic one-slice look-ahead loop | 567.6258 | 38.741 | Not run | Rejected: compiler retained excessive live state |
| Explicit four-slice, two-buffer schedule | 313.5145 | 70.141 | Passed at sequence 512 | Rejected: bounded look-ahead fragments still increase pressure |

## Explicit schedule

The final experimental fork uses two raw Q fragments, two raw K fragments, and
one reordered Q/K pair. For each D slice, it starts the next raw Q/K loads,
executes the current DPAS group, and only then reorders the next operands.
This avoids retaining multiple reordered DPAS operands and removes all dynamic
loop control.

Despite that bounded state, throughput is 44.7% below the original
126.890 TFLOP/s baseline and 46.7% below the best PV-width-16 observation.
The compiler's original fully unrolled schedule is therefore substantially
better than source-level double buffering for this register-heavy kernel.

## Conclusion

Do not pursue Q/K register double buffering further without first reducing
other live state. The kernel already operates in 256-GRF mode, and adding even
one look-ahead Q/K pair reduces performance sharply. More promising follow-up
work is:

1. Shorten PV accumulator or softmax-state live ranges before adding QK
   look-ahead.
2. Use compiler scheduling controls or lower-level VISA scheduling to move
   existing loads rather than expressing extra C++ fragments.
3. Compare the generated ISA and spill/fill diagnostics against the original
   schedule to determine whether the regression is occupancy loss, spills, or
   additional synchronization.
