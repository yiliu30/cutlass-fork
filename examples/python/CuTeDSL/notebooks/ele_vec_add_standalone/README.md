# CuTe DSL Elementwise Kernel Tutorial — Standalone Examples

Broken out from `ele_vec_add.py` into 5 standalone, runnable examples.
Study them in order — each builds on the previous one.

## Progression

| # | File | What It Teaches | Key New APIs |
|---|------|----------------|--------------|
| 1 | `01_naive_elementwise_add.py` | Baseline: 1 thread = 1 element | `@cute.kernel`, `@cute.jit`, `thread_idx/block_idx/block_dim`, `from_dlpack`, `cute.compile` |
| 2 | `02_vectorized_elementwise_add.py` | Vectorized 128-bit loads (4 elements/thread) | `cute.zipped_divide`, `.load()`, slice `(None, (mi, ni))` |
| 3 | `03_tv_layout_elementwise_add.py` | TV Layout: decouple thread mapping from data layout | `make_ordered_layout`, `recast_layout`, `make_layout_tv`, `cute.composition` |
| 4 | `04_tv_layout_block_remap.py` | Thread-block index remapping for row-major tensors | `cute.select`, `composition(tensor, (None, remap))` |
| 5 | `05_generic_elementwise_custom_ops.py` | Meta-programming: generic kernel with custom ops | `cutlass.Constexpr`, `List[cute.Tensor]`, `range_constexpr`, `make_identity_tensor`, `elem_less`, `cute.where` |

## How to Run

```bash
# Activate environment
source /mnt/disk1/yiliu7/sage/bin/activate

# Run examples one by one
python 01_naive_elementwise_add.py
python 02_vectorized_elementwise_add.py
python 03_tv_layout_elementwise_add.py
python 04_tv_layout_block_remap.py
python 05_generic_elementwise_custom_ops.py
```

## Concept Map

```
Example 1: Naive
  Each thread: 1 element, manual index math
  Problem: Low memory bandwidth utilization (Little's Law)
      |
      v
Example 2: Vectorized
  Each thread: 4 elements via 128-bit load/store
  Tool: zipped_divide(tensor, (1, 4)) to partition
  Problem: Index math still manual, layout tightly coupled
      |
      v
Example 3: TV Layout
  Decouple thread mapping from tensor layout:
    thr_layout: how threads are arranged (4 rows × 64 cols)
    val_layout: how many elements each thread handles (16 rows × 8 cols)
    tv_layout = make_layout_tv(thr, val)
  Two-level tiling: block-level (zipped_divide) + thread-level (composition)
      |
      v
Example 4: Block Remapping
  Same kernel, but remap block indices for row-major tensors:
    Default: blocks traverse column-first (bad for row-major)
    Remapped: blocks traverse row-first (better L2 cache hits)
      |
      v
Example 5: Generic + Custom Ops
  Meta-programming: pass any Python function as the GPU operation
  Variable input count via List[cute.Tensor]
  Boundary checking via coordinate tensors + predicates
  Examples: mul, mul_relu, lambda a,b: a+b
```
