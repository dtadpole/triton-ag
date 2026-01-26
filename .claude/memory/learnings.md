# Accumulated Learnings

This file captures insights from kernel optimization runs. Updated automatically by optimizer workers.

## Format

Each learning entry follows this structure:
```
## [Task Name] - [Date]
- **Operation type**: ...
- **Best strategy**: ...
- **Speedup achieved**: ...
- **Key insight**: ...
- **Failed approaches**: ...
```

## Learnings

(Entries will be added as optimization runs complete)

---

## Common Patterns Discovered

### Pattern: Memory Coalescing Critical for Large Tensors
- **Observed in**: ReLU, Sigmoid, element-wise ops
- **Finding**: Adjacent thread access to adjacent memory locations improves performance 2-3x
- **Apply when**: Tensor size > 1M elements

### Pattern: Block Size 1024 Often Optimal
- **Observed in**: Most element-wise operations
- **Finding**: 1024 threads per block provides best occupancy on modern GPUs
- **Exceptions**: Very small tensors (<1024 elements) or high register usage kernels

### Pattern: Warp-Level Reductions Faster for Small Dims
- **Observed in**: LayerNorm, attention
- **Finding**: For reduction dims < 1024, warp primitives beat tree reduction
- **Apply when**: Reducing over sequence length or small hidden dims

### Pattern: Single-Pass Statistics Always Wins
- **Observed in**: All normalization ops
- **Finding**: Welford's algorithm or online statistics beat two-pass approaches
- **Apply when**: Computing mean + variance

---

## Failed Strategies Log

Track strategies that consistently fail to help future optimization:

| Strategy | Task Type | Failure Mode | Notes |
|----------|-----------|--------------|-------|
| (To be populated) | | | |

---

## Top Performers

Track best speedups achieved for reference:

| Task | Speedup | Strategy Used | Date |
|------|---------|---------------|------|
| (To be populated) | | | |
