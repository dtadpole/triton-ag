# reduction patterns
<!-- Updated: 2026-02-14 | Source: 0212_v10_l1 -->

## What Works

### L1: 23_Softmax (1.32x, iter 2) -- Online 2-pass softmax
**Key insight**: Online softmax (combining max+sum in one pass via running max adjustment) saves one full memory traversal over naive 3-pass, critical for large dim (393216). Formula: s = s*exp(old_m-new_m) + sum(exp(x-new_m)).
**What worked**: Online softmax with inv_s = 1/s multiplication instead of division. More autotune configs with num_stages=2. Also: 24_LogSoftmax (1.33x, same online trick for logsumexp).

### L1: 51_Argmax_over_a_dimension (1.30x, iter 7) -- Coalesced tiling with unroll-by-4
**Key insight**: For argmax over a non-contiguous dimension, iterate one row at a time loading contiguous BLOCK_D2 elements per step, with unroll-by-4 to hide latency. Coalesced inner-dim access is the critical factor.
**What worked**: Each program handles BLOCK_D2 consecutive elements in contiguous dimension, iterating over reduction dim row-by-row. Unrolling by 4 gave ~10% improvement.

### L1: 93_masked_cumsum (1.45x, iter 0) -- Fusion eliminates intermediate tensor
**Key insight**: Fusing mask application (x * mask) with cumsum eliminates one full memory pass. Reference does x*mask (creates intermediate) then cumsum (reads back), 3 passes vs our 2.
**What worked**: Single Triton kernel: load x and mask, tl.where for masking, tl.cumsum with carry between blocks. First-try 1.45x.

## What Fails

### L1: 47_Sum_reduction (1.08x, iter 5) -- Bandwidth ceiling for large reductions
**Key insight**: For non-contiguous reduction over a middle dimension of 8GB tensor, PyTorch's optimized kernel already achieves ~71% of peak bandwidth. The 2D tiled approach with (BLOCK_REDUCE, BLOCK_INNER) is optimal but can only squeeze ~8% improvement.
**Why limited**: Transposing (0.089x -- 8GB copy), per-element (0.24x -- no coalescing), split-K with atomics (correctness failures from non-fresh output), two-pass with intermediate buffer (0.94x -- 32MB intermediate adds bandwidth). The fundamental bottleneck is reading 8GB of data at near-peak rate.
**Better approach**: Use (BLOCK_REDUCE=8-32, BLOCK_INNER=64-256) 2D tiles. Accept ~1.1x ceiling for pure sum/mean/max reduction of large tensors.

### L1: 89_cumsum (1.05x, iter 18) -- Sequential dependency kills parallelism
**Key insight**: Prefix scan has inherent sequential dependency. Two-phase parallel scan doubles memory traffic (0.54x). Decoupled lookback deadlocks (no tile ordering guarantee). tl.associative_scan unavailable.
**Why it fails**: CUB implementation is near-optimal for sequential scan. Only path to >1.0x is fusing with adjacent operations.
**Better approach**: Don't optimize pure cumsum. Focus on fusion with pre/post ops (masked_cumsum 1.45x, exclusive_cumsum 1.52x).

## Decision Framework for Reduction Tasks

1. **Softmax/LogSoftmax**: Online 2-pass (running max+sum) is mandatory for large dims. 33% bandwidth reduction over 3-pass. Expect 1.3x.
2. **Sum/Mean/Max over non-contiguous dim**: 2D tiled with (BLOCK_REDUCE, BLOCK_INNER) tiles. Coalesced inner-dim reads. NEVER transpose. Expect ~1.1x max.
3. **Argmax/Argmin over non-contiguous dim**: Same 2D tiled approach. Unroll-by-4 for latency hiding. Expect 1.3x.
4. **Prefix scans (cumsum)**: Single-pass sequential only. tl.cumsum is the only working primitive. Expect ~1.05x for pure cumsum. Fusion is the speedup path (1.4-1.5x).
5. **Cumprod**: Use exp(cumsum(log(x))). ~1.05x ceiling.
6. **Global reduction (Frobenius norm, L1/L2 norm)**: Single-kernel two-pass (compute, normalize) is best. Multi-kernel adds launch overhead. Expect 1.0-1.1x.
7. **Key anti-patterns**: Never transpose to make reduction contiguous (0.09-0.2x). Never use split-K with atomics on persistent output (correctness failure). Never use two-phase parallel prefix scan (0.54x doubles bandwidth).
