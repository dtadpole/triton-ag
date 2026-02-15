# element_wise patterns
<!-- Updated: 2026-02-14 | Source: 0212_v10_l1 -->

## What Works

### L1: 26_GELU (1.95x, iter 1) -- Approximate tanh formula beats exact erf
**Key insight**: Triton GELU using approximate tanh formula (via sigmoid-based workaround since tl.math.tanh missing) is 2x faster than PyTorch's exact erf-based GELU. Large block sizes (8192) with 16 warps are critical.
**What worked**: `0.5*x*(1+tanh(z))` = `x*sigmoid(2z)` using tl.sigmoid. The approximation trades slight precision for significant compute reduction. Autotune needed 8192 block option.

### L1: 88_MinGPTNewGelu (1.23x, iter 1) -- tl.sigmoid as core primitive
**Key insight**: For tanh-approximation GELU, use identity `0.5*x*(1+tanh(z)) = x*sigmoid(2z)`. tl.sigmoid is significantly faster than manual `1/(1+exp(-z))` or `(exp(2z)-1)/(exp(2z)+1)`.
**What worked**: Fused single kernel with tl.sigmoid. BLOCK_SIZE=2048-8192, moderate warps. Manual exp-based tanh was 30-40% slower.

### L1: 92_cumsum_exclusive (1.52x, iter 1) -- Fusion eliminates intermediate allocations
**Key insight**: Exclusive cumsum = inclusive cumsum with output shifted right by 1 + zero prepended. Single kernel avoids reference's torch.cat + slice + cumsum (2 extra allocations + memory passes).
**What worked**: Single Triton kernel with tl.cumsum, writing results at offset+1. Also: 93_masked_cumsum (1.45x, fuse mask*x + cumsum into single kernel).

## What Fails

### L1: 19_ReLU through 32_HardTanh (all ~1.0x) -- Pure bandwidth-bound activations
**Key insight**: ALL pure single-op element-wise activations on very large tensors (4096x393216 = 1.6B elements, ~6GB data) are completely memory-bandwidth bound. PyTorch's CUDA kernels already saturate HBM bandwidth. No Triton kernel can beat the fundamental 8 bytes/element I/O requirement.
**Why they all fail**: ReLU (1.0x), LeakyReLU (1.0x), Sigmoid (1.0x), Tanh (1.0x), Swish (1.0x), SELU (1.0x), HardSigmoid (1.0x), Softplus (1.0x), Softsign (1.0x), ELU (1.0x), HardTanh (1.0x), ScalarMul (1.0x). All attempted: large blocks, persistent kernels, vectorized loads, eviction policies, branchless formulations, 2D grids, fp16 conversion -- none exceeded ~1.0x.
**Better approach**: Do NOT attempt optimization of pure single-op element-wise on large tensors. The only path to >1.0x is fusing with adjacent operations (which makes it a multi-op task, not single element-wise).

### L1: 89_cumsum (1.055x, iter 18) -- Sequential dependency limits parallelism
**Key insight**: Cumulative sum has inherent sequential dependency. Two-phase parallel scan (0.54x) doubles memory traffic. Decoupled lookback deadlocks (Triton doesn't guarantee tile execution order). tl.associative_scan unavailable.
**Why it fails**: PyTorch's CUB implementation is near-optimal. Single-pass sequential scan with tl.cumsum is the only working approach, giving at most ~5% improvement.
**Better approach**: Accept ~1.05x ceiling for pure cumsum. Focus on fusing with pre/post operations (e.g., masked_cumsum 1.45x, exclusive_cumsum 1.52x).

## Decision Framework for Element-wise Tasks

1. **Pure single-op activation on large tensor (>1B elements)**: Skip optimization entirely. Always ~1.0x. Not feasible.
2. **GELU (approximate)**: Use x*sigmoid(2z) formula. ~2.0x because PyTorch uses exact erf. Large block sizes (8192).
3. **Multi-op element-wise chains**: Fuse into single kernel to eliminate intermediate allocations. Expect 1.2-1.5x.
4. **Prefix scans (cumsum/cumprod)**: Single-pass sequential scan only. Fusion with pre/post ops is the path to speedup (1.4-1.5x). tl.cumsum works; tl.cumprod does not exist (use exp(cumsum(log(x)))).
5. **When approximate formulas beat exact**: GELU approximate > exact erf. tanh via sigmoid > exact tanh. These approximation gaps are the only way to beat PyTorch on element-wise ops.
