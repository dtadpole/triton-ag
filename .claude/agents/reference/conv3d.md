# Conv3d Reference
<!-- Updated: 2026-02-15 | Source: chain_20260215_152436_b0 -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify this section. -->

## Tier 1: Algorithm Alternatives

### level2: 83_Conv3d_GroupNorm_Min_Clamp_Dropout (15.86x, iter 0) -- Algebraic dead-code elimination
**Key insight**: min(x, 0.0) followed by clamp(min=0.0, max=1.0) always produces 0.0 -- the only value that is both <= 0 and >= 0 -- so the entire pipeline (conv, GN, min, clamp, dropout) is dead code.
**What worked**: Return a pre-allocated zero tensor matching output shape. 15.86x speedup by eliminating all computation.

### level2: 23_Conv3d_GroupNorm_Mean (1.738x, iter 1) -- Algebraic GroupNorm-mean fusion
**Key insight**: mean(GroupNorm(conv(x))) can be computed algebraically from per-channel sums and group statistics, eliminating the full normalize-then-reduce pass entirely.
**What worked**: fp16 cuDNN conv (bias=None), then two Triton kernels: (1) compute group stats + per-channel sums with bias accounted, (2) algebraic mean from sums + stats + affine params. 1.738x speedup by removing one full data pass.

## Tier 2: Architecture Variants

No entries yet.

## Tier 3-4: Tuning Guide

- **fp16 conv with bias=None**: Pass bias=None to cuDNN conv3d and account for bias manually in subsequent Triton kernels. Faster cuDNN execution since bias-free conv has a simpler code path. (Source: 23_Conv3d_GroupNorm_Mean)
- **Two-kernel group stats + algebraic reduce**: When conv is followed by GroupNorm + reduction (mean/sum), split into (1) group stats + channel sums kernel and (2) final algebraic combination kernel. Avoids writing normalized tensor to global memory. (Source: 23_Conv3d_GroupNorm_Mean)

## Anti-Patterns

### level2: 23_Conv3d_GroupNorm_Mean (0x, iter 0) -- GN stats on wrong tensor
**Key insight**: GroupNorm statistics must be computed on the tensor that GroupNorm will normalize -- including any bias that was part of the conv output.
**Why it failed**: Conv was run with bias=None for speed, but group stats (mean, var) were computed on the no-bias output. GN normalizes the with-bias output, so stats were wrong, causing max_diff=0.046.
**Better approach**: When using bias=None in conv, add the bias back in the stats-computation kernel before computing group mean/variance.

## Decision Tree
1. Check for algebraic shortcuts first: does the post-conv pipeline collapse to a constant or simpler expression? (e.g., min+clamp producing zeros, or redundant normalize+reduce)
2. If conv is followed by GroupNorm + reduction: compute GN stats and reduction algebraically in fused Triton kernels, skipping the full normalize pass
3. Exploit phase: use fp16 conv with bias=None, fuse post-conv ops into minimal Triton kernels
