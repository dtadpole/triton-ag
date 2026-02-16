# Conv Transpose Reference
<!-- Updated: 2026-02-15 | Source: chain_20260215_152436_b0 -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify this section. -->

## Tier 1: Algorithm Alternatives

### level2: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool (1.833x, iter 6) -- Welford BN stats + fused normalize+pool on fp16
**Key insight**: Replace torch.batch_norm with Welford online stats computed directly on fp16 conv output, then fuse BN normalize + double pool into a single kernel that reads fp16 directly, eliminating the massive fp32 BN intermediate.
**What worked**: fp16 conv (no bias) -> Welford BN stats on fp16 data -> fused BN normalize + 4x4x4 pool reading fp16 with fp32 accumulation. Eliminates ~10ms of bandwidth cost from materializing 255M fp32 BN output elements. 1.833x speedup.

## Tier 2: Architecture Variants

### level2: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid (1.13x, iter 11) -- Online 2-pass softmax fused with all post-ops in single kernel
**Key insight**: Online 2-pass softmax (running max + sum_exp) fused with bias+scale+sigmoid in a single Triton kernel reading fp16 conv output directly eliminates intermediate softmax materialization that makes multi-kernel approaches 2-3x slower than PyTorch.
**What worked**: fp16 ConvTranspose2d via torch.convolution(bias=None) + single fused Triton kernel doing online softmax over C=128 channels + conv_bias add + post-op bias + scale(2.0) + sigmoid. Multi-kernel approach was 0.31-0.41x; single fused kernel reached 1.13x. Compute-bound ceiling from ConvTranspose2d (C_in=64) prevented reaching 1.3x.

## Tier 3-4: Tuning Guide

- **fp16 conv without bias**: Always run ConvTranspose with bias=None and handle bias in the fused Triton kernel. fp16 conv WITH bias is 20-40% slower. (Source: 91_ConvTranspose2d)
- **fp32 accumulation in fp16 pipelines**: Use fp32 accumulators for reductions (pooling, softmax sum_exp) even when reading fp16 data. Avoids precision failures while keeping bandwidth low. (Source: 72_ConvTranspose3d, 91_ConvTranspose2d)
- **Autotune block sizes for post-conv kernels**: Expanded autotune configs (varying BLOCK_SIZE, num_warps) added 0.02-0.11x on fused post-conv kernels. Worth trying when near the target. (Source: 91_ConvTranspose2d)
- **Minimize data passes**: Count memory passes over the conv output. Each pass over a large feature map (255M elements) costs ~3-5ms. Fuse operations to reduce pass count to 2 (stats + normalize+post-ops) or 1 where possible. (Source: 72_ConvTranspose3d)
- **Delegate conv to torch.convolution**: ConvTranspose itself is hard to beat; use torch.convolution for the conv and focus optimization on post-conv ops. (Source: 72_ConvTranspose3d, 91_ConvTranspose2d)

## Anti-Patterns

### level2: 91_ConvTranspose2d (0.31-0.41x, iters 0-7) -- Multi-kernel softmax cannot compete with PyTorch fused implementation
**Key insight**: Splitting softmax across separate kernels (max, subtract, exp, sum, normalize) materializes large intermediates and loses to PyTorch's fused implementation every time.
**Why it failed**: Each separate kernel pass reads/writes the full feature map. For softmax over 128 channels on a large spatial grid, the bandwidth cost of 5+ passes dwarfs the compute.
**Better approach**: Fuse entire softmax (online 2-pass: running max + sum_exp in one pass, then normalize in second pass) with all downstream ops into a single Triton kernel.

### level2: 91_ConvTranspose2d (0.856x, iter 12) -- fp16 ConvTranspose with bias is 20-40% slower
**Key insight**: Including bias in the fp16 ConvTranspose call changes the kernel dispatch path and adds 20-40% overhead.
**Why it failed**: cuDNN selects a slower kernel variant when bias is non-None for fp16 ConvTranspose2d.
**Better approach**: Call torch.convolution with bias=None and add the conv bias in the fused Triton post-processing kernel.

### level2: 91_ConvTranspose2d (0x, iter 13) -- static_range with large count causes compilation timeout
**Key insight**: tl.static_range(128) unrolls 128 iterations at compile time, causing Triton compilation to timeout.
**Why it failed**: Triton's compiler cannot handle 128-iteration static unrolling; the IR blows up.
**Better approach**: Use tl.arange-based vectorized loads instead of static_range for channel dimensions > 32.

## Decision Tree
1. Check Tier 1 algebraic shortcuts first: Can BN/layernorm stats + normalize be fused with downstream ops (pool, activation) to eliminate intermediate materialization?
2. If no shortcut: select from Tier 1-2 for explore phase -- try single-kernel fused approach for all post-conv ops (softmax, BN, pool, activations)
3. Exploit phase: apply Tier 3-4 tuning -- fp16 conv without bias, fp32 accumulation, autotune block sizes, minimize data passes
