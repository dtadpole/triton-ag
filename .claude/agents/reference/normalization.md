# Normalization Reference
<!-- Updated: 2026-02-14 | Source: 0212_v10_l1 -->

## Code Templates

No op-specific code templates. For Welford's LayerNorm kernel template, see `reference/reduction.md` (it's a reduction technique). For universal autotune configs and patterns, see `reference/common.md`.

## Tier 1: Algorithm Alternatives

### L1: 36_RMSNorm (1.58x, iter 14) -- 2D register tiling halves memory accesses
**Key insight**: 2D register tiling (BLOCK_S x BLOCK_F=64) loads all 64 features for a block of spatial positions into registers, enabling single-read compute+normalize. This halves global memory accesses compared to two-pass approaches.
**What worked**: Load 2D tile of (BLOCK_S, 64), compute sum_sq along feature axis with tl.sum, compute rsqrt, normalize in-place in registers, write. BLOCK_S autotune {16,32,64,128}. Feature stride = H*W = 262144 in NCHW, but 2D tiling loads contiguous spatial data for each feature.

## Tier 2: Architecture Variants

### L1: 40_LayerNorm (1.21x, iter 4) -- Multi-kernel for very large norm dim
**Key insight**: LayerNorm over 4M elements with only 16 rows needs multi-kernel for GPU utilization. Two-kernel approach: (1) parallel partial stats with 256 chunks, (2) fused reduce + normalize.
**What worked**: 256 chunks give 4096 blocks for stats phase. Each normalize block reads 256 partial sums from L1 cache. Single-kernel Welford only gets 0.4x (16 blocks for 16 rows).

## Tier 3-4: Tuning Guide

### L1: 34_InstanceNorm (1.04x, iter 1) -- Simple sum/sum-of-squares beats Welford
**Key insight**: For InstanceNorm2d on large tensors, simple sum/sum-of-squares approach is slightly faster than Welford due to simpler math per element. But fundamentally bandwidth-bound (3x memory traffic: 2 reads + 1 write).
**What worked**: Contiguous memory layout, autotune across block sizes 1024-8192. Only 2 iterations completed before eval server outage.

## Anti-Patterns

### L1: 33_BatchNorm (0.39x, iter 2) -- cuDNN BatchNorm unbeatable
**Key insight**: BatchNorm on large spatial inputs (64x64x512x512 = ~4GB) is extremely challenging. The stats computation requires reading all data twice, and PyTorch uses highly optimized fused CUDA kernels. Additionally, the eval server blocks "BatchNorm" strings everywhere including comments.
**Why it failed**: Single program per channel = only 64 programs for stats = terrible GPU utilization. Parallel reduction caused OOM. Welford compiled but was 2.5x slower than cuDNN.
**Better approach**: For pure BatchNorm with large spatial dims, accept failure. cuDNN's fused kernel is structurally superior.

### L1: 35_GroupNorm (1.02x, iter 4) -- Bandwidth-bound normalization on huge tensors
**Key insight**: GroupNorm on 7GB tensor (112x64x512x512) is fundamentally bandwidth-bound. Both implementations do 2-read + 1-write. The 3-kernel decomposition (per-channel stats -> per-group reduce -> per-channel apply) was best but only achieved 1.02x.
**Why limited**: fp16 conversion cost exceeds bandwidth savings, in-place writes fail correctness, cache hints useless for 7GB tensor (L2 is ~40MB), extensive autotune OOMs on busy GPUs.
**Better approach**: Accept ~1.0x for pure GroupNorm on huge tensors. 2D register tiling (like RMSNorm) might help if feature dim is small enough to fit in registers.

## Decision Tree

1. **RMSNorm with small feature dim (<=256)**: Tier 1 -- 2D register tiling (BLOCK_S x FEATURES). All features in registers. Expect 1.5x.
2. **LayerNorm with very large norm dim (4M+)**: Tier 2 -- Multi-kernel with chunked partial stats. Never single-kernel (insufficient parallelism). Expect 1.2x.
3. **LayerNorm with moderate norm dim**: Tier 3-4 -- BLOCK_SIZE must be >= D. Masked positions must be zeroed for variance. Expect 1.0-1.2x.
4. **InstanceNorm**: Tier 3-4 -- Sum/sum-of-squares (not Welford). Bandwidth-bound. ~1.0x.
5. **GroupNorm on huge tensors**: Anti-pattern -- 3-kernel (per-channel stats -> per-group reduce -> apply). ~1.0x ceiling.
6. **BatchNorm**: Anti-pattern -- cuDNN is unbeatable. Accept failure. Remember: eval server blocks "BatchNorm" strings.
7. **Key technique**: When features fit in registers, 2D register tiling halves memory accesses. This is the only way to significantly beat two-pass approaches.
