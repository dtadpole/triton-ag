# Normalization Reference
<!-- Updated: 2026-02-15 | Source: 0212_v10_l1+level2_20260214_232629 -->

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

### L2: GroupNorm with small group_size (<=32) -- Fits entirely in registers
**Key insight**: When group_size is small (e.g., 256 groups with 32 features/group for 8192 features), the entire group fits in registers. Load all group features, compute mean/var in one pass, normalize and apply affine in one pass. Zero global memory intermediate.
**What worked**: L2: 88_Gemm_GroupNorm_Swish_Multiply_Swish (8.1x), 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm (8.0x), 62_Matmul_GroupNorm_LeakyReLU_Sum (6.9x). Key: BLOCK_SIZE must exactly match group_size for correct variance computation.

### L2: Parallel split stats for BN/IN over large spatial -- N-way spatial decomposition
**Key insight**: For BatchNorm or InstanceNorm where spatial dim is 100M+ elements per channel, splitting spatial into N=16-32 chunks across separate programs gives much better parallelism. Each program computes partial sum/sum_sq, then a merge kernel combines them.
**What worked**: L2: 73_Conv2d went from 0.304x (single-program BN) to 0.987x (32-split parallel stats). Still <1.0x but a 3.2x improvement over naive approach. Key: precompute inv_count, use precomputed alpha=gamma*rstd and beta=bias-mean*alpha for normalize.

## Tier 3-4: Tuning Guide

### L1: 34_InstanceNorm (1.04x, iter 1) -- Simple sum/sum-of-squares beats Welford
**Key insight**: For InstanceNorm2d on large tensors, simple sum/sum-of-squares approach is slightly faster than Welford due to simpler math per element. But fundamentally bandwidth-bound (3x memory traffic: 2 reads + 1 write).
**What worked**: Contiguous memory layout, autotune across block sizes 1024-8192. Only 2 iterations completed before eval server outage.

### L2: Fuse MaxPool into normalize kernel for Conv+GN+MaxPool patterns
**Key insight**: In Conv+GroupNorm+MaxPool patterns, the stats pass must read the full conv output (GN needs all values), but the normalize pass only needs to read the pool window positions. MaxPool(4) means only 1/16 of conv output needs reading in normalize pass. Combined with fp16 conv output (halving stats bandwidth), total memory traffic reduced ~60%.
**What worked**: L2: 85_Conv2d (1.525x). Three kernels: (1) fp16 cuDNN conv no-bias, (2) per-channel stats from full conv output, (3) fused normalize+scale+MaxPool+clamp reading only pool window positions.

### L2: Pre-combine affine transforms to minimize per-element arithmetic
**Key insight**: When GroupNorm weight, external scale, and bias are all applied element-wise, precompute combined_scale = rstd * gn_weight * external_scale and combined_bias_offset to reduce per-element operations from 5-6 to 2 (multiply + add).
**What worked**: L2: 85_Conv2d, 79_Conv3d, 77_ConvTranspose3d. Small improvement (+0.01-0.03x) but free to implement.

## Anti-Patterns

### L1: 33_BatchNorm (0.39x, iter 2) -- cuDNN BatchNorm unbeatable
**Key insight**: BatchNorm on large spatial inputs (64x64x512x512 = ~4GB) is extremely challenging. The stats computation requires reading all data twice, and PyTorch uses highly optimized fused CUDA kernels. Additionally, the eval server blocks "BatchNorm" strings everywhere including comments.
**Why it failed**: Single program per channel = only 64 programs for stats = terrible GPU utilization. Parallel reduction caused OOM. Welford compiled but was 2.5x slower than cuDNN.
**Better approach**: For pure BatchNorm with large spatial dims, accept failure. cuDNN's fused kernel is structurally superior.

### L1: 35_GroupNorm (1.02x, iter 4) -- Bandwidth-bound normalization on huge tensors
**Key insight**: GroupNorm on 7GB tensor (112x64x512x512) is fundamentally bandwidth-bound. Both implementations do 2-read + 1-write. The 3-kernel decomposition (per-channel stats -> per-group reduce -> per-channel apply) was best but only achieved 1.02x.
**Why limited**: fp16 conversion cost exceeds bandwidth savings, in-place writes fail correctness, cache hints useless for 7GB tensor (L2 is ~40MB), extensive autotune OOMs on busy GPUs.
**Better approach**: Accept ~1.0x for pure GroupNorm on huge tensors. 2D register tiling (like RMSNorm) might help if feature dim is small enough to fit in registers.

### L2: GN variance bug with masked positions
**Key insight**: When BLOCK_SIZE does not exactly match group_size, masked positions in the tl.arange block can corrupt variance computation. Using tl.where with 0.0 for out-of-bounds loads but NOT excluding them from variance denominator gives wrong stats.
**What failed**: L2: 62_Matmul (correctness failure with max_diff=2.15 on first attempt). Fixed by using exact BLOCK_SIZE=16 matching group_size=16.
**Better approach**: Set BLOCK_SIZE equal to group_size (must be power of 2). If group_size is not power of 2, use next power of 2 and carefully mask variance denominator.

## Decision Tree

1. **RMSNorm with small feature dim (<=256)**: Tier 1 -- 2D register tiling (BLOCK_S x FEATURES). All features in registers. Expect 1.5x.
2. **GroupNorm with small group_size (<=32)**: Tier 2 -- All group features in registers. Single-pass compute+normalize. Expect 5-12x when fused with matmul (two-kernel).
3. **LayerNorm with very large norm dim (4M+)**: Tier 2 -- Multi-kernel with chunked partial stats. Never single-kernel (insufficient parallelism). Expect 1.2x.
4. **LayerNorm with moderate norm dim**: Tier 3-4 -- BLOCK_SIZE must be >= D. Masked positions must be zeroed for variance. Expect 1.0-1.2x.
5. **InstanceNorm**: Tier 3-4 -- Sum/sum-of-squares (not Welford). Bandwidth-bound. ~1.0x. InstanceNorm absorbs conv bias (algebraic insight).
6. **BN/IN over large spatial (100M+ per channel)**: Tier 3-4 -- Parallel split stats (16-32 splits). Still < 1.0x for pure BN but helps in composite patterns.
7. **GroupNorm on huge tensors**: Anti-pattern -- 3-kernel (per-channel stats -> per-group reduce -> apply). ~1.0x ceiling.
8. **BatchNorm**: Anti-pattern -- cuDNN is unbeatable. Accept failure. Remember: eval server blocks "BatchNorm" strings.
9. **Key technique**: When features fit in registers, 2D register tiling halves memory accesses. This is the only way to significantly beat two-pass approaches.
