# Pooling Reference
<!-- Updated: 2026-02-14 | Source: 0212_v10_l1 -->

## Code Templates

No op-specific code templates. See `reference/common.md` for universal patterns.

## Tier 1: Algorithm Alternatives

### L1: 41_Max_Pooling_1D (3.03x, iter 0) -- Simple spatial tiling beats cuDNN
**Key insight**: MaxPool1d with dilation>1 is a simple element-wise-like op where each output computes max over 8 non-contiguous positions. 2D grid (output_blocks, batch*channels) beats PyTorch's generic implementation. First-try success.
**What worked**: Each program handles a block of output positions for one (batch, channel) pair, iterating over kernel_size elements with dilation/padding handling. Also: 43_MaxPool3D (1.66x, flat spatial + 3x3x3 kernel).

### L1: 46_Average_Pooling_3D (2.0x, iter 0) -- Direct flat indexing with unrolled window
**Key insight**: Direct Triton avg pool3d with flat output indexing and unrolled 3x3x3 window is 2x faster than cuDNN for large 3D tensors. Clamped boundary handling for padding.
**What worked**: Flat index decomposition maps each thread to one output element. Unrolled window with mask-based boundary checks. First-try 2.0x.

## Tier 2: Architecture Variants

### L1: 42_Max_Pooling_2D (1.43x, iter 8) -- Flattened batch*channels + unit stride
**Key insight**: Flattening batch*channels into single contiguous dimension and using stride_w=1 for width access dramatically improves coalesced memory reads. Specialized kernel with 2D grid where pid encodes (oh, w_block).
**What worked**: Direct iw=ow-1+kw addressing, autotune BLOCK_W=64-512 and num_warps=2-8. Generic 4D decomposition was only 0.76-0.89x.

## Anti-Patterns

### L1: 45_Average_Pooling_2D (0x) -- Int32 overflow + compilation timeout
**Key insight**: Large tensors (16*64*2048*2048 = ~16GB) cause int32 pointer overflow AND large kernel sizes (11x11=121) cause compilation timeout from excessive unrolling.
**Why it failed**: Scalar per-element loop with int32 arithmetic: illegal memory access (>2B elements). constexpr kernel_size=11 causes 121 unrolled loads, Triton compilation >200s. Eval server was also down for most iterations.
**Better approach**: Use int64 pointer arithmetic (tl.cast to tl.int64). For large pool windows, accumulate row-sums first. Avoid constexpr for kernel_size>7.

### L1: 44_Average_Pooling_1D (1.07x, iter 11) -- cuDNN already efficient
**Key insight**: AvgPool1d with stride=1 and large input (65536) is well-optimized by cuDNN. Interior/boundary splitting (no bounds checks in hot path) gives ~6% but the fundamental bottleneck is 8 reads per output.
**Why limited**: Running sum (30x slower -- Triton not designed for sequential loops), explicit padding (4ms copy overhead), fp16 (cost > savings), manual unrolling (register pressure), flat 1D grid (division cost).
**Better approach**: Interior/boundary split + 2D grid (batch*channels, output_blocks) + small blocks (256-512). Accept ~1.07x ceiling.

## Decision Tree

1. **MaxPool (any dim)**: Tier 1 -- Flat spatial + per-element max over kernel window. 2D grid (spatial_blocks, batch*channels). Expect 1.4-3.0x. Usually first-try success.
2. **AvgPool3D (small kernel)**: Tier 1 -- Direct flat indexing with unrolled window. ~2.0x.
3. **AvgPool1D/2D (stride=1)**: Anti-pattern -- Interior/boundary split. ~1.07x ceiling (cuDNN already efficient).
4. **Large pool windows (11x11+)**: Anti-pattern -- Do NOT use constexpr for kernel_size (compilation timeout). Accumulate row-sums first to reduce iterations.
5. **Large tensors (>2GB)**: MUST use int64 pointer arithmetic. tl.cast to tl.int64.
6. **CUDA grid limits**: Grid dim z max = 65535. Flatten batch*channels into lower dimensions.
7. **Key technique**: Tier 2 -- Flatten batch*channels, use unit-stride width access for coalesced reads. Avoid 4D index decomposition (division/modulo overhead).
