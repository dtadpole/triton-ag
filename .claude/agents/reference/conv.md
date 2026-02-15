# Conv Reference
<!-- Updated: 2026-02-14 | Source: 0212_v10_l1+0212_v10_l2+0212_v10_l3+0212_v10_l3_retry+0212_v8_l2+0212_v3_l3+0212_l2 -->

## Code Templates

### 2D Spatial Autotune Config (post-conv processing)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=4),
    ],
    key=['C', 'HW'],
)
```

### Conv2d Decision Tree

**Note:** All PyTorch compute operations including `F.conv2d`, `F.conv_transpose2d`, `torch.matmul`, etc. are banned in `forward()` (rule 4). All convolution computation must be implemented in Triton.

**Case 1: Conv can be eliminated algebraically (check FIRST)**
- Spatial mean/sum after conv distributes into weights (7-14x speedup)
- AvgPool commutes with affine transforms
- Dead code elimination (output not used, or collapses to constant)
- See algebraic reasoning section in optimizer

**Case 2: Direct Triton convolution (im2col + matmul)**
- Use im2col to unfold input, then Triton tiled matmul on the unfolded matrix
- Fuse bias + activation into the matmul epilogue
- Best for tasks where post-conv ops are substantial enough to offset the im2col overhead

```python
class ModelNew(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.weight = torch.nn.Parameter(conv.weight.data.clone())  # (out_ch, in_ch, kH, kW)
        self.bias = torch.nn.Parameter(conv.bias.data.clone())
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        B, C_in, H, W = x.shape
        C_out, _, kH, kW = self.weight.shape
        H_out = H + 2 * self.padding - kH + 1
        W_out = W + 2 * self.padding - kW + 1
        # im2col via torch.as_strided or a Triton kernel to unfold patches
        # Reshape weight to (C_out, C_in*kH*kW) and do Triton matmul
        w_col = self.weight.view(C_out, -1)  # (C_out, C_in*kH*kW)
        # ... Triton matmul with epilogue fusion for bias + activation
```

**Case 3: Sliding-window Triton kernel (small kernels)**
- For small kernel sizes (1x1, 3x3), write a direct Triton kernel that computes convolution
- Each program handles one output spatial position, iterates over input channels and kernel elements
- Can fuse ALL post-ops into the same kernel

**Case 4: Conv + Matmul combos** — focus on optimizing the matmul side with epilogue fusion.

**Reality check:** Pure Triton convolution is significantly harder to optimize than cuDNN. For conv-dominated tasks with minimal post-ops, achieving 1.3x speedup may not be feasible. Focus effort on tasks where algebraic elimination or substantial post-op fusion is possible.

## Tier 1: Algorithm Alternatives

### L2: 42_ConvTranspose2d_GlobalAvgPool (15.8x, iter 1) -- Algebraic elimination
**Key insight**: When spatial mean/sum follows conv_transpose, entire convolution eliminated: mean_spatial(conv_transpose(x)) = conv_bias + (1/HW) * x_sum @ w_sum. Also: 44_ConvTranspose2d (4.1x), 83_Conv3d (27x, dead code).

### L1: 83_conv_depthwise_2D_asymmetric_kernel (15.2x, iter 0) -- Depthwise spatial tiling
**Key insight**: Depthwise conv with asymmetric kernel (3,1) and only 8 channels is trivially parallelizable. 2D grid (spatial_blocks, B*C), scalar weight broadcast per kernel position. cuDNN is not optimized for depthwise with asymmetric/small-channel inputs.
**What worked**: First-try 15.2x. Same pattern: 85_depthwise (5.6x, 3x7 kernel), 82_depthwise (2.0x, 3x3/unrolled+wide rows), 84_depthwise (1.3x, 3x3/128ch), 86_depthwise_separable (1.78x, fuse depthwise+pointwise).

### L1: 70_conv_transposed_3D (1.89x, iter 0) -- torch.convolution fp16 for ConvTranspose
**Key insight**: ConvTranspose3d stride=1 with fp16 tensor cores via torch.convolution gives clean ~1.9x. Triton kernel does fp16->fp32 cast. The critical factor for ConvTranspose infeasibility is stride>1, NOT C_in.
**What worked**: x.half() + weight_fp16 + torch.convolution(transposed=True). Also: 73_conv (1.53x, fp16 cuDNN + Triton bias), 80_conv (1.81x, fp16 cuDNN + Triton cast), 77_conv (1.20x, cuDNN+cast).

## Tier 2: Architecture Variants

### L1: 87_conv_pointwise_2D (2.82x, iter 1) -- NCHW-direct 1x1 conv as matmul
**Key insight**: Pointwise 1x1 conv is pure matmul per spatial position. Working directly on NCHW (no permute/contiguous) with fp16 tensor cores gives 2.8x. Index: stride_xc for C_in gather, direct NCHW writes.
**What worked**: Each program handles (batch, spatial_tile, channel_tile). K=64 (C_in) fits single tile. Permuting to (B*H*W, C_in) was 0.208x -- the 1GB permutation dominated.

### L2: 82_Conv2d_Tanh_Scaling_BiasAdd_Max (2.93x, iter 3) -- Fuse MaxPool into conv
**Key insight**: Fusing MaxPool(4) into conv kernel avoids materializing massive conv output. Each program computes 16 conv values (4x4 pool window) on-the-fly and takes max. Also: 50_ConvTranspose3d (1.45x), 96_ConvTranspose3d (1.96x).

### L1: 54_conv_3D_square (3.31x, iter 7) -- Single-pass implicit GEMM with small C_in
**Key insight**: Conv3d with small C_in (3), K=81 fits in single BLOCK_K=128 tile, enabling single-pass implicit GEMM with fp16 tensor cores. Key: narrow autotune key to ['M','N'] (exclude K, it's fixed).
**What worked**: All-OC single tile approach with BLOCK_K=32 (K=27 padded). Also: 59_conv3d (4.7x, K=27 single-pass), 66_conv3d (1.7x, kh/kw loop with K_inner=9), 60_conv3d (1.8x, kh/kw loop).

### L3: 21_EfficientNetMBConv (1.73x, iter 18) -- NCHW-native eliminates permutes
**Key insight**: Custom Triton matmul reading NCHW and writing NCHW directly for 1x1 conv eliminates ALL permute+contiguous ops. Also: 5_AlexNet (1.75x), 18_SqueezeNet (1.52x).

## Tier 3-4: Tuning Guide

### L1: 62_conv_2D_asymmetric_kernel (1.57x, iter 4) -- NHWC input for C_in=32-64
**Key insight**: Pre-converting input to NHWC makes C_in loads contiguous, giving much better coalescing for kpos loop. permute+contiguous cost amortized across kpos iterations. NCHW was only 1.13x; NHWC gave 1.57x.
**What worked**: Also: 69_convT (1.43x, NHWC), 71_convT (1.59x, NHWC), 78_convT (1.61x, NHWC). NHWC is key for C_in=32-64.

- **fp16 cast strategy**: Pre-cast input tensor to fp16 (x.half()) before kernel for C_in>=32 where amortized across many kpos iterations. In-kernel cast (.to(tl.float16)) for C_in<16. Pre-cast in forward() is faster than per-tile cast inside kernel.
- **Weight layout**: Pre-transpose to (KH*KW, C_in, C_out) or (K, C_out) in __init__, cached as fp16 via register_buffer. Contiguous weight reads are critical for tl.dot.
- **Always**: Wrap forward() in torch.cuda.device(x.device). Cache fp16 weights. Use nn.Parameter + nn.init, never nn.Conv*.

## Anti-Patterns

### L1: 61_conv_transposed_3D (0.61x, iter 8) -- Large C_in ConvTranspose structurally infeasible
**Key insight**: ConvTranspose3d with C_in=48, no post-ops is structurally infeasible. cuDNN tensor core implicit GEMM cannot be matched.
**Why it failed**: All-OC (0.61x best), flat K-loop (0.32x), kh/kw loop with padding (0.12x), pre-padding (worse), fp32 (worse than fp16).
**Better approach**: Pure ConvTranspose with large C_in and no algebraic shortcuts or post-op fusion caps at ~0.6x. Accept failure. Use torch.convolution for cuDNN if post-ops exist.

### L2: ConvTranspose3d(C_in=64+, stride=2) -- Structurally infeasible
**Key insight**: stride-2 in 3D wastes 87.5% of K-loop on alignment checks. cuDNN uses hardware-optimized implicit GEMM.
**Why it fails**: 100_ConvT (0.37x), 49_ConvT (0.22x), 3_ConvT (0.22x), 38_ConvT (0.39x), 72_ConvT (0.35x), 78_ConvT (0.12x).
**Better approach**: torch.convolution for cuDNN + fused Triton post-ops. Only ConvTranspose with small C_in (<=16) and K<=128 can beat cuDNN.

## Decision Tree

1. **Check algebraic elimination** (Tier 1): Spatial sum/mean after conv distributes into weights (7-27x). Dead code (27x). Pool fuses into conv (2.9x). Always check first.
2. **Depthwise conv** (Tier 1): Spatial tiling with scalar weight broadcast. 2D grid (spatial_blocks, B*C). Expect 1.3-15x. Usually first-try success.
3. **Depthwise separable** (Tier 1): Fuse depthwise+pointwise into single kernel, eliminate intermediate tensor. ~1.8x.
4. **Pointwise 1x1 conv** (Tier 2): NCHW-direct matmul, no permutes. ~2.8x.
5. **Conv2d/3d with small C_in (<=16) and 3x3 kernel** (Tier 2): Single-pass implicit GEMM. K=C_in*9 fits one BLOCK_K. cin-first K ordering. Expect 1.5-4.7x. (63_conv 1.87x, 54_conv 3.31x, 59_conv 4.74x).
6. **Conv2d with C_in=32-64 and C_out<=128** (Tier 3-4): kpos explicit loop with all-OC single tile + NHWC input + fp16 tensor cores. Expect 1.2-1.6x. (62_conv 1.57x, 56_conv 1.34x).
7. **Conv2d with C_in=64+ and large spatial** (Tier 1): Prefer torch.convolution() for cuDNN + fused Triton post-ops. Pure Triton caps at ~1.0-1.2x. (80_conv 1.81x via cuDNN fp16).
8. **Conv3d with small C_in (<=8)** (Tier 2): cin-first K ordering, pre-pad input, BLOCK_K=32. Single-pass when K<128. (54_conv 3.31x, 59_conv 4.74x).
9. **ConvTranspose stride=1 with moderate C_in** (Tier 1): torch.convolution fp16 + Triton fp16->fp32 cast. ~1.5-1.9x. (70_conv 1.89x, 73_conv 1.53x).
10. **ConvTranspose stride=1 with small C_in (<=32)** (Tier 2): Pure Triton kpos loop + fp16 tensor cores. ~1.3-3.0x. (64_conv 1.3x, 74_conv 1.84x, 79_conv 2.97x).
11. **ConvTranspose(C_in=48+, no post-ops)** (Anti-Pattern): Accept ~0.6x or use torch.convolution + cudnn.benchmark. (61_conv 0.6x, 68_conv 1.1x via cudnn.benchmark).
12. **Conv1d with very long sequences (65K+)** (Tier 3-4): Never do layout conversion (NCL->NLC). Use NCL directly with kpos loop. (64_conv 1.3x, NLC was 0.5x).
13. **fp16 cast strategy** (Tier 3-4): Pre-cast input tensor to fp16 (x.half()) before kernel for C_in>=32 where amortized across many kpos iterations. In-kernel cast (.to(tl.float16)) for C_in<16. Pre-cast in forward() is faster than per-tile cast inside kernel.
14. **Weight layout** (Tier 3-4): Pre-transpose to (KH*KW, C_in, C_out) or (K, C_out) in __init__, cached as fp16 via register_buffer. Contiguous weight reads are critical for tl.dot.
15. **Always** (Tier 3-4): Wrap forward() in torch.cuda.device(x.device). Cache fp16 weights. Use nn.Parameter + nn.init, never nn.Conv*.
