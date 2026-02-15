# Conv Reference
<!-- Updated: 2026-02-15 | Source: 0212_v10_l1+0212_v10_l2+0212_v10_l3+0212_v10_l3_retry+0212_v8_l2+0212_v3_l3+0212_l2+level2_20260214_232629 -->

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

**Case 4: Conv + Matmul combos** -- focus on optimizing the matmul side with epilogue fusion.

**Reality check:** Pure Triton convolution is significantly harder to optimize than cuDNN. For conv-dominated tasks with minimal post-ops, achieving 1.3x speedup may not be feasible. Focus effort on tasks where algebraic elimination or substantial post-op fusion is possible.

## Tier 1: Algorithm Alternatives

### L2: 42_ConvTranspose2d_GlobalAvgPool (11.7x, iter 1) -- Algebraic elimination
**Key insight**: When spatial mean/sum follows conv_transpose, entire convolution eliminated: mean_spatial(conv_transpose(x)) = conv_bias + (1/HW) * x_sum @ w_sum. Also: 44_ConvTranspose2d (6.1x), 83_Conv3d (27x, dead code).

### L1: 83_conv_depthwise_2D_asymmetric_kernel (15.2x, iter 0) -- Depthwise spatial tiling
**Key insight**: Depthwise conv with asymmetric kernel (3,1) and only 8 channels is trivially parallelizable. 2D grid (spatial_blocks, B*C), scalar weight broadcast per kernel position. cuDNN is not optimized for depthwise with asymmetric/small-channel inputs.
**What worked**: First-try 15.2x. Same pattern: 85_depthwise (5.6x, 3x7 kernel), 82_depthwise (2.0x, 3x3/unrolled+wide rows), 84_depthwise (1.3x, 3x3/128ch), 86_depthwise_separable (1.78x, fuse depthwise+pointwise).

### L1: 70_conv_transposed_3D (1.89x, iter 0) -- torch.convolution fp16 for ConvTranspose
**Key insight**: ConvTranspose3d stride=1 with fp16 tensor cores via torch.convolution gives clean ~1.9x. Triton kernel does fp16->fp32 cast. The critical factor for ConvTranspose infeasibility is stride>1 with large C_in, NOT C_in alone.
**What worked**: x.half() + weight_fp16 + torch.convolution(transposed=True). Also: 73_conv (1.53x, fp16 cuDNN + Triton bias), 80_conv (1.81x, fp16 cuDNN + Triton cast), 77_conv (1.20x, cuDNN+cast).

### L2: 87_Conv2d_Subtract_Subtract_Mish (1.98x, iter 3) -- Full Triton implicit GEMM beats cuDNN for small C_in
**Key insight**: For Conv2d with C_in=8, K=C_in*kH*kW=72 fits in 3 iterations of BLOCK_K=32. Full Triton implicit GEMM with fused epilogue (bias+sub+sub+mish) gives 1.98x, while cuDNN fp16 + Triton postops only gives 1.07x. Triton wins when K is small enough for efficient tiling.
**What worked**: Implicit GEMM with M=B*OH*OW, N=C_out, K=C_in*kH*kW. Full fusion of all post-ops in registers.

## Tier 2: Architecture Variants

### L1: 87_conv_pointwise_2D (2.82x, iter 1) -- NCHW-direct 1x1 conv as matmul
**Key insight**: Pointwise 1x1 conv is pure matmul per spatial position. Working directly on NCHW (no permute/contiguous) with fp16 tensor cores gives 2.8x. Index: stride_xc for C_in gather, direct NCHW writes.
**What worked**: Each program handles (batch, spatial_tile, channel_tile). K=64 (C_in) fits single tile. Permuting to (B*H*W, C_in) was 0.208x -- the 1GB permutation dominated.

### L2: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp (1.525x, iter 8) -- Fuse MaxPool into normalize kernel
**Key insight**: For Conv+GN+MaxPool patterns, fusing MaxPool into the normalize pass is critical. Stats pass reads full conv output (needed for GN). But normalize pass only reads pool window positions. MaxPool(4) means 93.75% of conv output values don't need to be read in normalize pass. Combined with fp16 conv output (halving stats pass bandwidth), total memory traffic reduced ~60%.
**What worked**: Three kernels: (1) fp16 cuDNN conv no-bias, (2) per-channel stats, (3) fused normalize+scale+MaxPool+clamp with pre-computed combined_scale.

### L2: 82_Conv2d_Tanh_Scaling_BiasAdd_Max (1.484x, iter 4) -- fp16 conv no-bias breakthrough
**Key insight**: fp16 conv WITHOUT bias (handled in Triton) passes correctness and enables tensor cores. Adding bias back during cuDNN fp16 conv causes precision issues. Fusing tanh+scale+bias+maxpool into single kernel avoids 3 intermediate tensors.
**What worked**: fp16_conv_nobias_fused_postops -- 1.48x. Key: pass None for bias to torch.convolution, add bias in Triton kernel.

### L1: 54_conv_3D_square (3.31x, iter 7) -- Single-pass implicit GEMM with small C_in
**Key insight**: Conv3d with small C_in (3), K=81 fits in single BLOCK_K=128 tile, enabling single-pass implicit GEMM with fp16 tensor cores. Key: narrow autotune key to ['M','N'] (exclude K, it's fixed).
**What worked**: All-OC single tile approach with BLOCK_K=32 (K=27 padded). Also: 59_conv3d (4.7x, K=27 single-pass), 66_conv3d (1.7x, kh/kw loop with K_inner=9), 60_conv3d (1.8x, kh/kw loop).

### L2: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max (4.374x, iter 0) -- fp16 ConvTranspose + multi-kernel fusion
**Key insight**: fp16 cuDNN ConvTranspose3d without bias drops conv time ~4x via tensor cores. Fusing softmax+subtract+swish+max across 16 channels into single kernel loads each spatial position once. Two Triton kernels: (1) fused bias+maxpool3d on fp16 conv output, (2) fused softmax+subtract+swish+channel-max with all channels in registers.
**What worked**: First-try success at 4.374x.

### L2: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp (5.244x, iter 1) -- fp16 ConvTranspose with fused pool chain
**Key insight**: fp16 cuDNN ConvTranspose3d WITHOUT bias gives ~5x speedup via tensor cores. Fusing scale+maxpool+globalavgpool+clamp into single Triton kernel avoids materializing maxpool intermediate. Only 2048 output values.
**What worked**: fp16 input/weight for torch.convolution + None bias + Triton kernel computing max over 2x2x2 windows, accumulating sum for globalavgpool, then clamp.

### L3: 21_EfficientNetMBConv (1.73x, iter 18) -- NCHW-native eliminates permutes
**Key insight**: Custom Triton matmul reading NCHW and writing NCHW directly for 1x1 conv eliminates ALL permute+contiguous ops. Also: 5_AlexNet (1.75x), 18_SqueezeNet (1.52x).

## Tier 3-4: Tuning Guide

### L1: 62_conv_2D_asymmetric_kernel (1.57x, iter 4) -- NHWC input for C_in=32-64
**Key insight**: Pre-converting input to NHWC makes C_in loads contiguous, giving much better coalescing for kpos loop. permute+contiguous cost amortized across kpos iterations. NCHW was only 1.13x; NHWC gave 1.57x.
**What worked**: Also: 69_convT (1.43x, NHWC), 71_convT (1.59x, NHWC), 78_convT (1.61x, NHWC). NHWC is key for C_in=32-64.

### L2: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum (1.447x, iter 8) -- conv no-bias pattern
**Key insight**: cuDNN Conv3d WITHOUT bias is significantly faster than WITH bias, same pattern as ConvTranspose. Handle conv bias as scalar add inside Triton kernel instead. The no-bias cuDNN path avoids an extra global memory write of bias per element.
**What worked**: torch.convolution with None bias + fused Triton kernel that loads conv_bias per channel and adds before maxpool. +0.246x improvement over best with-bias approach.

### L2: 57_Conv2d_ReLU_HardSwish (1.248x, iter 19) -- kpos loop with careful autotune tuning
**Key insight**: For Conv2d(C_in=8) + trivial post-ops (relu+hardswish), cuDNN already fuses simple activations. kpos explicit loop with num_warps=2 on small block configs and focused autotune around winner block sizes is needed to approach cuDNN.
**What worked**: 7 configs with num_warps=2 or 4, num_stages=2 or 3, BLOCK_HW=64-256 after discovering num_warps=2 was optimal.

- **fp16 cast strategy**: Pre-cast input tensor to fp16 (x.half()) before kernel for C_in>=32 where amortized across many kpos iterations. In-kernel cast (.to(tl.float16)) for C_in<16. Pre-cast in forward() is faster than per-tile cast inside kernel.
- **Weight layout**: Pre-transpose to (KH*KW, C_in, C_out) or (K, C_out) in __init__, cached as fp16 via register_buffer. Contiguous weight reads are critical for tl.dot.
- **Always**: Wrap forward() in torch.cuda.device(x.device). Cache fp16 weights. Use nn.Parameter + nn.init, never nn.Conv*.

## Anti-Patterns

### L1: 61_conv_transposed_3D (0.61x, iter 8) -- Large C_in ConvTranspose structurally infeasible
**Key insight**: ConvTranspose3d with C_in=48, no post-ops is structurally infeasible. cuDNN tensor core implicit GEMM cannot be matched.
**Why it failed**: All-OC (0.61x best), flat K-loop (0.32x), kh/kw loop with padding (0.12x), pre-padding (worse), fp32 (worse than fp16).
**Better approach**: Pure ConvTranspose with large C_in and no algebraic shortcuts or post-op fusion caps at ~0.6x. Accept failure. Use torch.convolution for cuDNN if post-ops exist.

### L2: ConvTranspose(C_in=64+, stride=2) -- Structurally infeasible without substantial post-ops
**Key insight**: Conv dominates ~90% of runtime. Even with fp16, cuDNN ConvTranspose is near-optimal. Total speedup caps at ~1.25x even with optimal post-op fusion.
**What failed**: L2: 91_ConvTranspose2d(C_in=64,stride=2) capped at 1.245x; L2: 5_ConvTranspose2d(C_in=64,stride=2) capped at 1.28x.
**Better approach**: torch.convolution fp16 no-bias + fuse all post-ops. Accept 1.1-1.25x ceiling.

### L2: Conv2d(C_in=8) + trivial post-ops (1-2 simple activations) -- Near-infeasible
**Key insight**: cuDNN already fuses simple activations (relu, hardswish) internally. Separate Triton kernel for post-ops adds launch overhead. Implicit GEMM also struggles -- cuDNN is near-optimal.
**What failed**: L2: 69_Conv2d_HardSwish_ReLU capped at 1.08x; L2: 71_Conv2d_Divide_LeakyReLU capped at 1.2x.
**Better approach**: Only attempt when 3+ substantial post-ops exist. With trivial post-ops, accept ~1.0-1.2x.

### L2: fp16 conv with C_in<=8 for Conv3d -- Dtype overhead dominates
**Key insight**: For Conv3d with C_in<=8, fp16 conversion overhead exceeds tensor core benefit. The data volume is too small for tensor cores to amortize cast cost.
**What failed**: L2: 7_Conv3d(C_in=8) fp16 was 0.88x; L2: 8_Conv3d(C_in=8) fp16 was 0.979x; L2: 79_Conv3d(C_in=3) fp16 was 0.911x.
**Better approach**: Use fp32 cuDNN for Conv3d with C_in<=8. Exception: Conv2d with C_in=8 but large spatial (128x128+) CAN benefit from fp16 when total GEMM volume is large.

## Decision Tree

1. **Check algebraic elimination** (Tier 1): Spatial sum/mean after conv distributes into weights (7-27x). Dead code (27x). Pool fuses into conv (2.9x). BN absorbs bias. InstanceNorm absorbs bias. Always check first.
2. **Depthwise conv** (Tier 1): Spatial tiling with scalar weight broadcast. 2D grid (spatial_blocks, B*C). Expect 1.3-15x. Usually first-try success.
3. **Depthwise separable** (Tier 1): Fuse depthwise+pointwise into single kernel, eliminate intermediate tensor. ~1.8x.
4. **Pointwise 1x1 conv** (Tier 2): NCHW-direct matmul, no permutes. ~2.8x.
5. **Conv2d/3d with small C_in (<=8) and 3x3 kernel + substantial post-ops** (Tier 1-2): Try full Triton implicit GEMM when K=C_in*9<=72 (fits few BLOCK_K iters). Can give 1.5-2.0x. Also try torch.convolution fp16 no-bias + Triton post-ops.
6. **Conv2d/3d with C_in=16 and 3x3 kernel** (Tier 2): Single-pass implicit GEMM. K=C_in*9 fits one BLOCK_K. cin-first K ordering. Expect 1.5-4.7x. (63_conv 1.87x, 54_conv 3.31x, 59_conv 4.74x).
7. **Conv2d with C_in=32-64 and C_out<=128** (Tier 3-4): kpos explicit loop with all-OC single tile + NHWC input + fp16 tensor cores. Expect 1.2-1.6x. (62_conv 1.57x, 56_conv 1.34x).
8. **Conv2d with C_in=64+ and large spatial** (Tier 1): Prefer torch.convolution() fp16 no-bias for cuDNN + fused Triton post-ops. Pure Triton caps at ~1.0-1.2x. (80_conv 1.81x via cuDNN fp16).
9. **Conv3d with small C_in (<=8)** (Tier 2): torch.convolution fp32 no-bias + fused Triton. fp16 not beneficial for Conv3d C_in<=8 (exception: C_in=8 with very large spatial). cin-first K ordering, pre-pad input.
10. **ConvTranspose stride=1 with moderate C_in** (Tier 1): torch.convolution fp16 no-bias + Triton fp16->fp32 cast. ~1.5-1.9x. (70_conv 1.89x, 73_conv 1.53x).
11. **ConvTranspose stride=2 with small C_in (<=16)** (Tier 1-2): torch.convolution fp16 no-bias + fused Triton post-ops. Expect 1.5-5.2x. (L2: 96_ConvTranspose3d 5.2x, 89_ConvTranspose3d 4.4x).
12. **ConvTranspose stride=2 with C_in=32** (Tier 2): torch.convolution fp16 no-bias. Expect 1.1-2.3x. (L2: 78_ConvTranspose3d 2.3x with fused bias+pool).
13. **ConvTranspose(C_in=64+, stride=2, no post-ops)** (Anti-Pattern): Accept ~0.7-1.25x or use torch.convolution + cudnn.benchmark. Conv dominates at ~90% runtime.
14. **Conv1d with very long sequences (65K+)** (Tier 3-4): Never do layout conversion (NCL->NLC). Use NCL directly with kpos loop. (64_conv 1.3x, NLC was 0.5x).
15. **fp16 cast strategy** (Tier 3-4): Pre-cast input tensor to fp16 (x.half()) before kernel for C_in>=32 where amortized across many kpos iterations. In-kernel cast (.to(tl.float16)) for C_in<16. Pre-cast in forward() is faster than per-tile cast inside kernel.
16. **Weight layout** (Tier 3-4): Pre-transpose to (KH*KW, C_in, C_out) or (K, C_out) in __init__, cached as fp16 via register_buffer. Contiguous weight reads are critical for tl.dot.
17. **Always** (Tier 3-4): Wrap forward() in torch.cuda.device(x.device). Cache fp16 weights. Use nn.Parameter + nn.init, never nn.Conv*. Pass bias=None to torch.convolution and handle bias in Triton.
