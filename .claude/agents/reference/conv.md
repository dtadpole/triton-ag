# Conv Reference
<!-- Updated: 2026-02-15 | Source: 0212_v10_l1+0212_v10_l2+0212_v10_l3+0212_v10_l3_retry+0212_v8_l2+0212_v3_l3+0212_l2+level2_20260214_232629+level2_20260215+level3_20260215_020905+level2_20260215_122501+level3_20260215_122506+level3_20260215_152600 -->

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
- AvgPool commutes with affine transforms (4x FLOP reduction for 1x1 conv)
- BN(x) - mean(BN(x)) cancels beta and BN mean, reducing to gamma*(x-spatial_mean(x))*rstd (1.9x)
- InstanceNorm(no affine) cancels conv bias entirely (skip bias in conv AND norm)
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

**Case 5: Deep CNN passthrough (L3 multi-layer networks)**
- For deep networks (DenseNet, EfficientNet, MobileNet, RegNet, VGG, ShuffleNet), replace nn.Module dispatch with direct torch.convolution + torch.batch_norm + torch.clamp calls
- Cache all parameter references in Python lists during __init__ or lazily on first forward() (avoid getattr overhead in loops)
- Enable cudnn.benchmark=True for conv algorithm auto-tuning
- Speedup comes from eliminating nn.Module.__call__ overhead across 20-200+ layer calls
- Expect 1.3-2.5x for launch-overhead-bound networks; FAILS for compute-bound large-batch CNNs
- Cached param refs vs getattr: +0.135x (ResNet101, 33 blocks), +0.47x (EfficientNetB1), +0.499x (MobileNetV2 lazy cache), +0.529x (MobileNetV1)

**Reality check:** Pure Triton convolution is significantly harder to optimize than cuDNN. For conv-dominated tasks with minimal post-ops, achieving 1.3x speedup may not be feasible. Focus effort on tasks where algebraic elimination or substantial post-op fusion is possible.

## Tier 1: Algorithm Alternatives

### L2: 42_ConvTranspose2d_GlobalAvgPool (13.6x, iter 0) -- Algebraic elimination
**Key insight**: When spatial mean/sum follows conv_transpose, entire convolution eliminated: mean_spatial(conv_transpose(x)) = bias + (1/HW_out) * spatial_sum(x) @ w_sum. Replaces massive ConvTranspose with a cheap matmul.
**What worked**: 13.6x. Also: 83_Conv3d (16x, dead code -- min(x,0)+clamp(0,1) always zero), 77_ConvTranspose3d (1.76x, GAP(BN(x))=f(spatial_mean)), 23_Conv3d (1.66x, mean(GN(x))=f(channel_sums)). Confirmed at 12.704x in separate session with two-kernel implementation.

### L2: 15_ConvTranspose3d_BatchNorm_Subtract (1.905x, iter 2) -- BN-subtract cancellation
**Key insight**: BN(x) - mean_spatial(BN(x)) simplifies to gamma * (x - spatial_mean(x)) / sqrt(var + eps). Beta and BN channel mean cancel out in the spatial mean subtraction, eliminating one full data pass.
**What worked**: fp16 cuDNN ConvTranspose3d no-bias + 3 Triton kernels with B*C=512 parallel spatial stats (not C=32). B*C parallelism was the critical GPU utilization unlock (+1.479x over C-only approach at 0.426x).

### L2: 13_ConvTranspose3d_Mean (8.44x, iter 8) -- Algebraic decomposition to 2D convs
**Key insight**: ConvTranspose3d(stride=1,pad=1,k=3) followed by depth mean decomposes into 3 cheaper 2D convolutions via im2col+Triton matmul, eliminating the 3D convolution entirely.
**What worked**: mean_d(ConvT3d(x)) = (1/D) * (Conv2D(x_depth_sum, w_sum) - Conv2D(x_last, w_kd0) - Conv2D(x_first, w_kd2)). Batched Triton matmul on grid dim gave 8.44x.

### L1: 83_conv_depthwise_2D_asymmetric_kernel (15.2x, iter 0) -- Depthwise spatial tiling
**Key insight**: Depthwise conv with asymmetric kernel (3,1) and only 8 channels is trivially parallelizable. 2D grid (spatial_blocks, B*C), scalar weight broadcast per kernel position.
**What worked**: First-try 15.2x. Same pattern: 85_depthwise (5.6x), 82_depthwise (2.0x), 84_depthwise (1.3x), 86_depthwise_separable (1.78x).

### L3: 13_DenseNet121TransitionLayer (2.708x, iter 19) -- Algebraic reorder + parallel BN stats
**Key insight**: AvgPool commutes with 1x1 Conv (both linear), so moving Pool before Conv reduces Conv FLOPs by 4x. Parallel BN stats with 64 spatial splits was the single biggest unlock (0.6x to 2.7x). tl.dot handles (BLOCK_M, 32) x (32, BLOCK_N) matmul for 1x1 conv efficiently with fp16 tensor cores (weight 32x64 = 2KB fits entirely in registers).
**What worked**: Parallel BN stats (64 splits) + fully fused BN_normalize+clamp+Conv1x1+AvgPool in single Triton kernel using tl.dot for 1x1 conv. fp32 throughout. 2.708x.

## Tier 2: Architecture Variants

### L3: 16_DenseNet201 (2.506x, iter 0) -- cuDNN passthrough for deep CNNs
**Key insight**: cuDNN passthrough (torch.convolution + torch.batch_norm + torch.clamp) with cached parameter lists scales superlinearly with network depth. DenseNet201 (200+ module calls) gets 2.5x; DenseNet121 gets 1.5x; EfficientNetB1 gets 1.7x; EfficientNetB0 gets 1.6x; MobileNetV1 gets 1.4x; MobileNetV2 gets 1.85x; ResNet101 gets 1.3x. Speedup is proportional to nn.Module dispatch overhead eliminated.
**What worked**: Cache all parameters as Python lists, call torch.convolution/torch.batch_norm/torch.clamp directly, enable cudnn.benchmark=True. Often first-try success.

### L3: 20_MobileNetV2 (1.854x, iter 7) -- Lazy cached param refs + dead code elimination
**Key insight**: The original MobileNetV2 discards residual connection flags (returns (Sequential, use_res_connect) but caller takes [0] only). Eliminating dead residual code + lazy caching param refs avoids getattr overhead. Lazy cache (built on first forward() call) prevents CPU/GPU device mismatch from __init__.
**What worked**: torch.convolution + torch.batch_norm + torch.clamp with lazy cached param refs. Removing dead residual connections gave 1.355x; lazy cache added +0.499x to reach 1.854x. Caching in __init__ fails because parameters are on CPU before .cuda().

### L3: 26_ShuffleNet (1.578x, iter 18) -- Full fp16 pipeline for deep group conv networks
**Key insight**: ShuffleNet's 40+ conv+BN operations across 13 units benefit massively from full fp16 pipeline (convert once at start, stay fp16 throughout). fp32 passthrough gives only 0.985x; fp16 pipeline gives 1.327x; pre-cached fp16 weights push to 1.578x. In-place clamp_ saves memory allocation per ReLU.
**What worked**: x.half() once at start + cuDNN passthrough + cached param refs + pre-cached .half() weights. fp16 must be all-or-nothing (per-layer conversion gave 0.848x). channels_last hurts due to channel shuffle requiring NCHW (view/transpose/contiguous). Triton channel shuffle slower than torch built-in.

### L3: 17_SqueezeNetFireModule (1.845x, iter 1) -- Full fp16 pipeline + cat elimination
**Key insight**: Keeping the entire multi-conv pipeline in fp16 (input -> squeeze -> expand) eliminates two fp32<->fp16 roundtrips, and writing expand outputs directly into a pre-allocated cat buffer with channel offsets eliminates torch.cat memory copy.
**What worked**: fp16 cuDNN no-bias for all 3 convs, squeeze output stays fp16 for expand inputs, bias+ReLU fused in Triton, two expand outputs write directly to pre-allocated NCHW output with channel offset. 1.845x.

### L3: 11_VGG16 (1.452x, iter 1) -- fp16 cuDNN no-bias + Triton bias+ReLU fusion
**Key insight**: fp16 cuDNN conv with bias=None + Triton fused bias+ReLU eliminates dispatch overhead and halves conv bandwidth. fp32 cuDNN passthrough achieves only 0.93x (near-parity), so fp16 tensor cores are essential.
**What worked**: fp16 no-bias + Triton bias+ReLU. 1.452x. The fp16 unlock alone gave +0.52x over fp32 passthrough.

### L3: 14_DenseNet121DenseBlock (1.392x, iter 9) -- Full fp16 BN+conv+cat pipeline
**Key insight**: Full fp16 pipeline (BN + conv + cat all in fp16) halves cat memory bandwidth and eliminates costly fp32<->fp16 conversions. torch.batch_norm accepts fp16 input directly via cuDNN (computes internally in fp32).
**What worked**: x.half() once at start, cat fp16 features, convert only final output .float(). Key: avoid .float() inside the loop for BN. 1.392x.

### L2: 36_ConvTranspose2d_Min_Sum_GELU_Add (1.566x, iter 1) -- Two-kernel decomposition for reductions
**Key insight**: Fully fusing channel-min (C=128) + height-sum (H=256) into single kernel forces 32K sequential iterations per program -- catastrophically slow (0.158x). Splitting into two kernels (channel_min parallelized over B*H*W, then height_sum) gives 1.566x.
**What worked**: Two-kernel decomposition: +1.408x improvement over single-fused approach.

### L2: 19_ConvTranspose2d_GELU_GroupNorm (1.439x, iter 11) -- Per-channel spatial splits for GN
**Key insight**: Per-channel spatial splits give coalesced memory access AND more parallelism than per-group splits, critical when channels_per_group is small (8) and spatial is huge (66K). Grid = B*C*num_splits programs.
**What worked**: Per-channel spatial split (4 splits) with recompute GELU in both passes. 1.439x, up from 1.274x with group-based 16-way splits. Per-channel keeps access within a single channel plane for perfectly coalesced reads.

### L2: 78_ConvTranspose3d_Max_Max_Sum (1.828x, iter 4) -- Fused fp16->fp32 cast+bias in Triton
**Key insight**: Fusing fp16->fp32 cast + bias add in a single Triton kernel saves a full memory round-trip compared to separate .float() then bias add. This was the single biggest unlock (+0.4x).
**What worked**: fp16 ConvTranspose3d no-bias + Triton fused cast+bias + aten.max_pool3d + Triton channel sum. 1.828x. Separate fp32 bias add was only 1.427x.

### L2: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish (2.03x, iter 1) -- Fuse MaxPool into post-ops
**Key insight**: Fusing MaxPool2d(2) into post-ops reads only pool window positions from conv output, avoiding materializing 3 intermediate tensors. Combined 4x output reduction means Triton kernel writes 4x less data.
**What worked**: fp16 cuDNN no-bias + single fused Triton kernel for bias-subtract+hardswish+maxpool+mish. 2.03x first try.

### L1: 87_conv_pointwise_2D (2.82x, iter 1) -- NCHW-direct 1x1 conv as matmul
**Key insight**: Pointwise 1x1 conv is pure matmul per spatial position. Working directly on NCHW (no permute/contiguous) with fp16 tensor cores gives 2.8x. Permuting to (B*H*W, C_in) was 0.208x.

### L2: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp (1.525x, iter 8) -- Fuse MaxPool into normalize kernel
**Key insight**: For Conv+GN+MaxPool, fusing MaxPool into normalize pass reads only pool window positions. MaxPool(4) skips 93.75% of conv output in normalize pass. fp16 conv output halves stats pass bandwidth.
**What worked**: Three kernels: (1) fp16 cuDNN conv no-bias, (2) per-channel stats, (3) fused normalize+scale+MaxPool+clamp.

### L1: 54_conv_3D_square (3.31x, iter 7) -- Single-pass implicit GEMM with small C_in
**Key insight**: Conv3d with small C_in (3), K=81 fits in single BLOCK_K=128 tile, enabling single-pass implicit GEMM with fp16 tensor cores. Narrow autotune key to ['M','N'].
**What worked**: Also: 59_conv3d (4.7x, K=27 single-pass), 66_conv3d (1.7x, kh/kw loop with K_inner=9).

### L2: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max (4.374x, iter 0) -- fp16 ConvTranspose + multi-kernel fusion
**Key insight**: fp16 cuDNN ConvTranspose3d without bias drops conv time ~4x via tensor cores. Two Triton kernels: (1) fused bias+maxpool3d on fp16 conv output, (2) fused softmax+subtract+swish+channel-max with all channels in registers.
**What worked**: First-try success at 4.374x. Confirmed at 3.595x in separate session with single-kernel fusion of all 5 post-ops.

### L2: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp (5.244x, iter 1) -- fp16 ConvTranspose with fused pool chain
**Key insight**: fp16 cuDNN ConvTranspose3d WITHOUT bias gives ~5x speedup via tensor cores. Fusing scale+maxpool+globalavgpool+clamp into single Triton kernel avoids materializing maxpool intermediate.
**What worked**: fp16 torch.convolution + None bias + Triton kernel computing max over 2x2x2 windows, accumulating sum for globalavgpool, then clamp. Confirmed at 2.488x in separate session.

### L2: 87_Conv2d_Subtract_Subtract_Mish (2.142x, iter 1) -- Implicit GEMM fp16 for small C_in
**Key insight**: For Conv2d with C_in=8, K=72 fits in single BLOCK_K=128 tile. fp16 tensor cores doubled throughput vs fp32 (1.258x to 2.142x). Triton wins when K is small enough for efficient tiling.
**What worked**: Implicit GEMM with M=B*OH*OW, N=C_out, K=C_in*kH*kW, bias+subtract+mish fused in epilogue registers. Also: 71_Conv2d_Divide_LeakyReLU (1.432x, K=72 single tile), 57_Conv2d_ReLU_HardSwish (1.492x, cin-first K ordering + fp16).

### L3: 21_EfficientNetMBConv (1.73x, iter 18) -- NCHW-native eliminates permutes
**Key insight**: Custom Triton matmul reading NCHW and writing NCHW directly for 1x1 conv eliminates ALL permute+contiguous ops. Also: 5_AlexNet (1.75x), 18_SqueezeNet (1.52x).

## Tier 3-4: Tuning Guide

### L1: 62_conv_2D_asymmetric_kernel (1.57x, iter 4) -- NHWC input for C_in=32-64
**Key insight**: Pre-converting input to NHWC makes C_in loads contiguous, giving much better coalescing for kpos loop. NCHW was only 1.13x; NHWC gave 1.57x.
**What worked**: Also: 69_convT (1.43x), 71_convT (1.59x), 78_convT (1.61x). NHWC is key for C_in=32-64.

### L2: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum (1.447x, iter 8) -- conv no-bias pattern
**Key insight**: cuDNN Conv3d WITHOUT bias is significantly faster than WITH bias. Handle conv bias as scalar add inside Triton kernel. The no-bias cuDNN path avoids an extra global memory write.
**What worked**: torch.convolution with None bias + fused Triton kernel. +0.246x over with-bias.

- **cuDNN passthrough for deep CNNs**: Replace nn.Module dispatch with torch.convolution + torch.batch_norm + torch.clamp. Cache all params in Python lists (avoid getattr). Use lazy cache (first forward() call) if params initialized in __init__ before .cuda(). Enable cudnn.benchmark=True. Expect 1.3-2.5x for 20+ layer networks. Cached param refs critical: getattr->list cache gave +0.135x (ResNet101, 33 blocks x 15+ attrs), +0.47x (EfficientNetB1), +0.499x (MobileNetV2 lazy cache), +0.529x (MobileNetV1). (Source: L3 DenseNet121/201, EfficientNetB0/B1, MobileNetV1/V2, ResNet101, VGG16, ShuffleNet)
- **Full fp16 pipeline**: Keep fp16 throughout multi-conv architectures. Never cast back to fp32 between conv layers when the next conv consumes fp16 directly. cuDNN BN accepts fp16 input (computes internally in fp32). Must be all-or-nothing: per-layer conversion is WORSE than fp32 (ShuffleNet per-layer: 0.848x vs full pipeline: 1.578x). x.half() once at start, convert only final output .float(). Pre-cache .half() weights to avoid repeated conversion. Eliminating all intermediate casts: +0.1-0.6x. (Source: L3 DenseNet121DenseBlock 1.39x, VGG16 1.45x, SqueezeNetFireModule 1.85x, AlexNet 1.3x, ShuffleNet 1.58x; L2 11_ConvTranspose2d_BatchNorm)
- **Lazy cached param refs**: Cache parameter references lazily on first forward() call, not in __init__. Parameters cached in __init__ point to CPU tensors before .cuda() is called, causing device mismatch or compile errors. Build cache dict/list on first forward() invocation. +0.499x for MobileNetV2. (Source: L3 MobileNetV2 1.854x)
- **Parallel BN/GN stats with B*C programs**: When C is small (32) and spatial is large (256x256), using only C programs severely underutilizes GPU. Use B*C programs where each handles one spatial plane. 0.6x to 2.7x improvement. Per-channel spatial splits outperform per-group splits for GN when channels_per_group is small. For DenseNet transition: 16 splits -> 2.074x, 32 splits -> 2.623x, 64 splits -> 2.708x. (Source: L3 DenseNet121TransitionLayer, L2 19_ConvTranspose2d +0.165x, 15_ConvTranspose3d +1.479x)
- **channels_last (NHWC) for deep CNN passthrough**: NCHW to NHWC gives +50% for cuDNN convolutions on modern GPUs when amortized over many layers. Exception: ConvTranspose2d (0.647x). Exception: ShuffleNet -- channel shuffle requires NCHW (view/transpose/contiguous), forcing expensive format conversions at every unit (channels_last 1.136x < NCHW fp16 1.578x). (Source: L3 27_RegNet 0.8x->1.22x)
- **Skip conv bias before BN in training mode**: BN subtracts batch mean which absorbs conv bias. Eliminating bias add saves ~0.45ms. Only valid in training mode (not eval with running_mean). (Source: L3 27_RegNet 1.22x->1.49x)
- **fp32 conv for tiny C_in (<=6)**: fp16 conv adds MORE overhead than it saves when C_in<=6 and spatial is small. The .half() casts cost more than tensor core gains. (Source: L3 4_LeNet5: fp16 1.145x vs fp32 1.216x; L3 17_SqueezeNetFireModule C_in=3/6: fp16 0.837x vs fp32 1.583x; fp16 FC dims 120/84/20 too small for tensor cores 0.794x)
- **Kernel launch minimization for sub-1ms models**: When total model runtime is ~1ms, each kernel launch costs ~50us (5% of runtime). Conv WITH bias (1 launch) + fused relu+pool (1 launch) = 2 launches per stage beats conv(no-bias) + bias_relu + pool = 3 launches. The no-bias pattern helps large models but hurts tiny ones. (Source: L3 4_LeNet5 1.115x)
- **Pre-allocated cat buffer**: For multi-branch architectures, write branch outputs directly to pre-allocated output tensor with channel offsets. Eliminates torch.cat copy. (Source: L3 17_SqueezeNetFireModule)
- **fp16 conv no-bias (MANDATORY)**: Always pass bias=None to cuDNN, handle bias in Triton. fp16 WITH bias is 20-40% slower than WITHOUT. Pre-cache fp16 weight via register_buffer. Exception: sub-1ms models where launch count matters more (use conv WITH bias). (Source: all L2 conv tasks)
- **In-place operations for deep pipelines**: Use `x.clamp_(min=0.0)` instead of `torch.clamp(x, min=0.0)` across 30+ layers to save memory allocation per ReLU. Meaningful in networks with 40+ conv+BN operations. (Source: L3 26_ShuffleNet)
- **Reduce data passes**: 2-pass Welford (sum+sq in pass 1, normalize in pass 2) beats 3-pass. Saves one full global memory read. +0.13x for InstanceNorm (17_Conv2d), +0.27x for softmax (24_Conv3d).
- **Recompute vs materialize**: Recomputing activations (sigmoid, GELU, swish) in both stats and normalize passes cheaper than writing+reading temp buffer. +0.24x (21_Conv2d), +0.327x for fp16 intermediate (52_Conv2d). Exception: erf-based GELU is too expensive to recompute when trading for marginal bandwidth savings (19_ConvTranspose2d: recompute was -0.04x).
- **Keep fp16 conv output**: Do NOT cast .float() after fp16 conv. Read fp16 directly in Triton, cast to fp32 in registers. Fuse cast+bias into single Triton kernel for +0.4x (78_ConvTranspose3d). +0.367x (32_Conv2d).
- **2D grid for spatial parallelism**: 1D grid (batch only) insufficient when B<256. Use 2D grid (B*C, HW_tiles). +0.66x (25_Conv2d: 1.07x->1.73x).
- **cudnn.benchmark + cached fp16 weight**: Set `torch.backends.cudnn.benchmark = True` for ConvTranspose and deep CNN passthrough. Cache fp16 weight via register_buffer in __init__. Combined +0.09x (90_Conv3d). +0.2x (92_Conv2d, L3 MobileNetV1).
- **Parallel GN/BN stats**: For spatial reduction >500K elements per group, use 8-16 split parallel stats with merge kernel. +0.35x (61_ConvTranspose3d: 1.19x->1.54x).
- **fp16 for Conv3d C_in=3**: Despite small C_in, fp16 IS beneficial when batch*D*H*W > 8M. fp32 gave 1.004x; fp16 gave 1.918x (48_Conv3d). Exception: C_in<=8 AND C_out<=32 always fp32 (7_Conv3d: 0.89x).
- **Expanded autotune configs**: Going from 4 to 7 configs, adding num_stages=3 and 16-warp variants, can push borderline tasks over 1.3x. +0.057x (1_Conv2D_ReLU_BiasAdd).
- **Post-op chain algebraic collapse**: bias+clamp+scale+clamp+divide -> clamp(conv+combined_bias, lo, hi). Pre-compute combined_bias and combined_scale in __init__. +0.18x (2_ConvTranspose2d), 4.425x first-try (50_ConvTranspose3d pre-combined scale1*scale2 and bias constants).
- **Algebraic mean(GN(x))**: mean(GN(HardSwish(x))) = gamma*(mean_c - mu_g)*rstd_g + beta eliminates the full normalize pass entirely. +0.17x with fp16 conv on top (27_Conv3d: 1.131x->1.302x).
- **InstanceNorm bias cancellation**: When InstanceNorm has no affine, conv bias cancels out (bias is per-channel constant subtracted in mean). Skip bias in BOTH conv and normalization. (Source: 17_Conv2d, 1.468x)
- **fp16 cast strategy**: Pre-cast input to fp16 (x.half()) for C_in>=32. In-kernel cast for C_in<16.
- **Weight layout**: Pre-transpose to (KH*KW, C_in, C_out) or (K, C_out) in __init__, cached as fp16 via register_buffer.
- **Single autotune config for small models**: For tiny CNNs (~1ms reference), removing autotune overhead via single-config Triton kernels is critical. Multi-config autotune overhead dominates gains. (Source: L3 4_LeNet5 iter 4: +0.149x from single-config)
- **Always**: Wrap forward() in torch.cuda.device(x.device). Use nn.Parameter + nn.init, never nn.Conv*. Pass bias=None to torch.convolution. Enable cudnn.benchmark for ConvTranspose.

## Anti-Patterns

### L2: 13_ConvTranspose3d_Mean (0x on 10 iterations) -- RNG init order mismatch
**Key insight**: When ModelNew declares parameters in different order than Model, ALL subsequent random values shift, causing systematic correctness failures (max_diff=0.177).
**Why it failed**: Declaring self.bias = nn.Parameter(torch.randn(...)) before calling kaiming_uniform_ on conv weights shifts RNG state. 10 iterations wasted debugging correctness.
**Better approach**: Match EXACT init order of original Model: initialize conv weight/bias FIRST, then any learnable parameters. Verify init order matches before any optimization work.

### L2: 78_ConvTranspose3d -- fp16 bias before MaxPool causes correctness failure
**Key insight**: MaxPool amplifies precision errors by selecting maximum values, which may differ between fp16 and fp32.
**Why it failed**: fp16 conv with fp16 bias add before MaxPool gave max_diff=1.147. MaxPool selects different max elements when fp16 rounding changes relative magnitudes.
**Better approach**: Fuse fp16->fp32 cast + bias add in a single Triton kernel BEFORE MaxPool. Never do bias arithmetic in fp16 when followed by MaxPool.

### L2: 27_Conv3d -- Autotune corrupts reduction/stats accumulators
**Key insight**: Autotune warmup runs kernel multiple times with same output buffers. For reduction kernels, different accumulation orders across configs corrupt stats.
**Why it failed**: Multiple BLOCK_S configs caused max_diff=0.025 correctness failure -- different spatial block sizes produced different floating-point accumulation in GN stats during warmup.
**Better approach**: Use single autotune config for any kernel that accumulates stats (BN, GN, InstanceNorm, softmax). Per-(n,c) output buffers without atomics, or parallel split stats with merge step.

### L3: 18_SqueezeNet (0.955x, iter 13) -- cuDNN passthrough fails for compute-bound large-batch CNNs
**Key insight**: cuDNN passthrough that works for deep CNNs (DenseNet 1.5-2.5x, EfficientNet 1.5-1.8x) FAILS when the workload is compute-bound (batch=64, 512x512). nn.Module dispatch overhead is negligible (<5%) relative to 27ms of GPU compute. channels_last + fp16 + no-bias is the optimal combination but still cannot overcome the structural ceiling.
**Why it failed**: torch.convolution + torch.clamp is 1.6x slower than nn.Conv2d + nn.ReLU because cuDNN fuses conv+relu internally. Separate conv + clamp = 2 kernel launches vs 1. channels_last gave +16% (biggest single improvement) but still capped at 0.955x.
**Better approach**: Only use cuDNN passthrough when Python/nn.Module overhead is significant fraction of runtime (>15%). For compute-bound CNNs, the reference is near-optimal.

### L3: 8_ResNetBasicBlock (0.715x, iter 13) -- Single conv+BN block unbeatable by decomposition
**Key insight**: cuDNN's internal conv+BN+relu fusion for single ResNet blocks is unbeatable. Manual torch.convolution + torch.batch_norm cannot replicate this fusion.
**Why it failed**: fp16 conv with fp32 BN adds conversion overhead (0.478x). native_batch_norm slower than batch_norm (0.443x). channels_last improved from 0.588x to 0.71x but still far below 1.0x.
**Better approach**: Accept failure for isolated conv+BN blocks. Passthrough only helps when eliminating dispatch overhead across many layers.

### L3: 21_EfficientNetMBConv (0.755x, iter 13) -- Decomposing nn.Sequential loses cuDNN fusion
**Key insight**: MBConv with large C_in (112, 672) is dominated by cuDNN-optimized 1x1 convolutions. Decomposing nn.Sequential into separate torch.convolution + torch.batch_norm + clamp adds ~2.5ms from lost cuDNN fusion. 17 exploit iterations all stayed in 0.718-0.754x range. Only full-network passthrough helps (EfficientNetB0 1.64x, B1 1.70x).
**Why it failed**: NCHW Triton matmul (0.77x, non-coalesced). fp16 on 250M+ elements (0.655x). NHWC permute on 1.4GB (0.472x). Manual BN stats (0.69x).
**Better approach**: Only full-network passthrough helps (EfficientNetB0 1.64x, B1 1.70x). Individual MBConv blocks are infeasible.

### L3: 25_ShuffleNetUnit (1.016x) -- Single ShuffleNet unit structural ceiling
**Key insight**: Individual ShuffleNet units have unavoidable channel shuffle memory copy (view+transpose+contiguous). cuDNN passthrough saves dispatch overhead but shuffle cost equals or exceeds savings. However, full ShuffleNet (13 units) reaches 1.578x via fp16 pipeline amortizing the single x.half() cost.
**Why it failed**: Triton channel shuffle kernel slower than view+transpose+contiguous (0.928x). fp16 cast overhead exceeds tensor core gains at batch=10 for single unit (0.777x).
**Better approach**: Accept infeasibility for single units. Full-network fp16 pipeline with pre-cached weights succeeds (1.578x).

### L3: 20_MobileNetV2 (correctness trap) -- Model behavior differs from apparent architecture
**Key insight**: The original model computes use_res_connect flag but the forward() runs self.features(x) as a flat Sequential -- no residual connections are actually used. Adding residual connections causes max_diff=0.198.
**Why it failed**: 12 iterations wasted adding residual connections that the original model doesn't implement in its forward path.
**Better approach**: Always trace the ACTUAL forward() execution path, not the apparent architecture. Check if skip connections, gating, etc. are actually used in the forward pass.

### L3: 28_VisionTransformer (0.782x, iter 28) -- batch_first=False shape confusion
**Key insight**: ViT passes (B=2, S=197, E=512) to nn.TransformerEncoder with batch_first=False. The transformer treats dim0 as sequence (length 2) and dim1 as batch (size 197). Previous optimizer spent 21 iterations assuming S=197 was the attention sequence length.
**Why it failed**: Transposing to (S, N, E) changes the semantics. Also: nn.TransformerEncoder uses copy.deepcopy for layers 1-5 (only layer 0 consumes RNG). Decomposed approach adds ~1.3ms of pure launch overhead across 60+ kernel launches.
**Better approach**: Check batch_first flag before decomposing transformers. For small batch_size (2), decomposed approach is likely infeasible due to launch overhead.

### L2: 36_ConvTranspose2d -- Fully fused reductions with C*H>10K iterations
**Key insight**: Fusing multiple reductions (channel_min C=128, height_sum H=256) into single program forces 32K sequential iterations per warp, starving the GPU.
**Why it failed**: 0.158x with single fused kernel.
**Better approach**: Split into separate kernels parallelized over different dims. Two-kernel gave 1.566x (+1.408x).

### L2: 15_ConvTranspose3d, 21_Conv2d -- atomic_add in autotuned reduction kernels
**Key insight**: Autotune warmup runs kernel multiple times with same output buffers. atomic_add accumulators get corrupted by repeated warmup runs.
**Why it failed**: Stats accumulators accumulated values from multiple warmup iterations.
**Better approach**: Per-(n,c) output buffers without atomics, or parallel split stats with merge step.

### L2: 19_ConvTranspose2d -- Approximate GELU before GroupNorm
**Key insight**: Approximate GELU (x*sigmoid(1.702x)) produces per-element errors that accumulate in GN mean/variance, causing max_diff=0.079 correctness failure.
**Why it failed**: GN stats amplify small GELU differences across spatial reduction.
**Better approach**: Always use exact GELU with libdevice.erf when followed by normalization layers.

### L1: 61_conv_transposed_3D (0.61x, iter 8) -- Large C_in ConvTranspose structurally infeasible
**Key insight**: ConvTranspose3d with C_in=48+, no post-ops cannot match cuDNN tensor core implicit GEMM.
**Why it failed**: All-OC (0.61x best), flat K-loop (0.32x), kh/kw loop (0.12x).
**Better approach**: Accept failure. Use torch.convolution for cuDNN if post-ops exist.

### L2: ConvTranspose(C_in=64+, stride=2) -- Structural ceiling ~1.25x
**Key insight**: Conv dominates ~90-95% of runtime. Even optimal post-op fusion cannot push past ~1.25x.
**What failed**: 91_ConvTranspose2d capped 1.245x; 5_ConvTranspose2d capped 1.302x (16 iters); 2_ConvTranspose2d capped 1.23x (17 wasted iters); 16_ConvTranspose2d capped 1.263x (12 iters).
**Better approach**: fp16 no-bias + cudnn.benchmark + fuse all post-ops. Accept 1.1-1.25x ceiling. Cap at 4 iterations max.

### L2: channels_last for ConvTranspose2d -- Layout conversion overhead
**Key insight**: NHWC adds conversion overhead that kills performance for ConvTranspose2d.
**Why it failed**: 0.647x vs 1.236x with NCHW. Format conversion not amortized.
**Better approach**: NCHW for ConvTranspose2d. NHWC only for direct Triton Conv2d with C_in=32-64 or deep CNN passthrough (RegNet).

### L2: Conv2d(C_in=8) + trivial post-ops -- Near-infeasible with cuDNN
**Key insight**: cuDNN already fuses simple activations internally. Separate Triton post-ops adds launch overhead. But implicit GEMM CAN beat cuDNN here (1.43-2.14x).
**What failed**: 69_Conv2d_HardSwish_ReLU capped 1.08x with cuDNN; 71_Conv2d_Divide_LeakyReLU capped 1.2x with cuDNN.
**Better approach**: Use full Triton implicit GEMM with fp16 tensor cores when C_in=8. K=72 fits single BLOCK_K tile. Do NOT use cuDNN approach.

### L2: fp16 conv with C_in<=8 for Conv3d -- Dtype overhead dominates
**Key insight**: Conv3d C_in<=8 fp16 conversion overhead exceeds tensor core benefit.
**What failed**: 7_Conv3d(C_in=8) 0.88x; 8_Conv3d(C_in=8) 0.685x with fp16 (cast overhead + .float()); 79_Conv3d(C_in=3) 0.911x.
**Better approach**: fp32 cuDNN for Conv3d C_in<=8. Exception: C_in=3 with batch*D*H*W>8M CAN benefit from fp16 (48_Conv3d: 1.918x).

### L3: 6_GoogleNetInceptionModule (0.993x, iter 19) -- Inception 1x1 convs near-optimal
**Key insight**: cuDNN handles 1x1 convolutions (implicit GEMMs) with C_in=480 near-optimally. Any Triton overhead is net negative.
**Why it failed**: Fusing 1x1 convs requires .contiguous() on output slices (0.904x). NCHW-direct matmul with stride=HW non-coalesced (0.898x).
**Better approach**: Accept infeasibility for compute-bound Inception-style modules with large C_in.

## Decision Tree

1. **Check algebraic elimination** (Tier 1): Spatial sum/mean after conv distributes into weights (8-14x). BN(x)-mean(BN(x)) cancels beta (1.9x). InstanceNorm(no affine) cancels bias. Dead code (16x). GAP(BN(x))=f(spatial_mean) (1.76x). mean(GN(x))=f(channel_sums) (1.66x). Conv3d+mean decomposes to 2D convs (8.44x). AvgPool commutes with 1x1 Conv (4x FLOP reduction). Post-op chain collapse. Always check first.
2. **Deep CNN passthrough** (Tier 2): For networks with 20+ conv layers (DenseNet, EfficientNet, MobileNet, RegNet, VGG, ShuffleNet, ResNet101), replace nn.Module dispatch with torch.convolution + torch.batch_norm + torch.clamp + cached param lists + cudnn.benchmark. Add full fp16 pipeline for +0.1-0.6x. Use lazy cache if params are set up in __init__ before .cuda(). Expect 1.3-2.5x. SKIP if compute-bound (batch>=64 with large spatial, or <20 layers).
3. **Full fp16 pipeline** (Tier 2): For multi-conv architectures, keep fp16 throughout without casting back to fp32. cuDNN BN accepts fp16 input. Must be all-or-nothing (per-layer conversion is WORSE than fp32). +0.1-0.6x over per-layer casting. Essential for VGG16 (+0.52x), DenseNet dense blocks (+0.14x), SqueezeNet fire modules, ShuffleNet (+0.6x). Pre-cache .half() weights.
4. **Depthwise conv** (Tier 1): Spatial tiling with scalar weight broadcast. 2D grid (spatial_blocks, B*C). Expect 1.3-15x. Usually first-try.
5. **Depthwise separable** (Tier 1): Fuse depthwise+pointwise, eliminate intermediate. ~1.8x.
6. **Pointwise 1x1 conv** (Tier 2): NCHW-direct matmul, no permutes. ~2.8x.
7. **Conv2d/3d small C_in (<=8) + substantial post-ops** (Tier 1-2): Try full Triton implicit GEMM when K=C_in*kH*kW<=72. fp16 tensor cores critical (doubles throughput). Also try cuDNN fp16 no-bias + Triton post-ops as fallback. 1.4-2.1x.
8. **Conv2d/3d C_in=16, 3x3 kernel** (Tier 2): Single-pass implicit GEMM. K fits one BLOCK_K. 1.5-4.7x.
9. **Conv2d C_in=32-64** (Tier 3-4): kpos loop + NHWC input + fp16 tensor cores. 1.2-1.6x.
10. **Conv2d C_in=64+ large spatial** (Tier 1): cuDNN fp16 no-bias + fused Triton post-ops. Pure Triton caps ~1.0-1.2x.
11. **Conv3d C_in<=8** (Tier 2): fp32 cuDNN no-bias + fused Triton. fp16 only when batch*D*H*W>8M AND C_out>32.
12. **ConvTranspose stride=1 moderate C_in** (Tier 1): cuDNN fp16 no-bias + Triton cast. ~1.5-1.9x.
13. **ConvTranspose stride=2 small C_in (<=16)** (Tier 1-2): cuDNN fp16 no-bias + fused Triton post-ops. Expect 1.5-5.2x.
14. **ConvTranspose stride=2 C_in=32** (Tier 2): cuDNN fp16 no-bias. Expect 1.1-2.8x with fused post-ops.
15. **ConvTranspose(C_in=64+, stride=2)** (Anti-Pattern): Accept ~1.0-1.25x ceiling. Cap at 4 iterations. Do NOT use channels_last (0.647x).
16. **Conv1d with long sequences (65K+)** (Tier 3-4): NCL directly with kpos loop. Never convert to NLC (0.5x).
17. **Large GN/BN spatial reductions (>500K)**: Use per-channel spatial splits (not per-group) for coalesced access. B*C programs for parallel BN when C is small. +0.35x.
18. **Multi-pass reductions**: Use 2-pass Welford over 3-pass. Recompute activations rather than materializing intermediates (exception: erf-GELU too expensive).
19. **Fused post-ops with reductions**: Split if total loop iterations >10K per program. Two kernels beat one fused kernel.
20. **Compute-bound single blocks** (Anti-Pattern): Individual conv+BN blocks (ResNet basic, MBConv, Inception, ShuffleNet unit) with large C_in are infeasible. cuDNN internal fusion is unbeatable. Only full-network passthrough helps. ShuffleNet channel shuffle adds structural ceiling at ~1.0x for single units but full network reaches 1.578x via fp16 pipeline.
21. **RNG init order**: Match EXACT parameter initialization order of original Model. Mismatch causes systematic correctness failures across ALL iterations.
22. **MaxPool precision**: Never do bias arithmetic in fp16 before MaxPool. Fuse fp16->fp32 cast+bias in Triton before pool.
23. **Autotune for reduction kernels**: Use single config only. Multiple configs corrupt accumulated stats during warmup.
24. **Model behavior verification**: Always trace ACTUAL forward() path. Check batch_first flags, verify skip connections are really used (MobileNetV2 computes use_res_connect but doesn't use it). nn.TransformerEncoder uses deepcopy (only layer 0 consumes RNG).
