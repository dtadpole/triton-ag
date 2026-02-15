# Common Reference
<!-- Updated: 2026-02-14 | Source: 0212_v10_l1+0212_v10_l2+0212_v10_l3+0212_v10_l3_retry+0212_v8_l2+0212_v3_l3+0212_l2 -->

## Code Templates

### Analysis Techniques (L2/L3)

Use these for multi-operation tasks where simple strategy selection isn't enough.

#### Computation Graph Analysis

Trace through `forward()` and build a mental computation graph:

1. List all operations in order with shapes
2. Identify fusion opportunities (which ops can share registers?)
3. Find the largest intermediate tensor (can you avoid materializing it?)
4. Map data flow dependencies (what's the critical path?)

#### Why PyTorch is Slow (identify the opportunity)

- **Multiple kernel launches**: PyTorch launches separate CUDA kernels per op (~5-10us overhead each). Your fused kernel eliminates this.
- **Memory round-trips**: PyTorch writes intermediates to global memory between ops. Your kernel keeps values in registers.
- **Bottleneck type**: Memory-bandwidth limited → reduce global memory accesses. Compute limited → use tensor cores. Latency limited → increase occupancy.

#### Multi-Kernel Decomposition (L3)

Use a single kernel when all ops fit in shared memory with linear data flow. Use multiple kernels when intermediates exceed shared memory or stages need different parallelization. Split at natural boundaries where data must go through global memory anyway.

#### Memory Hierarchy Planning

```
Registers: ~255 per thread (<64 for good occupancy) — accumulators, current tile
Shared memory: 48-164KB — tiles of A, B for matmul
L2 cache: implicit — proper tiling improves reuse
Global memory: input/output tensors — minimize traffic
```

Choose tile sizes to balance occupancy vs cache reuse. For matmul: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 uses ~16KB shared memory per tile pair.

## Environment Constraints

- **tl.math.tanh does not exist**: `tl.math.tanh`, `tl.math.fast_expf`, `tl.math.fast_dividef` all missing. `libdevice.tanh` exists but extremely slow (0.34x). Workaround: `tanh(x) = 2*sigmoid(2*x) - 1` using tl.sigmoid. tl.math.exp2 IS available and faster than tl.exp.
  (Source: L1: 22_Tanh, 26_GELU; L2: 11_ConvTranspose2d, 51_Gemm)

- **tl.cumprod, tl.associative_scan, tl.shift_left do NOT exist**: Only tl.cumsum works for prefix ops. Workaround for cumprod: `exp(cumsum(log(x)))`.
  (Source: L1: 89_cumsum, 90_cumprod)

- **tl.arange/BLOCK_SIZE must be power of 2**: Values like 384, 768 cause errors.
  (Source: L1: 89_cumsum; L2: 10_ConvTranspose2d)

- **tl.dot requires M, N, K >= 16**: Must pad K to 16 for small C_in (e.g., 3).
  (Source: L1: 75_conv; L2: 50_ConvTranspose3d)

- **tl.static_range with >50 iterations causes compilation timeout**: Use Python range() instead.
  (Source: L1: 58_conv_transposed_3D, 45_Average_Pooling_2D)

- **Triton "cpu tensor" pointer error on non-cuda:0 devices**: Fix: `with torch.cuda.device(x.device):` before any Triton kernel launch.
  (Source: universal)

- **Eval server blocks nn.* module strings via string matching**: Blocks nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d/3d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.Conv1d -- even in COMMENTS. Also blocks F.conv2d, F.linear, torch.matmul, torch.mm, torch.bmm via runtime detection. Workaround: nn.Parameter + nn.init.kaiming_uniform_. Do NOT use `getattr(nn, ...)` bypass (banned).
  (Source: L1: 33_BatchNorm, 40_LayerNorm, 50_conv, 82_depthwise; L2: extensive)

- **torch.convolution is NOT blocked**: Dispatches to cuDNN. Use for conv when pure Triton cannot match cuDNN. Critical for ConvTranspose and Conv2d with large C_in (64+).
  (Source: L1: 68_conv 1.1x, 70_conv 1.9x, 73_conv 1.5x, 80_conv 1.8x; L2: extensive)

- **torch.bmm and torch.addmm are NOT blocked**: Use for 1x1 conv and FC layers.
  (Source: L3: 19_MobileNetV1, 22_EfficientNetB0)

- **F.pad, F.avg_pool2d, F.adaptive_avg_pool2d, F.normalize, F.softmax are NOT blocked**.
  (Source: L3: 19_MobileNetV1, 22_EfficientNetB0)

- **Eval server runs models in training mode**: BatchNorm must compute batch statistics with momentum and Bessel correction.
  (Source: L1: 33_BatchNorm; L2: 33_Gemm, 39_Gemm)

- **Eval harness does NOT copy weights via load_state_dict**: Sets same random seed, instantiates both. Must replicate exact init: kaiming_uniform_(a=sqrt(5)) + uniform_ bias. Parameter creation ORDER must match.
  (Source: L1: 57_conv_transposed_2D; L2: 29_Matmul)

- **ConvTranspose fan_in uses out_channels**: weight dims (in_ch, out_ch, *kernel). fan_in = weight.size(1) * kernel_size^D.
  (Source: L1: 58_conv_transposed_3D; L2: 60_ConvTranspose3d)

- **Int32 pointer overflow for tensors >2GB**: Pointer offsets MUST use int64 (tl.cast to tl.int64) for >2B elements.
  (Source: L1: 45_Average_Pooling_2D)

- **CUDA grid dim z limit = 65535**: batch*channels may fit but batch*channels*spatial does not.
  (Source: L1: 42_Max_Pooling_2D)

- **Atomic_add with persistent output tensor corrupts results**: Output not fresh per trial during autotune. Use fresh torch.empty output per forward() call.
  (Source: L1: 47_Sum_reduction, 95_CrossEntropyLoss, 99_TripletMarginLoss)

- **GPU contention causes 2x runtime variance**: cuda:0 most reliable. OOM on busy GPUs.
  (Source: L1: 20_LeakyReLU, 35_GroupNorm)

- **kbEval server outages are common**: Extended outages (>5 min) frequent. Server may crash loop.
  (Source: L1: 27_SELU, 28_HardSigmoid, 31_ELU, 34_InstanceNorm, 52_Argmin, 53_Min)

- **Autotune incompatible with atomic_add to running stats**: Different configs during warmup corrupt accumulated values.
  (Source: L2: 77_ConvTranspose3d)

- **cuDNN ConvTranspose in fp16 WITH bias is slower than WITHOUT**: Pass None for bias, handle in Triton.
  (Source: L2: 91_ConvTranspose2d)

- **einops package not installed**. (Source: L3: 48_Mamba2)

- **fp16 crashes in deep networks**: Unreliable for 6+ layer transformer/CNN. (Source: L3: 28_VisionTransformer)

- **Triton kernel constexpr parameter ordering**: Float scalar args after all pointer/stride args but before constexpr block sizes. (Source: L2: 12_Gemm)

- **nn.Dropout is blocked; dropout is identity in eval mode**: Skip. (Source: L2: 66_Matmul)

## Anti-Patterns (Never Do This)

- **Bandwidth-bound element-wise ops cannot beat PyTorch**: For pure single-op activations (ReLU, LeakyReLU, Sigmoid, Tanh, Swish, SELU, ELU, HardTanh, HardSigmoid, Softplus, Softsign, ScalarMul) on very large tensors (>1B elements), PyTorch saturates HBM bandwidth. Max: ~1.0x. Do not waste iterations.
  (Source: L1: 19_ReLU, 20_LeakyReLU, 21_Sigmoid, 22_Tanh, 25_Swish, 27-32, 5_ScalarMul -- all ~1.0x)

- **Transposing large tensors to make reduction dim contiguous**: Copy cost (8GB+) dominates. Use 2D tiled access with coalesced inner-dim reads instead.
  (Source: L1: 47_Sum 0.089x, 48_Mean 0.2x, 49_Max 0.203x, 51_Argmax 0.087x)

- **Two-phase parallel prefix scan**: Doubles memory traffic. Always slower than single-pass sequential scan.
  (Source: L1: 89_cumsum 0.54x)

- **Explicit im2col for large spatial dims**: Creates 2-10GB matrix, always slower than implicit GEMM.
  (Source: L1: 67_conv 0.26x; L2: 1_Conv2D 0.43x, 10_ConvTranspose2d 0.32x)

- **Scalar weight loads for conv**: Per-element conv gives 0.01-0.15x. Always use tl.dot-based tiled approach.
  (Source: L1: 57_conv_transposed 0.031x; L2: 11_ConvTranspose2d 0.04x)

- **In-place Triton kernel writes**: Non-deterministic correctness failures. Use torch.empty_like().
  (Source: L1: 35_GroupNorm, 38_L1Norm; L2: 6_Conv3d)

- **Pre-padding large inputs (1024x1024+) for ConvTranspose**: Allocation cost (500MB+) exceeds boundary check savings.
  (Source: L1: 57_conv 0.78x, 65_conv 0.77x)

- **fp16 pre-conversion on very large tensors (>1B elements) for element-wise ops**: Cast overhead exceeds bandwidth savings.
  (Source: L1: 19_ReLU, 21_Sigmoid, 44_AvgPool, 94_MSELoss)

- **Permuting large NCHW tensors**: Cache transposed weights in __init__. Write NCHW directly.
  (Source: L1: 87_pointwise 0.208x; L3: 1_MLP 1.3x->4.4x)

- **Including fixed dims (K) in autotune key**: Causes poor config selection and overhead.
  (Source: L1: 54_conv_3D, 5.22ms vs 1.89ms)

- **constexpr kernel_size for pool windows 11x11+**: 121+ unrolled loads cause compilation timeout.
  (Source: L1: 45_Average_Pooling_2D)

- **Full deep CNN in Triton (13+ layers)**: Precision compounds. (Source: L3: 10_ResNet101, 11_VGG16)

- **.t().contiguous() in forward()**: Cache transposed weights in __init__. (Source: L3: 1_MLP)

- **tl.trans(b) in-kernel transpose**: Pre-transpose weight in __init__. (Source: L2: 1_Conv2D 0.58x vs 1.30x)

- **channels_last memory format**: Conversion overhead always exceeds benefit. (Source: L2: 91_ConvTranspose2d 0.54x)

- **fp16 when runtime <5ms or C_in<=8**: Dtype conversion overhead dominates. Use TF32. (Source: L2: 23_Conv3d 0.81x)

- **tl.where scatter for im2col**: O(K^2) work. Use vectorized gather. (Source: L2: 24_Conv3d 0.08x)

## Universal Techniques

- **Standard tiled matmul template**: Super-blocking GROUP_M=8, 7 autotune configs (32x32 to 128x128, BLOCK_K=32/64), fp32 accumulation. Gives 2-8x first try for square/rectangular matmuls. Works for batched matmul (batch on program_id(1)).
  (Source: L1: 1_Square 6.0x, 2_Rect 7.6x, 3_Batched 5.1x, 7_SmallK 2.5x, 13_Symmetric 6.7x)

- **Algebraic simplification before kernel writing**: Diagonal matmul = row scaling (104x). Triangular matmul = skip ~50% tiles (10-15x). Tensor matmul = reshape to 2D (4-5x). Sum/mean after matmul distributes into weights (20-74x). Dead code elimination.
  (Source: L1: 12_Diagonal 104x, 14_UpperTri 14.5x, 15_LowerTri 10.2x; L2: 80_Gemm 73.5x)

- **Implicit transpose via strides**: Pass A.stride(1), A.stride(0) instead of A.T.contiguous(). Avoids expensive copy.
  (Source: L1: 16_TransA 5.1x, 17_TransB 5.9x, 18_TransBoth 6.5x)

- **fp16 for tensor cores on large GEMMs (>1024x1024)**: Pre-convert in forward(), cache fp16 weights in __init__. Halves bandwidth for K-loop reads.
  (Source: L1: 6_LargeK 1.5x, 8_Irregular 3.2x; L2: 55_Matmul 11.1x)

- **Online softmax (2-pass vs 3-pass)**: Running max+sum saves one full memory pass. Critical for large reduction dims (>100K). 33% bandwidth reduction.
  (Source: L1: 23_Softmax 1.32x, 24_LogSoftmax 1.33x)

- **2D tiled reduction for non-contiguous dims**: (BLOCK_REDUCE, BLOCK_INNER) tiles with coalesced inner-dim reads. Never transpose, never reduce per-element.
  (Source: L1: 47_Sum 1.08x, 48_Mean 1.1x, 49_Max 1.12x, 51_Argmax 1.3x)

- **2D register tiling for normalization**: (BLOCK_S x FEATURES) loads all features into registers for single-read compute+normalize. Halves memory accesses vs two-pass.
  (Source: L1: 36_RMSNorm 1.58x)

- **Fused kernel for ops with intermediate tensors**: Eliminate intermediate allocations. Masked cumsum (1.45x), exclusive cumsum (1.52x), KL div (1.13x).
  (Source: L1: 92_ExclusiveCumsum 1.52x, 93_MaskedCumsum 1.45x)

- **kpos explicit loop for conv**: Loop over kernel positions, tl.dot per position with (BLOCK_HW, C_in) x (C_in, C_out). Better than flat K-loop for small C_in.
  (Source: L1: 50_conv 1.48x, 54_conv 3.3x, 62_conv 1.57x, 64_conv 1.3x)

- **NHWC input for conv with C_in=32-64**: permute+contiguous cost amortized across kpos iterations. Contiguous C_in loads.
  (Source: L1: 62_conv 1.57x, 69_conv 1.43x, 71_conv 1.59x, 78_conv 1.61x)

- **torch.convolution fp16 for ConvTranspose**: Cast input/weight to fp16, cuDNN tensor cores, Triton for fp16->fp32 cast. ~1.5-1.9x.
  (Source: L1: 70_conv 1.89x, 73_conv 1.53x, 80_conv 1.81x)

- **Depthwise conv spatial tiling**: 2D grid (spatial_blocks, B*C), scalar weight broadcast, unrolled small kernels. 2-15x.
  (Source: L1: 82_depthwise 2.0x, 83_depthwise 15.2x, 85_depthwise 5.6x)

- **Matmul epilogue fusion**: Fuse bias/activation/scaling. ~70% first-try success. 4-12x.
  (Source: L2: 12_Gemm 7.1x, 59_Matmul 11.9x)

- **Flash attention for T>=1024**: Avoid TxT materialization. BLOCK_M=64, BLOCK_N=64.
  (Source: L3: 31_VisionAttention 8.1x)

- **Interior/boundary splitting for pooling**: Eliminates 3 comparisons per load in hot path. (Source: L1: 44_AvgPool1D 1.07x)

- **cin-first K ordering for implicit GEMM**: Consecutive K elements access same spatial location, 2x better locality.
  (Source: L1: 63_conv 1.87x, 76_conv 1.32x; L2: 7_Conv3d 1.54x)

- **Pre-pad input to eliminate boundary checks**: F.pad before conv eliminates conditional loads, 30%+ improvement.
  (Source: L1: 50_conv 1.48x; L2: 43_Conv3d)

- **torch.cuda.device(device) context manager**: Essential for any forward() with Triton kernels.

- **torch.backends.cudnn.benchmark = True**: Set in __init__ for conv tasks.

- **Fast Mish**: mish(x) = x * e*(e+2)/(e*(e+2)+2) where e=exp(x). ONE exp(). (Source: L2: 29_Matmul, 94_Gemm)

- **Sub-pixel decomposition for ConvTranspose stride=2**: Group output by parity. Pre-compute compact weight per group. (Source: L2: 26_ConvTranspose3d 1.04x)

- **Split precision for matmul**: Large K (768+) needs input_precision="ieee". Small K can use TF32. (Source: L3: 50_ReLUSelfAttention 1.534x)

- **NCHW direct output writes**: Never write (M,N) then permute to NCHW. Embed NCHW index in store.
  (Source: L2: 1_Conv2D; L3: 21_EfficientNetMBConv)

- **Two-kernel for Gemm + Normalization**: Matmul epilogue, then fused GN/BN+acts. 5-12x.
  (Source: L2: 30_Gemm 11.7x, 62_Matmul 8.3x)

- **Implicit weight transpose via strides**: Pass w.stride(1), w.stride(0) to Triton. Saves memory.
  (Source: L3: 2_ShallowWideMLP 5.074x)

## L1 Structural Feasibility Guide

| Op Pattern | Feasible? | Expected Speedup | Key Strategy |
|---|---|---|---|
| Dense matmul (square/rect) | YES | 2-8x | Standard tiled matmul + super-blocking |
| Structured matmul (diag/tri) | YES | 10-104x | Algebraic simplification |
| Batched/tensor matmul | YES | 3-5x | Reshape to 2D + standard matmul |
| Transposed matmul | YES | 5-7x | Implicit transpose via strides |
| Matvec (large K) | NO | ~1.0x | Bandwidth-bound, cuBLAS near-optimal |
| Pure element-wise (1.6B elems) | NO | ~1.0x | Bandwidth-bound, PyTorch at ceiling |
| GELU (approximate) | YES | ~2.0x | Approximate tanh formula beats exact erf |
| Softmax/LogSoftmax (large dim) | YES | 1.3x | Online 2-pass softmax |
| Sum/Mean/Max reduction | MAYBE | 1.1x | 2D tiled, coalesced inner reads |
| Cumsum/Cumprod | MAYBE | 1.0-1.5x | Sequential scan + fusion with pre/post ops |
| Normalization (RMS/Layer/Group) | MAYBE | 1.0-1.6x | 2D register tiling or multi-kernel |
| BatchNorm (large spatial) | NO | 0.4x | cuDNN too optimized |
| MaxPool | YES | 1.4-3.0x | Flat spatial + per-element max |
| AvgPool | MAYBE | 1.0-2.0x | Interior/boundary split |
| Conv2d (small C_in<=16) | YES | 1.5-1.9x | Implicit GEMM, cin-first K |
| Conv2d (C_in=64+) | MAYBE | 1.2-1.6x | kpos loop + NHWC + fp16 |
| Conv3d (small C_in<=8) | YES | 1.7-4.7x | Single-pass K, all-OC tile |
| ConvTranspose (small C_in, stride=1) | YES | 1.3-3.0x | kpos + fp16 tensor cores |
| ConvTranspose (large C_in, stride=1) | MAYBE | 1.1-1.9x | torch.convolution fp16 |
| ConvTranspose (C_in=48+, no post-ops) | NO | 0.6x | cuDNN unbeatable |
| Depthwise conv | YES | 1.3-15x | Spatial tiling, scalar weight broadcast |
| Depthwise separable conv | YES | 1.8x | Fuse depthwise+pointwise |
| Pointwise 1x1 conv | YES | 2.8x | NCHW-direct matmul, no permutes |
| Loss functions (>1B elems) | NO | ~1.0-1.1x | Bandwidth-bound |
| Scaled dot-product attention | YES | 2.0x | Three-kernel decomposition |

## L2 Structural Feasibility Guide

| Op Pattern | Feasible? | Expected Speedup | Key Strategy |
|---|---|---|---|
| Gemm + pointwise chain | YES | 4-12x | Epilogue fusion, fp16 tensor cores |
| Gemm + sum/mean reduction | YES | 20-74x | Algebraic: distribute reduction into weights |
| Gemm + Normalization + acts | YES | 5-12x | Two-kernel: matmul + fused GN/BN+acts |
| Conv2d(C_in<=16) + post-ops | YES | 1.4-2.9x | Single-pass implicit GEMM, fp16, fused epilogue |
| Conv2d(C_in=64) + post-ops | MAYBE | 1.0-2.0x | kpos loop or torch.convolution hybrid |
| Conv3d(C_in<=8) + post-ops | YES | 1.3-1.9x | Implicit GEMM, cin-first K, pre-pad |
| Conv + spatial reduction | YES | 1.5-16x | Algebraic elimination or fused conv+reduce |
| Conv + MaxPool fusion | YES | 1.5-2.9x | Compute pool window on-the-fly in conv kernel |
| ConvTranspose(C_in<=16,stride=2) | YES | 1.0-6.0x | Sub-pixel decomp or single-pass implicit GEMM |
| ConvTranspose(C_in=64+,stride=2) | NO | 0.1-0.9x | cuDNN unbeatable |
| Dead code pattern | YES | 20-74x | Algebraic analysis |

## L3 Structural Feasibility Guide

| Architecture Type | Feasible? | Expected Speedup | Key Requirement |
|---|---|---|---|
| MLP (2-18 layers) | YES | 2-5x | Cached weight transpose + epilogue fusion |
| Single block (MBConv, Fire) | YES | 1.3-1.7x | NCHW-native kernels, eliminate permutes |
| Causal attention | YES | 1.5-8x | Flash attention, cached QKV weights |
| Shallow CNN (LeNet, AlexNet) | YES | 1.5-1.8x | Implicit im2col, fp16, epilogue fusion |
| RNN unidirectional | MAYBE | 0.5-2.5x | Persistent kernel; depends on batch size |
| Deep CNN (VGG, ResNet) | NO | 0.04-0.4x | cuDNN too fast, precision compounds |
| Bidirectional RNN | NO | 0.04-0.3x | Python loop overhead, cuDNN fusion |
| Complex transformers | NO | 0.3-0.7x | Many medium matmuls, cuBLAS unbeatable |

## Composite Patterns
<!-- Multi-op sequences with known best strategies. Extracted from exploration summaries by the learner agent. -->

- **matmul → pointwise(1-3)** → epilogue_fusion (4-12x). Fusing activation into matmul tile registers eliminates intermediate global memory write. Alternatives tried: two_kernel (2-5x) — intermediate write costs 40-60% of runtime.
  (Source: L2: 59_Matmul, 55_Matmul, 56_Matmul, 94_Gemm, 66_Matmul, 12_Gemm, 99_Matmul, 95_Matmul)

- **matmul → norm → activation** → two_kernel: matmul+bias epilogue + fused norm-act kernel (5-12x). BN/GN needs all values before normalizing, cannot fuse into matmul epilogue.
  (Source: L2: 30_Gemm 11.7x, 37_Matmul 5.5x, 62_Matmul 8.3x, 88_Gemm 5.2x)

- **matmul → reduction(sum/mean)** → algebraic_distribute: distribute reduction into weights, collapse matmul to matvec or constant (20-74x). Always check before writing any kernel.
  (Source: L2: 14_Gemm 61x, 18_Matmul 61x, 51_Gemm 70x, 80_Gemm 73.5x)

- **conv(C_in<=16) → post-ops** → torch.convolution fp16 + fused Triton post-ops (1.3-2.5x). cuDNN conv is near-optimal for small C_in; Triton handles post-ops only.
  (Source: L1: 70_conv 1.89x, 73_conv 1.53x, 80_conv 1.81x)

- **conv → pool** → fuse pool into conv kernel, avoid materializing full conv output (1.5-2.9x).
  (Source: L2: 82_Conv2d 2.93x, 50_ConvTranspose3d 1.45x)

- **reshape → matmul → softmax → matmul** → flash_attention for T>=1024 (2-8x); three-kernel decomposition for small T.
  (Source: L3: 31_VisionAttention 8.1x, L1: 97_ScaledDotProduct 1.97x)

- **pointwise(3+) → reduction** → single fused kernel. Fusing eliminates intermediate writes; individual ops are bandwidth-bound at ~1.0x.
  (Source: L1: 93_masked_cumsum 1.45x, L1: 92_cumsum_exclusive 1.52x)

## Strategy Selection Heuristics
<!-- Cross-cutting lessons about when to use which strategy class. Extracted from bottleneck + result data by the learner agent. -->

- **compute-bound + matmul chain** → prefer epilogue fusion over multi-kernel. Epilogue fusion wins 8/10 tasks because intermediate writes dominate at large GEMM sizes.
  (Source: L2: 12_Gemm 7.1x, 59_Matmul 11.9x, 55_Matmul 11.1x)

- **memory-bound + element-wise chain (3+ ops)** → prefer single fused kernel. Fusing 3+ pointwise ops is the ONLY way to beat PyTorch; individual ops are bandwidth-bound at ~1.0x.
  (Source: L1: 26_GELU 1.95x, L1: 92_cumsum_exclusive 1.52x)

- **infeasible pattern (deep CNN, bidir RNN)** → skip after 2 iterations. Spending 20 iterations on VGG16 or bidirectional GRU yields <0.4x every time.
  (Source: L3: 10_ResNet101 0.04x, 11_VGG16 0.4x, 15_GRU 0.04x)

- **conv with C_in<=16** → try full Triton first (single-pass implicit GEMM). For C_in>=32, prefer torch.convolution + Triton post-ops.
  (Source: L1: 54_conv 3.31x Triton, L1: 70_conv 1.89x cuDNN)

- **pure single-op element-wise on large tensors** → do not optimize. Bandwidth ceiling at ~1.0x. Only path to >1.0x is fusing with adjacent ops.
  (Source: L1: 19_ReLU through 32_HardTanh, all ~1.0x)
