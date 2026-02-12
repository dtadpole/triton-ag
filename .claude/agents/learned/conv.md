# conv patterns
<!-- Updated: 2026-02-12 | Top 5 by speedup -->

### 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp (1.991x, iter 0)
**Op type**: conv
**Key insight**: For some conv tasks, the reference model is NOT AOTI-compiled and the naive PyTorch sequential execution is slow (21.7ms). Simply using F.conv_transpose3d + torch scalar multiply + F.max_pool3d + F.adaptive_avg_pool3d + Triton clamp gives ~2x speedup on first try.
**What worked**: Standard approach of F.conv_transpose3d with bias, then PyTorch ops for pooling, with a minimal Triton clamp kernel. The reference was slow at 21.7ms suggesting it wasn't using AOTI compilation, making it easy to beat.
**What failed**: Nothing -- first try succeeded.

### 43_Conv3d_Max_LogSumExp_ReLU (1.316x, iter 8)
**Op type**: conv
**Key insight**: For conv-dominated 3D tasks, fp16 conv (tensor cores) + keeping intermediate in fp16 for the Triton kernel halves the memory bandwidth required for the fused post-conv kernel, providing the extra margin needed to cross 1.3x.
**What worked**: fp16 Conv3d via tensor cores + single-pass online LogSumExp algorithm that fuses MaxPool3d(2x2x2) + LogSumExp(dim=channels) + ReLU into one Triton kernel. The online algorithm tracks running_max and running_sum incrementally per channel, avoiding the two-pass approach. Keeping conv output in fp16 (halved memory) for the 8-load-per-channel pool window was the final key to hitting 1.3x.
**What failed**: (1) Separate F.max_pool3d + Triton logsumexp was slower than reference (0.77x) due to materializing the large pooled tensor. (2) Two-pass logsumexp (find max, then sum exp) was 0.9x due to re-reading 512 values per output. (3) Transpose for contiguous channels added copy overhead. (4) fp32 conv + fp32 Triton topped at 1.08x. (5) The single-pass online LogSumExp was the algorithmic key, but precision-related fp16 bandwidth reduction provided the final push.

### 35_Conv2d_Subtract_HardSwish_MaxPool_Mish (1.297x, iter 8)
**Op type**: conv
**Key insight**: For conv-dominated tasks, the post-conv fusion provides limited speedup since cuDNN conv takes 80%+ of the runtime. Eliminating bounds checks in the maxpool kernel (when H,W are exact multiples of pool size) and using larger BLOCK_SIZE gave the best marginal improvement.
**What worked**: F.conv2d (cuDNN) + single Triton kernel fusing subtract+hardswish+maxpool2x2+mish. Removing per-element bounds checks (since 126=2*63 exactly) and using BLOCK_SIZE up to 8192 with num_warps=8 gave the best 1.297x. Multiplying by 1/6 (0.16666667) instead of dividing by 6 also slightly helped.
**What failed**: (1) channels_last conv + contiguous conversion was slower (0.728x) due to the format conversion cost. (2) channels_last with stride-aware Triton (avoiding contiguous) was also slower (0.792x) due to non-coalesced memory access. (3) fp16 conv + float conversion (1.059x) -- the half/float conversion overhead negated tensor core gains. (4) Folding subtract into bias provided no benefit since the subtract was already cheap in-register.

### 46_Conv2d_Subtract_Tanh_Subtract_AvgPool (1.274x, iter 5)
**Op type**: conv + element_wise + pooling
**Key insight**: Fusing subtract+tanh+subtract+avgpool into a single Triton kernel eliminates three memory round-trips (writing/reading intermediate tensors between ops). The 2x2 avgpool is computed inline by reading 4 input pixels per output pixel, applying the fused pointwise chain, and averaging in registers.
**What worked**: Flat parallel kernel (one program per contiguous output block) with tanh approximated as 2*sigmoid(2x)-1 achieved 1.274x. Output tensor is 4x smaller than input (63x63 vs 126x126), so the kernel reads 4x per output element but writes 4x fewer elements.
**What failed**: (1) 7 out of 10 iterations failed due to device assignment to non-cuda:0 GPUs. (2) tl.math.tanh does not exist in this Triton version; used sigmoid-based tanh identity instead. (3) Per-channel grid (N*C programs) and flat grid gave similar performance (~1.27x). (4) Adding more autotune configs slightly hurt due to autotuning overhead. (5) Could not test bias-absorbed approach (subtract1 baked into conv bias) due to device errors.

### 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum (1.255x, iter 2)
**Op type**: conv
**Key insight**: sum_channels(avg_spatial(x)) = sum_all(x) / spatial_size. This algebraic identity collapses GlobalAvgPool + Sum(dim=1) into a single global sum reduction, and sum(bias) becomes a precomputed constant added once per batch.
**What worked**: Algebraic simplification: result[b] = sum_all(maxpool_out[b]) / (spatial_per_channel * divisor) + sum(bias). This replaced separate GlobalAvgPool + BiasAdd + Sum operations with a single Triton global sum kernel (one program per batch). Achieved 1.255x.
**What failed**: (1) Atomic_add approach for channel sum had precision issues (2/3 trials failed). (2) Absorbing divide into conv weights (1.229x) was slightly worse than keeping the divide separate (1.255x) due to changing conv computation characteristics. (3) The Conv3d + MaxPool3d still dominate, limiting further gains.
