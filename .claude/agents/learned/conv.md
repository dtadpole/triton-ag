# conv patterns
<!-- Updated: 2026-02-13 | Source: 0212_v8_l2, merged with 0212_v3_l3+0212_l2 -->

## What Works

### 42_ConvTranspose2d_GlobalAvgPool... (14.424x, iter 0) -- Algebraic elimination
**Key insight**: When spatial mean/sum follows conv_transpose, the entire convolution can be eliminated algebraically: mean_spatial(conv_transpose(x)) = (1/(OH*OW)) * x_sum @ w_sum + bias. Converts massive 2D transposed conv to tiny matmul.
**What worked**: Spatial sum of input, summed weights, small matmul. Post-ops fused into Triton kernel on tiny output. Also works with stride>1 (44_ConvTranspose2d at 7.68x). Always check if spatial reduction distributes into conv weights.

### 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool (4.162x, iter 5) -- BN+Pool commutativity
**Key insight**: AvgPool commutes with BN affine transform since both are linear. Pool BEFORE applying BN affine avoids materializing full BN output (256M elements). Three-kernel pipeline: (K1) per-channel sums for BN stats, (K2) batch stats + running mean/var update, (K3) vectorized pool4 + BN affine from conv output directly. fp16 conv + fused AvgPool(2)+AvgPool(2) into single Pool(4).
**What worked**: Algebraic BN elimination + pool commutativity saved ~0.5GB memory bandwidth = ~10ms. Reading conv output twice (stats + pool+affine) is much cheaper than materializing full BN output then pooling.

### 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling (9.051x, iter 6) -- Algebraic decomposition
**Key insight**: Mean over depth after ConvTranspose3d decomposes into three 2D convolutions: mean_d(conv3d(x,w)) = (1/D)*[conv2d(sum_d(x), sum_kd(w)) - boundary corrections]. Three 2D convs on (B,16,128,128) are much cheaper than one 3D conv on (B,16,32,128,128).
**What worked**: Derived algebraic identity eliminating 3D conv entirely. Post-conv ops (bias+softmax+tanh+scale) fused into single Triton kernel.

## What Fails

### 34_ConvTranspose3d_LayerNorm_GELU_Scaling (0.665x, iter 11) -- Conv dominated, no algebraic shortcut
**Key insight**: ConvTranspose3d with large outputs (268M elements) and fast reference (<6ms) are structurally infeasible when all conv operations must be written in Triton.
**Why it failed**: Triton convolution cannot match cuDNN for this shape. Conv dominates ~80% of runtime. Even fused Triton LN+GELU+Scale cannot compensate.
**Better approach**: Accept failure for conv_transpose-dominated tasks with large outputs. Focus optimization effort on tasks where algebraic elimination is possible.

### 21_Conv2d_Add_Scale_Sigmoid_GroupNorm (0.878x, iter 18) -- Conv + GroupNorm both need Triton
**Key insight**: Conv2d + GroupNorm tasks require writing both operations in Triton, which is fundamentally harder than when cuDNN conv was available.
**Why it failed**: Writing both conv and GroupNorm in Triton introduces overhead on both operations vs cuDNN.
**Better approach**: For Conv + GroupNorm, write custom two-kernel Triton GroupNorm (stats kernel + apply kernel) and im2col+matmul Triton conv.

## Decision Framework for Conv Tasks

1. **Check if conv can be eliminated algebraically** (e.g., spatial sum/mean after conv distributes into weights). If yes, massive speedup possible (7-14x).
2. **Check if algebraic reordering applies**: AvgPool before Conv1x1, AvgPool commutes with BN affine, dead code elimination (min(x,0)+clamp(0,1)=0), additive constants eliminated by LayerNorm.
3. **If conv cannot be eliminated**: Write im2col + Triton matmul (with epilogue fusion for bias + post-ops). Use `torch.as_strided` for im2col unfolding (shape manipulation, not compute). For small kernels (1x1, 3x3), consider direct sliding-window Triton kernel.
4. **If post-ops are just 1-2 cheap activations**: Conv tasks with only trivial post-ops will be harder since cuDNN fuses these internally. Focus on algebraic elimination or accept lower speedup.
5. **Always**: Wrap forward() in `torch.cuda.device(x.device)`. Use ParamHolder module to match state_dict keys. Cache fp16 weights in `__init__` when using tensor cores.
6. **Online softmax for conv + softmax tasks**: 2-pass algorithm with BLOCK_SIZE=8192, num_warps=16. Fuse bias+clamp+softmax+post-ops into single kernel. Recompute from fp16 input rather than storing intermediates.
7. **Conv_transpose tasks**: Write transposed convolution in Triton (im2col approach with transposed weight layout). Accept that these are structurally harder without cuDNN.
