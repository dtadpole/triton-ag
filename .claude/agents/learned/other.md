# other patterns (mixed op types)
<!-- Updated: 2026-02-12 | Source: 0212_l2 -->

## What Works

### 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum (1.255x, iter 2)
**Key insight**: sum_channels(avg_spatial(x)) = sum_all(x) / spatial_size. This algebraic identity collapses GlobalAvgPool + Sum(dim=1) into a single global sum reduction, and sum(bias) becomes a precomputed constant.
**What worked**: result[b] = sum_all(maxpool_out[b]) / (spatial_per_channel * divisor) + sum(bias). Replaced 3 separate ops with a single Triton global sum kernel (one program per batch).

### 46_Conv2d_Subtract_Tanh_Subtract_AvgPool (1.274x, iter 5)
**Key insight**: Fusing subtract+tanh+subtract+avgpool into a single Triton kernel eliminates three memory round-trips. The 2x2 avgpool is computed inline by reading 4 input pixels per output pixel.
**What worked**: Flat parallel kernel with tanh approximated as 2*sigmoid(2x)-1. Output tensor is 4x smaller than input, so reads 4x per output but writes 4x fewer. Achieved 1.274x.

### 27_Conv3d_HardSwish_GroupNorm_Mean (1.114x, iter 4)
**Key insight**: When the final op is spatial mean, GroupNorm and mean commute: mean(groupnorm(x)) = groupnorm_affine(per_channel_mean, group_stats). This eliminates materializing the normalized tensor.
**What worked**: Two-kernel approach: (1) per-(batch,channel) kernel computes sum and sum_sq of bias+hardswish(conv_out), (2) per-(batch,group) kernel aggregates and applies commuted formula. F.conv3d WITHOUT bias + fusing bias in kernel 1 saved ~1ms.

## What Fails

### 84_Gemm_BatchNorm_Scaling_Softmax (0x, all 10 failed)
**Key insight**: Device assignment is random and uncontrollable. Some tasks lose ALL iterations to the Triton device mismatch bug.
**Why it failed**: All 10 eval attempts landed on cuda:1/2/3 where Triton produces pointer errors. The code was correct in design but never got evaluated.
**Better approach**: Always use `with torch.cuda.device(x.device):` around the entire forward(). For extreme cases, consider minimizing Triton usage (use PyTorch ops for everything except the most beneficial fusion) to reduce exposure to device errors.

### 52_Conv2d_Activation_BatchNorm (0x, iter 2)
**Key insight**: BatchNorm training mode + device errors is a double penalty. Fusing BN in eval mode gives wrong results; fusing in training mode requires a reduction kernel; and device errors prevent iteration.
**Why it failed**: 9/10 iterations hit device errors. The 1 successful eval showed max_diff=9.59, likely from assuming eval mode BN (using running stats instead of batch stats). Could not iterate on the fix.
**Better approach**: Always use `F.batch_norm(training=self.training)`. For BN-heavy tasks, keep BN as PyTorch native and only fuse surrounding pointwise ops. Do not attempt custom Triton BN unless you have verified the training/eval mode behavior first.
