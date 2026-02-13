# matmul patterns
<!-- Updated: 2026-02-12 | Source: 0212_l2, merged with prior sessions -->

## What Works

### 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd (49.508x, iter 1)
**Key insight**: mean(Linear(x) - subtract, dim=1) = x @ W.sum(0)/N + mean(bias) - mean(subtract). Converts O(M*N*K) matmul to O(M*K) matvec -- ~4000x FLOP reduction for M=2048, N=K=8192.
**What worked**: Algebraic simplification exploiting linearity of mean. Precomputed w_sum = W.sum(dim=0) in __init__. Single Triton kernel does matvec + GELU + residual add. Always check if a post-matmul reduction (sum/mean) can be distributed into weights.

### 14_Gemm_Divide_Sum_Scaling (44.773x, iter 1)
**Key insight**: sum(x @ W.T, dim=1) = x @ W.sum(dim=0), reducing a full matmul to a matvec. Same algebraic pattern as task 51.
**What worked**: Precomputed w_sum in __init__, torch.mv for matvec (avoids Triton device issues), trivial Triton scale kernel. 44.7x speedup entirely from math simplification.

### 30_Gemm_GroupNorm_Hardtanh (10.176x, iter 4) / 64_Gemm_LogSumExp (10.252x, iter 0)
**Key insight**: fp16 autocast for large GEMMs (>1024x1024) enables tensor cores, giving ~10x speedup over fp32 cuBLAS. This dominates any kernel fusion benefit.
**What worked**: `torch.amp.autocast(device_type='cuda', dtype=torch.float16)` around the matmul + Triton kernel for post-ops. The GEMM goes from ~8ms (fp32) to ~0.6ms (fp16 tensor cores).

### Epilogue Fusion Pattern (3-7x, used in 13+ tasks)
**Key insight**: For Gemm + pointwise chain, fuse bias/activation/scaling into the Triton matmul epilogue -- compute in registers after tile accumulation, before writing to global memory. Eliminates N memory round-trips.
**What worked**: Standard tiled matmul with super-blocking (GROUP_M=8), autotune with 7 configs covering 32x32 to 128x128 tiles with K=32/64. In the epilogue: load bias, apply chain (scale, clamp, activation) all in-register. Reliable 3-7x for medium-to-large GEMMs. First-try success rate: ~60% (tasks 39, 40, 45, 63, 75, 94, 98, 99 all hit on iter 0).

## What Fails

### 12_Gemm_Multiply_LeakyReLU (0.967x, iter 6)
**Key insight**: For extremely large square GEMMs (8192x8192), cuBLAS is near-optimal and a custom Triton matmul barely matches it. When the pointwise epilogue (multiply + LeakyReLU) is trivially cheap, the kernel launch overhead makes the combined approach slightly slower.
**Why it failed**: The matmul itself is >99% of the compute. The epilogue fusion saves negligible memory traffic compared to the matmul cost. In-place modification of F.linear output caused correctness failures.
**Better approach**: For very large square GEMMs where the post-ops are trivially cheap (1-2 simple activations), use fp16 autocast with F.linear + a separate tiny Triton pointwise kernel. The fp16 tensor core speedup (~10x on the matmul) will dominate. Do NOT write a custom Triton matmul for these shapes unless you also fuse non-trivial post-ops.

### 55_Matmul_MaxPool_Sum_Scale (1.012x, iter 0) / 66_Matmul_Dropout_Softmax (1.033x, iter 5)
**Key insight**: When a single very large matmul (128x32768x32768 or 128x16384x16384) is >95% of runtime, no post-op fusion can reach 1.3x. Custom Triton matmul was 30% slower than cuBLAS for these shapes.
**Why it failed**: cuBLAS is extremely well-tuned for large square matmuls. The post-ops operate on the full output tensor but are pure memory-bandwidth-bound with trivial arithmetic, offering <5% potential savings.
**Better approach**: Focus on fp16 autocast for the matmul itself. If the matmul is already using tensor cores (or the task specifically uses fp32), there is no optimization path available for single-matmul-dominated tasks with large square shapes.

## Decision Framework for Matmul Tasks

1. **Check for algebraic simplification first**: If a reduction (sum/mean) follows matmul, distribute it into weights. This gives 20-50x and should be checked BEFORE writing any kernel.
2. **Check if fp16 autocast helps**: For large square GEMMs (>2048x2048), fp16 tensor cores give ~10x. Try this before custom Triton matmul.
3. **Matmul epilogue fusion**: The default strategy for Gemm + 2+ pointwise ops. Use the standard tiled matmul template with super-blocking. Fuse bias + activations into the epilogue. Expect 3-7x for medium matrices, less for very large ones.
4. **GroupNorm after matmul**: If group_size matches BLOCK_N, fuse into epilogue. Otherwise, use a separate Welford single-pass Triton kernel or F.group_norm.
5. **Autotune configs for asymmetric shapes**: For small-M, large-N (e.g., 128x32768), add BLOCK_N=256 with BLOCK_K=64 configs. Default configs assume roughly square tiles and miss optimal shapes.
6. **Never**: Write custom Triton matmul for very large square shapes (>8192x8192) unless you fuse significant post-ops. cuBLAS is near-optimal for these.
