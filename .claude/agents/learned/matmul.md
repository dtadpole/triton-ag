# matmul patterns
<!-- Updated: 2026-02-12 | Top 5 by speedup -->

### 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd (49.508x, iter 1)
**Op type**: matmul
**Key insight**: mean(Linear(x) - subtract, dim=1) = x @ W.sum(0)/N + mean(bias) - mean(subtract). This converts a full O(M*N*K) matmul into a O(M*K) matvec — a ~4000x FLOP reduction for M=2048, N=K=8192. The LogSumExp of a (2048,1) tensor is a no-op (single element per row).
**What worked**: Algebraic simplification exploiting the linearity of mean: mean(Wx + b - s) = x @ mean_of_W_columns + mean(b) - mean(s). Precomputed w_sum = W.sum(dim=0) in __init__. Single Triton kernel does matvec dot product, GELU (tanh approximation using exp), and broadcast residual add in one fused pass.
**What failed**: First attempt failed because tl.math.tanh doesn't exist in the Triton version on the server. Fixed by computing tanh manually as (exp(2x)-1)/(exp(2x)+1).

### 14_Gemm_Divide_Sum_Scaling (40.207x, iter 3)
**Op type**: matmul
**Key insight**: When matmul output is immediately summed along one dimension, the sum can be distributed into the weight matrix first (sum(x @ W.T, dim=1) = x @ sum(W, dim=0)), reducing a full matmul to a matvec -- ~8192x fewer FLOPs for hidden_size=8192.
**What worked**: Algebraic simplification reduced (1024, 8192) @ (8192, 8192) matmul to (1024, 8192) @ (8192,) matvec using torch.mv, then a trivial Triton kernel fused the divide-by-2 and scaling into a single multiply. The 40x speedup comes entirely from the math simplification, not from kernel optimization.
**What failed**: First 3 iterations hit "cpu tensor" Triton errors on non-default CUDA devices (cuda:1, cuda:2). The workaround was using torch.mv for the matvec and only using Triton for the trivial scale operation, which avoided device context issues with Triton pointer arguments.

### 30_Gemm_GroupNorm_Hardtanh (7.118x, iter 3)
**Op type**: matmul
**Key insight**: Triton matmul with super-blocking and bias epilogue fusion dramatically outperforms F.linear + separate PyTorch ops, even on large square matrices (8192x8192). The key is avoiding multiple kernel launches and memory round-trips.
**What worked**: Two-kernel approach: (1) Triton matmul with fused bias add in epilogue, (2) Fused GroupNorm (Welford single-pass) + HardTanh kernel. Using torch.cuda.device(device) context manager fixed the Triton "cpu tensor" error on non-default CUDA devices (cuda:1,2,3).
**What failed**: F.linear + fused GroupNorm+HardTanh only achieved 1.13x because F.linear was slower than the Triton matmul. The "cpu tensor" error on non-cuda:0 devices was fixed by wrapping forward() in torch.cuda.device(device) context.

### 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish (6.563x, iter 4)
**Op type**: matmul
**Key insight**: For matmul followed by pointwise ops then reduction, a Triton matmul kernel with fused epilogue (bias+scale+clamp in registers) dramatically outperforms PyTorch's separate kernel launches. The key win is avoiding writing the large (1024, 8192) intermediate to global memory.
**What worked**: Triton matmul with super-blocking for L2 locality, fusing bias add, 4x scaling (2x from scale_factor * 2x from x+x), and clamping into the epilogue. A separate logsumexp+mish kernel handles the reduction. This achieved 6.56x by eliminating 3 memory round-trips for the 64MB intermediate.
**What failed**: Using F.linear for matmul + Triton only for post-ops gave only 1.05x since the matmul dominated and the post-ops saved minimal memory traffic. The eval server blocks nn.Linear even as temporary objects in __init__. tl.math.tanh doesn't exist -- must compute tanh manually via exp.

### 59_Matmul_Swish_Scaling (5.452x, iter 8)
**Op type**: matmul
**Key insight**: For asymmetric matmul shapes (M=128, N=K=32768), the Triton autotune config set matters enormously -- adding BLOCK_N=256 with BLOCK_K=64 configs turned a 0.76x slowdown into a 5.5x speedup by letting autotune find the optimal tile shape for this small-M, large-N workload.
**What worked**: Triton tiled matmul with fused Swish+scale epilogue, using autotune configs with wide BLOCK_N (128, 256) and deep BLOCK_K (64) options. Pre-transposing the weight matrix in __init__ to avoid runtime transpose. Using torch.cuda.set_device(x.device) for multi-GPU compatibility.
**What failed**: (1) cuBLAS via torch.addmm/F.linear only reached ~1.18x -- PyTorch's own matmul for this shape is slow because it doesn't tune for the highly asymmetric M<<N shape. (2) Initial Triton matmul with standard configs (32-128 block sizes) was 0.76x slower. (3) In-place kernel modification caused non-deterministic correctness failures (2/3 trials). (4) Cached weight transpose with torch.addmm was ~1.19x but couldn't break 1.3x.
