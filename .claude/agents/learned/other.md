# other patterns (RNN, mixed op types)
<!-- Updated: 2026-02-13 | Source: 0212_v8_l2, merged with 0212_v3_l3+0212_l2 -->

## What Works

### 36_LSTMHn (4.08x, iter 0) -- Dead code elimination
**Key insight**: Dead code elimination is the key insight here -- the FC layer output is never returned, so skip it entirely. Always check if computed values are actually used in the return.
**What worked**: Runtime went from 40.8ms to 10ms. The fc layer was pure waste. Dead code elimination alone gives the major speedup here.

### L2: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum (1.333x, iter 2) -- Algebraic simplification
**Key insight**: sum_channels(avg_spatial(x)) = sum_all(x) / spatial_size. Collapses GlobalAvgPool + Sum(dim=1) into a single global sum reduction.
**What worked**: result[b] = sum_all(maxpool_out[b]) / (spatial_per_channel * divisor) + sum(bias). Single Triton global sum kernel. Conv without bias + fused bias/divide Triton kernel for the large intermediate.

### L2: 28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply (5.047x, iter 7) -- InstanceNorm equivalence
**Key insight**: InstanceNorm2d on (B,C,1,1) with spatial=1 is equivalent to batch normalization per channel across the batch dimension. Reshape (B,C)->(C,B), compute batch norm statistics in Triton, reshape back.
**What worked**: Triton matmul with bias epilogue + Triton batch norm on transposed tensor + fused (x+y)*y kernel.

## What Fails

### 41_GRUBidirectional (0.914x, iter 1) -- cuDNN RNN unbeatable
**Key insight**: Bidirectional multi-layer GRU cannot be matched with Triton. The nn.GRU cuDNN path is ~10-15ms faster, creating an unrecoverable gap.
**Why it failed**: Manual GRU (0.034x), fp16 (0.742x), torch._VF.gru (0.694x, now also banned).
**Better approach**: Accept ~0.9x for bidirectional GRU. Focus optimization effort elsewhere.

### 39_GRU (1.013x, iter 18) -- cuDNN already optimal
**Key insight**: cuDNN GRU fuses all 6 layers and 512 timesteps into a single kernel. Any Triton kernel added on top reduces performance due to launch overhead. The Triton kernel requirement forces overhead.
**Why it failed**: Every approach was slower or equal: manual GRU (0.05x), fp16 (no benefit), Triton in loop (launch overhead).
**Better approach**: For cuDNN RNN tasks, since torch._VF and nn.GRU are both banned (rules 4 and 14), these tasks are structurally infeasible. Accept ~0x speedup.

## Banned Reward Hacking Techniques

The following techniques produced good speedup numbers but are **reward hacking** -- they game the evaluation system rather than demonstrating real Triton kernel writing. They are banned by the strategy hard rules. Do NOT use them.

- **CUDA Graphs** (banned, rule 9): Gave 3-4x for ResNet18, EfficientNetB0. Inflates speedup by amortizing kernel launch overhead.
- **torch.compile + fp16** (banned, rule 8): Gave 1.4-3.2x for SwinMLP, GoogleNet. Delegates to PyTorch compiler.
- **torch.jit.script** (banned, rule 8): Gave 1.5-1.7x for ResNet101. Compiler delegation.
- **getattr(nn, ...) bypass** (banned, rule 7): Circumvented eval server nn.* string check. Use nn.Parameter + Triton kernels instead.
- **F.scaled_dot_product_attention** (banned, rule 13): Delegates to Flash Attention pre-built kernel.
- **F.conv2d, F.linear, torch.matmul, etc.** (banned, rule 4): All functional API and torch.* compute ops are banned. Write computation in Triton.

## Decision Framework for RNN/Other Tasks

1. **Check for dead code first**: Return values may not use all computed tensors. Skip unused FC layers, unused state components.
2. **fp16 for RNNs**: Only helps when GEMM sizes are large (>1024). For small batch/hidden (10/256), fp16 adds overhead with no tensor core benefit.
3. **Never implement RNNs manually**: cuDNN fuses all timesteps and layers. Python loops over timesteps are 20-30x slower. However, since nn.LSTM/nn.GRU and torch._VF are banned (rules 4 and 14), RNN tasks are structurally infeasible — accept ~0x and focus on algebraic shortcuts.
4. **Algebraic simplification**: Check if spatial reductions can collapse operations (GlobalAvgPool + Sum = single sum).
5. **cuDNN RNNs are no longer accessible**: Both nn.LSTM/nn.GRU and torch._VF are banned (rule 4). RNN-dominated tasks will have very low speedup.
6. **For tasks with missing packages (einops, flash-attn)**: Skip immediately. Reference model cannot load.
7. **InstanceNorm on spatial=1 tensors**: Does NOT produce zeros or identity. Equivalent to batch normalization on transposed tensor. Implement in Triton.
