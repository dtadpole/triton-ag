# Other Reference (RNN, SSM, ConvTranspose+post-ops, mixed op types)
<!-- Updated: 2026-02-15 | Source: 0212_v10_l1+0212_v10_l2+0212_v10_l3+0212_v10_l3_retry+level2_20260214_232629 -->

## Code Templates

No op-specific code templates. See `reference/common.md` for universal patterns.

## Tier 1: Algorithm Alternatives

### L3: 35_LSTM (2.463x, iter 11) -- Persistent kernel loops over all timesteps
**Key insight**: A persistent Triton kernel that loops over all 512 timesteps internally eliminates 3072 kernel launches per layer (512 timesteps x 6 gates). Two-phase: (1) batch precompute all input-to-hidden gate values with one large Triton matmul (5120x128 @ 128x1024), (2) persistent kernel per batch element loops over timesteps doing element-wise h@W_hh multiply-accumulate + sigmoid/tanh gates.
**What worked**: Wavefront synchronization between programs. fp16 for input matmul. Key: separate h_scratch buffer (NOT in-place on h0).

### L2: 36_LSTMHn (4.08x, iter 0) -- Dead code elimination
**Key insight**: The FC layer output is never returned (returns h_n only). Dead code elimination alone gives the major speedup. Always check if computed values are actually used in the return.
**What worked**: Skip FC layer entirely. Runtime went from 40.8ms to 10ms.

## Tier 2: Architecture Variants

### L3: 39_GRU (0.705x, iter 8) -- Persistent kernel 4x better than per-timestep launches
**Key insight**: Single persistent kernel per layer processing all 512 timesteps gave 0.705x vs 0.166x with per-timestep kernels (4.2x improvement). Each program handles one batch element, loops internally.
**What worked**: Precomputed input projections as batched matmul, persistent kernel with BLOCK_K=64 for H=256. Fixed configs (no autotune) to avoid warmup overhead.

### L2: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU (0.975x, iter 19) -- Two-kernel LN decomposition avoids full tensor write
**Key insight**: Separating LayerNorm into (1) a stats kernel that computes mean/rstd per row and (2) a fused kernel that applies LN inline + pool + GELU avoids writing the full LN output tensor. This cuts memory traffic from 536M+536M (torch.layer_norm write + read) to 268M (stats pass) + ~400M (fused apply reading only 8-of-64 elements per output).
**What worked**: fp16 cuDNN conv without bias (bias absorbed by LN), ROWS_PER_PROGRAM=8 in stats kernel, cached mean/rstd/output buffers to eliminate allocation overhead, preloaded LN weight/bias, view() instead of reshape(). Reached 0.975x (5.89ms vs 5.74ms reference).

## Tier 3-4: Tuning Guide

- **ROWS_PER_PROGRAM for LN stats kernels**: Use 8 rows per program. 16 rows causes register pressure and drops to 0.513x. (Source: L2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU)
- **fp16 conv bias**: Remove conv bias when followed by LayerNorm (LN mean subtraction absorbs bias). fp16 conv WITH bias = 0.574x; WITHOUT bias = 0.975x. cuDNN fp16 path with bias is slower. (Source: L2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU)
- **Buffer caching**: Pre-allocate mean/rstd/output buffers in __init__ and reuse across forward calls. Eliminates per-call allocation overhead for intermediate tensors. (Source: L2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU)
- **view() vs reshape()**: Use view() instead of reshape() for zero-copy tensor reshaping when contiguity is guaranteed. (Source: L2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU)
- **fp16 for RNNs**: Only helps when GEMM sizes are large (>1024). For small batch/hidden (10/256), adds overhead with no tensor core benefit. (Source: L3 RNN tasks)

## Anti-Patterns

### L3: Bidirectional RNN (38_LSTMBidirectional 0x, 41_GRUBidirectional 0.04x, 42_GRUBidirectionalHidden 0.276x)
**Key insight**: Bidirectional multi-layer RNNs are structurally infeasible. 6 layers x 512 timesteps x 2 directions = 6144 sequential kernel launches with ~62us Python loop overhead each = ~380ms minimum overhead vs cuDNN's single fused kernel at 83-106ms.
**Why it failed**: (1) nn.LSTM/nn.GRU banned by string matching. (2) Manual weight init may produce different weights due to PyTorch internal parameter creation consuming random state differently. (3) Python loop overhead is 4-5x the reference runtime.
**Better approach**: Accept as infeasible. Do NOT attempt manual reimplementation of bidirectional RNNs.

### L3: 37_LSTMCn (0.043x, iter 3) -- Cell state output exposes numerical differences
**Key insight**: Persistent kernel element-wise multiply-accumulate gives different FP results than cuDNN's tiled matmul. When returning raw cell state c (not h), accumulated error over 3072 timesteps produces max_diff=0.066-0.095.
**Why it failed**: Matmul accumulation order differences. Per-timestep tl.dot gives exact results but with 6150 kernel launches = 0.043x.
**Better approach**: Hybrid persistent kernel with tl.dot inside the timestep loop for numerical accuracy.

### L2: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU (0.028x, iter 6) -- Single fused kernel for LN+pool
**Key insight**: Fusing LayerNorm and pooling into a single kernel creates 33M programs, each loading 64 elements 4 times. The massive program count and redundant loads make this 36x slower than the reference.
**Why it failed**: The LN normalization requires a full row reduction (mean/variance), but each output position only needs 8-of-64 input positions for pooling. Combining these two access patterns in one kernel forces either redundant computation or excessive synchronization.
**Better approach**: Two-kernel decomposition: (1) stats kernel computes mean/rstd over full rows, (2) fused kernel reads only needed elements, applies LN inline, pools, and activates.

## Decision Tree

1. **Check for dead code first**: Tier 1 -- Return values may not use all computed tensors. Skip unused FC layers, unused state components. (36_LSTMHn: 4.08x from dead code elimination alone.)
2. **Triage bidirectional RNNs immediately**: Anti-pattern -- Bidirectional multi-layer RNNs are ALWAYS infeasible. Do not attempt.
3. **For unidirectional RNN/LSTM/GRU**: Tier 1/2 -- Use persistent Triton kernel (one program per batch element, loops over all timesteps internally). Precompute input projections as single large batched matmul. NEVER write h in-place -- use separate scratch buffer.
4. **Persistent kernel performance ceiling**: For batch_size=10 with H=256, expect ~0.5-0.7x. For batch_size >= 32 or tasks where input matmul dominates, can reach 2.5x.
5. **Cell state vs hidden state output**: Anti-pattern risk -- If returning raw cell state (c_n), use tl.dot inside persistent loop for numerical accuracy. If returning h_n, element-wise is acceptable (tanh squashes error).
6. **For conv + LayerNorm + post-ops**: Tier 2 -- Use two-kernel LN decomposition: stats kernel (mean/rstd) then fused apply+post-ops. Remove conv bias when followed by LN. Use fp16 conv. NEVER fuse LN stats and post-op selection into one kernel.
7. **tanh computation**: Use `2*sigmoid(2*x) - 1` for speed, or `from triton.language.extra.cuda import libdevice; libdevice.tanh(x)` for accuracy. For deep sequential computation (512+ timesteps), use libdevice.tanh.
8. **For tasks with missing packages (einops)**: Skip immediately.

## Banned Reward Hacking Techniques

The following techniques are **banned** by the strategy hard rules. Do NOT use them.

- **CUDA Graphs** (banned, rule 9): Inflates speedup by amortizing kernel launch overhead.
- **torch.compile + fp16** (banned, rule 8): Delegates to PyTorch compiler.
- **torch.jit.script** (banned, rule 8): Compiler delegation.
- **getattr(nn, ...) bypass** (banned, rule 7): Circumvents eval server nn.* string check.
- **F.scaled_dot_product_attention** (banned, rule 13): Delegates to Flash Attention pre-built kernel.
- **F.conv2d, F.linear, torch.matmul, torch.mm, torch.bmm** (banned, rule 4): Functional API and torch.* compute ops are banned.
