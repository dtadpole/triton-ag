# other patterns (RNN, SSM, mixed op types)
<!-- Updated: 2026-02-14 | Source: 0212_v10_l1+0212_v10_l2+0212_v10_l3+0212_v10_l3_retry -->

Note: L2 tasks are exclusively matmul and conv op types. This file covers L3 RNN/attention/SSM patterns.

## What Works

### L3: 35_LSTM (2.463x, iter 11) -- Persistent kernel loops over all timesteps
**Key insight**: A persistent Triton kernel that loops over all 512 timesteps internally eliminates 3072 kernel launches per layer (512 timesteps x 6 gates). Two-phase: (1) batch precompute all input-to-hidden gate values with one large Triton matmul (5120x128 @ 128x1024), (2) persistent kernel per batch element loops over timesteps doing element-wise h@W_hh multiply-accumulate + sigmoid/tanh gates.
**What worked**: Wavefront synchronization between programs. fp16 for input matmul. Key: separate h_scratch buffer (NOT in-place on h0).

### L3: 39_GRU (0.705x, iter 8) -- Persistent kernel 4x better than per-timestep launches
**Key insight**: Single persistent kernel per layer processing all 512 timesteps gave 0.705x vs 0.166x with per-timestep kernels (4.2x improvement). Each program handles one batch element, loops internally.
**What worked**: Precomputed input projections as batched matmul, persistent kernel with BLOCK_K=64 for H=256. Fixed configs (no autotune) to avoid warmup overhead.

### L2: 36_LSTMHn (4.08x, iter 0) -- Dead code elimination
**Key insight**: The FC layer output is never returned (returns h_n only). Dead code elimination alone gives the major speedup. Always check if computed values are actually used in the return.
**What worked**: Skip FC layer entirely. Runtime went from 40.8ms to 10ms.

## What Fails

### L3: Bidirectional RNN (38_LSTMBidirectional 0x, 41_GRUBidirectional 0.04x, 42_GRUBidirectionalHidden 0.276x)
**Key insight**: Bidirectional multi-layer RNNs are structurally infeasible. 6 layers x 512 timesteps x 2 directions = 6144 sequential kernel launches with ~62us Python loop overhead each = ~380ms minimum overhead vs cuDNN's single fused kernel at 83-106ms.
**Why it fails**: (1) nn.LSTM/nn.GRU banned by string matching. (2) Manual weight init may produce different weights due to PyTorch internal parameter creation consuming random state differently. (3) Python loop overhead is 4-5x the reference runtime.
**Better approach**: Accept as infeasible. Do NOT attempt manual reimplementation of bidirectional RNNs.

### L3: 37_LSTMCn (0.043x, iter 3) -- Cell state output exposes numerical differences
**Key insight**: Persistent kernel element-wise multiply-accumulate gives different FP results than cuDNN's tiled matmul. When returning raw cell state c (not h), accumulated error over 3072 timesteps produces max_diff=0.066-0.095.
**Why it fails**: Matmul accumulation order differences. Per-timestep tl.dot gives exact results but with 6150 kernel launches = 0.043x.
**Better approach**: Hybrid persistent kernel with tl.dot inside the timestep loop for numerical accuracy.

## Decision Framework for RNN/Other Tasks

1. **Check for dead code first**: Return values may not use all computed tensors. Skip unused FC layers, unused state components. (36_LSTMHn: 4.08x from dead code elimination alone.)
2. **Triage bidirectional RNNs immediately**: Bidirectional multi-layer RNNs are ALWAYS infeasible. Do not attempt.
3. **For unidirectional RNN/LSTM/GRU**: Use persistent Triton kernel (one program per batch element, loops over all timesteps internally). Precompute input projections as single large batched matmul. NEVER write h in-place -- use separate scratch buffer.
4. **Persistent kernel performance ceiling**: For batch_size=10 with H=256, expect ~0.5-0.7x. For batch_size >= 32 or tasks where input matmul dominates, can reach 2.5x.
5. **Cell state vs hidden state output**: If returning raw cell state (c_n), use tl.dot inside persistent loop for numerical accuracy. If returning h_n, element-wise is acceptable (tanh squashes error).
6. **tanh computation**: Use `2*sigmoid(2*x) - 1` for speed, or `from triton.language.extra.cuda import libdevice; libdevice.tanh(x)` for accuracy. For deep sequential computation (512+ timesteps), use libdevice.tanh.
7. **For tasks with missing packages (einops)**: Skip immediately.
8. **fp16 for RNNs**: Only helps when GEMM sizes are large (>1024). For small batch/hidden (10/256), adds overhead with no tensor core benefit.

## Banned Reward Hacking Techniques

The following techniques are **banned** by the strategy hard rules. Do NOT use them.

- **CUDA Graphs** (banned, rule 9): Inflates speedup by amortizing kernel launch overhead.
- **torch.compile + fp16** (banned, rule 8): Delegates to PyTorch compiler.
- **torch.jit.script** (banned, rule 8): Compiler delegation.
- **getattr(nn, ...) bypass** (banned, rule 7): Circumvents eval server nn.* string check.
- **F.scaled_dot_product_attention** (banned, rule 13): Delegates to Flash Attention pre-built kernel.
- **F.conv2d, F.linear, torch.matmul, torch.mm, torch.bmm** (banned, rule 4): Functional API and torch.* compute ops are banned.
