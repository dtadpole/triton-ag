# matmul patterns
<!-- Updated: 2026-02-12 | Source: 0212_v3_l3, merged with 0212_l2 -->

## What Works

### 33_VanillaRNN (5.314x, iter 0) -- fp16 on large GEMMs
**Key insight**: fp16 on large linear layers (32768->16384 and 16384->8192) enables tensor cores, giving ~5.3x speedup. The two large GEMMs dominate this RNN cell.
**What worked**: fp16 with F.linear for both i2h and h2o projections + Triton tanh kernel (2*sigmoid(2*x)-1). First-try success. Only valid when reference model also outputs fp16 (or when dtype is cast back to match).

### L2: 51_Gemm_Subtract... (49.508x, iter 1) -- Algebraic reduction
**Key insight**: mean(Linear(x) - subtract, dim=1) = x @ W.sum(0)/N + mean(bias) - mean(subtract). Converts O(M*N*K) matmul to O(M*K) matvec -- ~4000x FLOP reduction.
**What worked**: Precomputed w_sum = W.sum(dim=0) in __init__. Single Triton kernel does matvec + GELU + residual add. Always check if post-matmul reduction (sum/mean) can be distributed into weights.

### L2: Epilogue Fusion Pattern (3-7x, 13+ tasks)
**Key insight**: For Gemm + pointwise chain, fuse bias/activation/scaling into the Triton matmul epilogue (compute in registers after tile accumulation, before writing to global memory). Eliminates N memory round-trips. ~60% first-try success rate.
**What worked**: Standard tiled matmul with super-blocking (GROUP_M=8), autotune with 7 configs covering 32x32 to 128x128 tiles with K=32/64.

### 34_VanillaRNNHidden (1.403x, iter 5) -- Batched projections
**Key insight**: RNN sequential loop limits optimization. The key is to batch operations OUTSIDE the loop: precompute all input projections in one matmul, batch output projections after the loop.
**What worked**: Split i2h weight into input and hidden parts. Precompute input projections for all timesteps in one batched matmul. Batch h2o projections after loop. Use fp16 for the large input projection only.

## What Fails

### 39_GRU (1.013x, iter 18) -- cuDNN RNN is optimal
**Key insight**: Multi-layer cuDNN RNNs (GRU, LSTM) cannot be improved by Triton or any other approach. cuDNN fuses all 6 layers and 512 timesteps into one kernel.
**Why it failed**: Manual GRU (0.05x), fp16 (no benefit for batch=10, hidden=256), Triton kernels in loop (launch overhead dominates), torch._VF.gru (no flattened parameters).
**Better approach**: For multi-layer cuDNN RNNs, accept ~1.0x parity. Use nn.Parameter + functional API for weight extraction.

## Banned Reward Hacking Techniques

The following techniques produced good speedup numbers but are **reward hacking** — they game the evaluation system rather than demonstrating real Triton kernel writing. They are banned by the strategy hard rules. Do NOT use them.

- **F.scaled_dot_product_attention** (banned, rule 13): Gave 6-8x for MinGPTCausalAttention, MiniGPTBlock. Delegates to Flash Attention (a pre-built kernel) instead of writing your own Triton attention.
- **torch.compile** (banned, rule 8): Gave 4.28x for ConvolutionalVisionTransformer. Delegates to PyTorch's compiler instead of writing Triton kernels.
- **getattr(nn, ...) bypass** (banned, rule 7): Circumvented the eval server's nn.* string check via string concatenation. Use nn.Parameter + functional API instead.
- **CUDA Graphs for RNNs** (banned, rule 9): Gave 4.3x for GRUHidden, 4.1x for LSTMHn. Inflates speedup by amortizing kernel launch overhead, not by writing better kernels.

## Decision Framework for Matmul Tasks

1. **Check algebraic simplification first** (legitimate optimization): If a reduction (sum/mean) follows matmul, distribute it into weights. This gives 20-50x. **Verify the identity holds for ALL inputs** — document the proof in comments.
2. **Check for dead code**: Return values may not use all computed tensors (e.g., unused FC layers). Skip them.
3. **Matmul epilogue fusion**: The default strategy for Gemm + 2+ pointwise ops. Use the standard tiled matmul template with super-blocking. Fuse bias + activations into the epilogue. Expect 3-7x for medium matrices.
4. **fp16 for large GEMMs (>1024x1024)**: Only when reference output dtype matches. Explicit .half() casting preferred over autocast for short-runtime tasks.
5. **Attention tasks**: Write Q@K^T + softmax + att@V in Triton. Do NOT use F.scaled_dot_product_attention.
6. **RNN cells (vanilla RNN, custom)**: fp16 for large projections + batch operations outside the loop. Triton for activation only. Expect 1.4-5.3x depending on GEMM size.
7. **cuDNN RNNs (nn.LSTM, nn.GRU)**: Accept ~1.0x parity. cuDNN fuses all timesteps and layers internally.
8. **Never**: Write custom Triton matmul for very large square shapes (>8192x8192) unless fusing significant post-ops. Never use in-place writes (always allocate fresh output).
