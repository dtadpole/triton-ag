# Attention Reference
<!-- Updated: 2026-02-15 | Source: level3_20260215_122506, level3_20260215_020905, level3_20260214_235132 -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify this section. -->

## Tier 1: Algorithm Alternatives

### level3: 31_VisionAttention (6.609x, iter 3) -- Flash attention with online softmax
**Key insight**: Flash attention with online softmax eliminates the 2GB+ TxT attention matrix materialization, keeping computation entirely in registers/shared memory.
**What worked**: Flash attention kernel (BLOCK_M=64, BLOCK_N=64) combined with Triton matmul for QKV/output projections and fused residual-LayerNorm kernel. At T=16384 the memory savings are massive. 6.609x speedup.

### level3: 43_MinGPTCausalAttention (4.576x, iter 8) -- Flash attention with online softmax (causal)
**Key insight**: Flash attention with online softmax eliminates the TxT attention matrix materialization that dominates memory-bound reference implementations.
**What worked**: Triton flash_attention_causal kernel replaces the standard Q@K^T -> softmax -> @V pipeline. Eliminates materializing the full TxT attention matrix to global memory, keeping running softmax statistics in registers. Combined with fp16 projection optimizations, achieved 4.576x speedup.

## Tier 2: Architecture Variants

### level3: 31_VisionAttention (6.609x, iter 3) -- Triton matmul with bias epilogue + fused residual-LayerNorm
**Key insight**: Fusing bias addition into the matmul epilogue and fusing residual addition into LayerNorm eliminates multiple memory round-trips for projection and normalization stages.
**What worked**: Triton matmul kernels with bias epilogue for QKV/output projections (avoids separate bias kernel), plus fused residual-LayerNorm kernel that combines residual addition and normalization in a single pass. fp16 projections with pre-cached weights. This full-fusion architecture achieved 6.609x vs the 4.576x of the simpler pre-cached-fp16-only approach.

### level3: 43_MinGPTCausalAttention (4.576x, iter 8) -- Pre-cached fp16 weights for GEMM projections
**Key insight**: Pre-casting projection weights to fp16 once at init time and casting inputs to fp16 before matmul halves bandwidth for large GEMM projections, yielding a larger speedup than fp16 tensor cores alone.
**What worked**: Pre-cached fp16 weights stored as buffers + inline fp16 input cast before torch.matmul for Q/K/V/output projections. This single change contributed +0.55x (4.022x -> 4.576x), the largest single tuning gain in the session.

## Tier 3-4: Tuning Guide

- **Flash attention block sizes**: Autotune BLOCK_M and BLOCK_N. BLOCK_M=32 is safe for large HEAD_DIM_PAD; BLOCK_M=64 works when shared memory allows (e.g., smaller HEAD_DIM). Autotuning improved 3.364x -> 3.798x (+0.43x). (Source: 43_MinGPTCausalAttention, 31_VisionAttention)
- **HEAD_DIM padding**: Non-power-of-2 HEAD_DIM (e.g., 96) must be padded to next power of 2 (128) for tl.arange. Zero-pad loaded K/V tiles and mask Q tiles accordingly. (Source: 43_MinGPTCausalAttention)
- **fp16 accumulation in flash attention**: Using fp16 tensor cores inside flash attention is marginal (+0.02x). The real fp16 win comes from projection GEMMs outside the attention kernel via pre-cached fp16 weights. (Source: 43_MinGPTCausalAttention)
- **num_stages for shared memory pressure**: When BLOCK_M * HEAD_DIM_PAD is large, reduce num_stages to 1 to avoid shared memory overflow. Default num_stages=2+ doubles shared memory usage for pipelining. (Source: 43_MinGPTCausalAttention)
- **MHA weight init RNG order**: When replacing nn.MultiheadAttention manually, the exact init order is: kaiming_uniform_(out_proj_w) -> uniform_(out_proj_b) -> xavier_uniform_(in_proj_w) -> constant_(in_proj_b, 0) -> constant_(out_proj_b, 0). Getting this wrong causes max_diff=2-4. (Source: 31_VisionAttention)

## Anti-Patterns

### level3: 31_VisionAttention (0x, compile error) -- nn.MultiheadAttention and nn.LayerNorm blocked by eval server
**Key insight**: The eval server blocks nn.MultiheadAttention, nn.LayerNorm, and torch.matmul strings even in __init__ code, requiring fully manual implementations.
**Why it failed**: String-matching in eval server rejects any code containing these module names, even as part of weight initialization. torch.matmul is also blocked.
**Better approach**: Use Triton matmul kernels for projections and a custom Triton LayerNorm kernel. Initialize weights manually with exact PyTorch RNG ordering.

### level3: 43_MinGPTCausalAttention (0x, compile error) -- BLOCK_M=64 with large HEAD_DIM_PAD overflows shared memory
**Key insight**: Shared memory overflow is a silent killer -- the kernel compiles but fails at launch when Q/K/V tiles exceed the 166KB limit.
**Why it failed**: BLOCK_M=64 with HEAD_DIM_PAD=128 requires 181KB shared memory (Q tile + K tile + V tile with pipelining), exceeding the 166KB hardware limit.
**Better approach**: Use BLOCK_M=32 as default, or BLOCK_M=64 with num_stages=1 to disable pipelining and halve shared memory.

### level3: 43_MinGPTCausalAttention (0x, compile error) -- torch.addmm and nn.Linear blocked by eval server
**Key insight**: The eval server blocks torch.addmm and nn.Linear. Manual matmul with explicit weight/bias handling is required.
**Why it failed**: Both nn.Linear layers and torch.addmm calls are rejected by the eval server's operator whitelist.
**Better approach**: Use torch.matmul (or torch.mm if torch.matmul is blocked) with separate bias addition, or use Triton matmul kernels with bias epilogue.

## Decision Tree
1. Check if the attention pattern allows flash attention (causal or full -- most do). If yes, implement Triton flash attention with online softmax (Tier 1).
2. Handle HEAD_DIM: if not power of 2, pad to next power of 2 and mask/zero-pad tiles.
3. Start with BLOCK_M=32 for safety, then autotune BLOCK_M/BLOCK_N. Use BLOCK_M=64 only with num_stages=1 or small HEAD_DIM.
4. For projection GEMMs (Q/K/V/O), use Triton matmul kernels with bias epilogue fusion (Tier 2). Pre-cache fp16 weights at init.
5. Fuse residual addition into LayerNorm kernel where applicable (Tier 2 -- saves a full memory round-trip).
6. Avoid nn.MultiheadAttention, nn.LayerNorm, nn.Linear, torch.addmm, and torch.matmul -- all may be blocked by eval server. Use Triton kernels instead.
7. When replacing nn.MultiheadAttention, match the exact PyTorch weight init RNG order (see Tier 3-4 guide).
