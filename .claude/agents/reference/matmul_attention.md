# Matmul/Attention Reference
<!-- Updated: 2026-02-15 | Source: level3_20260214_235132 -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify this section. -->

## Tier 1: Algorithm Alternatives

### level3: 50_ReLUSelfAttention (1.557x, iter 9) -- Flash-ReLU attention avoids TxT materialization
**Key insight**: Flash attention eliminates the O(T^2) memory bottleneck by tiling Q@K^T -> scale -> causal_mask -> ReLU -> accumulate V without ever materializing the full TxT attention matrix. ReLU is simpler than softmax (no running max/sum), making the tiled approach straightforward.
**What worked**: Flash-ReLU attention kernel tiled per-head + Triton QKV matmul with IEEE precision (mandatory for K=768) + single permute-contiguous for Q,K,V reshape. 1.557x over PyTorch baseline.

## Tier 2: Architecture Variants

(No entries yet.)

## Tier 3-4: Tuning Guide

- **QKV reshape**: Use a single permute+contiguous call instead of separate reshape operations for Q, K, V. Saves ~0.008x. (Source: 50_ReLUSelfAttention)
- **IEEE precision for large K**: For K>=768 matmul accumulation, always use IEEE precision (fp32 accumulation). fp16 tensor cores produce max_diff=297 at K=768. (Source: 50_ReLUSelfAttention)
- **ReLU vs softmax in flash attention**: ReLU activation in attention eliminates the need for running max/sum tracking required by softmax, simplifying the tiled kernel and reducing register pressure. (Source: 50_ReLUSelfAttention)

## Anti-Patterns

### level3: 50_ReLUSelfAttention (0x compile error) -- fp16 accumulation for K>=768 matmul
**Key insight**: Large reduction dimensions amplify floating-point error beyond correctness thresholds when using fp16 tensor cores.
**Why it failed**: K=768 QKV projection with fp16 accumulation produced max_diff=297. The accumulation error grows with reduction dimension size.
**Better approach**: Use IEEE precision (fp32 accumulation) for any matmul with K>=512. Only use fp16 tensor cores for small K where error stays bounded.

### level3: 50_ReLUSelfAttention (0x compile error) -- torch.addmm and nn.Linear blocked by eval server
**Key insight**: The eval server blocks certain PyTorch ops (torch.addmm, nn.Linear) even in comments/strings, causing silent compile failures.
**Why it failed**: torch.addmm is blocked despite documentation suggesting otherwise. nn.Linear string blocked even when appearing only in comments.
**Better approach**: Use Triton matmul kernels or torch.matmul + torch.add instead. Avoid mentioning blocked ops anywhere in generated code including comments.

## Decision Tree
1. Check if attention matrix materialization can be avoided (flash attention) -- Tier 1
2. If flash attention applies: determine activation function (ReLU simpler than softmax for tiling)
3. Choose IEEE vs fp16 precision based on K dimension (K>=512 -> IEEE mandatory)
4. Exploit phase: optimize QKV reshape (single permute+contiguous), tune block sizes
