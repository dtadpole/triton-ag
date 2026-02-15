# Attention Reference
<!-- Updated: 2026-02-15 | Source: level3_20260214_235132 -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify this section. -->

## Tier 1: Algorithm Alternatives

### level3: 43_MinGPTCausalAttention (4.576x, iter 8) -- Flash attention with online softmax
**Key insight**: Flash attention with online softmax eliminates the TxT attention matrix materialization that dominates memory-bound reference implementations.
**What worked**: Triton flash_attention_causal kernel replaces the standard Q@K^T -> softmax -> @V pipeline. Eliminates materializing the full TxT attention matrix to global memory, keeping running softmax statistics in registers. Combined with fp16 projection optimizations, achieved 4.576x speedup.

## Tier 2: Architecture Variants

### level3: 43_MinGPTCausalAttention (4.576x, iter 8) -- Pre-cached fp16 weights for GEMM projections
**Key insight**: Pre-casting projection weights to fp16 once at init time and casting inputs to fp16 before matmul halves bandwidth for large GEMM projections, yielding a larger speedup than fp16 tensor cores alone.
**What worked**: Pre-cached fp16 weights stored as buffers + inline fp16 input cast before torch.matmul for Q/K/V/output projections. This single change contributed +0.55x (4.022x -> 4.576x), the largest single tuning gain in the session. By contrast, enabling fp16 tensor cores inside the flash attention kernel itself was marginal (+0.02x).

## Tier 3-4: Tuning Guide

- **Flash attention block sizes**: Autotune BLOCK_M and BLOCK_N for the flash attention kernel. BLOCK_M=32 is a safe starting point; BLOCK_M=64 may overflow shared memory with large HEAD_DIM_PAD. Autotuning improved 3.364x -> 3.798x (+0.43x). (Source: 43_MinGPTCausalAttention)
- **HEAD_DIM padding**: Non-power-of-2 HEAD_DIM (e.g., 96) must be padded to next power of 2 (128) for tl.arange. Zero-pad the loaded K/V tiles and mask Q tiles accordingly. (Source: 43_MinGPTCausalAttention)
- **fp16 accumulation in flash attention**: Using fp16 tensor cores (tl.dot with input_precision="tf32" -> fp16 cast) inside flash attention is marginal (+0.02x). The real fp16 win comes from projection GEMMs outside the attention kernel. (Source: 43_MinGPTCausalAttention)
- **num_stages for shared memory pressure**: When BLOCK_M * HEAD_DIM_PAD is large, reduce num_stages to 1 to avoid shared memory overflow. Default num_stages=2+ doubles shared memory usage for pipelining. (Source: 43_MinGPTCausalAttention)

## Anti-Patterns

### level3: 43_MinGPTCausalAttention (0x, compile error) -- BLOCK_M=64 with large HEAD_DIM_PAD overflows shared memory
**Key insight**: Shared memory overflow is a silent killer -- the kernel compiles but fails at launch when Q/K/V tiles exceed the 166KB limit.
**Why it failed**: BLOCK_M=64 with HEAD_DIM_PAD=128 requires 181KB shared memory (Q tile + K tile + V tile with pipelining), exceeding the 166KB hardware limit. The error manifests as a compile/launch failure, not a correctness issue.
**Better approach**: Use BLOCK_M=32 as default, or BLOCK_M=64 with num_stages=1 to disable pipelining and halve shared memory.

### level3: 43_MinGPTCausalAttention (0x, compile error) -- torch.addmm and nn.Linear are blocked by eval server
**Key insight**: The eval server blocks torch.addmm and nn.Linear despite documentation suggesting otherwise. Manual matmul with explicit weight/bias handling is required.
**Why it failed**: Both nn.Linear layers and torch.addmm calls are rejected by the eval server's operator whitelist, causing compile errors on the first attempt.
**Better approach**: Use torch.matmul (or torch.mm) with separate bias addition. Pre-transpose weights if needed. Store weights as raw tensors, not nn.Linear modules.

## Decision Tree
1. Check if the attention pattern allows flash attention (causal or full -- most do). If yes, implement Triton flash attention with online softmax (Tier 1).
2. Handle HEAD_DIM: if not power of 2, pad to next power of 2 and mask/zero-pad tiles.
3. Start with BLOCK_M=32 for safety, then autotune BLOCK_M/BLOCK_N.
4. For projection GEMMs (Q/K/V/O), pre-cache fp16 weights at init and cast inputs to fp16 before matmul (Tier 2 -- this is often the biggest single tuning gain).
5. Avoid nn.Linear and torch.addmm; use torch.matmul + manual bias instead.
