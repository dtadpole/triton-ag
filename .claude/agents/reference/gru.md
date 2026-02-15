# GRU Reference
<!-- Updated: 2026-02-15 | Source: level3_20260214_235132 -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify this section. -->

## Tier 1: Algorithm Alternatives

### level3: 39_GRU (0.136x, iter 19) -- cuDNN fused RNN is unbeatable for deep sequential GRU
**Key insight**: cuDNN's fused GRU kernel handles the sequential timestep loop internally with optimized register management; Triton cannot match it because GRU is inherently serial across timesteps and requires O(H^2) scalar extraction per step.
**What worked**: Nothing beat cuDNN. Best Triton attempt (persistent kernel with transposed weights, register-held h, precomputed input projections) reached only 0.136x. The fundamental barrier is that GRU with many layers (6) and long sequences (512 timesteps) is dominated by sequential recurrence that cuDNN handles with hardware-level optimizations Triton cannot replicate.

## Tier 2: Architecture Variants

### level3: 39_GRU (0.136x vs 0.066x, iter 12 vs 3) -- Persistent kernel with register h vs naive layout
**Key insight**: Holding hidden state `h` in registers and using a single persistent kernel (rather than per-timestep launches) avoids global memory round-trips for the hidden state between timesteps.
**What worked**: Persistent kernel processing all timesteps in a single launch, with `h` kept in registers and weight matrix transposed to (H, 3H) for coalesced row access. This doubled performance from 0.066x to 0.134x.

### level3: 39_GRU (0x correctness failure, iter 9) -- Per-timestep tl.dot kernel launches
**Key insight**: Per-timestep kernel launch with tl.dot provides faster matmul but accumulation order differences from cuDNN cause numerical divergence that compounds over deep sequential processing.
**What worked**: Nothing -- this approach fails correctness. tl.dot accumulation order differs from cuDNN, producing max_diff=0.019 per step that compounds over 6 layers x 512 timesteps to exceed tolerance.

## Tier 3-4: Tuning Guide

- **Weight matrix layout**: Transpose W_hh from (3H, H) to (H, 3H) for coalesced row access. Doubled throughput (0.066x to 0.134x). Interleaved layout with stride-3 access was worse (0.125x). (Source: 39_GRU)
- **Hidden state storage**: Keep h in registers, not global memory. Global memory h adds latency on every timestep access (0.103x vs 0.134x). (Source: 39_GRU)
- **Warp count**: num_warps=2 marginally better than num_warps=8 for serial inner loops. More warps don't help when the computation is sequentially dependent. (Source: 39_GRU)
- **Input projections**: Precompute input projections (x @ W_ih) via a separate Triton matmul before the recurrence loop to avoid redundant computation inside the timestep loop. (Source: 39_GRU)

## Anti-Patterns

### level3: 39_GRU (0x, iter 9) -- Per-timestep tl.dot with long sequences causes correctness failure
**Key insight**: Numerical precision differences in tl.dot compound exponentially over sequential steps in deep RNNs.
**Why it failed**: tl.dot uses different accumulation order than cuDNN, causing max_diff=0.019 per step. Over 6 layers x 512 timesteps (3072 sequential applications), error compounds past correctness threshold. Even `input_precision="ieee"` doesn't fix it.
**Better approach**: Use persistent kernel with manual scalar extraction via `tl.sum(tl.where(...))` which matches cuDNN's precision, or delegate to `torch.nn.GRU` / cuDNN entirely since Triton cannot beat it.

### level3: 39_GRU (0.103x, iter varies) -- Global memory for hidden state between timesteps
**Key insight**: Storing hidden state in global memory between timesteps adds unnecessary latency in a sequential recurrence.
**Why it failed**: Each timestep reads h from global memory, adding load latency to every step of the serial loop. With 512 timesteps x 6 layers, this overhead accumulates.
**Better approach**: Keep h in register variables within a persistent kernel that processes all timesteps in one launch.

## Decision Tree
1. Check if cuDNN/torch.nn.GRU can be used directly -- for deep GRU (many layers, long sequences), cuDNN is likely unbeatable
2. If Triton is required: use persistent kernel with register-held hidden state, transposed weight layout (H, 3H)
3. Avoid per-timestep kernel launches with tl.dot -- correctness risk from precision compounding
4. Exploit phase: tune weight layout (transpose > interleaved), num_warps=2 for serial loops, precompute input projections
