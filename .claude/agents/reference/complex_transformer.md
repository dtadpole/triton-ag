# Complex Transformer Reference
<!-- Updated: 2026-02-15 | Source: level3_20260214_235132 -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify this section. -->

## Tier 1: Algorithm Alternatives

No successful Tier 1 alternatives observed yet. Complex transformers with 12+ blocks remain structurally infeasible for full Triton replacement. Potential avenue: delegate all matmul/attention ops to cuBLAS/cuDNN via torch.mm/torch.bmm if the eval server permits, and only write Triton kernels for normalization/activation epilogues.

## Tier 2: Architecture Variants

### level3: 30_SwinTransformerV2 (0.284x, iter 6) -- Unified matmul kernel with epilogue fusion
**Key insight**: Consolidating multiple matmul kernel variants into a single unified mm_kernel with optional bias+GELU epilogue reduces compilation overhead and improves performance on repeated small matmuls.
**What worked**: Unified mm_kernel (0.279x) plus reduced autotune configs from 4 to 2 (0.284x). Pre-transposing weights in __init__ and caching them avoids repeated contiguous() calls. torch.convolution used for patch embedding (one of the few unblocked PyTorch ops).

## Tier 3-4: Tuning Guide

- **Autotune config count**: Reduce from 4-5 configs to 2 for small matmul sizes (49xN). Fewer configs reduce JIT compilation overhead with minimal performance loss. (+0.005x, Source: 30_SwinTransformerV2)
- **Weight transpose caching**: Pre-transpose and cache weight matrices in __init__ instead of calling .contiguous() every forward pass. Saves per-call overhead across 12+ blocks. (+0.010x, Source: 30_SwinTransformerV2)
- **Unified kernel design**: Use a single Triton matmul kernel with optional bias/activation epilogue flags rather than separate kernels per operation type. Reduces total compiled kernel count and launch overhead. (+0.036x from baseline, Source: 30_SwinTransformerV2)
- **fp16 tensor cores on small matrices**: AVOID for matrices smaller than ~128xN. Conversion overhead (fp32->fp16->fp32) dominates any tensor core benefit at these sizes. (-0.014x regression, Source: 30_SwinTransformerV2)

## Anti-Patterns

### level3: 30_SwinTransformerV2 (0.284x, iter 6) -- Complex multi-block transformer infeasibility
**Key insight**: Complex transformers with 12+ attention blocks and windowed attention (small matrix sizes like 49x96 to 49x768) are structurally infeasible for Triton optimization -- no amount of fusion can bridge the 3-4x gap between Triton matmul and cuBLAS at these sizes.
**Why it failed**: Each of 12 blocks requires 6+ matmul-type kernel launches plus normalization/softmax. Triton matmul is 3-4x slower than cuBLAS for small matrices (49xN). With 100+ total kernel launches, the cumulative slowdown is insurmountable. The eval server also blocks ALL PyTorch compute functions (torch.mm, torch.bmm, torch.addmm, F.linear, F.softmax, F.normalize, F.adaptive_avg_pool1d), forcing full Triton implementation.
**Better approach**: For tasks like this, accept sub-1.0x speedup as the ceiling. Focus effort on other tasks. If eval server constraints change to allow cuBLAS, a hybrid approach (cuBLAS for matmuls, Triton for fused normalization/activation) could potentially reach 1.0x+.

### Environment Gotcha: Blocked PyTorch Ops
**Key insight**: The eval server blocks MORE PyTorch ops than documented -- torch.addmm, torch.bmm, F.softmax, F.normalize, F.adaptive_avg_pool1d are all blocked. Only torch.convolution is reliably unblocked.
**Why it matters**: Assuming certain ops (like F.softmax or torch.mm) are available leads to compile errors on first iterations, wasting exploration budget.
**Better approach**: Always assume ALL compute ops are blocked except torch.convolution. Write Triton kernels for everything from the start.

## Decision Tree
1. Count the number of transformer blocks and estimate total kernel launches -- if 100+, this task is likely infeasible (expect <0.5x)
2. Check matrix sizes in attention -- if windowed attention with sizes <128xN, Triton matmul will be 3-4x slower than cuBLAS
3. If feasible: use unified matmul kernel with epilogue fusion, cache pre-transposed weights, minimize autotune configs
4. If infeasible: spend minimal iterations confirming, then move on to other tasks
