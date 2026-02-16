# Shallow CNN Reference
<!-- Updated: 2026-02-15 | Source: level3_20260215_152600 -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify this section. -->

## Tier 1: Algorithm Alternatives

No successful Tier 1 entries yet. Shallow CNNs (<=6 conv layers, small batch) have an extremely fast reference baseline (~3ms), making algorithmic alternatives difficult to exploit.

## Tier 2: Architecture Variants

No successful Tier 2 entries yet. Kernel fusion and decomposition approaches add more overhead than they save when the conv layer count is low and batch size is small.

## Tier 3-4: Tuning Guide

- **Memory format**: Use `channels_last` (NHWC) format to select better cuDNN algorithms. This was the least-bad approach at 0.82x but still below 1.0x for shallow nets. (Source: 27_RegNet)
- **Avoid fp16 per-layer casting**: Converting fp16<->fp32 at each layer boundary is devastating (0.52x). If attempting mixed precision, it must be end-to-end without per-layer casts. (Source: 27_RegNet)

## Anti-Patterns

### level3: 27_RegNet (0.821x, iter 1/20) -- cuDNN passthrough on shallow CNN
**Key insight**: cuDNN passthrough (replacing nn.Module with torch.convolution) only helps deep CNNs (60+ layers) where nn.Module dispatch overhead is significant; for shallow CNNs the dispatch savings are negligible.
**Why it failed**: With only 6 conv layers and batch=8 at 224x224, the reference runs in ~3ms. torch.convolution + torch.batch_norm adds MORE overhead than the nn.Module dispatch it replaces. All 20 iterations stayed below 1.0x with a structural ceiling at ~0.82x.
**Better approach**: For shallow CNNs with small batch, the reference PyTorch implementation is already near-optimal. Focus optimization effort on tasks where the reference has more overhead to eliminate (deeper networks, larger batches, or fused multi-op sequences).

### General: Triton custom kernels for post-ops on shallow CNN
**Key insight**: Adding Triton kernels for simple post-ops (e.g., ReLU) on top of cuDNN convolutions introduces extra kernel launch overhead that exceeds any compute savings.
**Why it failed**: At 0.565x, the Triton relu post-op kernel launch overhead outweighed any fusion benefit. The convolution already dominates runtime and cuDNN fuses activations internally when possible.
**Better approach**: Only fuse post-ops via Triton when the post-op is complex enough (multi-op epilogue) AND the model is deep enough that launch overhead is amortized across many layers.

## Decision Tree
1. Check layer count and batch size: if <=6 conv layers AND batch <=8, this is likely infeasible -- reference is already near-optimal
2. If deeper network (60+ layers): consider cuDNN passthrough to eliminate nn.Module dispatch overhead
3. For medium-depth networks: try channels_last memory format first (cheapest change)
4. Avoid per-layer dtype casting; if mixed precision, do it end-to-end
5. Only add Triton post-op kernels if the epilogue is complex and amortized over many layers
