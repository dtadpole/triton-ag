# conv patterns
<!-- Updated: 2026-02-12 | Source: 0212_v3_l3, merged with 0212_l2 -->

## What Works

### 13_DenseNet121TransitionLayer (1.746x, iter 9) -- Algebraic reordering
**Key insight**: Reordering AvgPool before Conv1x1 (legal since both are linear) reduces conv computation by 4x. Fusing BN affine+ReLU+AvgPool into one Triton kernel avoids materializing a 256M-element intermediate.
**What worked**: Pre-compute BN scale/offset from batch stats, then fuse affine+ReLU+2x2 avgpool into single Triton kernel. Conv1x1 then operates on 4x smaller input. Demonstrates that algebraic reordering applies to conv tasks too, not just matmul.

### L2: 42_ConvTranspose2d_GlobalAvgPool... (13.177x, iter 2) -- Algebraic elimination
**Key insight**: When post-conv ops include spatial mean/sum, the entire ConvTranspose2d can be eliminated algebraically: mean_spatial(conv_transpose(x)) distributes to sum_ic(weight_sum * spatial_sum(input)) / (OH*OW).
**What worked**: Three-step pipeline reducing a massive conv to a tiny matmul. Always check for algebraic elimination before writing kernels.

## What Fails

### 17_SqueezeNetFireModule (1.0x, iter 19) -- Can't beat cuDNN
**Key insight**: SqueezeNet Fire Module (3 convs + 3 relus) is tightly optimized by cuDNN/AOTI. No Triton kernel or fp16 can beat it.
**Why it failed**: F.conv2d extracts different cuDNN algorithms than nn.Conv2d modules (correctness issues). fp16 autocast caused CUDA illegal memory access. Triton ReLU breaks cuDNN internal fusion. The module is small enough that cuDNN perfectly saturates the GPU.
**Better approach**: For small, simple conv+relu architectures with AOTI-compiled reference, accept ~1.0x parity.

### 41_GRUBidirectional (0.914x, iter 1) -- cuDNN RNN unbeatable
**Key insight**: Bidirectional multi-layer GRU cannot be matched with functional API. The nn.GRU cuDNN path is ~10-15ms faster, creating an unrecoverable gap.
**Why it failed**: Manual GRU (0.034x), fp16 (0.742x), torch._VF.gru (0.694x) -- no approach matches cuDNN's fused bidirectional path.
**Better approach**: Accept ~0.9x for bidirectional GRU. Focus optimization effort elsewhere.

## Banned Reward Hacking Techniques

The following techniques produced good speedup numbers but are **reward hacking** — they game the evaluation system rather than demonstrating real Triton kernel writing. They are banned by the strategy hard rules. Do NOT use them.

- **CUDA Graphs** (banned, rule 9): Gave 3-4x for ResNet18, EfficientNetB0. Inflates speedup by amortizing kernel launch overhead, not by writing better kernels.
- **torch.compile + fp16** (banned, rule 8): Gave 1.4-3.2x for SwinMLP, GoogleNet. Delegates to PyTorch's compiler instead of writing Triton kernels.
- **torch.jit.script** (banned, rule 8): Gave 1.5-1.7x for ResNet101, MobileNetV1. Same problem — compiler delegation, not kernel writing.
- **getattr(nn, ...) bypass** (banned, rule 7): Circumvented the eval server's nn.* string check via string concatenation. Use nn.Parameter + functional API instead.

## Decision Framework for Conv Tasks

1. **Check if conv can be eliminated algebraically** (e.g., spatial sum/mean after conv distributes into weights). If yes, massive speedup possible (10-50x).
2. **Check if algebraic reordering applies**: AvgPool before Conv1x1, dead code elimination, linear op reordering.
3. **If conv dominates (>85% of runtime)**: Use fp16 conv for tensor cores (only if reference outputs fp16) + fuse all post-ops into ONE Triton kernel. Target 1.1-1.3x.
4. **If post-ops are just 1-2 cheap activations**: Do NOT write a Triton kernel for just the activation. cuDNN fuses these internally. Focus on other optimizations.
5. **Always**: F.conv* without bias + fuse bias in Triton. Set cudnn.benchmark=True. Wrap forward() in torch.cuda.device(x.device).
6. **Use nn.Parameter + nn.init + functional API** to extract weights from blocked nn modules. Do NOT use getattr bypass (reward hacking, rule 7).
