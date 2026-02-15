# Matmul Reference
<!-- Updated: 2026-02-15 | Source: 0212_v10_l1+0212_v10_l2+0212_v10_l3+0212_v10_l3_retry+0212_v8_l2+0212_v3_l3+0212_l2+level2_20260214_232629 -->

## Code Templates

### Matmul Autotune Config (with super-blocking)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
```

### Matmul Epilogue Fusion Template

The highest-value pattern for matmul + activation tasks. The accumulator is in registers after the tile loop -- applying bias + activation there is essentially FREE.

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_epilogue_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr,  # 0=none, 1=relu, 2=gelu, 3=silu
):
    # Super-blocking for L2 cache locality
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # EPILOGUE: bias + activation IN REGISTERS (free!)
    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    acc = acc + bias[None, :]
    if ACTIVATION == 1:  # ReLU
        acc = tl.maximum(acc, 0.0)
    elif ACTIVATION == 2:  # GELU (approximate)
        acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
    elif ACTIVATION == 3:  # SiLU / Swish
        acc = acc * tl.sigmoid(acc)

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        linear = torch.nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight.data.clone())  # (out_features, in_features)
        self.bias = nn.Parameter(linear.bias.data.clone())

    def forward(self, x):
        # weight is (N, K) -- must transpose for matmul
        M, K = x.shape
        N = self.weight.shape[0]
        c = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        matmul_epilogue_kernel[grid](
            x, self.weight.T.contiguous(), c,
            self.bias,
            M, N, K,
            x.stride(0), x.stride(1),
            K, 1,
            c.stride(0), c.stride(1),
            ACTIVATION=1,  # change per task: 0=none, 1=relu, 2=gelu, 3=silu
        )
        return c
```

**nn.Linear weight reminder:** `nn.Linear(in_f, out_f).weight` has shape `(out_features, in_features)`. Pass `weight.T.contiguous()` as the B matrix, or use strides for implicit transpose.

## Tier 1: Algorithm Alternatives

### L1: 12_Matmul_with_diagonal_matrices (104x, iter 0) -- Algebraic simplification
**Key insight**: Diagonal matrix times dense is just row scaling -- diag(A) @ B = A[:, None] * B. Eliminates O(N^2*M) matmul for O(N*M) element-wise multiply. Always check for structured matrix properties before writing matmul kernels.
**What worked**: Simple 2D-tiled Triton kernel for broadcast multiply. Also applies: 14_UpperTri (14.5x, skip below-diagonal tiles), 15_LowerTri (10.2x, skip above-diagonal tiles), 10_3DTensor (4.8x, reshape to 2D), 11_4DTensor (3.8x, reshape to 2D).

### L2: 80_Gemm_Max_Subtract_GELU (54.2x, iter 0) -- Algebraic collapse to zeros / distribute reduction into weights
**Key insight**: Two classes of algebraic elimination: (a) Dead code -- when dim-1 reduction produces (B,1) then x-mean(x) on scalar=0, so gelu(0)=0, output is always zeros (54.2x). (b) Distribute sum/mean -- when matmul output is summed/averaged over one dimension, the reduction distributes into the weight matrix, reducing O(B*K*N) matmul to O(B*K) matvec.
**What worked**: 80_Gemm dead-code detection (54.2x). Sum distribution: 18_Matmul (48.1x, sum->matvec + chain collapses to identity ops on scalar), 14_Gemm (33.8x, sum->matvec via w_sum=W.sum(dim=0)), 51_Gemm (25.6x, mean->matvec + logsumexp on (B,1) is identity), 40_Matmul (9.7x, x*s+x = x*(1+s) algebraic combine fused into epilogue).

### L3: 31_VisionAttention (8.1x, iter 7) -- Flash attention
**Key insight**: For T=16384, materializing TxT attention matrix (1GB) is bottleneck. Flash attention with online softmax and tiled Q/K/V avoids this entirely. BLOCK_M=64, BLOCK_N=64.

## Tier 2: Architecture Variants

### L1: 97_ScaledDotProductAttention (1.97x, iter 2) -- Three-kernel attention decomposition
**Key insight**: F.scaled_dot_product_attention is banned. Decompose into 3 Triton kernels (Q@K^T*scale, row softmax, attn@V) using batched matmul with super-blocking. fp16 inputs enable tensor cores.
**What worked**: Each kernel independently autotuned. SEQ_LEN=512 fits softmax in single block. Materializes attention matrix (512MB fp16) but simpler and 2x faster than reference.

### L2: 76_Gemm_Add_ReLU (11.4x, iter 0) -- fp16 epilogue fusion template (canonical)
**Key insight**: Standard tiled fp16 matmul with bias+activation fused into epilogue. Canonical first-try pattern for Gemm+pointwise. Cached fp16 weight, super-blocking GROUP_M=8, 7 autotune configs. Scales to 5+ chained pointwise ops at zero extra cost.
**What worked**: First-try success. Also: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh (11.5x, 5 ops), 63_Gemm_ReLU_Divide (11.3x), 29_Matmul_Mish_Mish (11.3x, double mish), 12_Gemm_Multiply_LeakyReLU (11.0x), 70_Gemm_Sigmoid_Scaling_ResidualAdd (10.8x), 59_Matmul_Swish_Scaling (10.9x), 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp (10.0x), 68_Matmul_Min_Subtract (7.2x), 9_Matmul_Subtract_Multiply_ReLU (6.4x), 56_Matmul_Sigmoid_Sum (11.7x, epilogue+separate sum), 53_Gemm_Scaling_Hardtanh_GELU (5.8x).

### L2: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish (9.87x, iter 0) -- Three-kernel matmul+BN+postops
**Key insight**: Three-kernel approach for matmul+BN+post-ops: (1) fp16 matmul with linear bias epilogue, (2) per-channel BN stats with Bessel correction, (3) fused normalize+bias+divide+swish. Training mode BN requires separate stats computation since it needs all batch values.
**What worked**: BN stats computed per-channel across batch, running stats updated with momentum between kernels. Also: 84_Gemm_BatchNorm_Scaling_Softmax (6.1x, three-kernel + fixed softmax), 33_Gemm_Scale_BatchNorm (7.2x), 41_Gemm_BatchNorm_GELU_ReLU (3.5x).

### L2: 88_Gemm_GroupNorm_Swish_Multiply_Swish (8.1x, iter 0) -- Two-kernel Gemm+GN+chain
**Key insight**: Two-kernel approach: matmul with bias epilogue + fused GN+activation chain. GroupNorm cannot fuse into matmul epilogue (needs all group values). Small group_size (16-32 features) fits entirely in registers.
**What worked**: First-try success. Also: 30_Gemm_GroupNorm_Hardtanh (11.0x), 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm (8.0x, pointwise in matmul epilogue + GN), 62_Matmul_GroupNorm_LeakyReLU_Sum (6.9x), 37_Matmul_Swish_Sum_GroupNorm (4.6x), 39_Gemm_Scale_BatchNorm (4.0x, two-kernel BN variant).

### L2: 99_Matmul_GELU_Softmax (10.4x, iter 0) -- Two-kernel matmul+epilogue+reduction
**Key insight**: Two-kernel: matmul+pointwise epilogue + online softmax/logsumexp. Row-wise reductions (softmax, logsumexp) need all N values in a row, so cannot fuse into matmul tile-level epilogue. But matmul + pointwise fusion is free.
**What worked**: First-try success. Also: 66_Matmul_Dropout_Softmax (6.3x, dropout=identity in eval), 64_Gemm_LogSumExp_LeakyReLU (6.7x, logsumexp+activation chain), 45_Gemm_Sigmoid_LogSumExp (9.1x, three-kernel for two matmuls + logsumexp), 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish (3.3x, online logsumexp during N-tile iteration).

### L2: 55_Matmul_MaxPool_Sum_Scale (4.8x, iter 0) -- Two-kernel matmul+pooling+reduction
**Key insight**: Two-kernel: matmul+bias then fused pool+pointwise+reduction. MaxPool prevents algebraic distribution through matmul. Pool+sum kernel is trivial overhead compared to the GEMM.
**What worked**: First-try success. Also: 98_Matmul_AvgPool_GELU_Scale_Max (5.9x, avgpool+gelu+scale+max in one pass).

### L1: 2_Standard_matrix_multiplication (7.6x, iter 0) -- Standard tiled matmul template
**Key insight**: Standard tiled Triton matmul with super-blocking (GROUP_M=8) and 7 autotune configs consistently gives 2-8x over torch.matmul for large matrices. This is the canonical first-try template.
**What worked**: Tiled matmul with fp32 accumulation, covering block sizes from 32x32 to 128x128. First-try success pattern seen across: 1_Square (6.0x), 3_Batched (5.1x), 7_SmallK (2.5x), 8_Irregular (3.2x), 9_TallSkinny (1.6x), 13_Symmetric (6.7x), 16_TransA (5.1x), 17_TransB (5.9x), 18_TransBoth (6.5x).

## Tier 3-4: Tuning Guide

- **fp16 for large GEMMs (>1024x1024)**: Cache fp16 weight in __init__ via register_buffer. Gives +20-50% over inline cast. Mandatory for very large K. (Source: 12_Gemm 9.26x->11.03x, 29_Matmul 7.1x->11.3x, 33_Gemm 4.9x->7.2x, 41_Gemm 2.8x->3.5x)
- **Cached fp16 weight vs inline cast**: Cached register_buffer is strictly better than inline .to(tl.float16) in kernel. Inline cast can actually degrade performance for some K sizes. (Source: 53_Gemm 5.79x->4.87x with inline cast)
- **For transposed inputs**: Implicit transpose via strides (pass stride_bk=N, stride_bn=1 for (N,K)-shaped weight). Never .T.contiguous() -- it copies.
- **GroupNorm BLOCK_SIZE**: Must match group_size exactly when group_size is small (16-32). BLOCK_SIZE > group_size causes variance bugs from masked positions contributing to stats. Use tl.where for masking if BLOCK_SIZE > group_size is necessary. (Source: 62_Matmul)
- **Softmax BLOCK_N**: Use a single fixed config matching the actual feature dimension. Autotune with multiple BLOCK_N configs causes correctness failures from warmup corruption. (Source: 84_Gemm)
- **BN training mode**: Must compute batch stats with Bessel correction (divide by N-1 for variance), update running_mean/var with momentum. (Source: 39_Gemm, 41_Gemm, 97_Matmul)
- **Fast mish formula**: x * e * (e + 2) / (e * (e + 2) + 2) where e = exp(x). Uses only ONE exp() call per element, keeping register pressure manageable. (Source: 29_Matmul, 94_Gemm)
- **GELU tanh workaround**: tl.math.tanh may not exist. Use 2*sigmoid(2*x)-1 as substitute for tanh in GELU formula. (Source: 53_Gemm, 64_Gemm)
- **Autotune budget**: 4-7 configs optimal. More = more warmup overhead.

## Anti-Patterns

### L1: 4_Matrix_vector_multiplication (1.02x, iter 7) -- Bandwidth-bound matvec
**Key insight**: Matrix-vector multiply with M=2048, K=1048576 is purely bandwidth-bound (8GB of A data). cuBLAS gemv is near-optimal.
**Why it failed**: Split-K (0.5x, partial-buffer overhead), 2D blocks (0.97x, register pressure), fp16 cast (0.5x, 5ms+ for 8GB), tl.dot (correctness failure from K=1M accumulation), persistent kernel (0.5x). Simple 1-row-per-program with large BLOCK_K is best you can do.
**Better approach**: For huge matvec (M<4K, K>100K), accept ~1.0x. Don't attempt multi-row, split-K, or fp16 casting.

### L1: 6_Matmul_with_large_K (1.55x, iter 6) -- Large K split-K failures
**Key insight**: For large K (524288) with small M,N (256x256), fp16 pre-conversion halves bandwidth for K-loop reads. But split-K failed: atomic_add loses fp16 precision over 524K elements.
**Why split-K failed**: Both atomic split-K (correctness failure) and 2-kernel split-K (0.49x from 8 sequential launches) are inferior to single-pass fp16 matmul.
**Better approach**: Pre-convert both inputs to fp16, standard tiled matmul. No split-K.

### L2: 84_Gemm_BatchNorm_Scaling_Softmax (0x, iter 0) -- Softmax autotune correctness failure
**Key insight**: Autotuning softmax with multiple BLOCK_N configs causes correctness failures. Different block sizes during autotune warmup may corrupt results or produce incorrect partial reductions.
**Why it failed**: fp16 matmul + autotuned softmax (max_diff=0.052). The autotuner tries multiple block sizes, and partial-row softmax at wrong block boundaries produces wrong normalization constants.
**Better approach**: Use a single fixed BLOCK_N config matching the actual feature dimension. Fixed-block softmax: 4.55x->6.09x with fp16 matmul.

### L3: 28_VisionTransformer (0.69x, iter 18) -- Triton slower than cuBLAS for medium GEMM
**Key insight**: For medium matrices (394x512), Triton matmul is ~2x slower than cuBLAS. With 24 such matmuls, overall caps at 0.5-0.7x.
**Better approach**: Accept cuBLAS is structurally better for medium GEMMs.

## Decision Tree

1. **Check algebraic simplification first** (Tier 1): Diagonal = row scaling (104x). Triangular = skip tiles (10-15x). Structured matrices always check first. Sum/mean after matmul distributes into weights (25-48x). Dead code collapse (54x). Residual x*s+x = x*(1+s) (9.7x).
2. **For dense matmul (any shape)** (Tier 2): Standard tiled template with super-blocking GROUP_M=8, 7 autotune configs. Expect 2-8x. ~90% first-try success rate for L1.
3. **For Gemm + pointwise chain (L2)** (Tier 2): Epilogue fusion template -- fp16 tensor cores, cached weight. Expect 4-12x. ~80% first-try. Scales to 5+ ops.
4. **For Gemm + Normalization + acts (L2)** (Tier 2): Two-kernel: matmul+bias epilogue, then fused GN/BN+activation. Expect 3-12x. Three-kernel for BN (separate stats pass). GroupNorm: match BLOCK_SIZE to group_size.
5. **For Gemm + softmax/logsumexp (L2)** (Tier 2): Two-kernel: matmul+epilogue then online softmax/logsumexp. Expect 3-11x. Use fixed BLOCK_N for softmax (never autotune).
6. **For Gemm + pooling + pointwise (L2)** (Tier 2): Two-kernel: matmul+bias then fused pool+acts. Expect 4-6x.
7. **For attention/transformer** (Tier 1-2): Three-kernel decomposition for small SEQ_LEN (512). Flash attention for T>=1024 (8x). Split precision (IEEE for large-K, TF32 for small-K).
8. **fp16 tuning** (Tier 3-4): Cache fp16 weight via register_buffer in __init__. Never inline-cast in kernel. Gives +20-50% on large GEMMs.
9. **For transposed inputs** (Tier 3-4): Implicit transpose via strides. Never .T.contiguous().
10. **For matvec (M<4K, K>100K)** (Anti-Pattern): Accept ~1.0x. Bandwidth-bound.
11. **Autotune budget** (Tier 3-4): 4-7 configs optimal. More = more warmup overhead.
12. **Never**: Write "nn.Linear" / "nn.BatchNorm1d" / "nn.GroupNorm" in source (eval server string filter). Never use in-place writes. Always replicate exact weight init via nn.Parameter + manual init.
