# Kernel Optimization Strategies

Reference guide for optimization strategies by operation type.

## Element-wise Operations

Operations: ReLU, Sigmoid, GELU, Tanh, Add, Mul, Div

### Strategy: Vectorized Loads
- **When**: Large tensors (>1M elements), memory-bound
- **How**: Load 4 elements per thread, process in parallel
- **Expected**: 1.2-1.5x speedup
- **Example**:
```python
@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)
```

### Strategy: Block Size Tuning
- **When**: Unknown optimal configuration
- **How**: Test BLOCK_SIZE in [256, 512, 1024, 2048]
- **Expected**: 1.1-1.3x from finding optimal

### Strategy: Fast Math
- **When**: Transcendental functions (exp, log, sin, cos)
- **How**: Use `tl.math.fast_*` variants
- **Expected**: 1.1-1.4x for heavy math operations

## Reduction Operations

Operations: Sum, Mean, Max, Min, Softmax, LogSumExp

### Strategy: Tree Reduction
- **When**: Single-dimension reduction
- **How**: Thread accumulation → warp reduction → block reduction
- **Expected**: 1.3-2.0x speedup

### Strategy: Two-Pass Reduction
- **When**: Large reductions (>1M elements)
- **How**: First pass produces partial sums, second pass aggregates
- **Expected**: 1.5-2.5x for very large inputs

### Strategy: Warp Primitives
- **When**: Small reductions (<1024 elements)
- **How**: Use `tl.sum`, `tl.max` directly
- **Expected**: 1.2-1.5x by avoiding shared memory

## Normalization Operations

Operations: LayerNorm, BatchNorm, RMSNorm, GroupNorm

### Strategy: Single-Pass Welford
- **When**: Mean + variance in one pass
- **How**: Welford's online algorithm
- **Expected**: 1.3-1.8x by avoiding second pass

### Strategy: Fused Norm + Scale
- **When**: Norm followed by linear/scale
- **How**: Compute norm and apply transformation in one kernel
- **Expected**: 1.5-2.0x from fusion

### Strategy: Parallel Statistics
- **When**: Large hidden dimensions
- **How**: Parallel mean computation with tree reduction
- **Expected**: 1.4-2.0x

## Matrix Operations

Operations: Linear, BMM, Matmul, Attention

### Strategy: Tiled + Shared Memory
- **When**: Standard matrix multiply
- **How**: Load tiles into shared memory, accumulate
- **Expected**: 2-5x for compute-bound matmuls

### Strategy: Register Blocking
- **When**: Small-medium matrices
- **How**: Each thread computes multiple output elements
- **Expected**: 1.5-3x from reduced memory traffic

### Strategy: Flash Attention Pattern
- **When**: Attention mechanisms
- **How**: Online softmax, tiled QKV
- **Expected**: 2-4x for attention

## Convolution Operations

Operations: Conv1d, Conv2d, DepthwiseConv

### Strategy: Im2Col + Matmul
- **When**: Standard convolutions
- **How**: Reshape to matmul, use tiled kernel
- **Expected**: 1.5-2.5x

### Strategy: Direct Convolution
- **When**: Small kernels (3x3, 5x5)
- **How**: Direct sliding window with register reuse
- **Expected**: 1.3-2.0x for specific sizes

## Operation Classification Guide

| Pattern | Indicators | Recommended Strategies |
|---------|------------|----------------------|
| Memory-bound | Low FLOPs/byte ratio | Vectorization, coalescing |
| Compute-bound | High FLOPs/byte ratio | Register blocking, tiling |
| Reduction | `sum`, `mean`, `max` calls | Tree reduction, warp primitives |
| Broadcast | Different input shapes | Careful indexing, avoid divergence |
| Transpose | `.T`, `.permute()` | Shared memory transpose |
