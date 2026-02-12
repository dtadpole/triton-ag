# Learned Patterns
<!-- Aggregated from session e2e_test_relu | Updated: 2026-02-12 -->

## element_wise

### 19_ReLU (1.005x, iter 1)
**Op type**: element_wise
**Key insight**: Pure ReLU on large tensors (1.6B elements, 6GB at fp32) is entirely memory-bandwidth-bound with near-zero arithmetic intensity (1 FLOP per 12 bytes). PyTorch's built-in torch.relu is already near-optimal and essentially impossible to beat with Triton.
**What worked**: Simple flat 1D kernel with autotune configs spanning 1024-16384 block sizes and 4-16 warps achieved parity at 1.005x. The autotuner selected the config that best saturates the memory bus, matching PyTorch's built-in CUDA kernel. Hardcoded BLOCK_SIZE=1024 without autotune also achieves the same ~1.005x.
**What failed**: (1) Persistent kernels with grid-stride loops (0.916x) -- loop overhead and reduced parallelism hurt for such a trivially simple op. (2) In-place operation with clone (0.474x) -- the clone() call doubles memory traffic, negating any in-place benefit. (3) 2D grid scheduling added overhead without improving coalescing since data is already row-major contiguous. (4) Unrolled multi-element kernels compiled but triggered OOM on memory-constrained GPUs. (5) ~50% of eval attempts failed due to intermittent Triton/CUDA context mismatch on non-default GPU devices (cuda:1,2,3 produce "cpu tensor" errors while cuda:0 works reliably). (6) The fundamental limitation is that ReLU has the lowest possible arithmetic intensity -- no amount of kernel tuning can beat a well-optimized bandwidth-saturating implementation.
