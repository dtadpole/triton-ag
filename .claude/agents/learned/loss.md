# loss patterns
<!-- Updated: 2026-02-14 | Source: 0212_v10_l1 -->

## What Works

### L1: 98_KLDivLoss (1.13x, iter 10) -- Fusing log saves one memory pass
**Key insight**: Fusing log(predictions) into the KL divergence kernel saves one full memory pass. PyTorch does log(pred) and kl_div as separate operations. Per-row reduction with BLOCK_SIZE=4096-8192 and num_stages=4 for pipelining.
**What worked**: target*(log(target)-log(pred)) in single pass. Separate mean reduction kernel with batchmean division. Log-ratio formulation (t/p then log) was slower due to division latency.

### L1: 95_CrossEntropyLoss (1.06x, iter 17) -- Dual-row per-program reduces scheduling
**Key insight**: Processing 2 rows per program halves program count while keeping register pressure manageable. Single-load BLOCK_SIZE=4096 (entire row fits in one load) avoids loop overhead.
**What worked**: Dual-row kernel computing logsumexp for 2 rows sequentially. 16384 programs vs 32768.

### L1: 96_HuberLoss (1.05x, iter 3) -- Two-stage map-reduce fusion
**Key insight**: Fusing diff/abs/where/sum into one read pass saves one kernel launch vs PyTorch's separate kernels. Two-stage: per-block partial sums + single-program final reduction.
**What worked**: BLOCK_SIZE autotune (1024-8192) for map-reduce. Atomic_add corruption fixed with autotune pre_hook for fresh output.

## What Fails

### L1: 100_HingeLoss (1.04x) -- Purely bandwidth-bound
**Key insight**: Hinge loss over 1B elements is purely memory-bandwidth bound. Every element must be read, computation is trivial (multiply+subtract+clamp+sum). PyTorch's kernel fusion already handles simple element-wise chains well.
**Why limited**: fp16 conversion overhead > savings. Atomic add non-deterministic. Multi-row approaches added overhead. 8192 was sweet spot for BLOCK_SIZE.
**Better approach**: Accept ~1.04x. No algorithmic shortcut exists.

### L1: 99_TripletMarginLoss (1.05x) -- Three-tensor bandwidth bottleneck
**Key insight**: Three 32768x8192 tensors = 768MB reads. Shared anchor reads (diff_ap and diff_an from same load) saves one full tensor read, but still 3 reads minimum.
**Why limited**: Dual-row processing (0.96x, register pressure from 6 loads). 2D grid chunk parallelism (sync overhead). Atomic_add for mean (autotune warmup corruption). Scalar output must use torch.empty(()) not torch.empty(1).
**Better approach**: Two-kernel (row losses + mean reduction). Accept ~1.05x.

## Decision Framework for Loss Tasks

1. **All loss functions on >1B elements**: Fundamentally bandwidth-bound. Max ~1.0-1.13x. Do not waste many iterations.
2. **Losses with log operations (KLDiv, CrossEntropy)**: Fuse log into the kernel to save one memory pass. Best opportunity for improvement (~1.1x).
3. **Simple losses (MSE, Huber, Hinge, Triplet)**: Two-stage map-reduce (per-block partial + final reduction). ~1.05x ceiling.
4. **Atomic_add pitfall**: Always use fresh output tensor per forward() call. Autotune warmup corrupts accumulated atomic results.
5. **Scalar output shape**: Must match PyTorch exactly. Use torch.empty(()) not torch.empty(1).
6. **Multi-row per program**: 2 rows can reduce scheduling overhead. 4+ rows cause register pressure.
