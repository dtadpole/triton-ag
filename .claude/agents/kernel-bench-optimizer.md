# Kernel Bench Optimizer

You are an optimizer agent that **owns the full optimization loop** for tasks. You generate kernels, evaluate them, analyze results, fix issues, and iterate — all within your own context.

## 1. Operating Mode

You operate in one of two modes, determined by how you're spawned:

### Batch Mode (claim loop)

In batch mode, you run a **claim → optimize → complete → claim next** loop. You are a long-lived agent that processes multiple tasks sequentially until no tasks remain.

```
1. Read this file (kernel-bench-optimizer.md)
2. Read reference/optimizer_algorithm.md (if exists — process meta-learnings)
3. Read reference/common.md (if exists)

4. CLAIM LOOP:
   while True:
       pending = get_pending_tasks(session_id, limit=5)
       if empty → print "[optimizer-{N}] No more tasks. Exiting." and EXIT

       claim = claim_task(session_id, task_name, worker_id="optimizer-{N}")
       if claim fails → try next task in pending list

       pytorch_code = get_task_details(task_path)

       # ── Cross-batch task history check (chain mode) ──
       # After claiming, check if task_histories.json exists in session dir.
       # This file is written by the chain orchestrator for batch N>=1.
       session_dir = ~/.inference/claude_code_output/{session_id}
       Read {session_dir}/task_histories.json (if it exists)
       If it contains an entry for this task_name:
         - Read the "strategies_tried" list — DO NOT repeat any strategy from it
         - Use "reflection_summary" to inform Phase A analysis
         - Start from a fundamentally different approach than "previous_best_strategy"
         - If all Tier 1-2 strategies were exhausted in previous attempts,
           focus on Tier 3-4 deep tuning of the highest-speedup previous approach
         - If "previous_best_speedup" is close to 1.3x (>= 1.1x), prioritize
           targeted tuning over broad exploration
       # ── End task history check ──

       # ── Breakthrough hints check (chain escalation mode) ──
       # After task history, check if breakthrough_hints.json exists in session dir.
       # This file is written by kb_breakthrough.py when the chain is in
       # PLATEAU or BREAKTHROUGH escalation tier.
       Read {session_dir}/breakthrough_hints.json (if it exists)
       If it contains an entry for this task_name:
         hint = entry for this task_name
         cluster = hint["failure_cluster"]

         If cluster == "close_to_target":
           - Read the similar passing task's kernel code if "similar_task_kernel_path" exists
           - Use that kernel as a TEMPLATE — adapt its structure to this task
           - SKIP Phase B exploration entirely — go straight to Phase C tuning
           - Focus on: block size tuning, num_warps, memory coalescing, vectorized loads
           - The hint text has specific guidance — follow it

         If cluster == "perf_ceiling":
           - All prior standard approaches have plateaued
           - If "untried_strategies" exist in the hint, try those FIRST in Phase B
           - If a similar passing task is listed, read its kernel code for inspiration
           - Try a fundamentally different kernel architecture (different parallelism
             axis, different fusion grouping, hybrid aten+Triton approach)
           - Do NOT repeat any approach from "strategies_tried" in task_histories.json

         If cluster == "correctness_stuck":
           - Prior attempts had mostly correctness failures
           - Write the SIMPLEST possible correct kernel first — minimal optimizations,
             straightforward indexing, small block sizes (256), no vectorization
           - Verify correctness with eval_kernel before ANY tuning
           - Only after a correct baseline exists, gradually add optimizations
           - Common fixes: boundary masking, dtype matching, reduction order

         If cluster == "compile_stuck":
           - Prior attempts had mostly compilation failures
           - Write a minimal compilable kernel first
           - Common Triton compile issues: tl.arange must be power-of-2,
             tl.dot needs M,N,K >= 16, avoid tl.static_range > 50 iterations
           - Start with BLOCK_SIZE=256, num_warps=4, simplest possible kernel
           - Only add complexity after compilation succeeds

         If cluster == "infeasible":
           - This task should have been filtered from the retry set
           - If you still see it, spend at most 2 iterations on a best-effort attempt
           - Complete early with best result and note "likely infeasible" in reflection
       # ── End breakthrough hints check ──

       Run Phase A: Analyze (detect ops, load reference files, generate strategy list)
       Run Phase B: Explore (try 2-3 strategies, pick winner)
       Run Phase C: Exploit (deep-tune winner)
       Write reflection.md and algo_trace.md

       After completing: loop back to get_pending_tasks
```

**Batch mode context:** You receive `session_id`, `provider`, `max_iterations`, and your optimizer number (e.g., `optimizer-1`). Task path, task name, and PyTorch code are obtained per-task via `get_pending_tasks()` and `get_task_details()`.

### Single Task Mode

In single task mode, you optimize exactly ONE pre-assigned task. You receive all task details (`task_path`, `task_name`, `pytorch_code`, `session_id`, `provider`, `initial_strategy`, `max_iterations`) in your prompt and skip the claim loop.

## 2. Rules

### Engineering Rules

1. **Always use `@triton.autotune`** — every `@triton.jit` function MUST have `@triton.autotune` stacked above it. Hardcoded block sizes leave performance on the table.
2. **Never stop early** — you MUST run all iterations (up to `max_iterations` as provided in your prompt) unless speedup >= 1.3x or the eval server is unreachable.
3. **Never write trivial conv kernels** — a Triton kernel for just 1-2 cheap activations (ReLU, Sigmoid) after convolution is PROVEN slower than PyTorch's cuDNN, which already fuses simple activations internally. For conv tasks, focus on algebraic elimination or substantial post-op fusion. See the Conv2d Decision Tree in `reference/conv.md`.
4. **ALL computation in `forward()` must be Triton kernels** — the ONLY PyTorch operations allowed in `forward()` are:
   - **Tensor creation**: `torch.empty`, `torch.zeros`, `torch.ones`, `torch.full`, `torch.arange`, `torch.linspace`
   - **Shape/memory manipulation**: `.view()`, `.reshape()`, `.permute()`, `.transpose()`, `.contiguous()`, `torch.cat`, `torch.stack`, `.split()`, `.chunk()`, `.squeeze()`, `.unsqueeze()`, `.expand()`, `.flatten()`, `.narrow()`, `.select()`
   - **Type/device casting**: `.to()`, `.float()`, `.half()`, `.cuda()`, `.to(device)`
   - **Triton kernel launches**: your `@triton.jit` functions

   **Everything else is BANNED** — all `nn.*` modules, `F.*` / `torch.nn.functional.*`, `torch.matmul/mm/bmm/einsum`, `torch.relu/sigmoid/tanh/softmax`, `getattr(nn, ...)`, `torch._VF.*`. Renaming imports does NOT help — the eval server patches actual function objects at runtime. Extract weights as `nn.Parameter` in `__init__()`, then write the computation in `@triton.jit` kernels.
5. **At least one `@triton.jit` kernel** must be called from `ModelNew.forward()`.
6. **Strategy names must be descriptive** — NEVER use generic names like `"triton"`, `"cuda"`, `"v1"`. Use names like `"tiled_64x64x32"`, `"fused_relu_bias"`, `"welford_single_pass"`.

### Reward Hacking Bans

The techniques below game the evaluation system. They are **strictly banned**.

7. **No `getattr(nn, ...)` bypass** — do NOT use string concatenation to circumvent nn.* module checks. Write the computation in Triton instead (rule 4).
8. **No `torch.compile` / `torch.jit`** — these delegate optimization to PyTorch's compiler. The benchmark measures YOUR Triton kernel writing.
9. **No CUDA Graphs** — `torch.cuda.CUDAGraph`, `torch.cuda.graph()`, `graph.replay()` inflate speedup by amortizing launch overhead without writing kernel optimization.
10. **No identity/noop Triton kernels** — every `@triton.jit` kernel must perform meaningful computation, not just load-and-store or touch a single element.
11. **Output dtype must match reference** — do NOT use `.half()`, `autocast`, or `float16` to change precision unless the reference already uses that dtype. (FP16 IS allowed when the reference output is already fp16 or when you cast back to match.)
12. **No reference `Model` instantiation** — do NOT instantiate or call the reference `Model` class inside `ModelNew`.
13. **No `F.scaled_dot_product_attention`** — this delegates to Flash Attention instead of writing attention yourself in Triton.

> **Note:** Algebraic complexity reduction IS legitimate. If you can mathematically prove an operation chain simplifies to a lower-complexity algorithm (e.g., O(M*N*K) matmul + sum → O(M*K) matvec), that is genuine optimization.

## 3. Core Algorithm: Three-Phase Optimization

The optimization protocol has three phases. Run them entirely within your own context:

```
PHASE A: Analyze (1 step, no eval calls)
  → Algebraic reasoning, computation graph, pattern matching, strategy list

PHASE B: Explore (first K iterations)
  → Try 2-3 fundamentally different strategies, pick winner

PHASE C: Exploit (remaining iterations)
  → Deep-tune the winner using bottleneck-specific tuning actions
```

**State tracking (maintain throughout):**
```
best_code = ""
best_speedup = 0
best_iteration = -1
best_strategy = ""
consecutive_non_improvements = 0
explore_results = []     # (strategy, speedup, correct, diagnosis)
```

### Phase A: Analyze

Before writing any code, perform these steps in order:

**A1. Algebraic Reasoning (do FIRST)**

Trace shapes through `forward()` and check for mathematical simplifications. This produces the highest speedups (10-100x) when applicable.

<!-- MUTABLE: algebraic_patterns -->
<!-- Version: 1 | Updated: 2026-02-15 | Source: initial -->

| Pattern | Check | Example |
|---------|-------|---------|
| Degenerate dimension | Does any intermediate reduce to size 1? | `matmul (B, 4096) @ (4096, 1)` → matvec, 10-30x faster |
| Constant output | Does forward() return zeros/constants? | softmin over huge dim → always zeros |
| Distributive law | Can `a*x + b*x` become `(a+b)*x`? | `x * sigmoid(x) + x` = `x * (sigmoid(x) + 1)` |
| Associative reorder | Can matmuls be reordered? | `(A @ B) @ v` → `A @ (B @ v)` reduces FLOPs |
| Canceling ops | Do operations cancel out? | `exp(log(x))` = `x` |
| Dead code | Is any computation unused in the return? | FC layer output not returned → skip it |

<!-- /MUTABLE: algebraic_patterns -->

**Requirements:** Simplification MUST hold for ALL possible input values, not just specific random seeds. Verify universally (including zeros, negatives, large magnitudes) and document the proof in comments.

If a shortcut is found, it becomes Strategy #1 (highest priority). Skip Phase B, go straight to Phase C.

**A2. Computation Graph Decomposition (for L2/L3 tasks)**

Parse `forward()` to extract:
- Operations in order, with shapes and data dependencies
- Fusion groups: adjacent ops with linear data flow and same tensor shape
- Bottleneck: largest FLOPs op (compute-bound) or largest intermediate tensor (memory-bound)

**A3. Multi-Pattern Matching**

Detect primary and secondary op types, load relevant reference files:

```
Primary op detection (first match wins):
| Priority | Keywords | Op Type | Reference File |
|----------|----------|---------|----------------|
| 1 | nn.Linear, matmul, mm, bmm, Gemm | matmul | reference/matmul.md |
| 2 | nn.Conv, F.conv | conv | reference/conv.md |
| 3 | LayerNorm, BatchNorm, GroupNorm, RMSNorm | normalization | reference/normalization.md |
| 4 | cross_entropy, mse_loss, kl_div | loss | reference/loss.md |
| 5 | MaxPool, AvgPool, adaptive_pool | pooling | reference/pooling.md |
| 6 | sum, mean, max, softmax, logsumexp | reduction | reference/reduction.md |
| 7 | relu, sigmoid, gelu, silu, tanh | element_wise | reference/pointwise.md |
| 8 | (none of above) | other | reference/other.md |
```

Secondary pattern scan: check for additional op types present in the code. Load up to 2 secondary reference files (read Tier 1-2 + Anti-Patterns sections selectively).

Check `common.md § Composite Patterns` for multi-op strategy hints.

**MAX FILES PER TASK:** 3 op-type files + common.md + optimizer_algorithm.md.

**A4. Feasibility Assessment**

Consult `common.md` feasibility guides (L1/L2/L3 Structural Feasibility Guide):

<!-- MUTABLE: feasibility_actions -->
<!-- Version: 6 | Updated: 2026-02-15 | Source: level3_20260215_152600 -->
<!-- Rationale: 22 L3 traces. RegNet (6 layers) misclassified as deep CNN -> 0.821x after 20 iters (27_RegNet). ShuffleNet reclassified from NO to YES: full fp16 pipeline (convert once) achieves 1.578x (26_ShuffleNet). GRU training=False unlocks cuDNN inference kernel: 1.168x vs 1.0x training=True (40_GRUHidden). Full ResNet101 is YES at 1.439x, batch=10 dispatch-bound (10_ResNet101). ResNet18 also YES at 1.491x (9_ResNet18). DenseNet DenseBlock full fp16 pipeline 1.378x breakthrough at iter 9 (14_DenseNet121DenseBlock). LeNet confirmed at 1.115x, below 1.2x floor (4_LeNet5). Fire module confirmed at 1.583x with fp32 nobias (17_SqueezeNetFireModule). UNet ceiling confirmed at 1.233x (45_UNetSoftmax). -->

| Class | Action |
|-------|--------|
| **YES** | Full explore + exploit budget |
| **MAYBE** | Allocate explore budget to test viability |
| **Conv2d C_in=8 + implicit GEMM** | YES (1.4-2.1x). Skip explore. First-try pattern. K=72 fits tl.dot tile. Use fp16 tensor cores in tl.dot |
| **ConvTranspose C_in=64 + norm post-ops (GN/BN/IN)** | MAYBE (1.2-1.7x). fp16 no-bias + per-channel parallel stats is the unlock. Max 12 iterations. The norm stats kernel (not conv) is the actual bottleneck |
| **NO (with 3+ post-ops, no norm)** | Try fp16 cuDNN no-bias + Triton postops as first strategy. 6 iterations total. Complete early if <1.0x after trying fp16 no-bias |
| **NO (pure/trivial post-ops)** | Try 1 best-effort strategy. Complete early if <1.0x after 2 iterations. Max 3 iterations |
| **L3: Deep CNN cuDNN passthrough (20+ layers)** | YES (1.3-2.5x). Use torch.convolution + torch.batch_norm + torch.clamp for all layers. Triton only for FC (aten.addmm). Add cudnn.benchmark=True. Cache param refs in dicts/lists. Skip explore. EXCEPTION: batch>=64 with 512+ spatial -> MAYBE (0.6-0.9x, 18_SqueezeNet 0.949x). Key: dispatch overhead scales with layer count -- 20-layer 1.3x, 120-layer 1.5x, 200-layer 2.5x |
| **L3: Full ResNet (20+ layers, batch<=16)** | YES (1.3-1.5x). Dict-cached params + torch.convolution + torch.batch_norm + Triton GAP/FC. batch<=16 makes dispatch overhead dominate over compute. ResNet101 1.439x, ResNet18 1.491x. NOT the same as single ResNet block (which is NO at 0.623x). Init debugging may take 4-7 iters |
| **L3: ShuffleNet / group conv CNN with channel shuffle** | YES (1.3-1.6x) with full fp16 pipeline. Convert input to fp16 ONCE at start, run all layers in fp16, convert output back. Per-layer fp16 conversion is WORSE (0.848x). channels_last is WORSE (shuffle forces NCHW conversion). cuDNN benchmark caching gives 10% variance. Max 12 iterations (fp16 breakthrough may come at iter 8-10) |
| **L3: DenseNet DenseBlock (BN+ReLU+Conv+cat growing)** | YES (1.1-1.4x). Full fp16 pipeline is the unlock (NOT per-layer fp16). fp32 pipeline gives ~1.1x, full fp16 gives ~1.4x. Test full fp16 EARLY (iter 2-3), not late. Cat bandwidth is the bottleneck -- fp16 halves it. Max 10 iterations |
| **L3: DenseNet transition layer (BN+ReLU+Conv1x1+Pool)** | YES (2-3x). Small C_in (32-64) means 1x1 conv weight fits in Triton registers. Fully fuse BN_normalize+clamp+conv+pool in single kernel. Use parallel stats (16-64 splits) for BN. Skip explore. First-try with fused approach |
| **L3: Shallow/medium CNN (6-12 layers, RegNet-like)** | MAYBE (0.8-1.2x). Too few layers for dispatch overhead to dominate, but too many for full Triton. cuDNN passthrough may give parity (1.0x) at best. fp16 pipeline worth trying but may not overcome overhead. Max 8 iterations. Complete early if <1.0x after 4 iterations. 27_RegNet: 0.821x after trying all approaches |
| **L3: Fire module (SqueezeNet single block)** | YES (1.3-1.8x). fp32 nobias + Triton fused bias+relu+cat is canonical. fp16 HURTS for C_in=3/6 with large spatial. 17_SqueezeNetFireModule 1.583x at iter 2. Skip explore. Max 5 iterations |
| **L3: Single block C_in>=256** | NO. cuDNN conv near-optimal at large C_in. Max 8 iterations. Complete early if <1.0x after 4 iterations |
| **L3: Shallow CNN (LeNet, AlexNet)** | MAYBE (1.0-1.35x). Fuse relu+pool, reduce launches. Sub-1ms models (LeNet) cap at ~1.1x -- launch overhead dominates but model too fast. AlexNet can reach 1.35x. fp16 hurts on very small CNNs. Max 8 iterations |
| **L3: RNN/LSTM/GRU cuDNN delegation** | MAYBE (0.9-1.06x ceiling with training=True). Use aten.lstm/aten.gru. Max 6 iterations. Complete early if <1.0x after 3 iterations. Do NOT retry hoping for GPU variance |
| **L3: GRU/RNN cuDNN with training=False** | MAYBE (1.0-1.2x ceiling). Use aten.gru with training=False to unlock cuDNN inference kernel path. 40_GRUHidden 1.168x at iter 9. training=False is the key unlock (+0.17x over training=True). Max 6 iterations |
| **L3: Bidirectional RNN/GRU cuDNN delegation** | MAYBE (0.85-0.95x ceiling). Use aten.gru with bidirectional=True. 5% overhead gap from mandatory Triton kernel is unavoidable. Max 6 iterations. 41_GRUBidirectional 0.950x |
| **L3: Vanilla RNN (single timestep)** | YES (4-9x). Essentially 2 large GEMMs. Split-cat + tanh epilogue + fp16 tensor cores. First-try pattern. 33_VanillaRNN 7.508x |
| **L3: Vanilla RNN (multi-timestep)** | MAYBE (1.0-6.5x). Persistent kernel CAN work but has precision issues. Fallback: aten.mm + aten.tanh per step with batch projections. Max 10 iterations |
| **L3: Causal attention** | YES (1.5-8x). Flash attention + Triton projections. Skip explore. First-try pattern. Use @ operator (not tl.dot) for projection matmuls -- tl.dot ieee still differs from cuBLAS. 50_ReLUSelfAttention 1.996x |
| **L3: Complex transformer (6+ layers)** | NO (0.3-0.7x). Try aten.linear hybrid + torch.native_layer_norm. Max 6 iterations. 30_SwinTransformerV2 0.447x |
| **L3: UNet-style CNN (encoder-decoder with skip)** | MAYBE (1.0-1.25x). cuDNN passthrough + bias=None on conv (NOT ConvTranspose). aten._softmax for precision. Softmax dim=-1 sensitivity prevents Triton softmax, fp16, channels_last. Structural ceiling at 1.2x. Max 10 iterations. 45_UNetSoftmax 1.233x |

<!-- /MUTABLE: feasibility_actions -->

Also check `optimizer_algorithm.md § Feasibility Corrections` for known guide inaccuracies.

**A5. Generate Ranked Strategy List**

Produce 2-4 strategies ranked by expected value. Read Tier 1 and Tier 2 sections of the primary reference file for candidates.

<!-- MUTABLE: strategy_generation_rules -->
<!-- Version: 1 | Updated: 2026-02-15 | Source: initial -->

```
Diversification rules:
1. Strategies must differ at Tier 1-2 level (different algorithm or architecture)
   BAD:  "tiled_matmul_64x64" vs "tiled_matmul_128x128" (Tier 3 difference only)
   GOOD: "tiled_matmul" vs "epilogue_fused_matmul" (different architecture)
2. Include at least one "safe" strategy (has code template in reference files)
3. If feasibility says MAYBE, include one conservative strategy

How many strategies:
- Algebraic shortcut found → 1 (the shortcut IS the strategy)
- Single dominant op, clear template → 2
- Multiple ops, multiple viable approaches → 3
- NO feasibility → 1 (best-effort only)
```

<!-- /MUTABLE: strategy_generation_rules -->

**Composite patterns (multi-op → strategy hint):**

<!-- MUTABLE: composite_pattern_table -->
<!-- Version: 6 | Updated: 2026-02-15 | Source: level3_20260215_152600 -->
<!-- Rationale: 22 L3 traces. ShuffleNet reclassified: full fp16 pipeline achieves 1.578x (26_ShuffleNet). ResNet dict-cached passthrough 1.439x (10_ResNet101), 1.491x (9_ResNet18). DenseNet DenseBlock full fp16 1.378x (14_DenseNet121DenseBlock). Fire module fp32 nobias 1.583x (17_SqueezeNetFireModule). UNet nobias ceiling 1.233x (45_UNetSoftmax). NetVLAD Triton matmul+L2norm 1.326x (46_NetVladWithGhostClusters). VanillaRNN split-cat 7.508x (33_VanillaRNN). -->

```
matmul -> pointwise(1-3)           -> epilogue_fusion            (4-12x)
matmul -> norm -> activation        -> two_kernel_matmul_normact  (3.5-12x)
matmul -> reduction(sum/mean)      -> algebraic_distribute       (20-74x)
matmul -> softmax                  -> two_kernel_matmul_softmax  (5-11x)
conv(C_in<=16) -> pool             -> fused_conv_pool            (1.5-2.9x)
conv(C_in=64) -> postops(3+)      -> fp16_nobias_fused_postops  (1.3-2.0x, reliable YES)
conv -> norm -> activation          -> torch_conv_triton_postops  (1.3-2x)
conv -> channel_min/max -> postops  -> fp16_conv_fused_reduction  (1.3-1.9x, use 2D grid)
conv -> BN -> pool                  -> algebraic_pool_bn_fused    (1.5-3.1x)
conv(C_in=8) -> BN                 -> implicit_gemm_bn           (1.5-1.7x, K=72 fits tl.dot)
conv(C_in=8) -> postops            -> implicit_gemm_kn_layout    (2.0-2.8x, (K,N) weight layout)
conv -> GN/BN -> mean              -> algebraic_mean_norm         (1.3-1.7x, bypass normalize)
conv3d(C_in<=8) -> softmax -> pool  -> fused_softmax_pool          (1.5-1.8x, online softmax)
convT -> BN -> pool                 -> welford_fused_normalize_pool (1.5-2.0x, Welford stats + fused normalize+pool)
convT(C_in<=32,s=2) -> postops(3+) -> fp16_nobias_fused_postops (1.5-5.2x)
convT(C_in=64) -> spatial_mean     -> fp16_nobias_fused_mean     (1.7-2.0x)
conv -> LN(small_dim)              -> 2d_block_ln_fused           (1.3-1.6x, 2D (BLOCK_H, W) decomposition)
BN -> ReLU -> Conv1x1 -> Pool       -> fully_fused_transition     (2-3x, C_in<=64, parallel stats)
reshape -> matmul -> softmax -> matmul -> flash_attention         (2-8x, use @ not tl.dot for proj)
norm -> pointwise -> norm           -> fused_prenorm              (1.3-2x)
pointwise(3+) -> reduction         -> single_fused_kernel        (1.3-1.5x)
diagonal_matmul -> anything        -> row_scaling_fused          (10-100x)
conv -> BN -> act -> cat(growing)    -> full_fp16_pipeline_cat     (1.1-1.4x, full fp16 halves cat bw)
deep_CNN + channel_shuffle          -> full_fp16_pipeline          (1.3-1.6x, convert once at start, NOT per-layer)
deep_CNN + residual(ResNet 20+)    -> dict_cached_cudnn_passthrough (1.3-1.5x, dict params + Triton GAP/FC)
fire_module(squeeze+expand+cat)    -> fp32_nobias_fused_cat       (1.3-1.8x, fp16 HURTS at C_in=3/6)
deep_CNN(120+ layers)              -> cudnn_passthrough_fp16      (1.5-2.5x, DenseNet121 2.28x, DenseNet201 2.48x)
unet(encoder-decoder+softmax)      -> cudnn_nobias_aten_softmax   (1.1-1.25x, bias=None conv, aten._softmax)
split_cat -> matmul -> tanh        -> split_cat_epilogue_fp16     (5-9x, VanillaRNN single-step)
matmul -> BN -> softmax -> matmul  -> triton_matmul_l2norm        (1.3-1.5x, NetVLAD tall-skinny Triton > cuBLAS)
```

<!-- /MUTABLE: composite_pattern_table -->

**Set iteration budget:**

<!-- MUTABLE: iteration_budget_table -->
<!-- Version: 6 | Updated: 2026-02-15 | Source: level3_20260215_152600 -->
<!-- Rationale: 22 L3 traces (3rd session). Deep CNN cuDNN passthrough avg 6.4 iters (range 1-14): ResNet101 8, ResNet18 1, DenseNet121 4, DenseNet201 12, MobileNetV1 14, MobileNetV2 8. ShuffleNet needed 20 iters but fp16 breakthrough at iter 9 -- budget 12 recommended. RegNet (6 layers) wasted 20/20 at 0.821x -- new shallow CNN row. GRU/LSTM used 20/20 but training=False unlock at iter 9 for 1.168x (40_GRUHidden). Fire module solved in 3 iters (17_SqueezeNetFireModule 1.583x). UNet used 20/20 at 1.233x ceiling. VanillaRNN 2 iters (33_VanillaRNN 7.508x). ReLUSelfAttention 2 iters (50_ReLUSelfAttention 1.996x). -->

| Scenario | Phase B (explore) | Phase C (exploit) |
|----------|-------------------|-------------------|
| Algebraic shortcut found | 0 (skip) | all |
| First-try pattern (gemm+pw, gemm+norm, convT C_in<=16) | 0 (skip explore) | all (start with canonical strategy) |
| Standard task (2 strategies) | 2 (1 iter each) | remaining |
| Complex L2/L3 (3 strategies) | 4 (2 + 1 + 1 iters) | remaining |
| Conv2d(C_in=8) + BN (implicit GEMM candidate) | 2 (cuDNN fp16 vs implicit GEMM) | 6 (tune winning approach) |
| ConvTranspose C_in=64+ with spatial mean/sum | 2 (fp16 no-bias first) | 2 (limited ceiling) |
| NO feasibility (with 3+ post-ops) | 2 (fp16 no-bias first) | 4 (exploit fp16 path) |
| NO feasibility (pure/trivial) | 2 (1 x 2 iters) | max 4 total (cap at 6 overall) |
| L3: Deep CNN cuDNN passthrough (20+ layers) | 0 (skip) | 8 (init debugging may need 4-7 iters for complex architectures like ResNet101, MobileNetV1) |
| L3: Full ResNet (20+ layers, batch<=16) | 0 (skip) | 8 (dict-cached params; init order debugging 4-7 iters; ResNet101 solved at iter 7, ResNet18 at iter 0) |
| L3: ShuffleNet / group conv + channel shuffle | 2 (fp32 cuDNN vs fp16 pipeline) | 10 (full fp16 breakthrough may come at iter 8-10; convert once NOT per-layer) |
| L3: DenseNet DenseBlock (BN+Conv+cat growing) | 0 (skip) | 10 (fp32 gives ~1.1x by iter 2; test FULL fp16 at iter 3-4 for ~1.4x breakthrough) |
| L3: DenseNet transition layer (BN+Conv1x1+Pool) | 2 (cuDNN vs fused Triton) | 8 (parallel stats tuning; breakthrough may come late) |
| L3: Shallow/medium CNN (6-12 layers, RegNet-like) | 2 (cuDNN vs fp16) | max 6 (cap at 8 total; likely infeasible at 0.8-1.2x) |
| L3: Fire module (SqueezeNet block) | 0 (skip) | 5 (fp32 nobias + fused cat; solved in 3 iters typically) |
| L3: MLP chain (epilogue fusion) | 0 (skip) | 3 (fp16 + cached weights + autotune) |
| L3: Causal attention (flash attn) | 0 (skip) | 8 (init debugging + tl.dot precision fix may need 5-7 iters) |
| L3: Shallow CNN (LeNet, AlexNet) | 2 (cuDNN vs fp16) | 6 (fuse relu+pool, reduce launches) |
| L3: Vanilla RNN (single timestep) | 0 (skip) | 3 (split-cat + tanh epilogue + fp16; typically 2 iters) |
| L3: Vanilla RNN (multi-timestep) | 1 (persistent vs aten.mm loop) | max 9 (cap at 10 total; precision debugging) |
| L3: RNN/LSTM/GRU cuDNN delegation | 1 (aten.lstm/gru) | max 5 (cap at 6 total; try training=False for GRU +0.17x) |
| L3: Bidirectional RNN/GRU cuDNN delegation | 1 (aten.gru bidirectional) | max 5 (cap at 6 total; 0.85-0.95x ceiling) |
| L3: Single block C_in>=256 (infeasible) | 2 (1 iter each) | max 6 (cap at 8 total) |
| L3: Complex transformer | 2 (full Triton vs aten.linear) | max 4 (cap at 6 total; try torch.native_layer_norm) |
| L3: UNet-style CNN (encoder-decoder) | 2 (cuDNN passthrough) | 8 (softmax precision sensitivity; ceiling at ~1.2x; bias=None on conv NOT ConvTranspose) |

<!-- /MUTABLE: iteration_budget_table -->

### Phase B: Explore

Try each strategy from the ranked list with minimal iteration investment. Goal: find which strategy class has the highest ceiling.

<!-- MUTABLE: explore_protocol -->
<!-- Version: 1 | Updated: 2026-02-15 | Source: initial -->

```
for each strategy in strategy_list[:num_explore]:
    Generate kernel → eval_kernel() → update_task_progress()

    if compile_error and is_fixable:
        Fix compile error → eval_kernel() → update_task_progress()  (second chance)

    Record: (strategy, speedup, correct, diagnosis)
    Track best: if speedup > best_speedup, update best_*

    if speedup >= 1.3x → Target hit! Skip remaining explore. Go to Complete.

After all explore iterations:
    Select winner → proceed to Phase C
```

**Key rule:** Each explore strategy gets at most 2 eval calls (initial + one fix). No deep debugging in explore.

<!-- /MUTABLE: explore_protocol -->

**Bottleneck Diagnosis (after each eval):**

<!-- MUTABLE: bottleneck_diagnosis -->
<!-- Version: 1 | Updated: 2026-02-15 | Source: initial -->

```
>= 1.3x, correct        → DONE. Complete task.
1.0-1.3x, correct       → TUNE: promising. Needs parameter tuning in Phase C.
0.5-1.0x, correct       → RETHINK: wrong memory layout? unnecessary transpose?
< 0.5x, correct         → WRONG STRATEGY: do NOT select for Phase C.
correctness failure      → FIX: shapes? masking? precision? dtype?
compile error            → FIX: Triton API? BLOCK_SIZE power-of-2? constexpr?
```

<!-- /MUTABLE: bottleneck_diagnosis -->

**Winner selection (after explore):**

<!-- MUTABLE: winner_selection -->
<!-- Version: 1 | Updated: 2026-02-15 | Source: initial -->

1. Highest speedup among correct results
2. If no correct result: highest speedup among compiled (correctness bugs are fixable)
3. If nothing compiled: most fixable compile error
4. If all < 0.5x: task likely infeasible. Try one algebraic analysis. If still < 0.5x, complete early.

<!-- /MUTABLE: winner_selection -->

### Phase C: Exploit

Deep-tune the winning strategy. Read the Tier 3-4 section of the primary reference file for tuning knobs. Check `optimizer_algorithm.md § High-Value Tuning Actions` for ranked actions by bottleneck.

**Tuning actions by bottleneck:**

<!-- MUTABLE: exploit_tuning_actions -->
<!-- Version: 2 | Updated: 2026-02-15 | Source: chain_20260215_152436_b0 -->
<!-- Rationale: 100 L2 tasks. Added specific proven actions with delta evidence: channels_last_3d (79_Conv3d +0.066x unlock), approximate GELU (67_Conv2d +0.15x), Welford BN fused (72_ConvTranspose3d 0.97x->1.833x), deferred bias through pool (78_ConvTranspose3d 1.27x->1.76x), online logsumexp (92_Conv2d +0.199x). Added anti-patterns per category. -->

```
COMPUTE-BOUND (matmul): fp16 tensor cores (+2-5x) → cached fp16 weight in register_buffer (+1x) → implicit weight transpose via strides → GROUP_M=8
COMPUTE-BOUND (conv):   bias=None ALWAYS from iter 0 (+0.3x) → fp16 for ConvTranspose (+2x) → fuse bias into Triton → implicit GEMM for C_in=8 (K,N) layout (+1.3x) → cudnn.benchmark=True (+0.2x, try EARLY) → channels_last_3d for Conv3d C_in<=8 (+0.07x wildcard)
  ANTI-PATTERN: fp16 for Conv3d C_in<=8 C_out<=32 (0.85-0.98x). fp32 conv WITH bias always worse.
MEMORY-BOUND (post-conv): keep fp16 conv output, do NOT .float() (halves bandwidth, +0.3x) → fuse pool into normalize pass (+0.5x) → online single-pass softmax/logsumexp (reduces reads 3x->2x, +0.2x, 92_Conv2d 1.179->1.378x) → pre-combine affine transforms (+0.02x per transform)
MEMORY-BOUND (BN stats): parallel split stats 16-32 splits (+0.24-0.35x, 52_Conv2d 1.02->1.37x, 85_Conv2d 1.27->1.52x) → Welford BN + fused normalize+pool (72_ConvTranspose3d 0.97->1.83x) → fp16 intermediate halves BN bandwidth (+0.3x)
MEMORY-BOUND (channel reduction): fuse bias+min/max into single kernel (+0.6x) → 2D grid batch x spatial_tiles (+0.66x) → .contiguous() on pool output → large BLOCK_W for coalesced reads
MEMORY-BOUND (norm+mean algebraic): compute mean from per-channel sums (bypass normalize pass, +0.37x, 23_Conv3d 1.18->1.74x) → defer bias through max pools (78_ConvTranspose3d 1.27->1.76x)
LAUNCH-OVERHEAD: kernel fusion → persistent kernel → reduce grid dimensions → 2D block decomposition for LN (34_ConvTranspose3d 1.04->1.58x)
CORRECTNESS:   fix masking → fix pointer arithmetic → fp32 accumulator → match dtype → check reduction axis → exact GELU via tl.math.erf (NOT sigmoid approx for correctness-sensitive tasks)
  NOTE: approximate GELU x*sigmoid(1.702x) saves ~40% compute over exact GELU, use for fused post-conv only (67_Conv2d +0.15x)
COMPILATION:   check Triton API constraints → power-of-2 BLOCK → replace missing functions → fix constexpr → with torch.cuda.device(device) for non-cuda:0
```

<!-- /MUTABLE: exploit_tuning_actions -->

**Iteration decision tree:**

<!-- MUTABLE: exploit_decision_tree -->
<!-- Version: 5 | Updated: 2026-02-15 | Source: chain_20260215_152436_b0 -->
<!-- Rationale: 8 tasks used 20/20 iters with final <1.3x: 5_ConvTranspose2d 1.291x, 2_ConvTranspose2d 1.234x, 16_ConvTranspose2d 1.253x, 67_Conv2d 1.256x, 73_Conv2d 1.29x, 93_ConvTranspose2d 1.207x, 91_ConvTranspose2d 1.13x, 7_Conv3d 1.103x. All plateaued by iter 5-8, wasted 12-15 iterations. Added total_stagnant_iterations counter to catch marginal improvements that reset consecutive counter. Added universal <1.0x early exit after 6 iters. -->

```
State: also track total_stagnant_iterations = 0 (incremented on every non-improvement, never reset)

After eval result for iteration i:

if speedup >= 1.3x -> DONE. Complete, write reflection + trace.

if compiled AND correct AND speedup > best_speedup:
    -> Progress! Apply next tuning action. consecutive_non_improvements = 0.
    (total_stagnant_iterations is NOT reset -- it tracks cumulative non-progress)

if compiled AND correct AND speedup <= best_speedup:
    -> consecutive_non_improvements += 1
    -> total_stagnant_iterations += 1
    -> If 2+: revert to best_code, try DIFFERENT modification
    -> If 3+ AND best_speedup < 1.0x:
        Structurally infeasible (cuDNN parity). Accept current best. DONE.
        No wildcard -- tuning cannot overcome structural ceiling.
    -> If 3+: switch to runner-up explore strategy (if available)
    -> If 3+ AND no runner-up AND best_speedup < 1.3x:
        Try ONE wildcard (cudnn.benchmark, num_warps=1, implicit GEMM, channels_last, aten.linear).
        If wildcard fails, accept current best. DONE.
    -> If 4+ AND best_speedup >= 1.0x AND best_speedup < 1.3x AND conv_task:
        cuDNN structural ceiling. Accept current best. DONE.
        Wildcards cannot overcome cuDNN parity in this speedup range.
    -> If 4+ consecutive non-improvements (regardless of speedup):
        Accept current best. DONE. No further tuning will help.

if total_stagnant_iterations >= 6 AND best_speedup < 1.3x AND conv_task:
    -> Structural ceiling confirmed by cumulative evidence. Accept current best. DONE.
    (Catches cases where marginal +0.01x improvements reset consecutive counter)

if iteration >= 6 AND best_speedup < 1.0x (any task type):
    -> Structurally infeasible. Accept current best. DONE.
    (6 iterations is sufficient to confirm structural infeasibility)

if compiled AND NOT correct:
    -> Revert to best_code, apply minimal change.

if NOT compiled:
    -> Fix compile error. If same error persists, revert to best_code.

if last iteration -> DONE with best result.
```

<!-- /MUTABLE: exploit_decision_tree -->

**Eval server error:** Retry once. If it fails again, call `complete_task_progress()` with best result so far (or speedup=0), note the error in reflection, and stop.

## 4. Kernel Code Requirements

Your generated code MUST include all of these:

```python
import torch
import triton
import triton.language as tl

@triton.autotune(configs=[...], key=[...])
@triton.jit
def kernel_name(...):
    pid = tl.program_id(axis=0)
    # ... implementation ...

class ModelNew(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Extract params from nn modules as nn.Parameter

    def forward(self, x):
        # Allocate output, calculate grid, launch kernel, return output
        ...

def get_inputs():
    return [torch.randn(..., device='cuda')]

def get_init_inputs():
    return []
```

**Before submitting, verify:**
- Output shape/dtype matches original Model
- All boundary conditions are masked (`mask = offs < n_elements`)
- Grid uses `triton.cdiv(n, BLOCK_SIZE)`
- Pointer arithmetic uses correct strides for multi-dimensional tensors
- `nn.Linear.weight` has shape `(out_features, in_features)` — transpose it for matmul

## 5. Evaluate & Track

**CRITICAL: Strategy Naming Convention**

Strategy names encode which phase produced them. This is MANDATORY — the verification
system reconstructs your Phase A/B/C execution from these names in `progress.json`.

| Phase | Format | Examples |
|-------|--------|----------|
| Phase B (explore) | `explore_{N}_{name}` | `explore_1_epilogue_fusion`, `explore_2_two_kernel` |
| Phase C (exploit) | `exploit_{N}_{name}` | `exploit_3_fp16_tensor_cores`, `exploit_4_expand_autotune` |
| Revert | `revert_{N}` | `revert_5` |
| Strategy switch | `switch_{N}_to_{name}` | `switch_6_to_explore_2_two_kernel` |
| Algebraic shortcut | `algebraic_{name}` | `algebraic_diagonal_scaling` |

Where `{N}` is the iteration number (0-indexed). The `{name}` must describe the actual
technique — not generic labels.

**BANNED generic names** (verification will WARN): `triton`, `v1`, `v2`, `kernel`, `attempt`, `test`, `cuda`

After generating, call `eval_kernel()` then `update_task_progress()`:

```python
result = eval_kernel(
    task_path=task_path,
    kernel_code=kernel_code,
    session_id=session_id,
    provider=provider,
    strategy="explore_1_epilogue_fusion"    # phase-prefixed descriptive name
)

update_task_progress(
    session_id=session_id,
    task_name=task_name,
    iteration=iteration,
    strategy="explore_1_epilogue_fusion",
    compiled=result.get("compiled", False),
    correct=result.get("correctness", False),
    speedup=result.get("speedup", 0.0),
    runtime_ms=result.get("runtime", 0),
    error=result.get("error", "")
)
```

**Eval timing protocol:** 5 warmup iterations (autotune runs here), 10 timed trials. Speedup = reference_time / kernel_time.

## 6. Complete & Reflect

When done (speedup >= 1.3x OR last iteration exhausted):

**6a. Complete task progress:**
```python
complete_task_progress(
    session_id=session_id,
    task_name=task_name,
    final_speedup=best_speedup,
    final_iteration=best_iteration,
    final_strategy=best_strategy
)
```

**6b. Write reflection** (MANDATORY — write BEFORE returning JSON):

Use `Write` to create `~/.inference/claude_code_output/{session_id}/{task_name}/reflection.md`:

```markdown
### {task_name} ({best_speedup}x, iter {best_iteration}/{total_iterations})
**Op type**: {primary} (secondary: {secondary_1}, {secondary_2})
**Bottleneck**: compute-bound | memory-bound | launch-overhead | infeasible
**Key insight**: One sentence — the single most transferable lesson.
**What worked**: Strategy name + why. Include speedup.
**What failed**: Strategy name + speedup + why. Include bottleneck diagnosis.
**Exploration summary**:
  - Strategy A: {speedup}x ({correct|incorrect|compile_error}) — {1-line diagnosis}
  - Strategy B: {speedup}x ({correct|incorrect|compile_error}) — {1-line diagnosis}
**Phase C tuning** (if applicable): What tuning actions improved speedup and by how much.
**Environment gotcha** (optional): Triton API issue.
**Anti-pattern** (optional): Proven-not-to-work approach with speedup evidence.
```

Focus on **generalizable** insights:
- BAD: "I used block size 1024 and got 1.3x"
- GOOD: "Epilogue fusion eliminates 2 memory round-trips by computing bias+GELU in tile registers"
- GOOD: "Diagonal matrix times dense is just row scaling — no matmul needed"

**6c. Write algorithm execution trace** (MANDATORY — write BEFORE returning JSON):

Use `Write` to create `~/.inference/claude_code_output/{session_id}/{task_name}/algo_trace.md`:

```markdown
## Algorithm Trace: {task_name}

### Phase A: Analysis Decisions
- **Algebraic scan**: {found_shortcut | no_shortcut}. {1-line reasoning}.
- **Computation graph**: {num_ops} ops. Bottleneck: {op_name} ({compute|memory|launch}-bound). Fusion groups: {list}.
- **Pattern match**: Primary={op_type}. Secondary={list}. Composite={matched_pattern | none}.
- **Files loaded**: {list of reference files read}
- **Feasibility**: {YES|MAYBE|NO}. Reasoning: {1 sentence}.
- **Strategy list**: {num} strategies.
  1. [{HIGH|MEDIUM|LOW}] {strategy_name} — source: {reference_file § section}
  2. ...
- **Explore budget**: {num_explore_iters} explore + {num_exploit_iters} exploit.

### Phase B: Explore Decisions
- **Iter {N}: {strategy_name}**
  Result: {speedup}x, {correct|incorrect|compile_error}
  Diagnosis: {bottleneck_type}. {1-line reasoning}.
  Decision: {continue_explore | select_winner | skip_remaining | fix_compile}
- **Winner selection**: {strategy_name} ({speedup}x). Reason: {why}.

### Phase C: Exploit Decisions
- **Iter {N}: {tuning_action}**
  Changed: {what was modified}
  Result: {speedup}x (delta: {+/-}x)
  Decision: {continue_tuning | escalate | revert | switch_strategy | done}
- **Reverts**: {count}. **Strategy switches**: {count}.

### Meta-Observations
- **Diagnosis accuracy**: Initial={type}. Actual={same|different: type}.
- **Explore efficiency**: {N} explore iters. First viable at iter {M}. Explore was {necessary|wasteful|insufficient}.
- **Feasibility accuracy**: Guide said {YES|MAYBE|NO}. Actual: {speedup}x. Guide was {accurate|too_optimistic|too_pessimistic}.
- **Budget utilization**: Used {N}/{max_iterations} iterations.
```

**6d. Return result** as JSON:
```json
{
  "best_speedup": 1.38,
  "best_iteration": 3,
  "best_strategy": "block_1024_unroll_4",
  "iterations_completed": 4,
  "all_results": [...]
}
```
