# Optimizer Algorithm Reference
<!-- Updated: 2026-02-15 | Source: level2_20260214_232629 -->
<!-- This file captures process-level meta-learnings about the optimization algorithm itself. -->
<!-- Populated by the learner agent from algo_trace.md files after each session. -->

## Diagnosis Calibration
<!-- Corrections to the bottleneck diagnosis framework. -->

- **Conv3d(C_in<=8) + post-ops**: Diagnosis often says compute-bound (conv dominates), but actual bottleneck can be memory-bound (conv output bandwidth for post-op reads). When conv output is 100M+ elements read multiple times for pool/norm, memory-bound tuning (reduce reads) is more effective than compute-bound tuning (fp16 tensor cores). Check: if conv output > 50M elements AND 2+ post-ops read it, diagnosis is memory-bound.
  (Source: L2: 8_Conv3d, 85_Conv2d, 43_Conv3d)

- **Conv2d(C_in=8) + BN**: Diagnosis says compute-bound (conv), actual bottleneck is BN stats reduction over large spatial (130M+ elements). The conv is fast; the Triton BN stats kernel is the bottleneck.
  (Source: L2: 73_Conv2d -- BN stats was the actual bottleneck, not conv)

- **ConvTranspose(C_in=64+, stride=2)**: Diagnosis says compute-bound and feasible. Actual: conv dominates at 90%+ of runtime and cuDNN is near-optimal. Structural ceiling at 1.0-1.25x regardless of post-op optimization.
  (Source: L2: 91_ConvTranspose2d 1.245x, 5_ConvTranspose2d 1.28x)

## Explore Budget Heuristics
<!-- How many explore strategies to try by task type. -->

- **Gemm + pointwise chain**: 1 strategy sufficient. First viable found by iter 0 in 85% of cases. Epilogue fusion is canonical -- skip explore when pattern matches.
  (Source: L2: 76_Gemm, 63_Gemm, 70_Gemm, 95_Matmul, 59_Matmul, 81_Gemm, 68_Matmul, 9_Matmul, 53_Gemm -- all first-try)

- **Gemm + Normalization + acts**: 1-2 strategies sufficient. First viable found by iter 0-1. Two-kernel is almost always the right approach. Main risk: BN correctness (Bessel correction, running stats momentum).
  (Source: L2: 88_Gemm iter 0, 94_Gemm iter 0, 30_Gemm iter 1, 84_Gemm iter 1, 97_Matmul iter 0)

- **Gemm + reduction (algebraic)**: 0 strategies needed. Algebraic shortcut found in Phase A analysis. Skip explore entirely.
  (Source: L2: 80_Gemm, 51_Gemm, 14_Gemm, 18_Matmul -- all 0-1 iters)

- **Conv + post-ops (C_in<=16)**: 2-4 strategies. First viable found by iter 0-3. Key decision: cuDNN fp16 no-bias vs full Triton implicit GEMM.
  (Source: L2: 48_Conv3d iter 0, 87_Conv2d iter 3, 82_Conv2d iter 4, 8_Conv3d iter 8)

- **Conv + trivial post-ops**: 3-6 strategies, often all < 1.0x. Infeasible tasks waste entire budget. Set max 6 explore iterations for MAYBE-feasibility tasks.
  (Source: L2: 69_Conv2d 20 iters, 71_Conv2d 20 iters, 57_Conv2d 20 iters, 73_Conv2d 20 iters)

- **ConvTranspose(C_in<=16, stride=2) + post-ops**: 1-2 strategies. fp16 no-bias is almost always optimal. First-try success in 70% of cases.
  (Source: L2: 89_ConvTranspose3d iter 0, 96_ConvTranspose3d iter 1, 50_ConvTranspose3d iter 0)

## Feasibility Corrections
<!-- Where the L1/L2/L3 feasibility guides in common.md are inaccurate. -->

- **Conv2d(C_in<=16) + post-ops**: Guide says YES (1.4-2.9x). Actual success rate is 80% at avg 1.6x. Guide is accurate for tasks with 3+ substantial post-ops but too optimistic for tasks with 1-2 trivial post-ops (relu+hardswish gives only 1.0-1.25x).
  (Source: L2: 46_Conv2d 1.53x, 82_Conv2d 1.48x, 87_Conv2d 1.98x; but 69_Conv2d 1.08x, 71_Conv2d 1.2x)

- **Conv3d(C_in<=8) + post-ops**: Guide says YES (1.3-1.9x). Actual success rate 75% at avg 1.3x. Accurate when post-ops are substantial. fp16 not beneficial for Conv3d C_in<=8 (use fp32).
  (Source: L2: 48_Conv3d 1.87x, 90_Conv3d 1.53x, 7_Conv3d 1.1x, 8_Conv3d 1.45x)

- **ConvTranspose(C_in<=16, stride=2) + post-ops**: Guide says YES (1.0-6.0x). Actual success rate 100% at avg 3.5x. Guide is accurate -- this is the most reliable conv pattern.
  (Source: L2: 96_ConvTranspose3d 5.2x, 89_ConvTranspose3d 4.4x, 50_ConvTranspose3d 4.4x, 72_ConvTranspose3d 3.1x)

- **ConvTranspose(C_in=64+, stride=2)**: Guide says NO (0.1-0.9x). Actual: ceiling is 1.0-1.25x with post-ops. Guide is slightly too pessimistic -- with fp16 no-bias + fused post-ops, 1.0-1.25x is achievable (but still below target).
  (Source: L2: 91_ConvTranspose2d 1.245x, 5_ConvTranspose2d 1.28x)

- **Gemm + pointwise chain**: Guide says YES (4-12x). Actual: 100% success rate at avg 9.2x. Guide is accurate. Most reliable pattern.
  (Source: 12 tasks, all >= 5.8x)

- **Gemm + Normalization + acts**: Guide says YES (5-12x). Actual: 100% success rate at avg 7.0x. Guide was slightly too optimistic at the low end (3.5x for some BN tasks) but accurate overall.
  (Source: L2: 30_Gemm 11.0x, 41_Gemm 3.5x, 88_Gemm 8.1x)

## High-Value Tuning Actions
<!-- Tier 3-4 actions ranked by impact, by bottleneck type. -->

- **compute-bound (matmul)**: Top actions: 1. fp16 tensor cores -- avg improvement +2.5x (25 tasks). 2. Implicit weight transpose via strides -- eliminates .T.contiguous() copy. 3. Super-blocking GROUP_M=8 -- consistent small improvement.
  (Source: L2 matmul tasks)

- **compute-bound (conv)**: Top actions: 1. bias=None in torch.convolution -- avg improvement +0.2x (15 tasks). 2. fp16 for ConvTranspose -- avg improvement +2.0x (8 tasks). 3. Fuse bias into first Triton post-op kernel.
  (Source: L2 conv tasks)

- **memory-bound (post-conv)**: Top actions: 1. Keep fp16 conv output (don't .float()) -- avg improvement +0.3x (6 tasks). 2. Fuse pool into normalize pass -- +0.5x (2 tasks). 3. Pre-combine affine transforms -- +0.02x per transform removed.
  (Source: L2: 85_Conv2d, 23_Conv3d, 27_Conv3d)

- **memory-bound (BN stats)**: Top actions: 1. Parallel split stats (16-32 splits) -- +0.24x (2 tasks). 2. fp16 conv + Triton BN -- +0.19x (2 tasks). 3. Precompute alpha=gamma*rstd, beta=beta-mean*alpha -- +0.01x.
  (Source: L2: 73_Conv2d, 84_Gemm)

## Process Anti-Patterns
<!-- Common causes of wasted iterations. -->

- **Persisting on infeasible conv tasks past 6 iterations**: Wastes 10-14 iterations on average. Detection: after 6 iterations, best speedup still < 1.1x for conv with trivial post-ops. Fix: stop and accept current best.
  (Source: L2: 69_Conv2d used 20 iters for 1.08x, 71_Conv2d used 20 iters for 1.2x, 57_Conv2d used 20 iters for 1.25x)

- **Trying fp16 for Conv3d C_in<=8**: Wastes 1-2 iterations. Detection: C_in<=8 AND Conv3d. Fix: skip fp16 strategy, use fp32 cuDNN.
  (Source: L2: 7_Conv3d 0.88x, 8_Conv3d 0.979x, 79_Conv3d 0.911x -- 3 wasted fp16 attempts)

- **Setting TF32 flags in eval code**: Wastes 4-5 iterations due to global state corruption. Detection: correctness failures appear on all subsequent iterations. Fix: NEVER set torch.backends.cuda.matmul.allow_tf32.
  (Source: L2: 79_Conv3d -- lost 5 iterations)

- **Trying cuDNN conv WITH bias after seeing it's slower WITHOUT**: Wastes 1 iteration per task. Detection: known pattern. Fix: always use bias=None from first attempt.
  (Source: 10+ tasks wasted 1 iter each trying with-bias first)

- **Trying multiple softmax BLOCK_N configs via autotune**: Wastes 1-2 iterations. Detection: softmax with large dim. Fix: use single config matching actual dim.
  (Source: L2: 84_Gemm lost 1 iter to autotune correctness issue)

- **Late discovery of cuDNN bias=None optimization**: Wastes 3-7 iterations before breakthrough. Detection: conv task with < 1.2x after 3 iterations. Fix: always pass bias=None from iter 0.
  (Source: L2: 8_Conv3d discovered at iter 8, 82_Conv2d at iter 4)

## Revert & Switch Effectiveness
<!-- When to revert vs. switch vs. persist. -->

- **Revert after 1 regression**: Success rate 70%. Best when: regression is > 0.2x AND you had a previous result > 1.0x. The previous approach is structurally better; the new variant failed for a reason.
  (Source: L2: 79_Conv3d, 8_Conv3d, 73_Conv2d -- reverts led to new attempts that succeeded)

- **Switch strategy after 3 consecutive sub-1.0x results**: Success rate 40%. Best when: exploring fundamentally different approaches (cuDNN -> implicit GEMM, fp32 -> fp16). Switching within same approach class is wasteful.
  (Source: L2: 87_Conv2d -- switched to implicit GEMM at iter 3, got 1.98x)

- **Persist on winning strategy with minor tuning variations**: Success rate 30% for improvements > 0.1x after first 3 exploit iterations. Marginal tuning rarely helps much after initial optimization. Consider stopping if 3 exploit iterations show < 0.05x improvement each.
  (Source: L2: 71_Conv2d -- 18 exploit iterations with no improvement over initial 1.2x)

- **Early exit when target reached**: Always correct. Never continue optimizing once 1.3x is achieved -- diminishing returns.
  (Source: 35+ tasks correctly exited early)

## First-Try Success Patterns
<!-- Patterns that work on first attempt, requiring no exploration. -->

- **Gemm + 1-5 pointwise ops**: Epilogue fusion template. 85% first-try success rate. Average: 9.2x. Budget: 1-2 iterations.
  (Source: 12 L2 tasks)

- **ConvTranspose(C_in<=16, stride=2) + post-ops**: fp16 no-bias + Triton. 70% first-try success rate. Average: 3.5x. Budget: 1-2 iterations.
  (Source: 6 L2 tasks)

- **Algebraic shortcuts (mean/sum distributes over conv/matmul)**: 100% first-try success rate (after algebraic analysis). Average: 25x. Budget: 1-2 iterations.
  (Source: 5 L2 tasks)

## Iteration Budget Allocation

Based on 60+ L2 tasks:

| Pattern | Avg iters used | Recommended max | Notes |
|---|---|---|---|
| Gemm + pointwise | 1.3 | 3 | Almost always first-try |
| Gemm + norm + acts | 1.8 | 5 | Occasional BN correctness fix |
| Algebraic shortcut | 1.4 | 3 | Analysis is the hard part |
| ConvTranspose small C_in | 1.5 | 5 | fp16 no-bias is key |
| Conv + substantial post-ops | 4.5 | 10 | Multiple strategies needed |
| Conv + trivial post-ops | 15+ | 8 | Usually infeasible -- cap budget |
| Conv + BN (large spatial) | 18 | 12 | Parallel split stats key |
