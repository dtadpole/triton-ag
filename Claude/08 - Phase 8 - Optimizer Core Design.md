# Phase 8: Optimizer Core Algorithm — Detailed Design

> **Status:** Design
> **Scope:** Reflect & iterate, pattern matching with memory loading, strategy exploration
> **Depends on:** Phase 7 (flat spawning), existing reference knowledge system
> **Last updated:** 2026-02-14

---

## 1. Problem Statement

The current optimizer (Section 3 of `kernel-bench-optimizer.md`) has a linear iteration loop: generate → eval → react → repeat. This design has three weaknesses:

1. **Shallow reflection** — The "How to React to Results" table is a 5-row lookup. The optimizer doesn't systematically diagnose *why* a kernel is slow (memory-bound? compute-bound? launch overhead? wrong algorithm?) or reason about what to try next based on quantitative signals.

2. **Coarse pattern matching** — Op-type detection is keyword-based with 8 categories. An L2 task like `Conv2d + BatchNorm + ReLU + MaxPool` matches "conv" and loads `reference/conv.md`, but the dominant optimization opportunity might be pool fusion (pooling knowledge) or algebraic elimination (common.md). The optimizer has no mechanism to load *multiple* relevant reference files or to match against *sub-patterns* within a task.

3. **No strategy exploration structure** — The optimizer picks one strategy at iteration 0, then iteratively debugs/tunes it. If the strategy is fundamentally wrong (e.g., tiled matmul for a bandwidth-bound matvec), it wastes all remaining iterations. There's no concept of "try a different approach class" vs. "tune parameters within this approach."

### 1.1 Goals

| Goal | Metric |
|------|--------|
| **Higher first-try success** | Reduce iterations-to-1.3x by 30%+ via better strategy selection |
| **Fewer wasted iterations** | Detect dead-end strategies within 2 iterations, pivot |
| **Better knowledge utilization** | Load 2-3 relevant reference files per task, not just 1 |
| **Structured exploration** | Explicit phases: explore (try diverse approaches) → exploit (tune best) |
| **Richer reflections** | Capture quantitative signals (bottleneck type, memory bandwidth utilization) for future learning |

### 1.2 Non-Goals

- Changing the claim loop, session management, or MCP server (Phase 7 scope)
- Adding new MCP tools (use existing `eval_kernel`, `update_task_progress`, etc.)
- Modifying the eval server or its string filtering

---

## 2. Design Overview

Replace the current flat iteration loop with a **three-phase optimization protocol**:

```
Phase A: Analyze (1 step, no eval)
  - Algebraic reasoning (existing, unchanged)
  - Computation graph decomposition
  - Multi-pattern matching → load relevant reference files
  - Feasibility assessment → set iteration budget allocation
  - Generate ranked strategy list (2-4 candidates)

Phase B: Explore (first K iterations)
  - Try top 2-3 fundamentally different strategies
  - 1-2 iterations each (generate + eval + quick fix if compile error)
  - Record quantitative profile per strategy (speedup, bottleneck diagnosis)
  - After explore phase: select winner strategy

Phase C: Exploit (remaining iterations)
  - Deep-tune the winning strategy
  - Guided by bottleneck diagnosis from Phase B
  - Systematic parameter sweep (block sizes, num_warps, fusion depth)
  - Structured reflection at each iteration
```

**Iteration budget allocation** (default `max_iterations=20`):

| Scenario | Phase A | Phase B (explore) | Phase C (exploit) |
|----------|---------|--------------------|-------------------|
| High-confidence strategy (e.g., diagonal matmul → row scaling) | 1 | 0 (skip) | 19 |
| Standard task (1 dominant op type) | 1 | 4 (2 strategies × 2 iters) | 15 |
| Complex L2/L3 (multiple viable approaches) | 1 | 6 (3 strategies × 2 iters) | 13 |
| Known-infeasible (feasibility guide says NO) | 1 | 2 (1 best-effort) | 2 (minimal) |

---

## 3. Phase A: Analyze (Detailed Design)

### 3.1 Algebraic Reasoning (unchanged)

Existing algebraic reasoning (Section 3, Step 1 of optimizer.md) remains the first substep. It already handles:
- Degenerate dimensions, constant outputs, distributive law, associative reorder, canceling ops, dead code

No changes needed. If algebraic simplification is found, it becomes strategy #1 (highest priority).

### 3.2 Computation Graph Decomposition (new)

For L2/L3 tasks, build an explicit computation graph before strategy selection:

```
Input: PyTorch forward() code
Output: Ordered list of operations with shapes, data dependencies, and fusion groups

Algorithm:
1. Parse forward() to extract operations in order
2. For each operation, record:
   - Op type (matmul, conv, activation, reduction, norm, etc.)
   - Input/output shapes (from get_inputs() and weight dimensions)
   - Whether it reads from / writes to global memory
   - Data dependency: which previous op's output does it consume?
3. Identify fusion groups:
   - Adjacent ops with linear data flow (output of A → only input of B)
   - Same tensor dimensions (no shape change between ops)
   - Both memory-bound (fusion eliminates intermediate write+read)
4. Identify bottleneck:
   - Largest FLOPs operation → compute-bound candidate
   - Largest intermediate tensor → memory-bound candidate
   - Most kernel launches → latency-bound candidate
```

**Example** — L2 task `Gemm + BatchNorm + GELU`:
```
Op 1: Linear(4096, 4096)    → matmul, (B, 4096) @ (4096, 4096)  → (B, 4096)  [compute-bound]
Op 2: BatchNorm1d(4096)     → norm, (B, 4096) → (B, 4096)       [memory-bound]
Op 3: GELU()                → pointwise, (B, 4096) → (B, 4096)  [memory-bound]

Fusion groups: {Op2, Op3} can fuse (same shape, linear flow, both memory-bound)
Bottleneck: Op1 (matmul dominates FLOPs)
Strategy implication: Two-kernel approach — matmul with epilogue for BN+GELU
```

This decomposition is performed mentally by the optimizer (it's a thinking process, not a tool call). The optimizer writes the decomposition as a comment in its analysis before generating code.

### 3.3 Multi-Pattern Matching (new)

**Problem:** Current system loads exactly 1 reference file based on primary op type. A task like `Conv2d + MaxPool + ReLU` loads `conv.md` but misses relevant patterns in `pooling.md` (MaxPool fusion) and `common.md` (anti-patterns for conv + cheap activation).

**Solution:** Match against multiple patterns and load up to 3 reference files.

#### 3.3.1 Pattern Matching Algorithm

```
Input: PyTorch forward() code, computation graph from 3.2
Output: Ordered list of (pattern_name, reference_file, relevance_score)

Step 1: Primary op detection (existing keyword match, unchanged)
  → Always loads reference/{primary_op_type}.md

Step 2: Secondary pattern scan
  For each operation in the computation graph:
    - Check against SECONDARY_PATTERNS table (see below)
    - If matched, add (pattern, file, score) to candidate list

Step 3: Composite pattern matching
  Check the FULL operation sequence against COMPOSITE_PATTERNS table
  These match multi-op combinations that have specific optimization strategies

Step 4: Select top 3 files
  - Always include: reference/{primary_op_type}.md
  - Add highest-scoring secondary matches (deduplicated by file)
  - common.md is always read once at startup (not counted in the 3)
```

#### 3.3.2 Secondary Patterns Table

| Pattern | Keywords/Signals | Reference File | Score |
|---------|-----------------|----------------|-------|
| Has matmul + reduction | `linear` + `sum`/`mean` | `reduction.md` | 0.9 |
| Has normalization post-op | `LayerNorm`/`BatchNorm`/`GroupNorm` after compute | `normalization.md` | 0.7 |
| Has pooling post-op | `MaxPool`/`AvgPool`/`AdaptiveAvgPool` | `pooling.md` | 0.7 |
| Has attention pattern | `softmax` + `matmul` + `matmul` | `matmul.md` (flash attention section) | 0.9 |
| Has loss function | `cross_entropy`/`mse`/`kl_div` | `loss.md` | 0.6 |
| Has element-wise chain (3+) | 3+ consecutive pointwise ops | `pointwise.md` | 0.5 |
| Has conv + spatial reduction | `conv` + `pool` or `adaptive_pool` | `pooling.md` | 0.8 |

#### 3.3.3 Composite Patterns Table

These are multi-op sequences with known optimization strategies:

| Composite Pattern | Detection | Strategy Hint | Expected Speedup |
|-------------------|-----------|---------------|-----------------|
| `matmul → pointwise chain` | Linear followed by 1-3 activations/bias | Epilogue fusion | 4-12x |
| `matmul → norm → activation` | Linear + BN/LN/GN + act | Two-kernel (matmul + fused norm-act) | 5-12x |
| `matmul → reduction` | Linear + sum/mean | Algebraic: distribute reduction into weights | 20-74x |
| `conv → pool` | Conv2d + MaxPool/AvgPool | Fuse pool into conv kernel | 1.5-2.9x |
| `conv → norm → act` | Conv + BN + ReLU | torch.convolution + fused BN-ReLU kernel | 1.3-2x |
| `multi-head attention` | reshape + matmul + softmax + matmul | Flash attention | 2-8x |
| `norm → pointwise → norm` | LN + act + LN (e.g., pre-norm transformer) | Fused pre-norm | 1.3-2x |
| `element-wise → reduction` | Pointwise ops + sum/mean | Single fused kernel | 1.3-1.5x |

The composite pattern table lives in the optimizer prompt (not a separate file). It provides *strategy hints* — the optimizer still makes the final strategy decision.

### 3.4 Feasibility Assessment (new)

After pattern matching, consult the feasibility guides in `common.md` (L1/L2/L3 Structural Feasibility Guide) to set expectations:

```
Feasibility classes:
  YES    → High confidence. Allocate full explore+exploit budget.
  MAYBE  → Possible but uncertain. Allocate explore budget to test.
  NO     → Structurally infeasible (e.g., deep CNN, bidirectional RNN).
           Attempt 1 best-effort strategy. Don't waste iterations.
```

**For NO-feasibility tasks:**
- Try 1 strategy (usually `torch.convolution` fp16 or algebraic shortcut)
- If <1.0x after 2 iterations, complete early with best result
- Reflection should document *why* it's infeasible (cuDNN ceiling, precision compounding, etc.)

### 3.5 Strategy List Generation (new)

The analysis phase concludes by producing a **ranked strategy list**:

```
Strategy list (ordered by expected value):
1. [HIGH] epilogue_fusion_matmul_gelu — fuse GELU into matmul epilogue
   Reference: matmul.md → epilogue fusion template
   Expected: 4-8x (from feasibility guide + past results)

2. [MEDIUM] two_kernel_matmul_norm_act — separate matmul + fused BN+GELU
   Reference: matmul.md → two-kernel pattern
   Expected: 5-12x (from common.md → universal techniques)

3. [LOW] full_fused_single_kernel — everything in one kernel
   Reference: common.md → memory hierarchy planning
   Expected: uncertain (register pressure may be too high)
```

**How many strategies?**

| Task complexity | # Strategies | Reasoning |
|----------------|-------------|-----------|
| Algebraic shortcut found | 1 | The shortcut IS the strategy. Go straight to exploit. |
| Single dominant op, clear template | 2 | Template + one variant. |
| Multiple ops, multiple viable approaches | 3 | Need explore phase to pick winner. |
| NO feasibility | 1 | Best-effort only. |

---

## 4. Phase B: Explore (Detailed Design)

### 4.1 Purpose

Phase B answers: **Which strategy class has the highest ceiling?** Instead of committing to one approach and spending 20 iterations debugging it, try 2-3 fundamentally different approaches with minimal iteration investment to find the one worth deep-tuning.

### 4.2 Explore Protocol

```
for each strategy in strategy_list[:num_explore]:
    Iteration N: Generate kernel using this strategy
    result = eval_kernel(...)

    if compile_error and is_fixable(error):
        Iteration N+1: Fix compile error, re-eval
        result = eval_kernel(...)  # second chance

    Record: (strategy, speedup, bottleneck_diagnosis, fixability)

    if speedup >= 1.3x:
        → Target hit! Skip remaining explore, go to complete.

After all explore iterations:
    Select winner = strategy with highest speedup (among correct results)
    If all failed correctness: select most promising (compiled, closest to correct)
    Proceed to Phase C with winner strategy
```

**Key rule:** Each explore iteration gets at most 2 eval calls (initial + one fix attempt). No deep debugging in explore phase — that's what exploit phase is for.

### 4.3 Bottleneck Diagnosis

After each explore eval, the optimizer performs a **bottleneck diagnosis** to guide Phase C tuning:

```
Diagnosis framework (based on eval result):

1. If speedup > 1.3x → DONE, no diagnosis needed

2. If speedup 1.0-1.3x (correct, close):
   Diagnosis: "Correct algorithm, needs tuning"
   Likely bottleneck:
   - If matmul-heavy → compute-bound: try fp16, larger tiles, more autotune configs
   - If element-wise/memory → bandwidth-bound: try vectorized loads, fewer global mem trips
   - If many small kernels → launch-overhead: try kernel fusion
   Phase C action: Parameter sweep on block sizes, num_warps, tile dimensions

3. If speedup 0.5-1.0x (correct, slow):
   Diagnosis: "Fundamentally wrong approach or inefficient memory access"
   Check:
   - Is there unnecessary transposition? (anti-pattern: large tensor permute)
   - Is there redundant global memory traffic? (writing intermediates)
   - Is occupancy too low? (register spill → reduce tile size)
   Phase C action: May need different strategy variant, not just tuning

4. If speedup < 0.5x (correct, very slow):
   Diagnosis: "Strategy is unsuitable for this workload"
   Phase C action: Do NOT select this strategy. Try next explore candidate.

5. If correctness failure:
   Diagnosis: Classify error type:
   - Shape mismatch → grid/block calculation bug (fixable)
   - Numerical difference → precision issue (may need fp32 accumulation)
   - NaN/Inf → missing masking or division by zero (fixable)
   - Wrong values → algorithmic bug (may be fixable, may indicate wrong approach)
   Phase C action: If shape/masking issue, likely fixable. If algorithmic, deprioritize.

6. If compile error:
   Diagnosis: Classify error type:
   - Triton API misuse (tl.math.tanh, wrong constexpr) → fixable with env constraints
   - Shape incompatibility → grid/block bug, fixable
   - OOM → tile too large, fixable by reducing
   - Timeout → too many unrolled iterations, fixable by dynamic loop
   Phase C action: Usually fixable. Worth selecting if approach is sound.
```

### 4.4 Strategy Selection After Explore

After Phase B completes, select the winner strategy:

```
Selection criteria (priority order):
1. Highest speedup among correct results
2. If no correct result: highest speedup among compiled results (correctness bugs are often fixable)
3. If nothing compiled: the strategy with the most fixable compile error
4. If all strategies got < 0.5x: consider that the task may be infeasible; check feasibility guide
```

**Edge cases:**

| Scenario | Action |
|----------|--------|
| Strategy 1 gets 1.35x on first try | Skip remaining explore, go to complete |
| All strategies get < 0.5x | Task likely infeasible. Try one more creative approach (algebraic?). If still <0.5x, complete early. |
| Strategy 2 gets 1.2x, strategy 1 got 0.8x | Select strategy 2 for exploit. The 0.4x gap suggests fundamentally better approach. |
| Two strategies both get ~1.1x | Select the one with more tuning headroom (e.g., fewer autotune configs tried, more parameters to tune) |

---

## 5. Phase C: Exploit (Detailed Design)

### 5.1 Purpose

Phase C answers: **How fast can we make the winning strategy?** This is systematic deep-tuning, guided by the bottleneck diagnosis from Phase B.

### 5.2 Tuning Actions by Bottleneck Type

```
Bottleneck: COMPUTE-BOUND (matmul-heavy, low arithmetic intensity insufficient)
  Actions (try in order):
  1. fp16 conversion for tensor core utilization (if not already)
  2. Expand autotune configs (more tile sizes, 7-10 configs)
  3. Increase BLOCK_K to improve K-loop efficiency
  4. Try different super-blocking GROUP_M values (4, 8, 16)
  5. Try split-K for large K dimensions

Bottleneck: MEMORY-BANDWIDTH (element-wise, reduction on large tensors)
  Actions (try in order):
  1. Fuse more operations into the kernel (eliminate intermediate writes)
  2. Vectorized loads (BLOCK_SIZE=1024-4096, process 4 elements per thread)
  3. Reduce number of global memory reads (cache in registers/shared mem)
  4. Coalesce memory access patterns (ensure threads read contiguous addresses)
  5. If at HBM bandwidth ceiling (~1.0x for pure element-wise): accept and stop

Bottleneck: LAUNCH-OVERHEAD (many small kernels, latency-dominated)
  Actions (try in order):
  1. Kernel fusion (combine 2-3 kernels into 1)
  2. Persistent kernel (one kernel handles multiple batches)
  3. Reduce grid dimensions (fewer program IDs, more work per thread)

Bottleneck: CORRECTNESS (compiled but wrong)
  Actions (try in order):
  1. Fix boundary masking (mask = offs < n_elements)
  2. Fix pointer arithmetic (check strides for multi-dimensional tensors)
  3. Use fp32 accumulator for matmul (prevent precision loss)
  4. Match reference dtype exactly (.to(ref_dtype) before return)
  5. Check reduction axis direction and initial accumulator values
  6. For training-mode BN: verify momentum and Bessel correction

Bottleneck: COMPILATION (doesn't compile)
  Actions (try in order):
  1. Check Triton API constraints (common.md environment constraints)
  2. Ensure BLOCK_SIZE is power of 2
  3. Replace missing functions (tl.math.tanh → 2*sigmoid(2*x)-1)
  4. Fix constexpr parameter ordering
  5. Reduce tl.static_range iterations (<50)
```

### 5.3 Iteration-Level Decision Tree

At each exploit iteration, the optimizer follows this decision tree:

```
After eval result for iteration i:

if speedup >= 1.3x:
    → DONE. Complete task, write reflection.

if compiled AND correct AND speedup > previous_best:
    → Progress! Apply next tuning action for diagnosed bottleneck.
    → Record what improved and by how much.

if compiled AND correct AND speedup <= previous_best:
    → Regression or plateau.
    → If 2+ consecutive non-improvements: escalate.
       Escalation: try a more aggressive change (different tile dimension,
       add/remove fusion, change memory layout).
    → If 3+ consecutive non-improvements: switch to next-best explore strategy
       (if one exists and hasn't been fully explored).

if compiled AND NOT correct:
    → Correctness regression from a correct iteration.
    → Revert to last correct code as base, apply minimal change.

if NOT compiled:
    → Fix compile error (usually 1 iteration to fix).
    → If same error persists after fix: likely fundamental incompatibility.
       Revert to last compiling code.

if iteration == max_iterations - 1:
    → Last iteration. Complete with best result.
```

### 5.4 Revert Mechanism

The optimizer should mentally track the "best working version" — the kernel code that achieved the highest speedup with correct results. If a modification causes regression:

```
Track:
  best_code: str         # kernel code that achieved best_speedup
  best_speedup: float
  best_iteration: int
  best_strategy: str
  current_base: str      # kernel code being modified (may differ from best)
  consecutive_non_improvements: int

On regression or compile failure:
  consecutive_non_improvements += 1
  if consecutive_non_improvements >= 2:
      current_base = best_code  # revert to best working version
      consecutive_non_improvements = 0
      # Apply a DIFFERENT modification than what caused the regression
```

This prevents the optimizer from spiraling away from a working solution through accumulated regressions.

---

## 6. Pattern Matching and Memory Loading (Detailed Design)

### 6.1 Current System

```
Optimizer startup:
  1. Read kernel-bench-optimizer.md
  2. Read reference/common.md

Per task:
  3. Detect primary op type (keyword scan)
  4. Read reference/{op_type}.md
```

**Limitation:** Only 1 op-type file loaded. For `Conv2d + BatchNorm + ReLU + MaxPool`, only `conv.md` is loaded. Relevant patterns in `normalization.md` (BN handling), `pooling.md` (MaxPool fusion), and `pointwise.md` (activation fusion) are missed.

### 6.2 New System

```
Optimizer startup:
  1. Read kernel-bench-optimizer.md (includes composite pattern table, bottleneck diagnosis framework)
  2. Read reference/common.md (once, cached)

Per task:
  3. Build computation graph (Phase A, Step 3.2)
  4. Primary op detection (existing keyword match) → primary_file
  5. Secondary pattern scan → secondary_files (0-2 additional files)
  6. Composite pattern matching → strategy hints
  7. Read reference files: [primary_file] + secondary_files (deduplicated, max 3 total)
  8. Feasibility assessment using guides in common.md
  9. Generate ranked strategy list
```

### 6.3 Reference File Loading Protocol

```
Files to load (in order):

0. reference/optimizer_algorithm.md — ALWAYS (read once at startup, reused across tasks)
   Contents used:
   - Diagnosis Calibration: correct bottleneck diagnosis biases
   - Explore Budget Heuristics: calibrate explore/exploit split per task type
   - Feasibility Corrections: override feasibility guide where it's known to be wrong
   - High-Value Tuning Actions: prioritize Phase C tuning actions
   - Process Anti-Patterns: avoid known causes of wasted iterations

1. reference/common.md — ALWAYS (read once at startup, reused across tasks)
   Contents used:
   - Environment Constraints: check BEFORE writing any kernel
   - Anti-Patterns: check BEFORE selecting strategy
   - Universal Techniques: scan for applicable patterns
   - Composite Patterns: check for multi-op strategy hints
   - Strategy Selection Heuristics: cross-cutting strategy guidance
   - Feasibility Guides: check during feasibility assessment

2. reference/{primary_op_type}.md — ALWAYS (per task)
   Contents used:
   - Code Templates: starting point for kernel generation
   - Tier 1-2: strategy candidates for Phase B explore
   - Tier 3-4: tuning actions for Phase C exploit
   - Anti-Patterns: avoid wasting iterations
   - Decision Tree: strategy ranking

3. reference/{secondary_op_type_1}.md — IF secondary pattern matched
   Contents used selectively:
   - Only the sections relevant to the secondary pattern
   - e.g., for conv + pool task: read pooling.md's "What Works" section for fusion patterns

4. reference/{secondary_op_type_2}.md — IF second secondary pattern matched
   Same selective reading as above.

MAX FILES PER TASK: 3 op-type files + common.md + optimizer_algorithm.md
DEDUPLICATION: If secondary matches the primary, skip (already loaded)
CACHING: Don't re-read a file if the optimizer already has it in context from the previous task
```

### 6.4 Selective Reading Strategy

For secondary reference files, the optimizer reads sections based on the current phase:

**Phase A (strategy selection):** Read Tier 1-2 sections → pick explore strategies
**Phase B (explore):** Already loaded from Phase A
**Phase C (exploit):** Read Tier 3-4 section of the winning strategy's op type → get tuning knobs

For secondary files (not the primary op type), read selectively:
- **Code Templates** — only if the secondary op is written in Triton (not delegated to PyTorch)
- **Tier 1-2 sections** — always read (may contain fusion patterns relevant to the composite)
- **Anti-Patterns** — always read (avoid wasting iterations)
- **Tier 3-4 section** — skip unless the winning Phase B strategy is from this op type

This keeps context usage reasonable even when loading 3 files.

### 6.5 Reference File Internal Structure (Tier-Based)

The current internal structure of op-type files is:

```
## Code Templates
## What Works       ← flat list mixing algorithm changes and parameter tips
## What Fails       ← flat list
## Decision Framework
```

This creates a mismatch with the Phase B/C protocol. The optimizer must mentally sort entries into tiers during Phase A, wasting reasoning on classification instead of strategy selection.

**New internal structure** — reorganize each `reference/{op_type}.md` by strategy tier:

```markdown
# {Op Type} Reference
<!-- Updated: {date} | Source: {session_ids} -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify. -->
{existing code templates — unchanged}

## Tier 1: Algorithm Alternatives
<!-- Fundamentally different approaches. Phase B explores these. -->

### {pattern_name} ({speedup}x) — {source_tasks}
**When to use**: {conditions}
**Key insight**: {one sentence}
**Evidence**: {task results}

### {pattern_name_2} ...

## Tier 2: Architecture Variants
<!-- Same algorithm, different decomposition/structure. Phase B explores these. -->

### {pattern_name} ({speedup}x) — {source_tasks}
**When to use**: {conditions}
**Key insight**: {one sentence}
**Evidence**: {task results}

## Tier 3-4: Tuning Guide
<!-- Parameter sweeps and micro-optimizations. Phase C uses these. -->

- **{tuning_knob}**: {what to try}, {expected impact}. (Source: {tasks})
- **{tuning_knob_2}**: ...

## Anti-Patterns
<!-- Approaches PROVEN to never work. Check before wasting iterations. -->

- **{anti-pattern}**: {why it fails}, {measured speedup}. (Source: {tasks})

## Decision Tree
<!-- Prioritized strategy selection. Updated to reference tiers. -->
1. Check Tier 1 algebraic shortcuts first
2. If no shortcut: select Tier 1-2 strategies for Phase B explore
3. Phase C: apply Tier 3-4 tuning to winner
```

**How this maps to the Phase B/C protocol:**

| Optimizer Phase | Reference Section Read | Purpose |
|-----------------|----------------------|---------|
| Phase A (analyze) | Tier 1 + Tier 2 + Anti-Patterns + Decision Tree | Generate ranked strategy list |
| Phase B (explore) | (already loaded from Phase A) | Know what to try |
| Phase C (exploit) | Tier 3-4: Tuning Guide | Know what knobs to turn |

#### Migration: Existing Content → New Structure

Here's how existing entries in each reference file reclassify:

**matmul.md:**

| Current Section | Entry | New Section |
|-----------------|-------|-------------|
| What Works | Diagonal matrix → row scaling (104x) | **Tier 1**: algebraic |
| What Works | Algebraic collapse to zeros (73.5x) | **Tier 1**: algebraic |
| What Works | Flash attention (8.1x) | **Tier 1**: algorithm |
| What Works | Three-kernel attention (1.97x) | **Tier 2**: architecture |
| What Works | Epilogue fusion (11.9x) | **Tier 2**: architecture |
| What Works | Two-kernel Gemm+Norm (11.7x) | **Tier 2**: architecture |
| What Works | Standard tiled matmul (7.6x) | **Tier 2**: architecture (baseline) |
| Decision Framework #6 | fp16 for large GEMMs | **Tier 3-4**: tuning |
| Decision Framework #7 | Implicit transpose via strides | **Tier 3-4**: tuning |
| Decision Framework #9 | Autotune budget (4-7 configs) | **Tier 3-4**: tuning |
| What Fails | Bandwidth-bound matvec (1.02x) | **Anti-Patterns** |
| What Fails | Split-K correctness failures | **Anti-Patterns** |
| What Fails | Medium GEMM slower than cuBLAS (0.69x) | **Anti-Patterns** |

**conv.md:**

| Current Section | Entry | New Section |
|-----------------|-------|-------------|
| What Works | Algebraic elimination (15.8x) | **Tier 1**: algebraic |
| What Works | Depthwise spatial tiling (15.2x) | **Tier 1**: algorithm (specialized) |
| What Works | torch.convolution fp16 (1.89x) | **Tier 1**: algorithm (delegate to cuDNN) |
| What Works | NCHW-direct 1x1 as matmul (2.82x) | **Tier 2**: architecture |
| What Works | Fuse MaxPool into conv (2.93x) | **Tier 2**: architecture |
| What Works | Single-pass implicit GEMM (3.31x) | **Tier 2**: architecture |
| What Works | NHWC input for C_in=32-64 (1.57x) | **Tier 3-4**: tuning (memory layout) |
| Decision Framework #13 | fp16 cast strategy | **Tier 3-4**: tuning |
| Decision Framework #14 | Weight layout pre-transpose | **Tier 3-4**: tuning |
| What Fails | Large C_in ConvTranspose (0.61x) | **Anti-Patterns** |
| What Fails | stride-2 3D ConvTranspose (0.12-0.39x) | **Anti-Patterns** |

**reduction.md:**

| Current Section | Entry | New Section |
|-----------------|-------|-------------|
| What Works | Online 2-pass softmax (1.32x) | **Tier 1**: algorithm |
| What Works | Fused mask+cumsum (1.45x) | **Tier 2**: architecture (fusion) |
| What Works | Coalesced tiling + unroll-by-4 (1.30x) | **Tier 3-4**: tuning |
| What Fails | Bandwidth ceiling (1.08x) | **Anti-Patterns** |
| What Fails | Two-phase parallel scan (0.54x) | **Anti-Patterns** |
| What Fails | Sequential dependency kills cumsum (1.05x) | **Anti-Patterns** |

**Classification rules for the learner:**

```
Tier 1: The entry describes a fundamentally different ALGORITHM.
  Signal: Different asymptotic complexity, different mathematical formulation,
  or delegating to a completely different backend (cuDNN vs Triton).
  Examples: algebraic elimination, flash attention, torch.convolution delegation

Tier 2: The entry describes a different DECOMPOSITION of the same algorithm.
  Signal: Same math, but different kernel count, different data flow,
  different fusion boundary.
  Examples: epilogue fusion vs two-kernel, single-pass vs multi-pass,
  fused conv+pool vs separate kernels

Tier 3-4: The entry describes a PARAMETER or LAYOUT choice within a fixed architecture.
  Signal: Same kernel structure, just different block sizes, dtypes,
  memory layouts, or loop unroll factors.
  Examples: fp16 conversion, NHWC layout, autotune config expansion,
  super-blocking GROUP_M values

Anti-Pattern: The entry describes something that NEVER works.
  Signal: Measured speedup < 0.5x, or proven to fail for structural reasons.
  Examples: transpose large tensor for reduction, explicit im2col, bandwidth-bound ceiling
```

---

## 7. Strategy Exploration Structure (Detailed Design)

### 7.1 Strategy Taxonomy

Strategies fall into four tiers based on how different they are:

```
Tier 1: Algorithm change (fundamentally different approach)
  Examples:
  - Tiled matmul → algebraic elimination (for matmul + reduction)
  - Triton conv → torch.convolution + Triton post-ops
  - Flash attention → three-kernel decomposition
  → Phase B explore: always try at least 2 Tier 1 strategies

Tier 2: Architecture change (same algorithm, different decomposition)
  Examples:
  - Single fused kernel → two-kernel (matmul + fused post-ops)
  - Flat 1D grid → 2D tiled grid
  - Sequential scan → decoupled lookback scan
  → Phase B explore: try as variant of Tier 1 strategy

Tier 3: Parameter change (same architecture, different config)
  Examples:
  - BLOCK_SIZE 256 → 1024
  - Add/remove autotune configs
  - num_warps 4 → 8
  → Phase C exploit: systematic sweep

Tier 4: Micro-optimization (within a kernel, cosmetic)
  Examples:
  - Unroll inner loop
  - Vectorized loads
  - Register blocking
  → Phase C exploit: late-stage polish
```

**Rule:** Phase B explores Tier 1-2 differences. Phase C tunes Tier 3-4 within the winner.

### 7.2 Strategy Diversification

When generating the strategy list in Phase A, ensure diversity:

```
Diversification rules:
1. No two strategies should share the same bottleneck mitigation
   BAD:  Strategy 1 = "tiled_matmul_64x64", Strategy 2 = "tiled_matmul_128x128"
         (both are same algorithm with different block sizes — that's Tier 3, not Tier 1)
   GOOD: Strategy 1 = "tiled_matmul_64x64", Strategy 2 = "epilogue_fused_matmul"
         (different decomposition — Tier 1/2 difference)

2. Include at least one "safe" strategy (has code template in reference files)
   and optionally one "creative" strategy (novel combination not in templates)

3. If feasibility guide says MAYBE, include a conservative strategy
   (e.g., torch.convolution + Triton post-ops) alongside the aggressive one
   (e.g., full Triton implicit GEMM)
```

### 7.3 Exploration vs. Exploitation Budget

The optimizer dynamically adjusts the explore/exploit split based on early results:

```
Dynamic budget adjustment:

if explore_iter_0 hits 1.3x:
    → Skip remaining explore. Phase C = complete.
    Budget: 1 explore + 0 exploit = done.

if explore_iter_0 hits 1.0-1.3x (promising):
    → Still try second strategy (it might be better).
    → But reduce exploit budget for first strategy.
    Budget: 2-4 explore + remaining exploit.

if explore_iter_0 hits < 0.5x (bad):
    → Strategy is wrong. Move to next explore strategy immediately.
    → Don't spend a fix iteration.
    Budget: skip to next explore.

if all explore strategies hit < 0.5x:
    → Task is likely structurally infeasible.
    → Try one algebraic analysis (if not already done).
    → If still < 0.5x: complete with best result, mark as infeasible in reflection.
    Budget: 2-3 more attempts max.
```

### 7.4 Worked Example: L2 Task `Conv2d + BatchNorm + ReLU + MaxPool`

```
PHASE A: Analyze
  Computation graph:
    Op 1: Conv2d(3, 64, 3, padding=1)     → conv, (B,3,H,W) → (B,64,H,W)     [compute-bound for small C_in]
    Op 2: BatchNorm2d(64)                  → norm, (B,64,H,W) → (B,64,H,W)    [memory-bound]
    Op 3: ReLU()                           → pointwise, (B,64,H,W) → (B,64,H,W) [memory-bound]
    Op 4: MaxPool2d(2)                     → pooling, (B,64,H,W) → (B,64,H/2,W/2) [memory-bound]

  Fusion groups: {Op2, Op3} → fused BN-ReLU; {Op3, Op4} → fused ReLU-Pool

  Primary op: conv → load reference/conv.md
  Secondary patterns: pooling (MaxPool present) → load reference/pooling.md
  Composite: conv + pool → "Fuse pool into conv kernel" (expected 1.5-2.9x)

  Feasibility: Conv2d(C_in=3) + post-ops → YES, expected 1.4-2.9x

  Strategy list:
  1. [HIGH] implicit_gemm_fused_bnrelu_pool — full Triton: implicit GEMM with fused BN+ReLU+MaxPool
     Expected: 1.5-2.9x (conv.md: single-pass implicit GEMM for small C_in)
  2. [MEDIUM] torch_conv_triton_postops — torch.convolution for conv, Triton kernel for BN+ReLU+MaxPool
     Expected: 1.3-2.0x (conv.md: torch.convolution not blocked + common.md: fuse post-ops)
  3. [LOW] torch_conv_fp16 — torch.convolution in fp16, no Triton post-ops
     Expected: 1.1-1.5x (conv.md: fp16 tensor cores)

PHASE B: Explore (4 iterations)
  Iter 0: implicit_gemm_fused_bnrelu_pool → eval → 1.8x, correct ✓
    Diagnosis: correct and fast! But try #2 to see if easier to tune.
  Iter 1: torch_conv_triton_postops → eval → 1.45x, correct ✓
    Diagnosis: correct, decent. But #1 is better.

  Winner: implicit_gemm_fused_bnrelu_pool (1.8x > 1.45x)

  Wait — 1.8x already exceeds 1.3x target!
  → DONE. Complete task.

PHASE C: Exploit (skipped — target already reached)

Reflection:
  ### conv_bn_relu_pool (1.8x, iter 0)
  **Op type**: conv
  **Key insight**: Small C_in (3) makes implicit GEMM viable; fusing BN+ReLU+Pool saves 3 memory round-trips.
  **What worked**: Single-pass implicit GEMM with fused epilogue for all post-ops.
  **What failed**: N/A (first strategy succeeded).
```

---

## 8. Reflection Format Enhancement

### 8.1 Current Reflection Format

```markdown
### {task_name} ({best_speedup}x, iter {best_iteration})
**Op type**: element_wise | matmul | reduction | ...
**Key insight**: One sentence.
**What worked**: 1-2 sentences.
**What failed**: 1-2 sentences.
**Environment gotcha** (optional): Triton API issue.
**Anti-pattern** (optional): Proven-not-to-work approach.
```

### 8.2 Enhanced Reflection Format

Add quantitative signals to make reflections more useful for the learner agent:

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

**Example:**

```markdown
### 59_Matmul_BiasAdd_GELU (11.9x, iter 2/8)
**Op type**: matmul (secondary: element_wise)
**Bottleneck**: compute-bound (large GEMM dominates)
**Key insight**: Epilogue fusion eliminates 2 memory round-trips for bias+GELU by computing them in matmul tile registers before store.
**What worked**: epilogue_fused_matmul_bias_gelu — 11.9x. fp16 inputs + fp32 accumulator. 7-config autotune with super-blocking GROUP_M=8.
**What failed**: two_kernel_matmul_then_fused_bias_gelu — 5.2x. The intermediate tensor write between kernels costs ~45% of runtime.
**Exploration summary**:
  - epilogue_fused_matmul_bias_gelu: 11.9x (correct) — single kernel, all compute in registers
  - two_kernel_matmul_then_fused_bias_gelu: 5.2x (correct) — memory-bound on intermediate write
**Phase C tuning**: Expanded autotune from 4 to 7 configs (+0.8x). fp16 input conversion (+1.2x).
**Anti-pattern**: Separate bias add kernel after matmul (5.2x vs 11.9x) — the intermediate write dominates.
```

### 8.3 Algorithm Execution Trace

The enhanced reflection (Section 8.2) captures *what kernel strategies worked*. But the optimizer also makes meta-decisions throughout Phase A/B/C — pattern matching choices, feasibility judgments, explore/exploit budget splits, bottleneck diagnoses, pivot decisions — that are equally valuable to learn from. Currently these decisions happen in the optimizer's reasoning and are lost when the agent exits.

**Problem:** The system learns kernel knowledge (Tier 1-4 strategies, anti-patterns) but not optimization process knowledge. For example:
- "Bottleneck diagnosis said compute-bound, but the real bottleneck was memory layout" — this is a calibration error in the diagnosis framework
- "Explored 3 strategies for a simple L1 task, but the first one hit 1.3x" — the explore budget was too high
- "Feasibility guide said NO, but algebraic shortcut got 40x" — the feasibility guide had a gap
- "Reverted to best after 2 regressions, then the revert-based variant hit 1.5x" — the revert mechanism worked well here

**Solution:** The optimizer writes a structured **algorithm execution trace** (`algo_trace.md`) to the task folder alongside `reflection.md`. The learner reads these traces and distills meta-learnings about the optimization process into `reference/optimizer_algorithm.md`.

#### 8.3.1 Execution Trace Format

Written by the optimizer to `~/.inference/claude_code_output/{session_id}/{task_name}/algo_trace.md`:

```markdown
## Algorithm Trace: {task_name}

### Phase A: Analysis Decisions
- **Algebraic scan**: {found_shortcut | no_shortcut}. {1-line reasoning}.
- **Computation graph**: {num_ops} ops. Bottleneck: {op_name} ({compute|memory|launch}-bound). Fusion groups: {list}.
- **Pattern match**: Primary={op_type}. Secondary={list}. Composite={matched_pattern | none}.
- **Files loaded**: {list of reference files read}
- **Feasibility**: {YES|MAYBE|NO}. Reasoning: {1 sentence from guide}.
- **Strategy list**: {num} strategies.
  1. [{HIGH|MEDIUM|LOW}] {strategy_name} — source: {reference_file § section}
  2. [{HIGH|MEDIUM|LOW}] {strategy_name} — source: {reference_file § section}
  3. ...
- **Explore budget**: {num_explore_iters} explore + {num_exploit_iters} exploit. Reason: {why this split}.

### Phase B: Explore Decisions
- **Iter {N}: {strategy_name}**
  Result: {speedup}x, {correct|incorrect|compile_error}
  Diagnosis: {bottleneck_type}. {1-line reasoning}.
  Decision: {continue_explore | select_winner | skip_remaining | fix_compile}
- **Iter {N+1}: {strategy_name}**
  ...
- **Winner selection**: {strategy_name} ({speedup}x). Reason: {why this over alternatives}.
  Runner-up: {strategy_name} ({speedup}x).

### Phase C: Exploit Decisions
- **Iter {N}: {tuning_action}**
  Changed: {what was modified from previous iteration}
  Result: {speedup}x (delta: {+/-change}x from previous)
  Decision: {continue_tuning | escalate | revert | switch_strategy | done}
- **Iter {N+1}: ...**
- **Reverts**: {count} (reverted at iters {list}, reason: {consecutive_non_improvements | correctness_regression})
- **Strategy switches**: {count} (switched at iter {N} from {A} to {B}, reason: {3+_non_improvements})

### Meta-Observations
- **Diagnosis accuracy**: Initial diagnosis was {bottleneck_type}. Actual bottleneck turned out to be {same | different: actual_type}. Evidence: {1 sentence}.
- **Explore efficiency**: {num_explore_iters} explore iters used. First viable strategy found at iter {N}. Explore was {necessary | wasteful | insufficient}.
- **Feasibility accuracy**: Guide said {YES|MAYBE|NO}. Actual result: {speedup}x. Guide was {accurate | too_optimistic | too_pessimistic}.
- **Budget utilization**: Used {N}/{max_iterations} iterations. Hit target at iter {N} | exhausted budget at {best_speedup}x.
```

#### 8.3.2 Example Execution Trace

```markdown
## Algorithm Trace: 59_Matmul_BiasAdd_GELU

### Phase A: Analysis Decisions
- **Algebraic scan**: no_shortcut. Linear + bias + GELU has no simplification.
- **Computation graph**: 3 ops. Bottleneck: linear (compute-bound, large GEMM). Fusion groups: {bias, gelu} can fuse.
- **Pattern match**: Primary=matmul. Secondary=element_wise. Composite=matmul→pointwise(2) → epilogue_fusion hint.
- **Files loaded**: reference/matmul.md, reference/pointwise.md
- **Feasibility**: YES. Reasoning: Gemm + pointwise chain, expected 4-12x (L2 guide).
- **Strategy list**: 2 strategies.
  1. [HIGH] epilogue_fused_matmul_bias_gelu — source: matmul.md § Tier 2
  2. [MEDIUM] two_kernel_matmul_then_fused_bias_gelu — source: matmul.md § Tier 2
- **Explore budget**: 4 explore + 16 exploit. Reason: 2 strategies, standard L2 split.

### Phase B: Explore Decisions
- **Iter 0: epilogue_fused_matmul_bias_gelu**
  Result: 9.1x, correct
  Diagnosis: compute-bound. GEMM dominates, epilogue is free.
  Decision: continue_explore (try second strategy to compare)
- **Iter 1: two_kernel_matmul_then_fused_bias_gelu**
  Result: 5.2x, correct
  Diagnosis: memory-bound on intermediate write between kernels.
  Decision: select_winner (epilogue 9.1x > two_kernel 5.2x)
- **Winner selection**: epilogue_fused_matmul_bias_gelu (9.1x). Reason: 74% faster, single kernel avoids intermediate write.
  Runner-up: two_kernel (5.2x).

### Phase C: Exploit Decisions
- **Iter 2: expand_autotune_configs**
  Changed: 4 configs → 7 configs (added 128x128 BLOCK_K=64)
  Result: 9.9x (delta: +0.8x)
  Decision: continue_tuning
- **Iter 3: fp16_input_conversion**
  Changed: Added x.half() + cached fp16 weight in __init__
  Result: 11.9x (delta: +2.0x)
  Decision: done (>= 1.3x already, but large improvement suggests more headroom — however 11.9x is excellent)

### Meta-Observations
- **Diagnosis accuracy**: Initial diagnosis was compute-bound. Confirmed: fp16 tensor cores gave +2.0x, characteristic of compute-bound.
- **Explore efficiency**: 2 explore iters used. First viable at iter 0 (9.1x). Explore was necessary — confirmed epilogue >> two-kernel.
- **Feasibility accuracy**: Guide said YES (4-12x). Actual: 11.9x. Guide was accurate.
- **Budget utilization**: Used 4/20 iterations. Hit target at iter 0 (9.1x >> 1.3x). Could have stopped earlier but tuning added +2.8x.
```

#### 8.3.3 What the Learner Extracts from Traces

The learner reads `algo_trace.md` files alongside `reflection.md` files and distills process-level learnings into `reference/optimizer_algorithm.md`. This file is read by optimizers at startup alongside `common.md`.

**Categories of meta-learning:**

| Category | What to extract | Example |
|----------|----------------|---------|
| **Diagnosis calibration** | When initial bottleneck diagnosis was wrong and what the actual bottleneck was | "For conv + pooling tasks, initial diagnosis is often 'compute-bound' but actual bottleneck is memory layout (NCHW vs NHWC). Check layout first." |
| **Explore budget tuning** | How many explore iterations are needed by task type | "L1 single-op tasks: 1 strategy is sufficient 80% of the time. L2 fused chains: 2 strategies needed. L3 full models: 3 strategies." |
| **Feasibility calibration** | Where the feasibility guide is wrong | "Guide says Conv2d(C_in=64+) is MAYBE, but with NHWC + fp16 it's YES for 70% of tasks." |
| **Phase C patterns** | What tuning actions produce the biggest improvements | "fp16 conversion is the single highest-value Tier 3 action: +1.5-3.0x for compute-bound matmul, +0.5-1.5x for conv." |
| **Revert/switch patterns** | When reverts and strategy switches happen and whether they help | "Reverts after 2 regressions recover 85% of the time. Strategy switches after 3 non-improvements succeed 40% of the time." |
| **Wasted iteration patterns** | Common causes of wasted iterations | "60% of wasted iterations are from trying fp16 on memory-bound element-wise tasks. Check bottleneck type before applying fp16." |

#### 8.3.4 `reference/optimizer_algorithm.md` Format

```markdown
# Optimizer Algorithm Reference
<!-- Updated: {date} | Source: {session_ids} -->

## Diagnosis Calibration
<!-- Corrections to the bottleneck diagnosis framework -->

- **{task_pattern}**: Diagnosis says {X}, but actual bottleneck is {Y} in {N}% of cases. Check {signal} first.
  (Source: {task_list})

## Explore Budget Heuristics
<!-- How many explore strategies to try by task type -->

- **{task_type}**: {N} strategies sufficient. First viable found by iter {M} in {P}% of cases.
  (Source: {task_list})

## Feasibility Corrections
<!-- Where the L1/L2/L3 feasibility guides are inaccurate -->

- **{pattern}**: Guide says {YES|MAYBE|NO}, actual success rate is {P}% at {avg_speedup}x.
  (Source: {task_list})

## High-Value Tuning Actions
<!-- Tier 3-4 actions ranked by impact, by bottleneck type -->

- **{bottleneck_type}**: Top actions in order:
  1. {action} — avg improvement {delta}x ({N} tasks)
  2. {action} — avg improvement {delta}x ({N} tasks)

## Process Anti-Patterns
<!-- Common causes of wasted iterations -->

- **{pattern}**: Wastes {N} iterations on average. Detection: {signal}. Fix: {what to do instead}.
  (Source: {task_list})

## Revert & Switch Effectiveness
<!-- When to revert vs. switch vs. persist -->

- **Revert after {N} regressions**: Success rate {P}%. Best when: {conditions}.
- **Switch strategy after {N} non-improvements**: Success rate {P}%. Best when: {conditions}.
```

---

## 9. Implementation Plan

### 9.1 Changes to `kernel-bench-optimizer.md`

| Section | Change | Scope |
|---------|--------|-------|
| Section 3 (Core Algorithm) | Replace flat loop with Phase A/B/C protocol | Major rewrite |
| Step 1 (Analyze) | Add computation graph, multi-pattern matching, feasibility, strategy list; reference Tier 1-2 sections for strategy candidates | Expand |
| Step 2 (Generate) | No change (code generation rules unchanged) | None |
| Step 3 (Evaluate) | Add bottleneck diagnosis after each eval; reference Tier 3-4 for tuning actions | Add |
| Step 4 (Complete & Reflect) | Enhanced reflection format with tier classification; write algo_trace.md | Modify |
| New: Composite Patterns table | Add multi-op pattern → strategy hint mapping | Add |
| New: Bottleneck Diagnosis framework | Add diagnosis → tuning action mapping | Add |
| New: Exploration protocol | Add explore/exploit budget, strategy diversification rules | Add |
| Section on reading reference files | Update to reference tier-based sections (read Tier 1-2 for Phase B, Tier 3-4 for Phase C); read optimizer_algorithm.md at startup | Modify |

### 9.2 Changes to `kernel-bench-learner.md`

The enhanced reflection format (Section 8.2) introduces composite patterns, bottleneck diagnoses, and exploration summaries. The current learner categorizes each reflection into exactly one `{op_type}.md` by primary op type, and puts cross-cutting patterns into `common.md`. This still works for single-op insights, but two categories of knowledge now have no home:

1. **Composite pattern insights** — "matmul → norm → activation works best as two-kernel" isn't a matmul-only insight. An optimizer working on `conv → norm → activation` won't find it in `matmul.md`.
2. **Strategy selection heuristics** — "for compute-bound tasks, epilogue fusion beats two-kernel 80% of the time" is a cross-cutting strategy lesson, not an op-type-specific one.

**Changes to learner.md:**

| Section | Change |
|---------|--------|
| "What to Look For" | Add "Composite patterns", "Strategy selection heuristics", and **"Algorithm process patterns"** categories |
| "Output Format" for `common.md` | Add `## Composite Patterns` section and `## Strategy Selection Heuristics` section |
| "Output Format" for `{op_type}.md` | Replace `What Works` / `What Fails` / `Decision Framework` with `Tier 1` / `Tier 2` / `Tier 3-4` / `Anti-Patterns` / `Decision Tree` (see Section 6.5) |
| **New output file** | Add `reference/optimizer_algorithm.md` — distilled from `algo_trace.md` files (see Section 8.3.4) |
| "Procedure" | Add steps: read algo_trace.md files, extract composite patterns, extract strategy heuristics, extract process meta-learnings, classify entries by tier |
| "Rules" | Add: "Composite patterns go to common.md"; "Classify each insight by tier using the classification rules in Section 6.5"; **"Process-level learnings go to optimizer_algorithm.md, never to op_type files"** |

**New sections in `common.md` output format:**

```markdown
## Composite Patterns
<!-- Multi-op sequences with known best strategies. Extracted from exploration summaries. -->

- **{op_sequence}** → {best_strategy} ({speedup}x). {why}.
  Alternatives tried: {alt_strategy} ({alt_speedup}x) — {why_worse}.
  (Source: {task_name1}, {task_name2})

## Strategy Selection Heuristics
<!-- Cross-cutting lessons about when to use which strategy class. -->

- **{bottleneck_type} + {op_pattern}** → prefer {strategy_class}. {evidence}.
  (Source: {task_name1}, {task_name2}, ...)
```

**Example entries:**

```markdown
## Composite Patterns

- **matmul → pointwise(1-3)** → epilogue_fusion (4-12x). Fusing activation into matmul tile registers eliminates intermediate global memory write.
  Alternatives tried: two_kernel (2-5x) — intermediate write costs 40-60% of runtime.
  (Source: L2: 12_Gemm, 59_Matmul, 22_Matmul)

- **conv(C_in<=16) → norm → activation** → torch_conv + fused_triton_norm_act (1.5-2.5x). cuDNN conv is near-optimal for small C_in; fusing BN+act in Triton saves 2 memory passes.
  Alternatives tried: full_triton_implicit_gemm (1.0-1.8x) — competitive but harder to get correct.
  (Source: L2: 3_Conv2d, 77_ConvTranspose3d)

## Strategy Selection Heuristics

- **compute-bound + matmul chain** → prefer epilogue fusion over multi-kernel. Epilogue fusion wins 8/10 tasks because intermediate writes dominate at large GEMM sizes.
  (Source: L2: 12_Gemm 7.1x, 59_Matmul 11.9x, 55_Matmul 11.1x)

- **memory-bound + element-wise chain (3+ ops)** → prefer single fused kernel. Fusing 3+ pointwise ops into one kernel is the ONLY way to beat PyTorch; individual ops are bandwidth-bound at ~1.0x.
  (Source: L1: 26_GELU 1.95x, L2: 29_Matmul 2.3x)

- **infeasible pattern (deep CNN, bidir RNN)** → skip after 2 iterations. Spending 20 iterations on VGG16 or bidirectional GRU yields <0.4x every time.
  (Source: L3: 10_ResNet101 0.04x, 11_VGG16 0.4x, 15_GRU 0.04x)
```

**Learner procedure change:**

```
Current:
  1. Read reflections
  2. Categorize by op type
  3. Identify cross-cutting patterns → common.md
  4. Select top 3+2 per op type → {op_type}.md (flat What Works / What Fails)

New:
  1. Read reflections (now include exploration summaries + bottleneck diagnoses)
  2. Read algo_trace.md files from all tasks in the session
  3. Categorize reflections by PRIMARY op type
  4. Identify cross-cutting patterns → common.md (Environment Constraints, Anti-Patterns, Universal Techniques)
  5. Extract composite patterns from exploration summaries → common.md § Composite Patterns
  6. Extract strategy selection lessons from bottleneck + result data → common.md § Strategy Selection Heuristics
  7. For each op type:
     a. Classify each insight by tier (Tier 1 / Tier 2 / Tier 3-4 / Anti-Pattern)
        using the classification rules from Section 6.5
     b. Select top entries per tier: 2-3 for Tier 1, 2-3 for Tier 2,
        3-5 for Tier 3-4, 2-3 for Anti-Patterns
     c. Write {op_type}.md with tier-based sections
  8. Extract process meta-learnings from algo_trace.md files:
     a. Aggregate diagnosis accuracy across tasks (how often was initial diagnosis correct?)
     b. Compute explore efficiency (how many explore iters before viable strategy, by task type?)
     c. Check feasibility accuracy (guide prediction vs actual result)
     d. Rank tuning actions by average improvement delta
     e. Compute revert/switch success rates
     f. Identify common wasted-iteration patterns
  9. Write optimizer_algorithm.md (merging with existing file if present)
  10. Merge all other files with existing versions (keep best from both old and new per tier)
```

**New `{op_type}.md` output format:**

```markdown
# {Op Type} Reference
<!-- Updated: {date} | Source: {session_id} -->

## Code Templates
<!-- Preserved from existing file. Do NOT delete or modify. -->

## Tier 1: Algorithm Alternatives

### {pattern_name} ({speedup}x) — {source_tasks}
**When to use**: {conditions}
**Key insight**: {one sentence}

## Tier 2: Architecture Variants

### {pattern_name} ({speedup}x) — {source_tasks}
**When to use**: {conditions}
**Key insight**: {one sentence}

## Tier 3-4: Tuning Guide

- **{knob}**: {what to try}, {expected impact}. (Source: {tasks})

## Anti-Patterns

- **{anti-pattern}**: {why it fails}, {measured speedup}. (Source: {tasks})

## Decision Tree
1. Check Tier 1 algebraic shortcuts first
2. If no shortcut: select from Tier 1-2 for explore phase
3. Exploit phase: apply Tier 3-4 tuning
```

**Key rule:** A composite pattern insight belongs in `common.md` if the multi-op sequence appears across 2+ different primary op types. If it's specific to one primary type (e.g., "matmul + bias is always epilogue fusion"), it can stay in `{op_type}.md` under Tier 2 — but the learner should cross-reference by noting "see also: common.md § Composite Patterns" when the same pattern generalizes.

### 9.3 Changes to Reference Files

| File | Change |
|------|--------|
| `reference/common.md` | Add `## Composite Patterns` and `## Strategy Selection Heuristics` sections (initially empty; populated by learner after first Phase 8 session) |
| `reference/{op_type}.md` (all 8 files) | **Restructure internal sections** from `What Works / What Fails / Decision Framework` to `Tier 1 / Tier 2 / Tier 3-4 / Anti-Patterns / Decision Tree` (see Section 6.5). Code Templates section preserved unchanged. |
| **`reference/optimizer_algorithm.md`** (new) | **New file.** Process-level meta-learnings about the optimization algorithm itself. Sections: Diagnosis Calibration, Explore Budget Heuristics, Feasibility Corrections, High-Value Tuning Actions, Process Anti-Patterns, Revert & Switch Effectiveness. Initially empty; populated by learner after first Phase 8 session. ~3KB budget. |

**Migration approach:** One-time manual reclassification of existing entries using the migration tables in Section 6.5. This is a content reorganization, not a rewrite — all existing knowledge is preserved, just moved to the correct tier section. The migration should be done before the first Phase 8 session run so that optimizers see the tier-based layout immediately.

### 9.4 Changes to `kernel-bench.md` (Skill Controller)

| Section | Change |
|---------|--------|
| Optimizer spawn prompt | Include `num_explore_strategies` parameter (default: 2 for L1, 3 for L2/L3) |
| Finalize step | Update `kb_reflect.py` to also collect `algo_trace.md` files into `all_algo_traces.md`; pass both files to learner |
| Learner spawn prompt | Add `algo_traces_file` parameter pointing to `all_algo_traces.md` |

### 9.5 No MCP Server Changes

The optimizer core algorithm operates entirely within the agent's reasoning. No new tools, no server changes. The optimizer uses existing tools:
- `get_task_details()` — read PyTorch code
- `eval_kernel()` — evaluate kernel (unchanged)
- `update_task_progress()` — record results (unchanged)
- `complete_task_progress()` — mark done (unchanged)

---

## 10. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Explore phase wastes iterations on bad strategies | Medium | Medium | Cap explore at 2 iters per strategy; skip if <0.5x |
| Multi-file loading increases context usage per task | Medium | Low | Cap at 3 files + selective reading; common.md cached |
| Bottleneck diagnosis is wrong, leading to wrong tuning | Medium | Low | Diagnosis is a guide, not a hard constraint; optimizer can override |
| Feasibility assessment is wrong (says NO but 1.3x is achievable) | Low | Medium | Feasibility says "allocate less budget" not "skip entirely"; always try at least 1 strategy |
| Added complexity makes optimizer prompt too long | Medium | Medium | Keep additions focused: composite pattern table (~30 lines), diagnosis framework (~40 lines), explore protocol (~20 lines). Total: ~90 lines added. |
| Phase A analysis takes too long (optimizer thinks instead of generating) | Low | Low | Phase A is 1 iteration (no eval calls). Analysis is fast compared to GPU eval round-trips. |
| Learner misclassifies composite vs. op-specific pattern | Medium | Low | Clear rule: composite goes to common.md if multi-op sequence appears across 2+ primary op types; otherwise stays in op_type.md |
| common.md grows too large with new sections | Medium | Medium | Composite Patterns and Strategy Selection Heuristics share the existing ~3KB budget; learner enforces size limit by cutting least-transferable entries |
| Reference file migration misclassifies tier | Low | Low | Classification rules are concrete (Section 6.5); borderline entries can go in either adjacent tier without harm; learner corrects on next session |
| Tier-based sections are less scannable than flat What Works/Fails | Low | Low | Decision Tree section provides the quick-scan entry point; tier sections provide depth when needed |
| algo_trace.md writing consumes optimizer context/tokens | Medium | Low | Trace is structured (fill-in-the-blanks), not free-form prose; ~30-50 lines per task. Written once at completion, not per-iteration. |
| Learner overloaded with two input streams (reflections + traces) | Medium | Medium | Process meta-learning is a separate pass from kernel knowledge extraction; learner writes optimizer_algorithm.md as a distinct step after op-type files |
| optimizer_algorithm.md gives wrong calibration (small sample size) | Medium | Medium | Require minimum 10 tasks before drawing conclusions; include sample size in each entry; mark low-confidence entries |

---

## 11. Success Criteria

Phase 8 is successful if, compared to the current optimizer on the same task set:

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Median iterations to 1.3x | ~6 | ~4 | Count iterations for tasks that reach 1.3x |
| Success rate (clean, no gaming) | 55.7% | 65%+ | Corrected >=1.3x rate |
| Wasted iterations (wrong strategy, <0.5x) | ~15% of all iters | <5% | Count iters with <0.5x speedup |
| Tasks with 0x (all failed) | ~8% | <5% | Count tasks with completion_reason=all_iterations_failed |
| Reference file utilization | 1 file/task | 2+ files/task | Count reference files read per task |
| Diagnosis accuracy | unknown | 70%+ | Compare initial diagnosis in algo_trace to actual bottleneck |
| Explore efficiency | unknown | <3 iters to viable | Count explore iters before first correct result >= 1.0x |

---

## Appendix A: Decision Flowchart

```
START: Optimizer claims task, reads PyTorch code

┌─────────────────────────────────────────────────┐
│ PHASE A: ANALYZE                                │
│                                                 │
│ 1. Algebraic reasoning                          │
│    └─ Shortcut found? → Strategy #1 (highest)   │
│                                                 │
│ 2. Computation graph decomposition              │
│    └─ Ops, shapes, fusion groups, bottleneck    │
│                                                 │
│ 3. Multi-pattern matching                       │
│    ├─ Primary op → load reference/{op}.md       │
│    ├─ Read Tier 1-2 + Anti-Patterns sections    │
│    ├─ Secondary patterns → load 0-2 more files  │
│    └─ Composite patterns → strategy hints       │
│                                                 │
│ 4. Feasibility assessment                       │
│    └─ YES / MAYBE / NO → set budget             │
│                                                 │
│ 5. Generate ranked strategy list (2-4)          │
└─────────────────────┬───────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │ Algebraic shortcut?   │
          └───┬───────────────┬───┘
           YES│               │NO
              ▼               ▼
    ┌─────────────┐  ┌────────────────────────────┐
    │ Skip explore │  │ PHASE B: EXPLORE           │
    │ Go to exploit│  │                            │
    │ with shortcut│  │ For each strategy (2-3):   │
    └──────┬──────┘  │   Generate → Eval           │
           │         │   If compile error: 1 fix   │
           │         │   Record speedup + diagnosis│
           │         │   If >= 1.3x → DONE         │
           │         │                            │
           │         │ Select winner strategy      │
           │         └────────────┬───────────────┘
           │                      │
           └──────────┬───────────┘
                      ▼
    ┌─────────────────────────────────────────────┐
    │ PHASE C: EXPLOIT                            │
    │                                             │
    │ Read Tier 3-4: Tuning Guide for winner op   │
    │ Deep-tune winner strategy:                  │
    │ ┌─────────────────────────────────────────┐ │
    │ │ Generate variant → Eval → Diagnose      │ │
    │ │                                         │ │
    │ │ if >= 1.3x → DONE                       │ │
    │ │ if improved → continue tuning           │ │
    │ │ if 2+ non-improvements → revert to best │ │
    │ │ if 3+ non-improvements → try next       │ │
    │ │     explore strategy (if available)      │ │
    │ │ if last iteration → DONE with best      │ │
    │ └─────────────────────────────────────────┘ │
    └──────────────────┬──────────────────────────┘
                       ▼
    ┌─────────────────────────────────────────────┐
    │ COMPLETE & REFLECT                          │
    │                                             │
    │ complete_task_progress(best_speedup, ...)   │
    │ Write enhanced reflection.md                │
    │ Write algo_trace.md (algorithm decisions)   │
    │ Return JSON result                          │
    └─────────────────────────────────────────────┘
```

## Appendix B: Composite Pattern Quick Reference

For embedding in the optimizer prompt:

```
COMPOSITE PATTERNS (multi-op → strategy hint):

matmul → pointwise(1-3)           → epilogue_fusion          (4-12x)
matmul → norm → activation        → two_kernel_matmul_normact (5-12x)
matmul → reduction(sum/mean)      → algebraic_distribute      (20-74x)
conv(C_in<=16) → pool             → fused_conv_pool           (1.5-2.9x)
conv → norm → activation          → torch_conv_triton_postops (1.3-2x)
reshape → matmul → softmax → matmul → flash_attention        (2-8x)
norm → pointwise → norm           → fused_prenorm             (1.3-2x)
pointwise(3+) → reduction         → single_fused_kernel       (1.3-1.5x)
diagonal_matmul → anything        → row_scaling_fused         (10-100x)
triangular_matmul → anything      → skip_upper_tiles_fused    (5-15x)
```

## Appendix C: Bottleneck Diagnosis Quick Reference

For embedding in the optimizer prompt:

```
BOTTLENECK DIAGNOSIS (after eval):

>= 1.3x                → DONE
1.0-1.3x, correct      → TUNE: more autotune configs, fp16, larger tiles
0.5-1.0x, correct      → RETHINK: wrong memory layout? unnecessary transpose? low occupancy?
< 0.5x, correct        → WRONG STRATEGY: skip, try next
correctness failure     → FIX: shapes? masking? precision? dtype?
compile error           → FIX: Triton API? BLOCK_SIZE power-of-2? constexpr order?
```
