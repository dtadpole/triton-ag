# Phase 9: Iterative Algorithm Learning

> **Status:** Design complete. Pending implementation.
> **Date:** 2026-02-15
> **Prerequisite:** Phase 8 (optimizer core algorithm — Phase A/B/C protocol, dual knowledge streams)

---

## 1. Problem Statement

The system learns **what** kernels to write (kernel knowledge via reference files) but NOT **how** to run the optimization loop. The Phase A/B/C protocol in `kernel-bench-optimizer.md` is static — iteration budgets, explore counts, revert thresholds, diagnosis-action mappings never improve from empirical evidence, even though `algo_trace.md` captures exactly the data needed to improve them.

**Secondary problem:** The existing learner is a single blocking agent that processes all reflections and all algo traces sequentially. With 100+ tasks per session, this is slow.

**Phase 9 addresses both:** split learning into two independent agents (kernel + algorithm) that run in parallel, with further parallelism within kernel learning via per-op-type sub-agents.

---

## 2. Design Overview

### 2.1 Two Separate Learner Agents

| Agent | Input | Output | Focus |
|-------|-------|--------|-------|
| **Kernel Learner** | `all_reflections.md` | `reference/common.md`, `reference/{op_type}.md` | What strategies work for which operations |
| **Algorithm Learner** | `all_algo_traces.md`, current `optimizer.md`, session metrics | Updated `kernel-bench-optimizer.md`, `algorithm_changelog.md` | How to run the optimization loop better |

These have independent inputs and outputs → run in parallel.

### 2.2 Parallel Execution Architecture

```
Skill Controller (finalize step or /kernel-bench learn)
  │
  ├── [script] kb_reflect.py collect → all_reflections.md
  ├── [script] kb_reflect.py collect_traces → all_algo_traces.md
  ├── [script] kb_reflect.py classify → reflections_by_op/{op_type}.md
  │
  │   ── ONE message, all background ──
  │
  ├── Kernel Learner: common    → reads all_reflections.md → writes common.md
  ├── Kernel Learner: matmul    → reads reflections_by_op/matmul.md → writes reference/matmul.md
  ├── Kernel Learner: conv      → reads reflections_by_op/conv.md → writes reference/conv.md
  ├── Kernel Learner: ...       → (one per op-type WITH reflections)
  ├── Algorithm Learner         → reads all_algo_traces.md + optimizer.md → writes optimizer.md
  │
  └── Wait for all to complete
```

The skill controller spawns all learner agents in a **single message** for maximum parallelism. Only op-types that have reflections get a sub-agent.

### 2.3 Why Two Agents (Not One With Two Steps)

The previous design used one agent with two sequential steps. Problems:
- **Slow**: Sequential execution of two independent tasks
- **Context waste**: Kernel learning doesn't need optimizer.md; algorithm learning doesn't need reflections
- **Coupling**: A failure in one step blocks the other

Two separate agents:
- Run in parallel (~2x faster for the learning phase)
- Each has focused context (reads only what it needs)
- Independent failure — kernel learning succeeds even if algorithm learning fails

### 2.4 Learning Loop (Cross-Session)

```
Session N:
  Optimizers read optimizer.md (algorithm v_k)
  → produce reflection.md + algo_trace.md per task
  → kernel learner: update reference files (parallel per op-type)
  → algorithm learner: analyze traces, update optimizer.md → v_{k+1}
  (both learners run in parallel)

Session N+1:
  Optimizers read optimizer.md (v_{k+1}) + updated reference files
  → improved decisions → better traces
  → learners produce v_{k+2}
```

---

## 3. LEARN Mode (New Skill Input Style)

### 3.1 Invocation

```
/kernel-bench learn {session_id}
/kernel-bench learn {session_id} --kernel-only
/kernel-bench learn {session_id} --algo-only
```

This mode runs learning on an **existing completed session** without running a batch. Useful for:
- Re-running learning after fixing the learner prompt
- Running learning on a session where finalization was skipped
- Running only algorithm learning after a kernel-only session
- Iterating on learning quality without re-running optimization

### 3.2 Parsing

Add to the parsing logic (between "Detect verify" and "Detect single task"):

```
3b. **Detect learn**: Starts with `learn` → LEARN MODE
```

### 3.3 Execution

```
1. Parse session_id and flags (--kernel-only, --algo-only)

2. Validate session exists:
   state = get_session_state(session_id)
   if state.completed == 0:
       "No completed tasks in session '{session_id}'. Nothing to learn from."
       exit

3. Collect artifacts (if not already present):
   session_dir = ~/.inference/claude_code_output/{session_id}

   if --algo-only is NOT set:
       if all_reflections.md does NOT exist:
           Run: python3 kb_reflect.py {session_id}
       Run: python3 kb_reflect.py classify {session_id}
       # → produces reflections_by_op/{op_type}.md files

   if --kernel-only is NOT set:
       if all_algo_traces.md does NOT exist:
           Run: python3 kb_reflect.py collect_traces {session_id}

4. Snapshot optimizer.md (if algorithm learning will run):
   if --kernel-only is NOT set:
       Bash: cp .claude/agents/kernel-bench-optimizer.md \
             {session_dir}/optimizer_snapshot.md

5. Spawn learner agents (ALL in ONE message, background):
   agents = []

   if --algo-only is NOT set:
       # Kernel learner: common patterns
       agents.append(kernel_common_agent)

       # Kernel learner: per-op-type (only for op-types with reflections)
       for each reflections_by_op/{op_type}.md that exists and is non-empty:
           agents.append(kernel_op_agent(op_type))

   if --kernel-only is NOT set:
       # Algorithm learner
       agents.append(algorithm_learner_agent)

   Spawn all agents in ONE message (background)

6. Wait for all agents to complete (read output files)

7. Generate score report:
   Run: python3 kb_score.py {session_id}

8. Display summary of learning results
```

---

## 4. Kernel Learner Agents

### 4.1 Pre-Classification (`kb_reflect.py classify`)

Before spawning kernel learner agents, `kb_reflect.py classify` splits `all_reflections.md` into per-op-type files. The classification uses the `**Op type**: {type}` field already present in each reflection.

Output directory: `{session_dir}/reflections_by_op/`
- `matmul.md`, `conv.md`, `reduction.md`, `pointwise.md`, `normalization.md`, `loss.md`, `pooling.md`, `other.md`

Only files with content are created. The script also outputs a list of populated op-types for the skill controller to know which agents to spawn.

### 4.2 Common Patterns Agent

**Agent file:** `.claude/agents/kernel-bench-learner-common.md`

**Input:** `all_reflections.md` (needs all reflections to find cross-cutting patterns)

**Output:** `reference/common.md`

**What it does:**
- Reads ALL reflections from the session
- Identifies cross-cutting patterns (environment constraints, anti-patterns, universal techniques)
- Extracts composite patterns (multi-op sequences with known best strategies)
- Extracts strategy selection heuristics
- Merges with existing `common.md` (keep best from both)
- Preserves the Code Templates section

**Scope rule:** Only write cross-cutting patterns that span 2+ op-types. Op-specific patterns go to the op-type files.

### 4.3 Per-Op-Type Agents

**Agent file:** `.claude/agents/kernel-bench-learner-op.md` (single template, parameterized by op_type)

**Input:** `reflections_by_op/{op_type}.md` + existing `reference/{op_type}.md`

**Output:** Updated `reference/{op_type}.md`

**What it does:**
- Reads only the reflections for its assigned op-type
- Classifies each entry by tier (Tier 1 / Tier 2 / Tier 3-4 / Anti-Pattern)
- Selects top entries per tier (2-3 for Tier 1, 2-3 for Tier 2, 3-5 for Tier 3-4, 2-3 for Anti-Patterns)
- Merges with existing reference file (keep best from both)
- Preserves Code Templates section
- Enforces ~3KB budget (excluding Code Templates)

**Spawning:** The skill controller spawns one instance per op-type that has reflections:

```
for op_type in populated_op_types:
    Task(
      subagent_type="general-purpose",
      description=f"kernel learner {op_type}",
      prompt=f"Read .claude/agents/kernel-bench-learner-op.md — your full instructions.

              Op type: {op_type}
              Reflections: {session_dir}/reflections_by_op/{op_type}.md
              Existing reference: .claude/agents/reference/{op_type}.md
              Output: .claude/agents/reference/{op_type}.md

              Classify entries by tier, merge with existing, write updated file.",
      run_in_background=true
    )
```

### 4.4 Agent Prompt Factoring

The current monolithic `kernel-bench-learner.md` (274 lines) is split into three focused prompts:

| File | Lines (est.) | Responsibility |
|------|-------------|----------------|
| `kernel-bench-learner-common.md` | ~80 | Cross-cutting patterns → `common.md` |
| `kernel-bench-learner-op.md` | ~80 | Per-op-type patterns → `reference/{op_type}.md` (parameterized) |
| `kernel-bench-learner-algo.md` | ~150 | Algorithm trace analysis → `optimizer.md` |

Each prompt is focused and self-contained. The shared rules (tier classification, merge protocol, Code Templates preservation, size budget) are duplicated in each prompt rather than referenced — sub-agents can't read other agent files during execution.

---

## 5. Algorithm Learner Agent

### 5.1 Agent File

**New file:** `.claude/agents/kernel-bench-learner-algo.md`

### 5.2 Input

1. **`all_algo_traces.md`** — per-task Phase A/B/C decisions and Meta-Observations
2. **Current `kernel-bench-optimizer.md`** — the algorithm (with mutable section markers)
3. **Session metrics** — from `kb_score.py` report or `progress.json` files

### 5.3 Output

1. **Updated `kernel-bench-optimizer.md`** — mutable sections rewritten
2. **`{session_dir}/algorithm_changelog.md`** — what changed, why, expected impact
3. **Updated `reference/optimizer_algorithm.md`** — advisory process knowledge (audit trail)

### 5.4 Architecture: Mutable Section Rewriting

Split `kernel-bench-optimizer.md` into **immutable** (safety rules, code format, tool calling) and **mutable** (algorithm heuristics, budgets, thresholds) sections. The algorithm learner rewrites only mutable sections, identified by `<!-- MUTABLE: {id} -->` markers.

**Why this approach:**
- **Not patch-based:** LLM agents produce unreliable diff syntax
- **Not full regeneration:** Risks losing safety rules and tested protocol
- **Structured section rewriting:** Constrained blast radius, explicit boundaries, coherent per-section rewrites

### 5.5 Immutable Sections (Never Change)

| Section | Why |
|---------|-----|
| Section 1: Operating Mode (claim loop, modes) | Infrastructure |
| Section 2: Rules 1-13 | Safety / reward hacking bans |
| Section 4: Kernel Code Requirements | Code format |
| Section 5: Evaluate & Track (tool calls, naming) | Protocol |
| Section 6: Complete & Reflect (output formats) | Trace/reflection templates |
| Phase A: A1 instructions, A3 pattern matching table | Structural |
| State tracking variable definitions | Variable schema |

### 5.6 Mutable Sections (10 Section IDs)

| Section ID | What It Controls | Location |
|------------|-----------------|----------|
| `feasibility_actions` | YES/MAYBE/NO response actions | Phase A, step A4 |
| `strategy_generation_rules` | Diversification rules, how many strategies | Phase A, step A5 |
| `composite_pattern_table` | Multi-op → strategy mapping + expected speedups | Phase A, step A5 |
| `iteration_budget_table` | Phase B/C allocation by scenario | Phase A, step A5 |
| `explore_protocol` | Max evals per strategy, early exit | Phase B |
| `bottleneck_diagnosis` | Speedup range → diagnosis + action mapping | Phase B |
| `winner_selection` | Criteria for picking explore winner | Phase B |
| `exploit_tuning_actions` | Tuning actions ranked by bottleneck type | Phase C |
| `exploit_decision_tree` | Post-eval decision tree (revert/switch thresholds) | Phase C |
| `algebraic_patterns` | Algebraic reasoning pattern table | Phase A, step A1 |

### 5.7 Marker Format

```markdown
<!-- MUTABLE: iteration_budget_table -->
<!-- Version: 1 | Updated: 2026-02-15 | Source: initial -->

| Scenario | Phase B (explore) | Phase C (exploit) |
|----------|-------------------|-------------------|
| Algebraic shortcut found | 0 (skip) | all |
| Standard task (2 strategies) | 4 (2 × 2 iters) | remaining |

<!-- /MUTABLE: iteration_budget_table -->
```

### 5.8 What the Algorithm Learner Analyzes

From `all_algo_traces.md` Meta-Observations, aggregate per mutable section:

| Target Section | What to Extract from Traces |
|---------------|----------------------------|
| `iteration_budget_table` | Budget utilization: used/max, plateau detection in late iterations |
| `bottleneck_diagnosis` | Diagnosis accuracy: initial vs actual bottleneck type |
| `explore_protocol` | Explore efficiency: iters to first viable, was explore wasteful |
| `winner_selection` | Cases where explore winner wasn't the best long-term strategy |
| `exploit_decision_tree` | Revert/switch counts, success rates, optimal thresholds |
| `feasibility_actions` | Feasibility accuracy: guide prediction vs actual speedup |
| `exploit_tuning_actions` | Per-action improvement deltas by bottleneck type |
| `composite_pattern_table` | Which composite patterns led to 1.3x, which didn't |
| `strategy_generation_rules` | How many strategies needed by task complexity/level |
| `algebraic_patterns` | New algebraic shortcuts discovered in traces |

### 5.9 Change Constraint Rules

1. **Only write within mutable markers** — never modify text outside markers
2. **Preserve section IDs** — must not change
3. **Increment version number** — each rewrite increments
4. **Document rationale** — `<!-- Rationale: ... -->` comment per changed section
5. **Minimum 10 tasks** with relevant evidence before any change
6. **Max 3 sections** changed per session (limit blast radius)
7. **Threshold bounds** on numeric parameters:

| Parameter | Range | Current |
|-----------|-------|---------|
| `max_evals_per_explore_strategy` | [1, 4] | 2 |
| `consecutive_non_improvements_to_revert` | [1, 5] | 2 |
| `consecutive_non_improvements_to_switch` | [2, 6] | 3 |
| `explore_budget_per_strategy` | [1, 4] | 2 |
| Feasibility NO total iteration allocation | [1, 4] | 4 |

---

## 6. Versioning

### 6.1 Primary: Git

Agent prompts are version-controlled in git. Each algorithm update is a git-trackable change to `kernel-bench-optimizer.md`.

### 6.2 Per-Section Version Metadata

```markdown
<!-- MUTABLE: iteration_budget_table -->
<!-- Version: 3 | Updated: 2026-02-15 | Source: session_l1_20260215 -->
<!-- Previous: Version 2, Source: session_l2_20260214 -->
```

### 6.3 Session-Level Snapshots

Before algorithm learning, the skill saves:
```
~/.inference/claude_code_output/{session_id}/optimizer_snapshot.md
```

### 6.4 Rollback

```bash
# Via git
git diff HEAD~1 .claude/agents/kernel-bench-optimizer.md
git checkout HEAD~1 -- .claude/agents/kernel-bench-optimizer.md

# Via snapshot
cp ~/.inference/claude_code_output/{session_id}/optimizer_snapshot.md \
   .claude/agents/kernel-bench-optimizer.md
```

---

## 7. Safety Guardrails

### 7.1 Structural

1. **Immutable section protection** — algorithm learner prompt explicitly lists immutable sections
2. **Section ID preservation** — set of mutable section IDs must be identical before and after
3. **Format preservation** — each mutable section preserves its format type

### 7.2 Content

1. **No rule removal** — 13 rules in Section 2 are immutable
2. **Threshold bounds** — numeric thresholds constrained to allowed ranges
3. **Minimum evidence** — 10+ tasks before any change
4. **Max 3 sections** changed per session

### 7.3 Evaluation

1. **Pre/post comparison** — compare fast1.3 rate across sessions; flag if drop >5pp
2. **Changelog auditing** — `algorithm_changelog.md` is human-auditable
3. **Snapshot rollback** — always available

---

## 8. Evaluation: Cross-Session Comparison

### 8.1 Metrics

| Metric | Definition | What It Measures |
|--------|-----------|-----------------|
| `fast1.3_rate` | count(speedup ≥ 1.3x) / total_tasks | Primary success metric |
| `median_iters_to_1.3x` | Median iterations among successes | Convergence speed |
| `explore_waste_rate` | explore_iters_on_non_winners / total_explore_iters | Exploration quality |
| `exploit_plateau_rate` | exploit_iters_with_zero_delta / total_exploit_iters | Exploitation quality |
| `diagnosis_accuracy` | correct_diagnoses / total_diagnoses | Analysis quality |

### 8.2 Attribution

```
Session A (algorithm v2): fast1.3=72%, avg_speedup=2.4x
Session B (algorithm v3): fast1.3=78%, avg_speedup=2.6x
  Changed: iteration_budget_table (v2→v3), bottleneck_diagnosis (v1→v2)
  Impact: +6pp fast1.3, +0.2x avg_speedup
```

---

## 9. Concrete Examples of Algorithm Evolution

### 9.1 Iteration Budget Table

**Trace evidence:** First viable strategy found at iter 0 in 78% of L1 tasks.

```
Before: Standard task (2 strategies) | 4 (2 × 2 iters) | remaining
After:  Standard L1 task             | 2 (1 × 2 iters) | remaining
```

### 9.2 Revert/Switch Thresholds

**Trace evidence:** Switching at 2 non-improvements has 55% success rate vs 35% at 3.

```
Before: if 3+ non-improvements → switch to runner-up
After:  if 2+ non-improvements → switch to runner-up (if within 0.8x of winner)
```

### 9.3 Feasibility Actions

**Trace evidence:** 80% of MAYBE successes happen in first 4 explore iters.

```
Before: MAYBE → full explore + exploit budget
After:  MAYBE → 4 explore iters. If <1.0x after 4, complete early.
```

### 9.4 Bottleneck Diagnosis

**Trace evidence:** 60% of 0.5-1.0x tasks needed algorithm change, not layout fix.

```
Before: 0.5-1.0x, correct → RETHINK: wrong memory layout?
After:  0.5-1.0x, correct → INVESTIGATE: If explore phase, move to next strategy.
        If exploit phase, check memory layout first, then consider strategy switch.
```

---

## 10. Implementation: File Changes

### 10.1 New Agent Files

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `.claude/agents/kernel-bench-learner-common.md` | ~80 | Cross-cutting patterns → `common.md` |
| `.claude/agents/kernel-bench-learner-op.md` | ~80 | Per-op-type patterns → `reference/{op_type}.md` (parameterized) |
| `.claude/agents/kernel-bench-learner-algo.md` | ~150 | Algorithm trace analysis → `optimizer.md` |

### 10.2 Modified Files

| File | Changes |
|------|---------|
| `.claude/agents/kernel-bench-optimizer.md` | Add `<!-- MUTABLE -->` markers around 10 sections. No content changes. |
| `.claude/commands/kernel-bench.md` | Add LEARN MODE (Section 3). Update finalize step to use parallel learner agents instead of single blocking learner. |
| `kb_reflect.py` | Add `collect_traces` and `classify` subcommands. |
| `Claude/Architecture.md` | Update to document Phase 9. |

### 10.3 Deprecated Files

| File | Status |
|------|--------|
| `.claude/agents/kernel-bench-learner.md` | Replaced by three focused learner agents. Keep for reference but no longer spawned. |

### 10.4 `kb_reflect.py` Additions

**`collect_traces` subcommand:**
```python
def collect_traces(session_id):
    """Concatenate all algo_trace.md files into all_algo_traces.md."""
    # Same pattern as collect() but for algo_trace.md files
```

**`classify` subcommand:**
```python
def classify(session_id):
    """Split all_reflections.md into per-op-type files for parallel learning."""
    # Parse each reflection entry's **Op type**: field
    # Write to {session_dir}/reflections_by_op/{op_type}.md
    # Print list of populated op-types (for skill controller to know which agents to spawn)
```

### 10.5 Skill Controller: Updated Finalize Step

Replace the single blocking learner spawn (current step 5b) with:

```
# 5a. Collect and classify
Run: python3 kb_reflect.py {session_id}              # → all_reflections.md
Run: python3 kb_reflect.py collect_traces {session_id} # → all_algo_traces.md
Run: python3 kb_reflect.py classify {session_id}       # → reflections_by_op/*.md

# 5b. Snapshot optimizer.md
Bash: cp .claude/agents/kernel-bench-optimizer.md \
      {session_dir}/optimizer_snapshot.md

# 5c. Spawn ALL learner agents in ONE message (background)
agents = []

# Kernel learner: common patterns
agents.append(Task(kernel-bench-learner-common, background=true))

# Kernel learner: per-op-type (only populated types)
for op_type in populated_op_types:
    agents.append(Task(kernel-bench-learner-op(op_type), background=true))

# Algorithm learner
agents.append(Task(kernel-bench-learner-algo, background=true))

# 5d. Wait for ALL agents to complete
for agent in agents:
    Read(agent.output_file) until complete

# 5e. Score report
Run: python3 kb_score.py {session_id}
```

### 10.6 Skill Controller: LEARN Mode

Add between VERIFY MODE and SINGLE TASK MODE in parsing:

```markdown
### LEARN MODE

When user says "learn {session_id}" or provides learning flags:

1. Parse session_id and flags:
   - `--kernel-only`: only run kernel knowledge learning
   - `--algo-only`: only run algorithm learning

2. Validate session:
   state = get_session_state(session_id)
   if state.completed == 0 → error: "No completed tasks"

3. Collect artifacts (if not present):
   (same as finalize steps 5a)

4. Snapshot optimizer.md (if algorithm learning will run):
   (same as finalize step 5b, skip if --kernel-only)

5. Spawn learner agents in parallel:
   (same as finalize step 5c, filtered by flags)

6. Wait for completion:
   (same as finalize step 5d)

7. Display summary
```

---

## 11. Implementation Order

1. Add `collect_traces` and `classify` subcommands to `kb_reflect.py`
2. Add `<!-- MUTABLE -->` markers to `kernel-bench-optimizer.md` (10 sections)
3. Create `kernel-bench-learner-common.md` (from existing learner, common.md scope)
4. Create `kernel-bench-learner-op.md` (from existing learner, per-op-type scope)
5. Create `kernel-bench-learner-algo.md` (new: algorithm trace analysis)
6. Update `kernel-bench.md` skill controller:
   - Add LEARN MODE
   - Update finalize step to use parallel learner agents
7. Update `Architecture.md`

---

## 12. Verification Plan

1. Run `kb_reflect.py classify` on an existing session → verify per-op-type files are correct
2. Run `/kernel-bench learn {session_id} --kernel-only` → verify reference files updated correctly
3. Run `/kernel-bench learn {session_id} --algo-only` → verify optimizer.md mutable sections updated
4. Run `/kernel-bench learn {session_id}` → verify all agents run in parallel and complete
5. Verify immutable sections untouched: `diff optimizer_snapshot.md optimizer.md` (only mutable markers differ)
6. Run a batch session with updated optimizer.md → verify optimizers read new algorithm
7. Compare fast1.3 metrics between sessions

---

## Appendix A: Relationship to `optimizer_algorithm.md`

The existing `reference/optimizer_algorithm.md` remains as an advisory artifact. Its role shifts:

| | Phase 8 | Phase 9 |
|-|---------|---------|
| **Primary mechanism for algorithm improvement** | optimizer_algorithm.md (advisory) | Direct rewriting of optimizer.md mutable sections |
| **optimizer_algorithm.md role** | Primary process knowledge output | Audit trail / analysis artifact |
| **What changes the algorithm** | Nothing (static) | Algorithm learner rewrites mutable sections |

The algorithm learner still writes `optimizer_algorithm.md` as an intermediate analysis artifact documenting what was found and why the algorithm changed.

## Appendix B: Agent Count Summary

| Scenario | Agents Spawned | Parallelism |
|----------|---------------|-------------|
| Minimum (1 op-type + algo) | 3 | common + 1 op-type + algo |
| Typical (5 op-types + algo) | 7 | common + 5 op-types + algo |
| Maximum (8 op-types + algo) | 10 | common + 8 op-types + algo |

All agents spawned in ONE message. Each is small and focused (~80-150 line prompts). Expected wall-clock time: dominated by the slowest agent (likely common patterns or algorithm learner), not the sum.
