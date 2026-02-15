# Phase 10: Cross-Batch Learning (Automated Multi-Batch Orchestration)

> **Status:** Design draft
> **Depends on:** Phase 9 (iterative algorithm learning)
> **Last updated:** 2026-02-15

---

## 1. Problem Statement

Today's workflow requires manual intervention between batch sessions:

```
1. User: /kernel-bench level1                  # kick off batch
2. [~45 min] Optimizers run, finalize, learn   # reference files updated on disk
3. User: restart Claude Code                   # context bloat / stale context
4. User: /kernel-bench level1                  # manually kick off next batch
5. [repeat]
```

**Hypothesis:** Iterative batches with learning between them will improve results, because:
- Failed tasks get a second chance with better knowledge (new anti-patterns, refined strategies)
- Algorithm heuristics improve between batches (iteration budgets, diagnosis calibration)
- Composite patterns from early successes inform later attempts

**Blocker:** This loop is manual, tedious, and the user believes they need to restart Claude to pick up updated `.md` files.

### Why Restart Is Not Architecturally Necessary

New `Task` agents spawned after learning always start fresh and read files from disk. The skill controller doesn't cache reference file contents — it just orchestrates. Already-running optimizer agents hold stale context, but they finish before the next batch starts. The "need to restart" comes from:

1. No automation of the batch → learn → batch loop
2. Context accumulation in the main conversation (sluggishness, not incorrectness)

Phase 10 solves both by automating the loop within a single skill invocation, where each batch cycle spawns fresh agents that read the latest files.

### How Updated Knowledge Reaches the Next Batch

The mechanism is specific and worth tracing step by step:

```
Batch 0 optimizers finish
    ↓
Skill controller spawns learner agents (Task, background)
    ↓
Learner agents call Write/Edit tool → .md files updated ON DISK
    (reference/common.md, reference/{op}.md, kernel-bench-optimizer.md)
    ↓
Skill controller waits until ALL learner agents complete
    ↓
Skill controller spawns Batch 1 optimizers (new Task agents, background)
    ↓
Each new optimizer agent starts with BLANK context (fresh subprocess)
    ↓
Optimizer's first actions use the Read tool:
    Read → kernel-bench-optimizer.md   → latest version FROM DISK
    Read → reference/common.md         → latest version FROM DISK
    Read → reference/optimizer_algorithm.md → latest version FROM DISK
    ↓
These are the UPDATED versions the learners just wrote
```

**Why this works:** The spawn prompt tells agents to *read the file themselves* (`"Read .claude/agents/kernel-bench-optimizer.md — your full instructions."`), not embedding file contents in the prompt. The `Read` tool always hits the filesystem with no caching layer. A new `Task` agent starts with an empty context — it has no memory of what the files contained in a prior batch.

**Why the old manual flow seemed to require a restart:** It didn't, architecturally. Typing `/kernel-bench level1` again after learning finishes would spawn new optimizers that read fresh files. The restart was about context cleanliness (accumulated monitor output, agent results) rather than file freshness.

### Context Management Across Batches

**The concern:** After 3 batch cycles within a single skill invocation, the skill controller's own conversation accumulates ~3 rounds of: monitor polling output (every 30s × ~45 min = ~90 messages per batch), agent completion messages (15 per batch), learner results (3-10 per batch), and finalize/scoring output. This is 300+ messages of accumulated context.

**Claude Code's automatic summarization** compresses older conversation history, so the context window doesn't literally overflow. However, practical degradation is possible:

- Summarization may lose details the skill controller needs for chain manifest updates
- The skill controller's decision-making quality may degrade as its context becomes summary-heavy
- Token throughput may slow as the conversation grows

**Mitigations built into the design:**

1. **Chain manifest as ground truth** — The skill controller writes all cross-batch state to `chain_manifest.json` on disk rather than relying on conversation memory. After summarization, it can re-read the manifest to recover batch history, retry sets, and cumulative stats.

2. **Terse inter-batch loop** — The batch loop should be deliberately terse between cycles. After learning completes and before the next batch spawns, the skill controller only needs to:
   - Read `chain_manifest.json` (ground truth for prior batches)
   - Call `get_batch_progress()` on the just-completed session (compute retry set)
   - Write `task_histories.json` (for the next batch)
   - Spawn agents

   All verbose output (monitor logs, individual agent results, scoring reports) is written to files and not retained in the conversation.

3. **No re-reading of reference files** — The skill controller never reads `optimizer.md` or `reference/*.md` itself. It only tells agents to read them. So stale context in the controller is not a correctness issue.

**Risk level:** Low for 2-3 batches (the expected use case). If chains grow to 5+ batches, context degradation should be monitored. A hard mitigation (not in Phase 10 scope) would be to shell out each batch as an entirely separate Claude Code invocation via a wrapper script, but the in-process loop should work for the target scale.

---

## 2. New Input Styles

```
/kernel-bench level1 --batches=3
/kernel-bench level1 --batches=3 --retry=all
/kernel-bench chain level1,level2,level3
/kernel-bench chain level1,level1,level2,level3 --workers=8
```

Two modes:

| Mode | Syntax | Behavior |
|------|--------|----------|
| **Repeat** | `--batches=N` | Run the same level N times, retrying below-target tasks each round |
| **Chain** | `chain L1,L2,...` | Run levels sequentially; all tasks for a level's first appearance, retry set for repeats |

---

## 3. Retry Modes

| Flag | Behavior | Use Case |
|------|----------|----------|
| `--retry=below_target` (default) | Re-run tasks with speedup < 1.3x | Standard iterative improvement |
| `--retry=failed` | Only tasks with 0x (compile/correctness/server error) | Conservative, when algorithm didn't change much |
| `--retry=all` | Re-run every task | After major algorithm changes, or to push passing tasks higher |

---

## 4. Chain Manifest

New file at `~/.inference/claude_code_output/{chain_id}/chain_manifest.json`:

```json
{
  "chain_id": "chain_20260215_143022",
  "config": {
    "retry_mode": "below_target",
    "target_speedup": 1.3,
    "max_batches": 3,
    "workers": 15,
    "original_command": "/kernel-bench level1 --batches=3"
  },
  "batches": [
    {
      "batch_index": 0,
      "session_id": "chain_20260215_143022_b0",
      "level": "level1",
      "task_count": 100,
      "status": "completed",
      "success_rate": 0.45,
      "avg_speedup": 2.1,
      "knowledge_versions": {"common.md": "v3", "optimizer.md": "v4"}
    },
    {
      "batch_index": 1,
      "session_id": "chain_20260215_143022_b1",
      "level": "level1",
      "task_count": 55,
      "status": "completed",
      "success_rate": 0.38,
      "avg_speedup": 1.9,
      "knowledge_versions": {"common.md": "v4", "optimizer.md": "v5"}
    }
  ],
  "cumulative": {
    "tasks_total": 100,
    "tasks_passing": 66,
    "cumulative_success_rate": 0.66,
    "improvement_over_batch_0": "+21pp"
  }
}
```

The `cumulative` section tracks the **union** of best results across all batches. If task X got 1.5x in batch 0, it stays at 1.5x even if it wasn't retried or regressed in batch 1.

---

## 5. Task History Injection

When re-optimizing a task, the optimizer needs to know what was already tried to avoid repeating failures. The skill controller writes `{session_dir}/task_histories.json` before spawning optimizers:

```json
{
  "19_ReLU": {
    "previous_best_speedup": 1.15,
    "previous_best_strategy": "exploit_4_unrolled_load_store",
    "strategies_tried": [
      "explore_1_fused_relu_bias (1.15x, correct)",
      "explore_2_vectorized_pointwise (0.92x, correct)",
      "exploit_3_block_sweep (1.08x, correct)"
    ],
    "reflection_summary": "Memory-bound; approaches fuse well but can't beat PyTorch's vectorized path for simple activations.",
    "from_batches": ["chain_20260215_143022_b0"]
  }
}
```

For batch 2+, histories are **merged** across all prior batches in the chain. A task retried in both batch 0 and batch 1 will have strategies from both listed.

### Optimizer Protocol Addition

New step inserted into the optimizer claim loop, after claiming a task:

```
After claiming a task, check if {session_dir}/task_histories.json exists.
If it contains an entry for this task:
  - Read the previous strategies tried
  - DO NOT repeat any strategy from the "strategies_tried" list
  - Use the reflection_summary to inform Phase A analysis
  - Start from a fundamentally different approach than previous_best_strategy
  - If all Tier 1-2 strategies were exhausted in previous attempts,
    focus on Tier 3-4 deep tuning of the highest-speedup previous approach
```

---

## 6. Execution Flow

```
/kernel-bench level1 --batches=3

┌─ BATCH 0 (100 tasks, seed knowledge) ──────────────────┐
│  init_session(chain_xxx_b0, level1)                     │
│  Spawn 15 optimizers + monitor                          │
│  Wait loop (existing)                                   │
│  Finalize: reflections → learn → files updated on disk  │
│  Record: 45/100 pass (45%)                              │
└─────────────────────────────────────────────────────────┘
         │
         │ [reference files updated, task_histories.json written]
         ▼
┌─ BATCH 1 (55 retry tasks, batch 0 knowledge) ──────────┐
│  Compute retry set: tasks from b0 with speedup < 1.3x   │
│  Write task_histories.json from b0 progress data         │
│  init_session(chain_xxx_b1, level1, tasks=retry_set)     │
│  Spawn 15 optimizers + monitor (read FRESH files)        │
│  Wait loop                                               │
│  Finalize: reflections → learn → files updated           │
│  Record: 21/55 newly pass → cumulative 66/100 (66%)      │
│  Convergence check: +21pp improvement → continue         │
└──────────────────────────────────────────────────────────┘
         │
         │ [reference files updated again]
         ▼
┌─ BATCH 2 (34 retry tasks, batch 0+1 knowledge) ────────┐
│  Compute retry set: tasks still < 1.3x across b0+b1     │
│  Write task_histories.json (merged from b0 and b1)       │
│  init_session(chain_xxx_b2, level1, tasks=retry_set)     │
│  Spawn optimizers (read FRESH files with 2 rounds learn) │
│  Wait loop                                               │
│  Finalize: reflections → learn                           │
│  Record: 10/34 newly pass → cumulative 76/100 (76%)      │
│  Convergence check: +10pp → above min threshold          │
└──────────────────────────────────────────────────────────┘

Chain Summary:
  Batch 0: 45% → Batch 1: 66% → Batch 2: 76%
  Knowledge files updated 3 times
```

---

## 7. Convergence Detection

The chain stops early if any of:

| Condition | Logic |
|-----------|-------|
| **All tasks pass** | Cumulative success rate = 100% |
| **Diminishing returns** | Last batch improved cumulative success rate by < 3pp AND avg speedup by < 0.1x |
| **Empty retry set** | No tasks below target remain |
| **Max batches reached** | `batch_index >= max_batches` |

The diminishing returns threshold is intentionally low (3pp) because even small improvements are valuable — each task passing represents a real kernel optimization.

---

## 8. Worker Count Scaling

Later batches have fewer tasks. The skill controller scales workers proportionally:

```
workers = min(configured_workers, max(4, len(retry_tasks) // 3))
```

This avoids spawning 15 agents when there are only 10 tasks. Minimum 4 workers to maintain parallelism.

---

## 9. Resume Support

Chain resume works by reading `chain_manifest.json`:

```
/kernel-bench --resume chain_20260215_143022
```

The skill controller finds the chain manifest, identifies the last completed batch, and continues from the next batch. If a batch was interrupted mid-flight, it resumes that batch first (using existing session resume logic), then continues the chain.

---

## 10. Chain-Level Reporting

### Progress Query

```
/kernel-bench progress chain_20260215_143022
```

Prints cross-batch improvement:

```
Chain: chain_20260215_143022 (level1 × 3 batches)

Batch  Tasks  Passed  Rate    Cumulative  Δ
─────  ─────  ──────  ──────  ──────────  ──────
  0     100     45    45.0%     45.0%      —
  1      55     21    38.2%     66.0%    +21.0pp
  2      34     10    29.4%     76.0%    +10.0pp

Per-task improvement examples:
  19_ReLU:          b0=1.15x → b1=1.42x ✓
  42_Softmax:       b0=0.88x → b1=0.92x → b2=1.35x ✓
  77_LayerNorm:     b0=0.00x → b1=1.21x → b2=1.21x (no retry)
```

### Best-Result Carry-Forward

Each task's "best result" is the maximum speedup from any batch. The chain report shows:

```
Task 19_ReLU:
  Batch 0: 1.15x (best: exploit_4_unrolled_load_store)
  Batch 1: 1.42x (best: explore_1_tiled_512_vectorized)  ← NEW PASS
  Final: 1.42x ✓
```

---

## 11. Verification

`/kernel-bench verify` extended to chain-level:

```
/kernel-bench verify chain_20260215_143022
```

Cross-batch checks in addition to existing per-batch verification:

- Did retry tasks actually try different strategies from previous batches?
- Did knowledge files grow between batches?
- Were convergence decisions correct?

---

## 12. Changes Required

| File | Change | Size |
|------|--------|------|
| `.claude/commands/kernel-bench.md` | Add `chain` and `--batches` parsing; batch loop orchestration; `task_histories.json` generation; convergence detection; chain summary report | Medium |
| `.claude/agents/kernel-bench-optimizer.md` | Add task history check after claiming (read `task_histories.json`, avoid repeating strategies) | Small |
| `Claude/Architecture.md` | Phase 10 section | Small |
| `kb_score.py` | Optional: chain-level reporting (`--chain` flag) | Small |

**No new MCP server tools needed.** `init_session(task_names=...)` already supports selective task lists. `get_batch_progress()` provides the data to build retry sets and task histories.

**No new scripts needed.** The skill controller handles chain manifest management and task history generation using existing MCP tools and Bash for file I/O.

---

## 13. Pseudocode: Skill Controller Batch Loop

```python
# After parsing: have levels[], max_batches, retry_mode, workers
chain_id = f"chain_{timestamp}"
chain_dir = f"~/.inference/claude_code_output/{chain_id}"
chain_manifest = {"chain_id": chain_id, "config": {...}, "batches": [], "cumulative": {...}}

all_best_results = {}  # task_name → best speedup across all batches

for batch_index in range(max_batches):
    level = levels[batch_index % len(levels)]

    # Determine task set
    if batch_index == 0:
        task_names = None  # all tasks from level
    else:
        retry_set = compute_retry_set(all_best_results, retry_mode, target=1.3)
        if len(retry_set) == 0:
            print("All tasks passed. Chain complete.")
            break
        task_names = retry_set

        # Write task histories for retry tasks
        histories = merge_histories_from_prior_batches(chain_manifest, task_names)
        write_json(f"{session_dir}/task_histories.json", histories)

    # Scale workers
    effective_workers = scale_workers(workers, len(task_names or all_tasks))

    # Session ID
    session_id = f"{chain_id}_b{batch_index}"

    # Init session
    init_session(session_id, level, task_names=task_names,
                 num_workers=effective_workers, ...)

    # === EXISTING BATCH FLOW (steps 2-5 from current skill) ===
    spawn_optimizers(session_id, effective_workers)
    spawn_monitor(session_id)
    wait_for_completion()        # existing wait loop with recovery
    finalize_and_learn()         # existing finalize: reflections → learners
    # ==========================================================

    # Record batch results
    batch_stats = get_session_summary(session_id)
    update_all_best_results(all_best_results, session_id)
    chain_manifest.batches.append({...batch_stats...})

    # Update cumulative stats
    chain_manifest.cumulative = compute_cumulative(all_best_results)
    save_json(f"{chain_dir}/chain_manifest.json", chain_manifest)

    # Convergence check
    if batch_index > 0:
        delta_pp = current_cumulative_rate - previous_cumulative_rate
        delta_speedup = current_avg - previous_avg
        if delta_pp < 0.03 and delta_speedup < 0.1:
            print(f"Converged after batch {batch_index} (Δ={delta_pp:.1%}pp)")
            break

# Print chain summary
print_chain_summary(chain_manifest)
```

---

## 14. Future Work (Not In Scope)

**Mid-batch learning (Phase 11?):** Optimizers read `reference/{op_type}.md` per-task during Phase A. If learners ran mid-batch (after 50% of tasks complete) and updated these files, later tasks claimed by the same optimizer would benefit from fresher knowledge without waiting for the full batch to finish. Requires careful synchronization.

**Cross-level task similarity:** L2 task "Conv2d + BatchNorm + ReLU" is directly relevant to L1 "Conv2d" and L3 "ResNet18". Explicit task-to-task similarity matching (beyond op-type grouping) could inform strategy selection more precisely than the current keyword-based detection.

**Adaptive batch sizing:** Instead of fixed `--batches=N`, the system could dynamically decide how many batches to run based on the learning rate curve — more batches when knowledge is growing fast, stop when plateauing.
