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

**Risk level:** Low for 2-3 batches. For 10 batches, the mitigations above are sufficient because:

- **Correctness never depends on conversation memory.** Every cross-batch decision (retry set, task histories, convergence) is computed from files on disk (`chain_manifest.json`, `progress.json`, `best_result.json`). The skill controller re-reads the manifest at the start of each batch cycle. Even if summarization loses all details of batch 3, the manifest faithfully records its stats.
- **Each batch is self-contained.** The skill controller's per-batch work is the same every time: read manifest → compute retry set → write task histories → spawn agents → wait → finalize → update manifest. No batch depends on conversation context from a prior batch.
- **Verbose output is write-only.** Monitor logs, agent results, scoring reports go to files. The skill controller doesn't re-read them for future batches. Summarization can aggressively compress these without affecting future behavior.
- **Agents are isolated.** Each optimizer/learner is a fresh `Task` subprocess. They read files from disk and have no connection to the skill controller's conversation history.

**The 10-batch constraint demands one discipline:** the skill controller's inter-batch loop must be strictly terse. No "let me summarize what happened in batch 3" — just read the manifest, compute the next step, act. The pseudocode in Section 14 enforces this.

---

## 2. Core Concepts

### Chain vs. Batch

| Concept | Definition |
|---------|------------|
| **Batch** | A single session of parallel optimization — what `/kernel-bench level1` does today. Spawn N optimizers, process tasks, finalize, learn. One `session_id`, one `session_manifest.json`. |
| **Chain** | An ordered sequence of batches linked by a `chain_manifest.json`. Each batch in the chain builds on the knowledge from previous batches. One `chain_id`, multiple `session_id`s. |

A chain is the Phase 10 orchestration unit. A batch is the Phase 9 execution unit. Phase 10 doesn't change how batches work — it automates running multiple batches sequentially with learning between each one.

```
Chain "chain_20260215_143022"
├── Batch 0: session "chain_20260215_143022_b0" (100 tasks, level1)
│   └── learn → reference files updated on disk
├── Batch 1: session "chain_20260215_143022_b1" (55 retry tasks, level1)
│   └── learn → reference files updated on disk
└── Batch 2: session "chain_20260215_143022_b2" (34 retry tasks, level1)
    └── learn → reference files updated on disk
```

**`--batches=N`** creates a chain of N batches on the same level. `--batches=3` on level1 is equivalent to `chain level1,level1,level1`.

**`chain L1,L2,...`** creates a chain of batches on different levels. Each level's first appearance runs all tasks; if the same level appears again later in the chain, it runs only the retry set from its previous appearance.

Both forms produce the same data structure (a chain manifest with batch entries). The only difference is how the level sequence is specified.

---

## 3. New Input Styles

```
/kernel-bench level1 --batches=3
/kernel-bench level1 --batches=3 --retry=all
/kernel-bench chain level1,level2,level3
/kernel-bench chain level1,level1,level2,level3 --workers=8
```

Both forms create a chain. `--batches=N` is syntactic sugar for `chain` with the same level repeated N times:

| Syntax | Equivalent Chain | Level Sequence |
|--------|-----------------|----------------|
| `level2 --batches=3` | `chain level2,level2,level2` | l2 → l2 → l2 |
| `chain level1,level2,level3` | (as written) | l1 → l2 → l3 |
| `chain level1,level1,level2` | (as written) | l1 → l1 → l2 |

Same manifest, same orchestration loop, same learning pipeline. The retry set logic handles both patterns:

- **Same-level repeat** (l2→l2→l2): batch 1 retries batch 0's below-target l2 tasks. Learning between batches means each retry has better knowledge.
- **Cross-level** (l1→l2→l3): each level runs all its tasks on first appearance. Learning still runs between batches, so l2 benefits from l1 patterns and l3 benefits from l1+l2 patterns.
- **Mixed** (l1→l1→l2): batch 1 retries l1 failures. Batch 2 runs all l2 tasks (first appearance of l2) with accumulated l1 knowledge.

---

## 4. Retry Modes

| Flag | Behavior | Use Case |
|------|----------|----------|
| `--retry=below_target` (default) | Re-run tasks with speedup < 1.3x | Standard iterative improvement |
| `--retry=failed` | Only tasks with 0x (compile/correctness/server error) | Conservative, when algorithm didn't change much |
| `--retry=all` | Re-run every task | After major algorithm changes, or to push passing tasks higher |

---

## 5. Chain Manifest

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
      "knowledge_versions": {"common.md": "v3", "optimizer.md": "v4"},
      "completed_at": "2026-02-15T15:12:00Z"
    },
    {
      "batch_index": 1,
      "session_id": "chain_20260215_143022_b1",
      "level": "level1",
      "task_count": 55,
      "status": "learning",
      "success_rate": 0.38,
      "avg_speedup": 1.9,
      "knowledge_versions": null
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

### Batch Status State Machine

Each batch entry progresses through four statuses, written to the manifest at each transition:

```
running → tasks_done → learning → completed
```

| Status | Meaning | Written When |
|--------|---------|--------------|
| `running` | Optimizers are processing tasks | Batch spawns |
| `tasks_done` | All tasks complete, learning not started | Monitor reports ALL_DONE |
| `learning` | Learner agents are running | Before learner spawn |
| `completed` | Learning done, ready for next batch | All learners finish |

These write-ahead transitions are critical for resume (see Section 10).

---

## 6. Task History Injection

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

## 7. Execution Flow

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

## 8. Convergence Detection

The chain stops early if any of:

| Condition | Logic |
|-----------|-------|
| **All tasks pass** | Cumulative success rate = 100% |
| **Diminishing returns** | Last batch improved cumulative success rate by < 3pp AND avg speedup by < 0.1x |
| **Empty retry set** | No tasks below target remain |
| **Max batches reached** | `batch_index >= max_batches` |

The diminishing returns threshold is intentionally low (3pp) because even small improvements are valuable — each task passing represents a real kernel optimization.

---

## 9. Worker Count Scaling

Later batches have fewer tasks. The skill controller scales workers proportionally:

```
workers = min(configured_workers, max(4, len(retry_tasks) // 3))
```

This avoids spawning 15 agents when there are only 10 tasks. Minimum 4 workers to maintain parallelism.

---

## 10. Resume and Recovery

### 10.1 Invocation

```
/kernel-bench --resume chain_20260215_143022
```

The skill controller reads `chain_manifest.json`, finds the last batch entry, and resumes based on its status.

### 10.2 Failure Scenarios

Phase 10 introduces failure points that don't exist in single-batch mode. The batch status state machine (Section 5) ensures the skill controller can always determine where it left off.

| Failure Point | Manifest Status | What Happened | Resume Action |
|---|---|---|---|
| Mid-batch (optimizers running) | `running` | Some tasks done, some in-progress/pending | Resume the session: respawn optimizers for remaining tasks (existing session resume logic) |
| After all tasks done, before learning | `tasks_done` | All `best_result.json` written, no reflections collected | Run the full learning pipeline from scratch |
| During learning (learners running) | `learning` | Some learner agents finished, others crashed | Detect which learners completed, re-run only the missing ones |
| After learning, before next batch | `completed` | Everything done for this batch | Start next batch |
| Between batches (manifest updated) | Last batch `completed`, no next entry | Clean state | Compute retry set, start next batch |

### 10.3 Resume Flow

```python
def resume_chain(chain_id):
    manifest = read_json(f"{chain_dir}/chain_manifest.json")
    last_batch = manifest.batches[-1]

    if last_batch.status == "running":
        # Mid-batch crash. Delegate to existing session resume.
        resume_session(last_batch.session_id)
        # After session completes, fall through to tasks_done handling
        last_batch.status = "tasks_done"
        save_manifest()

    if last_batch.status == "tasks_done":
        # Tasks done but learning never started. Run full learning.
        collect_reflections(last_batch.session_id)
        last_batch.status = "learning"
        save_manifest()
        spawn_all_learners(last_batch.session_id)
        wait_for_learners()
        last_batch.status = "completed"
        save_manifest()

    elif last_batch.status == "learning":
        # Partial learning. Detect what finished, re-run the rest.
        missing = detect_incomplete_learners(last_batch.session_id)
        if missing:
            spawn_learners(missing)
            wait_for_learners()
        last_batch.status = "completed"
        save_manifest()

    # last_batch.status is now "completed"
    # Continue the chain from the next batch
    next_batch_index = last_batch.batch_index + 1
    if next_batch_index < max_batches:
        continue_chain_from(next_batch_index)
    else:
        print_chain_summary(manifest)
```

### 10.4 Detecting Incomplete Learners

When resuming from `learning` status, the skill controller checks artifacts to determine which learners finished:

| Artifact | Learner | If Missing |
|---|---|---|
| `{session_dir}/all_reflections.md` | Collection step | Re-run `kb_reflect.py` |
| `{session_dir}/all_algo_traces.md` | Trace collection | Re-run `kb_reflect.py collect_traces` |
| `{session_dir}/algorithm_changelog.md` | Algorithm learner | Re-spawn algorithm learner |
| `reference/common.md` modified after session start | Common kernel learner | Re-spawn common learner |
| `reference/{op}.md` modified after session start | Per-op kernel learner | Re-spawn that op's learner |

**Idempotency:** Re-running a learner is safe. Learners read reflections and merge with existing reference files. Running one twice produces the same (or equivalent) result because the merge is additive — the same entries deduplicate, and entries are only upgraded, never deleted.

### 10.5 Chain Extension

A completed chain can be extended by adding more batches:

```
/kernel-bench --resume chain_20260215_143022 --batches=2
```

This appends 2 more batches to the existing chain, using the chain's cumulative results to compute the initial retry set. The chain manifest grows with new batch entries.

---

## 11. Chain-Level Reporting

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

## 12. Verification and Testing

Chain verification follows the same three-layer pattern as single-batch verification (mechanical script → semantic agent → human review), extended with cross-batch checks.

### 12.1 Invocation

```
/kernel-bench verify chain_20260215_143022              # mechanical only
/kernel-bench verify chain_20260215_143022 --semantic    # + semantic agent
```

This runs the existing per-batch verification on each batch in the chain, then runs chain-level cross-batch checks on top.

### 12.2 Layer 1: Mechanical Checks (`kb_verify.py --chain`)

Per-batch checks (existing, run on each batch in the chain):
- All 17 existing per-task checks (file existence, phase compliance, strategy names, etc.)

Chain-level structural checks (new):

| # | Check | Severity | What It Catches |
|---|---|---|---|
| C1 | `chain_manifest.json` exists and is valid JSON | FAIL | Corrupted or missing manifest |
| C2 | All batch entries have valid status (`running`/`tasks_done`/`learning`/`completed`) | FAIL | Bad state machine transition |
| C3 | Batch statuses are monotonically progressed (no `completed` → `running`) | FAIL | Manifest corruption |
| C4 | Each batch `session_id` maps to an existing session directory | FAIL | Missing session data |
| C5 | Cumulative stats are consistent (union of per-batch bests matches `cumulative` section) | WARN | Manifest bookkeeping error |
| C6 | Retry set is correct: batch N's tasks are a subset of below-target tasks from batches 0..N-1 | WARN | Retry logic bug |
| C7 | `task_histories.json` exists for batch N≥1 | WARN | History injection skipped |
| C8 | Task histories reference only strategies that actually appear in prior batches' `progress.json` | WARN | Fabricated history |
| C9 | Knowledge files (`reference/*.md`) were modified between batches (mtime check) | WARN | Learning produced no output |
| C10 | Convergence decision was justified: if chain stopped early, verify Δ < 3pp AND Δ speedup < 0.1x | WARN | Premature convergence |
| C11 | No batch has 0 tasks (empty retry set should stop the chain, not create an empty batch) | FAIL | Loop logic bug |
| C12 | Worker count scaled appropriately (not 15 workers for 5 tasks) | WARN | Wasted agent spawns |

**Output:** Console summary + `chain_verification_YYYYMMDD_HHMMSS.md` in chain directory.

**Score format (matching existing pattern):**
```
Chain: chain_20260215_143022 (3 batches)

Per-batch: 100/100 PASS, 0 WARN, 0 FAIL (batch 0)
           55/55 PASS, 0 WARN, 0 FAIL   (batch 1)
           34/34 PASS, 0 WARN, 0 FAIL   (batch 2)

Cross-batch: 10/12 PASS, 2 WARN, 0 FAIL
  WARN C9:  reference/loss.md not modified between batch 0 and batch 1
  WARN C12: batch 2 used 15 workers for 34 tasks (>3:1 ratio)
```

### 12.3 Layer 2: Cross-Batch Semantic Checks (Verifier Agent)

Spawned with `--semantic` flag. Extends the existing verifier agent with chain-aware checks:

| Check | What It Verifies | How |
|---|---|---|
| **Strategy non-repetition** | Retry tasks in batch N actually tried different Tier 1-2 strategies from batch N-1 | Compare `progress.json` strategy names across batches for the same task; flag if >50% overlap |
| **Knowledge growth quality** | Learning between batches produced meaningful updates, not noise | Diff reference files between batches; check that new entries have speedup evidence, not just rewording |
| **History utilization** | Optimizers in batch N actually read and acted on `task_histories.json` | Check if Phase A analysis references prior attempts; check if strategy choices diverge from history |
| **Diminishing returns coherence** | Tasks that failed across all batches have consistent failure reasons | Read reflections across batches for the same task; flag if the optimizer kept trying the same approach class despite the history saying not to |
| **Cumulative best accuracy** | The reported "best across chain" for each task matches the actual best | Spot-check 10 tasks: read `progress.json` from all batches, verify the chain manifest's cumulative best is correct |

**Output:** `chain_semantic_verification.md` in chain directory.

### 12.4 Layer 3: End-to-End Smoke Tests

These are **manual tests** to run when first implementing Phase 10, to validate the core mechanism works. Not automated — run once during development, document results.

| Test | Setup | Expected Result | Validates |
|---|---|---|---|
| **T1: Fresh file read** | Run batch 0 (2 tasks). Manually edit `reference/common.md` to add a marker comment. Run batch 1 (same tasks). | Batch 1 optimizers' algo_trace.md references content from the marker (or at least the file read timestamp shows it was read after the edit) | New agents read updated files from disk |
| **T2: Retry set correctness** | Run chain with 5 tasks, `--batches=2`. Batch 0: 3 pass, 2 fail. | Batch 1 has exactly 2 tasks (the failures). Chain manifest shows correct cumulative (3 + however many pass in batch 1). | Retry set computation, cumulative tracking |
| **T3: Task history injection** | Run chain with 3 tasks, `--batches=2`. After batch 0, inspect `task_histories.json` in batch 1's session dir. | Histories contain correct strategy names and speedups from batch 0's `progress.json`. | History generation from prior batch data |
| **T4: Convergence stop** | Run chain with `--batches=5` on a level where most tasks pass in batch 0. | Chain stops before batch 5 (either empty retry set or diminishing returns). | Convergence detection |
| **T5: Resume from `running`** | Start a 2-batch chain. Kill Claude mid-batch-0 (after some tasks complete). Resume with `--resume`. | Batch 0 resumes (remaining tasks processed), then batch 1 runs. | Mid-batch resume + chain continuation |
| **T6: Resume from `learning`** | Start a 2-batch chain. Kill Claude during learning (after some learners finish). Resume. | Missing learners re-run. Batch 1 starts. | Partial learning recovery |
| **T7: Resume from `tasks_done`** | Start a 2-batch chain. Kill Claude after all tasks done but before learning starts. Resume. | Learning runs from scratch. Batch 1 starts. | tasks_done → learning transition |
| **T8: Chain extension** | Complete a 2-batch chain. Then `--resume chain_id --batches=1`. | Third batch appends to existing chain. Manifest has 3 entries. | Chain extension |
| **T9: Worker scaling** | Run `--batches=3` with `--workers=15` on 50 tasks. Batch 0: 30 fail. Batch 1: 15 fail. | Batch 1 uses ~10 workers. Batch 2 uses ~5 workers. | Proportional worker scaling |

### 12.5 Regression Checks

After Phase 10 is implemented, existing single-batch behavior must still work unchanged:

| Check | Command | Expected |
|---|---|---|
| Single batch (no `--batches`) | `/kernel-bench level1` | Works exactly as before. No chain manifest created. |
| Single task | `/kernel-bench level1/19_ReLU.py` | Interactive mode, unaffected. |
| Existing resume | `/kernel-bench --resume my_session` | Resumes single session, not treated as chain. |
| Existing verify | `/kernel-bench verify my_session` | Per-batch verification, no chain checks. |
| Existing learn | `/kernel-bench learn my_session` | Single-session learning, unaffected. |

---

## 13. Changes Required

| File | Change | Size |
|------|--------|------|
| `.claude/commands/kernel-bench.md` | Add `chain` and `--batches` parsing; batch loop orchestration; `task_histories.json` generation; convergence detection; chain summary report | Medium |
| `.claude/agents/kernel-bench-optimizer.md` | Add task history check after claiming (read `task_histories.json`, avoid repeating strategies) | Small |
| `.claude/agents/kernel-bench-verifier.md` | Add cross-batch semantic checks (strategy non-repetition, knowledge growth quality, history utilization) | Small |
| `kb_verify.py` | Add `--chain` flag with 12 chain-level mechanical checks (C1-C12) | Medium |
| `Claude/Architecture.md` | Phase 10 section | Small |
| `kb_score.py` | Optional: chain-level reporting (`--chain` flag) | Small |

**No new MCP server tools needed.** `init_session(task_names=...)` already supports selective task lists. `get_batch_progress()` provides the data to build retry sets and task histories.

**No new scripts needed.** The skill controller handles chain manifest management and task history generation using existing MCP tools and Bash for file I/O.

---

## 14. Pseudocode: Skill Controller Batch Loop

```python
# After parsing: have levels[], max_batches, retry_mode, workers
chain_id = f"chain_{timestamp}"
chain_dir = f"~/.inference/claude_code_output/{chain_id}"
chain_manifest = {"chain_id": chain_id, "config": {...}, "batches": [], "cumulative": {...}}
save_json(f"{chain_dir}/chain_manifest.json", chain_manifest)

all_best_results = {}  # task_name → best speedup across all batches

for batch_index in range(max_batches):
    level = levels[batch_index % len(levels)]

    # Determine task set (handles both first-appearance and retry)
    level_tasks = get_tasks_for_level(level)
    previously_attempted = [t for t in level_tasks if t in all_best_results]

    if not previously_attempted:
        # First appearance of this level — run all tasks
        task_names = None
    else:
        # Level seen before — compute retry set
        retry_set = [t for t in level_tasks
                     if all_best_results.get(t, 0) < 1.3]  # filter by retry_mode
        if len(retry_set) == 0:
            print(f"All {level} tasks pass. Skipping.")
            continue  # skip to next level in chain
        task_names = retry_set

    # Scale workers
    task_count = len(task_names or level_tasks)
    effective_workers = scale_workers(workers, task_count)

    # Session ID
    session_id = f"{chain_id}_b{batch_index}"
    session_dir = f"{chain_dir}/{session_id}"

    # Init session (creates session directory)
    init_session(session_id, level, task_names=task_names,
                 num_workers=effective_workers, ...)

    # Write task histories AFTER init_session creates the directory
    if previously_attempted and task_names:
        histories = merge_histories_from_prior_batches(chain_manifest, task_names)
        write_json(f"{session_dir}/task_histories.json", histories)

    # ── Write-ahead: mark batch as running ──
    batch_entry = {"batch_index": batch_index, "session_id": session_id,
                   "level": level, "task_count": task_count,
                   "status": "running"}
    chain_manifest.batches.append(batch_entry)
    save_json(f"{chain_dir}/chain_manifest.json", chain_manifest)

    # === EXISTING BATCH FLOW (steps 2-5 from current skill) ===
    spawn_optimizers(session_id, effective_workers)
    spawn_monitor(session_id)
    wait_for_completion()        # existing wait loop with recovery

    # ── Write-ahead: tasks done ──
    batch_entry["status"] = "tasks_done"
    save_json(f"{chain_dir}/chain_manifest.json", chain_manifest)

    # ── Write-ahead: learning ──
    batch_entry["status"] = "learning"
    save_json(f"{chain_dir}/chain_manifest.json", chain_manifest)

    finalize_and_learn()         # existing finalize: reflections → learners
    # ==========================================================

    # Record batch results
    batch_stats = get_session_summary(session_id)
    update_all_best_results(all_best_results, session_id)
    batch_entry.update({
        "status": "completed",
        "success_rate": batch_stats.success_rate,
        "avg_speedup": batch_stats.avg_speedup,
        "knowledge_versions": snapshot_knowledge_versions(),
        "completed_at": datetime.now().isoformat()
    })

    # ── Write-ahead: completed ──
    chain_manifest.cumulative = compute_cumulative(all_best_results)
    save_json(f"{chain_dir}/chain_manifest.json", chain_manifest)

    # Convergence check (only for same-level retries)
    if previously_attempted:
        prev_cumulative = chain_manifest.batches[-2].cumulative_rate if len(chain_manifest.batches) > 1 else 0
        delta_pp = chain_manifest.cumulative.cumulative_success_rate - prev_cumulative
        delta_speedup = current_avg - previous_avg
        if delta_pp < 0.03 and delta_speedup < 0.1:
            print(f"Converged after batch {batch_index} (Δ={delta_pp:.1%}pp)")
            break

# Print chain summary
print_chain_summary(chain_manifest)
```

---

## 15. Future Work (Not In Scope)

**Mid-batch learning (Phase 11?):** Optimizers read `reference/{op_type}.md` per-task during Phase A. If learners ran mid-batch (after 50% of tasks complete) and updated these files, later tasks claimed by the same optimizer would benefit from fresher knowledge without waiting for the full batch to finish. Requires careful synchronization.

**Cross-level task similarity:** L2 task "Conv2d + BatchNorm + ReLU" is directly relevant to L1 "Conv2d" and L3 "ResNet18". Explicit task-to-task similarity matching (beyond op-type grouping) could inform strategy selection more precisely than the current keyword-based detection.

**Adaptive batch sizing:** Instead of fixed `--batches=N`, the system could dynamically decide how many batches to run based on the learning rate curve — more batches when knowledge is growing fast, stop when plateauing.
