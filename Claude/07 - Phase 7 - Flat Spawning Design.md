# Phase 7 Design: Flat Spawning with Claim Loop + Monitor

**Date:** 2026-02-14
**Status:** Implementing

---

## 1. The Constraint

**General-purpose subagents do NOT have the Task tool.** They have 36 tools (Bash, Read, Edit, MCP tools, etc.) but cannot spawn further subagents. Only the root-level skill agent can spawn subagents.

This means the previous 4-level hierarchy (skill → supervisor → worker → optimizer) was broken. The "working" setup actually worked because the worker, unable to spawn an optimizer, read optimizer.md and did the optimization work itself — it was effectively a 2-level system already.

---

## 2. Design: Claim Loop + Monitor

### Architecture

```
Skill Controller (has Task tool)
  │
  ├── Optimizer 1 (background) ─── claim → optimize → complete → claim next → ...
  ├── Optimizer 2 (background) ─── claim → optimize → complete → claim next → ...
  ├── Optimizer 3 (background) ─── claim → optimize → complete → claim next → ...
  ├── Optimizer N (background) ─── claim → optimize → complete → claim next → ...
  │
  ├── Monitor (background) ─── poll session state → print progress → detect ALL_DONE / STALL
  │
  ├── [Stall recovery: spawn replacement optimizers + new monitor]
  │
  ├── Learner (blocking, post-batch)
  └── [Verify]
```

### Key Properties

| Property | Design Choice |
|----------|---------------|
| **Task assignment** | Optimizer calls `get_pending_tasks()` + `claim_task()` in a loop |
| **Optimizer lifespan** | Long-lived — processes many tasks sequentially |
| **Context accumulation** | Optimizer context grows across tasks (trade-off for zero idle time) |
| **GPU utilization** | Maximum — no idle gaps between tasks (claim next immediately) |
| **Progress monitoring** | Separate monitor agent polls `get_session_state()` |
| **Skill context** | Clean (~25-35 tool calls total) |

### Why Claim Loop?

Each optimizer runs a `claim → optimize → complete → claim next` loop until no tasks remain. This maximizes GPU utilization because there's no idle gap between tasks — the optimizer immediately claims the next task after completing one.

The alternative (one-task-per-optimizer, centralized scheduling from the skill) would require the skill to repeatedly spawn batches of optimizers, wait for each batch to finish, then spawn the next. This creates idle gaps when fast tasks finish before slow ones, and adds significant tool calls to the skill context.

### Why Monitor?

The skill agent needs to know when all work is done. Polling `get_session_state()` directly from the skill adds ~10-15 tool calls per batch, pushing the skill to ~130-180 total. The monitor offloads this:

- Monitor polls `get_session_state()` every ~30s
- Prints `[progress]` lines to its output file
- Exits with `[monitor] ALL_DONE` when all tasks complete
- Exits with `[monitor] STALL` if no progress for 2+ consecutive checks
- Skill reads monitor output to detect completion (~3-5 reads total)

**Skill context estimate:**
- Init: ~3 calls
- Spawn: 1 message (N optimizers + 1 monitor)
- Wait loop: ~5-8 reads of monitor output
- Finalize + verify: ~10 calls
- **Total: ~25-35 tool calls**

---

## 3. Optimizer Agent Design

The optimizer combines the previous worker's dispatch role with the optimizer's kernel generation role into a single agent.

### Flow

```
1. Read kernel-bench-optimizer.md (protocol, rules, templates)
2. Read reference/common.md (if exists)

3. CLAIM LOOP
   while True:
       pending = get_pending_tasks(session_id)
       if empty → EXIT

       task = claim_task(session_id, task_name, worker_id)
       if claim fails → try next

       pytorch_code = get_task_details(task_path)
       op_type = detect_op_type(pytorch_code)
       Read reference/{op_type}.md (if exists, and if different from last)

       4. OPTIMIZE (full iteration loop)
          best_speedup = 0
          for iteration in 0..max_iterations-1:
              generate kernel based on previous results

              result = eval_kernel(task_path, kernel_code, session_id, provider, strategy)
              update_task_progress(...)
              track best

              if speedup >= 1.3x → break

          complete_task_progress(session_id, task_name, best_speedup, ...)
          Write reflection.md

       5. Loop back to step 3
```

### Content from Previous Agents

**Kept from optimizer.md (ALL unchanged):**
- Hard rules (engineering rules 1-6, reward hacking bans 7-13)
- Algebraic reasoning section
- Strategy selection table
- Autotune config blocks
- Matmul epilogue fusion template
- Reduction templates
- Conv2d decision tree
- L2/L3 analysis techniques
- Reflection format specification

**Added from worker.md:**
- Op type detection table (keyword → op type mapping)
- Operating mode section (batch mode with claim loop)

**Removed (no longer needed):**
- Supervisor agent entirely (absorbed by skill controller)
- Worker agent entirely (merged into optimizer)
- Claim loop from worker (moved into optimizer)
- Iteration enforcement / re-spawn logic (irrelevant — optimizer runs its own loop)

---

## 4. Monitor Agent Design

~60 lines. Lightweight agent that polls session state and reports progress.

```
1. Poll get_session_state(session_id) every ~30s
2. Print: [progress] {completed}/{total} completed, {in_progress} active, avg {avg_speedup}x
3. Track consecutive zero-progress checks

Exit conditions:
- [monitor] ALL_DONE — pending=0 and in_progress=0
- [monitor] STALL — 3 consecutive checks with zero progress change
- [monitor] STUCK — 5 consecutive checks (deeper stall, optimizer agents likely dead)
```

---

## 5. Skill Controller Flow

```
1. PARSE args (session_id, level, num_workers, iterations, provider)

2. INIT session
   init_session(session_id, level, ...)

3. SPAWN (ONE message)
   - N optimizer agents (background, each runs claim loop)
   - 1 monitor agent (background, polls progress, knows expected optimizer count)

4. WAIT LOOP (progress-gated recovery)
   Read monitor output file periodically.
   Recovery continues as long as each round makes progress:
   - ALL_DONE → finalize
   - STALL / LOW_ACTIVE with progress since last round → spawn recovery optimizers + new monitor
   - STALL / LOW_ACTIVE with zero progress → stop (true stall)
   - Max 5 recovery rounds (safety cap)

5. FINALIZE
   5a. Bash: python3 kb_reflect.py {session_id}
   5b. Task: learner agent (blocking)
   5c. Bash: python3 kb_score.py {session_id}

6. VERIFY
   Check artifacts exist, run missing steps
```

---

## 6. Agent Hierarchy

```
Skill Controller (has Task tool)
  │
  ├── [Single task mode: unchanged — skill spawns optimizer directly]
  │
  └── [Batch / Resume mode — flat spawning]
      │
      ├── Optimizer 1 (claim loop, background)
      ├── Optimizer 2 (claim loop, background)
      ├── ...Optimizer N
      ├── Monitor (background)
      │
      ├── [Recovery round if STALL: new optimizers + monitor]
      │
      ├── Learner (leaf, blocking)
      └── [Verify]
```

**Agent types:**

| Agent | File | Has Task? | Role |
|-------|------|-----------|------|
| **Skill Controller** | `kernel-bench.md` | Yes | Parse, spawn, wait, finalize, verify |
| **Optimizer** | `kernel-bench-optimizer.md` | No | Claim loop + optimize (full iteration loop per task) |
| **Monitor** | `kernel-bench-monitor.md` | No | Poll session state, detect completion/stalls |
| **Learner** | `kernel-bench-learner.md` (unchanged) | No | Distill reflections → learned files |

---

## 7. Files Changed

### MODIFY
| File | Changes |
|------|---------|
| `.claude/agents/kernel-bench-optimizer.md` | Add operating mode, op type detection. Keep all optimization content. |
| `.claude/commands/kernel-bench.md` | BATCH/RESUME: replace supervisor with optimizer + monitor spawning. |
| `claude/Architecture.md` | Phase 5 → Phase 7, new hierarchy, optimizer claim loop, updated appendix. |

### CREATE
| File | Purpose |
|------|---------|
| `.claude/agents/kernel-bench-monitor.md` | Session progress monitor |

### DELETE
| File | Reason |
|------|--------|
| `.claude/agents/kernel-bench-supervisor.md` | Absorbed by skill controller |
| `.claude/agents/kernel-bench-worker.md` | Merged into optimizer |

### KEEP (unchanged)
| File | Reason |
|------|--------|
| `.claude/agents/kernel-bench-learner.md` | Still spawned by skill for finalization |
| `.claude/agents/reference/*.md` | Renamed from learned/ — reference knowledge for optimizers |
| `claudeCodeKernelBenchServer.py` | MCP tools work as-is |
| `kb_reflect.py`, `kb_score.py`, `kb_server.py` | No changes needed |
