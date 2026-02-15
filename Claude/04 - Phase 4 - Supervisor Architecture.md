# Phase 4: Supervisor Architecture

> Proposed redesign of agent responsibilities to fix post-batch reliability and add retry capability.

**Date:** 2026-02-14
**Status:** Implemented — see [Architecture.md](Architecture.md) for the living reference

---

## 1. Problem Statement

The `0212_v10_l3` run exposed a structural problem: the skill agent (main Claude Code) forgot to run post-batch steps (reflection + learning) after workers completed. Fix 1 (bold warnings in prompt) is fragile because the root cause is architectural, not instructional:

| Root Cause | Why Prompt Fixes Don't Help |
|---|---|
| **Context overload** | Monitoring loop generates 10-20 status check outputs. By the time workers finish, the post-batch instructions are 4000+ tokens above the agent's attention. |
| **Notification distraction** | Background worker completion notifications interrupt the monitoring flow, causing the agent to react to the notification rather than follow its checklist. |
| **Mixed responsibilities** | The skill agent is parser, orchestrator, monitor, evaluator, and finalizer. Any single role can crowd out the others. |
| **No retry logic** | Even if post-batch ran, there's no agent responsible for evaluating "are these results good enough?" and deciding to retry failed tasks. |

### What We Need

An agent that:
1. **Starts with clean context** — no conversation history, no parsing artifacts
2. **Owns the full session lifecycle** — init through finalization, no gaps
3. **Cannot skip post-batch** — it's embedded in the agent's core loop, not an afterthought
4. **Makes retry decisions** — evaluates results and spawns additional workers if needed
5. **Is bounded** — has a finite lifecycle (init → run → evaluate → finalize → return)

---

## 2. Agent Naming

### 2.1 The Naming Problem

The Phase 3 names are misleading:

| Phase 3 Name | What It Actually Does | Problem |
|---|---|---|
| "Optimizer worker" | Claims tasks, dispatches to sub-agents, collects results | Doesn't optimize anything — it's a dispatcher |
| "Strategy sub-agent" | Runs 10-20 iteration deep optimization loop in one context window | This is the actual optimizer, not a "strategy" |

The deep, multi-iteration optimization loop in a single context window is the system's core value proposition. Calling it a "strategy sub-agent" buries this. Meanwhile, the task claimer gets the name "optimizer" despite doing no optimization.

### 2.2 Renamed Agents

| Phase 3 Name | Phase 4 Name | Rationale |
|---|---|---|
| Optimizer worker | **Worker** | Accurately describes its role: claims tasks, dispatches work, collects results. It's a worker in the pool. |
| Strategy sub-agent | **Optimizer** | It's the agent doing the actual optimization — 10-20 iterations of write → eval → analyze → fix in one context window. |

The rest keep their names:

| Agent | Name | Rationale |
|---|---|---|
| Skill | **Skill** | Standard Claude Code skill entry point. Thin dispatcher. |
| Supervisor *(new)* | **Supervisor** | Owns session lifecycle. New in Phase 4. |
| Learner | **Learner** | Distills cross-task patterns. Unchanged. |

### 2.3 File Rename

| Phase 3 File | Phase 4 File |
|---|---|
| `kernel-bench-optimizer.md` | `kernel-bench-worker.md` |
| `kernel-bench-strategy.md` | `kernel-bench-optimizer.md` |
| *(new)* | `kernel-bench-supervisor.md` |

---

## 3. Proposed Architecture

### 3.1 Agent Roles

| # | Agent | File | Responsibility | Spawns |
|---|-------|------|----------------|--------|
| 1 | **Skill** | `kernel-bench.md` | Parse input, route to mode, display results | Supervisor (batch/resume) |
| 2 | **Supervisor** *(new)* | `kernel-bench-supervisor.md` | Session lifecycle: init → workers → monitor → evaluate → retry? → finalize → return | Workers, Learner |
| 3 | **Worker** | `kernel-bench-worker.md` | Claim tasks, detect op type, spawn optimizers, enforce iteration count, move to next task | Optimizers |
| 4 | **Optimizer** | `kernel-bench-optimizer.md` | Deep optimization: own full write → eval → analyze → fix loop for one task (10-20 iterations in one context window) | None |
| 5 | **Learner** | `kernel-bench-learner.md` | Read reflections, distill cross-task patterns into learned files | None |

### 3.2 What Changed from Phase 3

| Responsibility | Phase 3 Owner | Phase 4 Owner | Why |
|---|---|---|---|
| Initialize session | Skill | **Supervisor** | Remove from skill to keep it thin |
| Spawn workers | Skill | **Supervisor** | Co-located with monitoring |
| Monitor progress | Skill | **Supervisor** | Clean context, no conversation noise |
| Evaluate batch quality | **Nobody** | **Supervisor** | New: decides if retry needed |
| Decide retry | **Nobody** | **Supervisor** | New: spawns retry workers |
| Run reflect + learn + score | Skill (forgotten) | **Supervisor** | Guaranteed: part of core protocol |
| Report to user | Skill | Skill (from supervisor output) | Unchanged — skill tails output file |
| Parse user input | Skill | Skill | Unchanged |
| Single task mode | Skill | Skill | Unchanged — no monitoring noise |

### 3.3 Agent Tree

```
User
└── Skill (thin dispatcher)
    │
    ├── [Single task mode: handled directly by skill — interactive, no supervisor]
    │
    └── [Batch / Resume mode]
        └── Supervisor (background, owns session lifecycle)
            │
            ├── Phase 1: Run
            │   ├── Worker-1 (background)
            │   │   └── Optimizer(s) per task — deep 10-20 iteration loop
            │   ├── Worker-2 (background)
            │   │   └── Optimizer(s) per task
            │   ├── Worker-3 (background)
            │   └── Worker-4 (background)
            │
            ├── Phase 2: Evaluate
            │   └── Check for retryable tasks (server_error, all_iterations_failed)
            │
            ├── Phase 3: Retry (if needed, max 1 round)
            │   ├── Worker-R1 (background)
            │   └── Worker-R2 (background)
            │
            ├── Phase 4: Finalize (ALWAYS runs)
            │   ├── python3 kb_reflect.py {session_id}
            │   ├── Learner (blocking)
            │   └── python3 kb_score.py {session_id}
            │
            └── Return: final summary JSON
```

### 3.4 Why This Partition Works

**Names match reality.** The agent that does deep, multi-iteration kernel optimization is called "Optimizer." The agent that claims tasks and dispatches is called "Worker." No ambiguity.

**Skill agent is trivially simple.** Its batch-mode logic reduces to:
1. Parse args
2. Spawn supervisor (background)
3. Periodically read supervisor's output file, display to user
4. When supervisor returns, display final summary

Even if the skill agent gets confused, the worst case is the user doesn't see intermediate progress. The post-batch pipeline still runs inside the supervisor.

**Supervisor starts clean.** It has no conversation history, no tool-call noise from parsing, no prior monitoring outputs. Its entire context is its protocol prompt + MCP tool results. This gives it maximum headroom for the monitoring loop.

**Post-batch is structural, not optional.** The supervisor's protocol is a linear pipeline: init → workers → monitor → evaluate → (retry?) → finalize → return. Finalization isn't a "remember to do this" addendum — it's the path to returning a result. The supervisor literally cannot return without running through finalization.

**Retry is a first-class phase.** Instead of hoping someone notices retryable tasks, the supervisor explicitly checks after each worker round and spawns retry workers if needed. Bounded to 1 retry round to prevent loops.

---

## 4. Supervisor Protocol

### 4.1 Lifecycle

```
RECEIVE: session_id, level, num_workers, num_strategies, max_iterations, provider, mode (batch|resume)

PHASE 1 — INIT:
  if mode == batch:
    init_session(session_id, level, num_workers, num_strategies, ...)
  elif mode == resume:
    state = get_session_state(session_id)
    extract config from state

  print("[init] Session {session_id}: {total} tasks, {num_workers} workers")

PHASE 2 — RUN WORKERS:
  spawn N workers in ONE message (background)

  monitor_loop:
    while True:
      state = get_session_state(session_id)
      print("[progress] {completed}/{total} completed, {in_progress} active, avg {avg_speedup}x")

      if state.pending == 0 and state.in_progress == 0:
        break

      wait ~30 seconds (read worker output files to fill time)

  print("[workers] All workers finished")

PHASE 3 — EVALUATE:
  state = get_session_state(session_id)
  retryable = [t for t in state.incomplete_tasks
               if t has best_result.json with retryable completion_reason]

  if retryable AND retry_round == 0:
    print("[evaluate] {len(retryable)} retryable tasks found, spawning retry workers")
    retry_workers = min(len(retryable), num_workers)
    spawn retry_workers workers (background)
    goto monitor_loop  (retry_round = 1)
  else:
    print("[evaluate] No retryable tasks (or retry already done)")

PHASE 4 — FINALIZE (always runs):
  print("[finalize] Collecting reflections...")
  run: python3 kb_reflect.py {session_id}

  print("[finalize] Running learning agent...")
  spawn learner agent (blocking), wait for completion

  print("[finalize] Generating score report...")
  run: python3 kb_score.py {session_id}

  print("[finalize] Done")

RETURN:
  final_state = get_session_state(session_id)
  return {
    session_id, total, completed, retryable, failed,
    avg_speedup, retry_rounds_used,
    post_batch: {reflections: true, learning: true, score_report: path}
  }
```

### 4.2 Retry Logic

| Condition | Action |
|---|---|
| retryable > 0 AND retry_round == 0 | Spawn min(retryable_count, num_workers) workers, retry |
| retryable > 0 AND retry_round == 1 | Accept results, proceed to finalize (prevent loop) |
| retryable == 0 | Proceed to finalize |

Retryable tasks are those with `completion_reason` in `("server_error", "all_iterations_failed")`. These were not legitimately optimized — they failed due to infrastructure or compile issues that may be transient.

Tasks with `completion_reason: "max_iterations"` and speedup > 0 are NOT retryable — the optimizer did try and produced a real (if low) result.

### 4.3 Supervisor Outputs

The supervisor prints structured progress lines to stdout. The skill agent reads these via the output file to display to the user:

```
[init] Session 0214_l3: 50 tasks, 4 workers, strategies=1
[progress] 10/50 completed, 4 active, avg 1.82x
[progress] 25/50 completed, 4 active, avg 2.14x
[progress] 48/50 completed, 0 active, avg 2.31x
[workers] All workers finished (round 1)
[evaluate] 3 retryable tasks (server_error: 2, all_iterations_failed: 1)
[retry] Spawning 2 workers for 3 retryable tasks
[progress] 50/50 completed, 1 active, avg 2.28x
[workers] All workers finished (round 2)
[evaluate] 1 retryable task remaining — max retries reached, accepting
[finalize] Collecting reflections...
[finalize] Running learning agent...
[finalize] Generating score report...
[complete] Session done: 49/50 succeeded, 1 failed, avg 2.35x
```

---

## 5. Updated Skill Agent

### 5.1 Batch/Resume Mode (simplified)

The skill agent's batch mode reduces from a 6-step protocol with inline monitoring and post-batch to:

```
1. Parse args (workers, strategies, iterations, session_id, level)
2. Spawn supervisor:
   Task(
     subagent_type="general-purpose",
     prompt="Read .claude/agents/kernel-bench-supervisor.md — it contains your full protocol.
             Mode: batch
             Session: {session_id}
             Level: {level}
             Workers: {num_workers}
             Strategies: {num_strategies}
             Max iterations: {max_iterations}
             Provider: {provider}",
     run_in_background=true
   )
3. Monitor supervisor output:
   while supervisor running:
     Read output_file, display new lines to user
4. Display supervisor's final summary
```

The skill agent's batch logic is now 4 steps, all trivial. There's nothing to forget.

### 5.2 Single Task Mode (unchanged)

Single task mode stays in the skill agent because it's interactive — the user wants to see each iteration result in real time. There's no monitoring loop, so the context-overload problem doesn't apply.

### 5.3 Progress/Server Modes (unchanged)

These are simple one-shot operations that don't need a supervisor.

---

## 6. What Stays the Same

| Component | Changes? | Notes |
|---|---|---|
| Worker (was "Optimizer") | Renamed only | Still claims tasks, spawns optimizers, enforces iteration count |
| Optimizer (was "Strategy") | Renamed only | Still owns full write → eval → fix loop. This is the core agent. |
| Learner | No | Still reads reflections, writes learned files |
| MCP server | No | All tools unchanged (Fix 2/3/4 already added completion_reason, retry, claim enforcement) |
| File-based state | No | Same directory structure, same JSON files |
| kb_score.py | No | Already updated with retryable display (Fix 4) |
| kb_reflect.py | No | Unchanged |

---

## 7. Risk Analysis

| Risk | Mitigation |
|---|---|
| **Supervisor itself gets overloaded by monitoring** | Starts with clean context (~2K tokens of prompt vs main agent's 10K+). Monitoring output is repetitive `get_session_state()` calls, not reading worker output. Much less noise than current design. |
| **Supervisor crashes mid-session** | All state is on disk. User runs `/kernel-bench --resume session_id`, skill spawns new supervisor, which picks up from get_session_state(). |
| **Retry loop runs forever** | Hard-capped at 1 retry round. After round 2, finalize regardless. |
| **User wants real-time progress** | Skill agent tails supervisor's output file. Supervisor prints `[progress]` lines. User sees updates, just through an extra hop. |
| **Adds one more agent hop (latency)** | Supervisor is spawned once and runs for the session duration. The extra spawn is ~2 seconds overhead on a multi-hour batch. Negligible. |
| **Single task mode doesn't benefit** | Kept in skill agent intentionally — no monitoring noise, interactive UX preserved. |
| **worker_id naming in MCP calls** | Workers use `worker-1` through `worker-N` instead of `optimizer-1`. The MCP server doesn't care about the string format — it's just an opaque identifier for claim ownership. |

---

## 8. Implementation Plan

### Files to create:
1. `.claude/agents/kernel-bench-supervisor.md` — supervisor protocol

### Files to rename:
2. `.claude/agents/kernel-bench-optimizer.md` → `.claude/agents/kernel-bench-worker.md`
3. `.claude/agents/kernel-bench-strategy.md` → `.claude/agents/kernel-bench-optimizer.md`

### Files to modify:
4. `.claude/commands/kernel-bench.md` — simplify batch/resume mode to spawn supervisor; update agent references
5. `.claude/agents/kernel-bench-worker.md` — update internal references (spawn "optimizer" not "strategy sub-agent")
6. `Claude/Phase 3 - current architecture.md` — update agent hierarchy diagram and naming

### Files unchanged (content):
- `.claude/agents/kernel-bench-learner.md`
- `claudeCodeKernelBenchServer.py` (Fix 2/3/4 already in place)

### Verification:
1. Run a small batch (3 tasks, 2 workers) and verify supervisor completes all 4 phases
2. Simulate a retryable task (create best_result.json with completion_reason: "server_error"), verify supervisor retries it
3. Verify skill agent displays progress from supervisor output file
4. Verify single-task mode still works (unchanged path)
