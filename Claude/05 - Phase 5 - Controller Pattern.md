# Phase 5: Controller Pattern

> Skill agent becomes a lightweight verification controller that catches supervisor failures.

**Date:** 2026-02-14
**Status:** Proposed

---

## 1. Problem Statement

Phase 4 introduced the supervisor to guarantee post-batch steps run. But the supervisor has the same structural vulnerability as the skill agent it replaced: it runs a monitoring loop that generates many tool-call outputs, pushing the finalization instructions out of attention.

| Phase 4 Improvement | Remaining Risk |
|---|---|
| Supervisor starts with clean context (~2K tokens vs skill's 10K+) | After 20+ `get_session_state()` calls + worker output reads, supervisor's context is 8-10K tokens of monitoring noise |
| Post-batch is embedded in the protocol, not an afterthought | The protocol is markdown instructions — the LLM can still skip steps under context pressure |
| Retry logic is a first-class phase | If the supervisor forgets Phase 5 (finalize), there's no agent to catch it |

**Root cause:** There is no agent that (a) stays lightweight enough to never get overwhelmed, and (b) has a checklist of expected outcomes to verify against.

### What We Need

An agent that:
1. **Does almost nothing** — too simple to get confused
2. **Knows the expected end-state** — has a concrete checklist
3. **Verifies after execution** — checks that outcomes match expectations
4. **Can intervene** — spawns corrective agents for missing outcomes
5. **Is the outermost agent** — nothing depends on it remembering; it IS the memory

---

## 2. Design Decision: Who Does What?

The core question is: given supervisor, workers, and the skill agent — who is responsible for **worker respawning** and **learner spawning**?

### 2.1 Worker Spawning and Respawning

| Scenario | Owner | Rationale |
|---|---|---|
| **Initial worker spawn** | **Supervisor** | Workers are the supervisor's subordinates. It knows the config, tracks their output files, monitors their completion. |
| **Retry spawn** (retryable tasks after round 1) | **Supervisor** | This is Phase 4's evaluate step. The supervisor already has the retryable task list from `get_session_state()`. Spawning retry workers is the natural next step. |
| **Supervisor crash recovery** | **Skill controller** | If the supervisor itself crashes (background task returns with incomplete results), the controller spawns a **new supervisor in resume mode**. It does NOT spawn workers directly — it delegates to a fresh supervisor. |

**Principle:** The skill controller never spawns workers directly. It only spawns supervisors. This keeps the controller trivially simple (it doesn't need to know about worker IDs, claiming, monitoring).

### 2.2 Learner Spawning

| Scenario | Owner | Rationale |
|---|---|---|
| **Normal operation** | **Supervisor** | Learner spawn is Step 5b of the supervisor's Phase 5 (finalize). The supervisor collects reflections, spawns the learner (blocking), then generates the score report. This keeps finalization self-contained. |
| **Supervisor forgot/crashed** | **Skill controller** | Controller checks if learning ran. If not, spawns the learner as a corrective action. This is the safety net. |

**Principle:** The supervisor is the **primary** executor of all lifecycle steps. The skill controller is the **verification layer** that catches supervisor failures. The supervisor should succeed ~95% of the time. The controller handles the other ~5%.

### 2.3 Why Not Move Learner to the Controller?

We considered making the controller always spawn the learner (removing it from the supervisor entirely):

| Approach | Pros | Cons |
|---|---|---|
| Supervisor spawns learner, controller verifies | Supervisor is self-contained; controller is simpler; normal path is faster | If supervisor forgets, adds ~2 min latency for controller to catch it |
| Controller always spawns learner | Guaranteed to run; supervisor is lighter | Controller needs to know reflection file paths, learner prompt, output dir — it becomes a mini-orchestrator |

**Decision:** Keep learner in supervisor. The supervisor's finalize phase is a linear sequence (reflect → learn → score) that runs after the monitoring loop exits. By this point, the monitoring noise has stopped — the supervisor's remaining instructions are short and clear. The controller is the safety net.

---

## 3. Skill Controller Design

### 3.1 Overview

The skill agent's batch/resume logic becomes a controller that manages the supervisor through a **unified control loop**. Rather than treating monitoring and verification as separate phases, the controller runs a single outer loop that transitions between monitoring (while the supervisor is active) and verification (after it exits). The loop only terminates when all outcomes are verified.

```
LAUNCH supervisor in background
CONTROL LOOP:
    MONITOR — relay output, detect stalls via session state
    if stalled → stop supervisor, fall through to verify
    if supervisor done → fall through to verify
    VERIFY — check outcomes (tasks, reflections, learning, score)
    if tasks incomplete → restart supervisor (resume, workers=min(remaining, N))
                          continue loop → back to monitor
    if post-batch missing → run corrective actions directly
    if all checks pass → break
DISPLAY final summary
```

The controller can restart the supervisor from either trigger:
- **Stall detection** (during monitoring): no progress across 2 consecutive checks
- **Incomplete tasks** (during verification): supervisor exited but tasks remain

On restart, the controller scales the worker count to match remaining work:
```
remaining = state.pending + len(state.incomplete)
restart_workers = min(remaining, num_workers)
```
This avoids wasting agents — no point spawning 12 workers for 3 leftover tasks.

At most 2 supervisor attempts total (original + 1 restart). If the second also fails, the controller proceeds with verification and reports gaps to the user.

### 3.2 Verification Checklist

After the supervisor exits (or is stopped), the controller checks:

```
CHECKLIST:
  session_dir = ~/.inference/claude_code_output/{session_id}/

  1. TASKS COMPLETE?
     state = get_session_state(session_id)
     remaining = state.pending + len(state.in_progress) + len(state.incomplete)
     ✓ if remaining == 0
     → if remaining > 0 AND attempts < 2: restart supervisor, continue loop
     → if remaining > 0 AND attempts >= 2: warn user, continue to checks 2-4

  2. REFLECTIONS COLLECTED?
     ✓ if {session_dir}/all_reflections.md exists AND is non-empty
     → run: python3 kb_reflect.py {session_id}

  3. LEARNING RAN?
     ✓ if any file in .claude/agents/learned/ was modified after session start
     → spawn learner agent (blocking)

  4. SCORE REPORT GENERATED?
     ✓ if {session_dir}/progress_*.md exists with timestamp after session start
     → run: python3 kb_score.py {session_id}
```

### 3.3 Why This Works

**The controller cannot get overwhelmed.** Its entire execution is:
1. One Task spawn (supervisor)
2. A monitor loop (read file, print — minimal MCP tool accumulation)
3. Four filesystem checks
4. 0-4 corrective actions

That's ~15 tool calls total across the entire session.

**The controller cannot forget.** The checklist is inside a `while true` loop — the controller cannot exit without passing all checks or exhausting restart attempts. There's no way to "skip to the end."

**Restart uses scaled workers.** When restarting, `workers = min(remaining, N)` ensures efficient resource use. 3 tasks remaining = 3 workers, not 12.

**The corrective actions are idempotent.** Running `kb_reflect.py` twice just overwrites the file. Running `kb_score.py` twice generates a new report. Spawning the learner twice just re-merges (merge is additive). There's no harm in the controller running a corrective action that the supervisor already completed.

---

## 4. Updated Agent Hierarchy

```
User
└── Skill Controller
    │
    ├── [Single task mode: handled directly — interactive, no supervisor]
    │
    └── [Batch / Resume mode]
        │
        └── Control Loop:
            │
            ├── MONITOR ←──────────────────────────────────┐
            │   └── Supervisor (background)                │
            │       ├── Phase 1: Init                      │
            │       ├── Phase 2: Spawn Workers             │
            │       │   ├── Worker-1 → Optimizer(s)        │
            │       │   ├── Worker-2 → Optimizer(s)        │
            │       │   └── Worker-N → Optimizer(s)        │
            │       ├── Phase 3: Monitor                   │
            │       ├── Phase 4: Evaluate + Retry          │
            │       └── Phase 5: Finalize                  │
            │           ├── kb_reflect.py                  │
            │           ├── Learner (blocking)             │
            │           └── kb_score.py                    │
            │                                              │
            ├── VERIFY                                     │
            │   ├── Check 1: Tasks complete? ──── no ──────┘
            │   │              (restart supervisor,          (workers=min(remaining,N))
            │   │               max 2 attempts)
            │   ├── Check 2: Reflections? → run kb_reflect.py
            │   ├── Check 3: Learning? → spawn learner
            │   └── Check 4: Score? → run kb_score.py
            │
            └── DONE → display summary
```

### 4.1 What Changed from Phase 4

| Responsibility | Phase 4 Owner | Phase 5 Owner | Why |
|---|---|---|---|
| Worker spawning (initial + retry) | Supervisor | **Supervisor** (unchanged) | Workers are supervisor's subordinates |
| Learner spawning | Supervisor | **Supervisor** (primary), **Controller** (fallback) | Supervisor handles normal path; controller catches failures |
| Supervisor lifecycle management | Nobody | **Skill controller** | New: unified control loop monitors, detects stalls, restarts with scaled workers |
| Post-completion verification | Nobody | **Skill controller** | New: 4-point checklist integrated into control loop |
| Dynamic worker scaling on restart | N/A | **Skill controller** | New: `workers = min(remaining, N)` on every restart |

### 4.2 Responsibility Matrix

| Agent | Primary Role | Spawns | Spawned By | Complexity |
|---|---|---|---|---|
| **Skill Controller** | Parse input, launch supervisor, control loop (monitor + verify + correct) | Supervisor (up to 2), Learner (corrective) | User | Trivial (~15 tool calls) |
| **Supervisor** | Session lifecycle: init → workers → monitor → evaluate → retry → finalize | Workers (scaled count), Learner, retry Workers | Skill controller | Medium (~30-50 tool calls) |
| **Worker** | Claim tasks, dispatch optimizers, enforce iteration count | Optimizers | Supervisor | Low (~10 tool calls/task) |
| **Optimizer** | Deep optimization: write → eval → analyze → fix (10-20 iterations) | None | Worker | High (~40-60 tool calls) |
| **Learner** | Read reflections, distill patterns, write learned files | None | Supervisor (primary) or Skill controller (corrective) | Low (~10 tool calls) |

---

## 5. Skill Controller Protocol

### 5.1 Batch Mode

```
1. PARSE args (session_id, level, workers, strategies, iterations, provider)

2. LAUNCH supervisor (background)
   max_supervisor_attempts = 2
   supervisor_attempts = 1

3. CONTROL LOOP:
   while true:
       # ─── MONITOR (while supervisor is running) ───
       while supervisor running:
           Read and relay output to user
           if stale_reads >= 10:
               state = get_session_state(session_id)
               if all tasks done: break
               if 2 consecutive checks with 0 progress:
                   TaskStop(supervisor)
                   break  # fall through to verify

       # ─── VERIFY ───
       state = get_session_state(session_id)
       remaining = state.pending + len(state.in_progress) + len(state.incomplete)

       # Check 1: Tasks complete?
       if remaining > 0 and supervisor_attempts < max_supervisor_attempts:
           restart_workers = min(remaining, num_workers)
           Spawn new supervisor (resume, workers=restart_workers)
           supervisor_attempts += 1
           continue  # back to MONITOR

       # Check 2: Reflections?
       if missing → run kb_reflect.py

       # Check 3: Learning?
       if missing → spawn learner (blocking)

       # Check 4: Score report?
       if missing → run kb_score.py

       break  # all verified

4. DISPLAY final summary
```

### 5.2 Resume Mode

Same as batch mode, except:
- Step 1 reads config from session state
- Step 2 launches supervisor with `Mode: resume` and `workers = min(remaining, num_workers)`

### 5.3 Single Task Mode

Unchanged from Phase 4. No supervisor, no controller pattern. Interactive UX.

---

## 6. Failure Scenarios and Recovery

| Failure | Detection | Recovery |
|---|---|---|
| **Supervisor completes normally** | Background task returns, all 4 checks pass | No action needed. Display summary. |
| **Supervisor forgets finalize** | Checks 2/3/4 fail in verify phase | Controller runs the missing steps directly (idempotent). |
| **Supervisor crashes mid-monitoring** | Supervisor exits, verify detects remaining > 0 | Control loop restarts supervisor in resume mode with `workers = min(remaining, N)`. |
| **Supervisor stuck: all workers exited** | Supervisor's own monitor loop detects all worker output files complete while tasks remain | Supervisor self-recovers: breaks to Phase 4 (evaluate), spawns retry workers. If supervisor itself fails to self-recover, controller's stall detection catches it (see next row). |
| **Supervisor stuck: unrecoverable** | Controller detects 2 consecutive stall checks with 0 progress change | Controller stops the supervisor, falls through to verify. Verify detects remaining > 0, restarts supervisor with `workers = min(remaining, N)`. At most 2 total attempts. |
| **Supervisor stuck: stale markers** | `get_session_state()` auto-cleans markers (PID-based: immediate; time-based: 30 min) | Markers cleaned on next poll. Tasks move from `in_progress` to `incomplete`, which the verify phase counts as remaining work. |
| **Supervisor spawns learner but learner crashes** | Check 3 fails (learned files not updated) | Controller spawns learner again. Merge is idempotent. |
| **Supervisor output file unreadable** | Monitor gets empty reads, stall counter increments | Controller falls through to verify phase after stall threshold, checks session state directly. |
| **Supervisor hits max_turns** | Claude Code kills supervisor. Background task returns. | Control loop exits monitor, enters verify. Remaining > 0 triggers restart with scaled workers. |
| **Worker crashes** | Supervisor detects via monitoring (stale markers cleaned) | Supervisor handles this (Phase 4 retry logic). Not controller's concern. |
| **Optimizer quits early** | Worker detects via iteration count check | Worker handles this (re-spawn logic). Not controller's concern. |

---

## 7. What Stays the Same

| Component | Changes? | Notes |
|---|---|---|
| Supervisor | Minor update | Monitor loop now tracks worker task IDs and detects "all workers exited" to prevent deadlock. Evaluate phase also retries pending tasks (not just retryable). |
| Worker | No | Same claim → dispatch → enforce → loop protocol. |
| Optimizer | No | Same deep optimization loop. |
| Learner | No | Same reflection distillation. May be spawned by controller instead of supervisor in failure cases. |
| MCP server | No | All tools unchanged. |
| File-based state | No | Same directory structure, same JSON files. |
| kb_score.py | No | Unchanged. |
| kb_reflect.py | No | Unchanged. |

---

## 8. Risk Analysis

| Risk | Mitigation |
|---|---|
| **Controller itself forgets verification** | Verification is inside a `while true` loop — the controller cannot exit without passing all checks or exhausting restart attempts. With ~15 tool calls total, context overload is not possible. |
| **Corrective learner conflicts with supervisor's learner** | Learner merge is additive. Running twice just re-processes the same reflections. No data loss. |
| **False positive on Check 3** (mtime check unreliable) | Use a generous threshold (within 1 hour of session start). If in doubt, run the learner — it's idempotent. |
| **Recovery supervisor also gets stuck** | At most 2 supervisor attempts. If both fail, the controller exits the loop, runs checks 2-4 on whatever was completed, and reports the gap to the user. |
| **False stall detection** (supervisor is slow, not stuck) | Requires 2 consecutive stall checks with zero progress change (completed count unchanged). A supervisor making any progress — even 1 task between checks — will not be restarted. |
| **Old supervisor and new supervisor both running** | Controller stops the old supervisor (best-effort via TaskStop). Even if the old one lingers, claim enforcement prevents duplicate work. The old supervisor eventually hits max_turns. |
| **Wasted workers on restart** | Dynamic scaling: `workers = min(remaining, N)`. 3 tasks left = 3 workers, not 12. |
| **Adds latency for corrective actions** | Only when the supervisor fails (~5% of runs). Normal path: supervisor completes everything, all 4 checks pass instantly, no corrective actions. |

---

## 9. Implementation Plan

### Files to modify:
1. `.claude/commands/kernel-bench.md` — Add unified control loop (monitor + verify + correct) to BATCH MODE and RESUME MODE sections

### Files unchanged:
- `.claude/agents/kernel-bench-supervisor.md` — No changes. Supervisor still runs finalize.
- `.claude/agents/kernel-bench-worker.md` — No changes.
- `.claude/agents/kernel-bench-optimizer.md` — No changes.
- `.claude/agents/kernel-bench-learner.md` — No changes.
- `claudeCodeKernelBenchServer.py` — No changes.

### Verification:
1. Run a batch (3 tasks, 2 workers). Verify controller passes all 4 checks on normal completion.
2. Simulate supervisor forgetting finalize: manually delete `all_reflections.md` before controller checks. Verify controller re-runs `kb_reflect.py`.
3. Simulate supervisor crash: kill supervisor task mid-run. Verify controller spawns recovery supervisor.

---

## Appendix: Design Alternatives Considered

### Alternative A: Move ALL post-batch to controller

Remove finalize from supervisor. Controller always runs reflect + learner + score after supervisor returns.

**Rejected because:** Makes the controller more complex (needs to know reflection paths, learner prompt, etc.). The supervisor's finalize is a natural part of its lifecycle — removing it makes the supervisor feel incomplete. The 95% normal case gets slower (controller adds an extra hop for steps that usually succeed).

### Alternative B: Dedicated Finalizer agent

Spawn a separate "finalizer" agent that only does reflect + learn + score.

**Rejected because:** Adds an agent to the hierarchy without solving the core problem (who verifies the finalizer ran?). The controller pattern is simpler — it verifies ALL outcomes, not just finalization.

### Alternative C: Controller spawns workers directly (no supervisor)

The controller itself does init → workers → verify. No supervisor.

**Rejected because:** The controller needs to stay trivially simple. Spawning workers requires monitoring, which requires a loop, which creates context noise — the exact problem we're solving. The supervisor exists to absorb that complexity.

### Alternative D: Supervisor verifies itself

Add a self-check at the end of the supervisor protocol.

**Rejected because:** If the supervisor's context is overloaded enough to skip finalize, it's also overloaded enough to skip the self-check. Self-verification can't catch context-induced failures.
