---
name: kernel-bench
description: Run kernel benchmark optimization with multiple input styles
---

# Kernel Bench

Optimize CUDA/Triton kernels for kernel_bench tasks with crash recovery and **parallel execution**.

## Default Behavior: Parallel Execution

**CRITICAL: Parallel execution is the default and expected behavior.**

- **Default workers**: 4 concurrent workers (configurable via `--workers=N`)
- **Default strategies**: 1 strategy per optimizer per iteration (configurable via `--strategies=N`)
- **Parallel spawning**: ALL workers MUST be spawned in a SINGLE message with multiple Task tool calls
- **Concurrent tasks**: Each worker claims and processes tasks independently in parallel

## Strategy Modes

Each optimizer worker can explore 1 or more strategies per iteration:

| Mode | Flag | Behavior | Use Case |
|------|------|----------|----------|
| **Simple (default)** | `--strategies=1` | 1 optimizer → 1 strategy per iteration | Fast, lower cost |
| **Exploration** | `--strategies=3` | 1 optimizer → 3 parallel strategies per iteration | Better coverage, higher cost |

### Simple Mode (1:1 ratio) - DEFAULT
```
/kernel-bench level1 --session=my_run
/kernel-bench level1 --session=my_run --strategies=1
```
Each optimizer generates ONE kernel per iteration, evaluates it, then iterates if needed.
- Faster per-task completion
- Lower API cost
- Good for simple operations (element-wise, basic reductions)

### Exploration Mode (1:3 ratio)
```
/kernel-bench level1 --session=my_run --strategies=3
```
Each optimizer spawns 3 strategy sub-agents IN PARALLEL per iteration, picks the best result.
- Better chance of finding optimal kernel
- Higher API cost (3x more generations per iteration)
- Recommended for complex operations (matmul, attention, fused ops)

## Input Styles (all supported)

### Style 1: Single Task (path)
```
/kernel-bench kernel_bench/level1/19_ReLU.py
/kernel-bench level1/19_ReLU.py
```
→ Direct optimization, no coordinator, interactive feedback

### Style 2: Directory Batch
```
/kernel-bench level1/
/kernel-bench kernel_bench/level2/
```
→ Spawns **4 parallel workers** directly

### Style 3: Full Parameters
```
/kernel-bench level1 --session=te --workers=4 --strategies=3 --resume
/kernel-bench level1 my_run --workers=8 --strategies=1
```
→ Batch mode with explicit session ID, worker count, and strategy mode

### Style 4: Quick Parallel (ad-hoc)
```
/kernel-bench 4 random tasks from level1
/kernel-bench tasks 1-10 from level1 using 2 agents
```
→ Natural language interpreted, spawns appropriate agents

### Style 5: Resume Session
```
/kernel-bench --resume my_run
/kernel-bench resume my_run
```
→ Resumes existing session from last checkpoint

## Parsing Logic

When invoked with `/kernel-bench [args]`, parse the input:

1. **Detect single task**: Path ends with `.py` → SINGLE TASK MODE
2. **Detect directory**: Path ends with `/` or is a level name (level1, level2, level3) → BATCH MODE
3. **Detect resume**: Contains `--resume` or starts with `resume` → RESUME MODE
4. **Detect parameters**: Contains `--session`, `--workers`, or `--strategies` → BATCH MODE with config
5. **Detect natural language**: Contains numbers + keywords ("tasks", "agents", "random") → interpret and route

**Parameter defaults:**
- `--workers=4` (if not specified)
- `--strategies=1` (if not specified, simple mode)
- `--session={level}_{timestamp}` (if not specified)

## Execution Modes

### SINGLE TASK MODE (Interactive)

When user provides a single .py file path:

1. **Read task** via `get_task_details(task_path)`
2. **Run deep optimization loop** directly (no coordinator needed):
   - UNDERSTAND: Classify operation, retrieve similar kernels
   - STRATEGIZE: Generate 3 candidate strategies
   - PARALLEL GENERATION: Spawn 3 strategy sub-agents via Task tool
   - AGGREGATE: Pick best result
   - ITERATE: Up to 3 iterations
3. **Show progress** interactively to user after each iteration
4. **Save result** via `save_benchmark_result()`

### BATCH MODE (Parallel Workers)

When user provides directory, level name, or session parameters:

1. **Parse parameters** (with defaults):
   - `--workers=N` → N concurrent workers (default: 4)
   - `--strategies=N` → N strategies per optimizer per iteration (default: 1)
   - `--session=ID` → session identifier (default: auto-generated)

2. **Initialize or resume session**:
   - If `--resume`: Call `get_session_state(session_id)` to check progress
   - Else: Call `init_session(session_id, level, config)` to create new session
   - Store `strategies` count in session config for workers to use

3. **Spawn workers DIRECTLY in parallel** (skip coordinator for efficiency):

   **CRITICAL: You MUST spawn ALL workers in a SINGLE message with MULTIPLE Task tool calls.**

   Example with 4 workers - send ONE message containing ALL of these Task calls:
   ```
   Task(
     subagent_type: "general-purpose",
     prompt: "You are optimizer worker 1 for session {session_id}.
              Strategies per iteration: {num_strategies}
              Follow .claude/agents/kernel-bench-optimizer.md protocol.
              Use claim_task() to get work, process, repeat until done.",
     name: "optimizer-1",
     run_in_background: true
   )
   Task(
     subagent_type: "general-purpose",
     prompt: "You are optimizer worker 2 for session {session_id}.
              Strategies per iteration: {num_strategies}
              Follow .claude/agents/kernel-bench-optimizer.md protocol.
              Use claim_task() to get work, process, repeat until done.",
     name: "optimizer-2",
     run_in_background: true
   )
   Task(
     subagent_type: "general-purpose",
     prompt: "You are optimizer worker 3 for session {session_id}.
              Strategies per iteration: {num_strategies}
              Follow .claude/agents/kernel-bench-optimizer.md protocol.
              Use claim_task() to get work, process, repeat until done.",
     name: "optimizer-3",
     run_in_background: true
   )
   Task(
     subagent_type: "general-purpose",
     prompt: "You are optimizer worker 4 for session {session_id}.
              Strategies per iteration: {num_strategies}
              Follow .claude/agents/kernel-bench-optimizer.md protocol.
              Use claim_task() to get work, process, repeat until done.",
     name: "optimizer-4",
     run_in_background: true
   )
   ```

   **DO NOT spawn workers one at a time. DO NOT wait for one worker to finish before spawning the next.**

4. **Monitor progress loop** (after spawning workers):

   After spawning all workers in the background, YOU (the skill/main agent) must monitor:

   ```
   while True:
       # Check progress
       state = get_session_state(session_id)

       # Report to user
       print(f"Progress: {state.completed}/{state.total} (avg {state.avg_speedup}x)")
       print(f"  - In progress: {state.in_progress}")
       print(f"  - Pending: {state.pending}")

       # Check if done
       if state.pending == 0 and state.in_progress == 0:
           break

       # Wait before next check (use Read on worker output files to check status)
       # Workers write to their output_file paths returned from Task tool
   ```

5. **Generate final summary** when all workers complete:

   ```
   final_state = get_session_state(session_id)
   summary = get_session_summary(session_id)
   
   print("=== Session Complete ===")
   print(f"Completed: {final_state.completed}/{final_state.total}")
   print(f"Failed: {final_state.failed}")
   print(f"Average speedup: {summary.avg_speedup}x")
   print(f"Top performer: {summary.top_performers[0]}")
   ```

**Key point:** The SKILL itself handles monitoring - no separate coordinator agent needed.

### RESUME MODE

When user provides `--resume my_session`:

1. **Check session exists**: Call `get_session_state(my_session)`
2. **Show current state**: Display completed, in_progress, pending counts
3. **Clean stale markers**: Auto-cleanup happens in get_session_state()
4. **Spawn workers directly**: Same as BATCH MODE step 3 - spawn N workers in parallel
5. **Monitor and complete**: Same as BATCH MODE steps 4-5

## Output Format

All skill invocations return structured JSON for consistency:

```json
{
  "mode": "single | batch | resume",
  "session_id": "...",
  "status": "initializing | running | completed",
  "progress": {
    "total": 100,
    "completed": 45,
    "in_progress": 2,
    "pending": 53
  },
  "current_task": "...",
  "avg_speedup": 1.31
}
```

## Error Handling

- **Invalid path**: "Task not found: {path}. Check path or use list_kernel_bench_tasks()."
- **Session not found** (on resume): "Session '{session_id}' not found. Available sessions: [...]"
- **No pending tasks**: "All {total} tasks completed. Average speedup: {avg}x"

## Examples

**Example 1: Single task (uses simple mode by default)**
```
User: /kernel-bench level1/19_ReLU.py

Response:
Reading task... PyTorch ReLU activation.
Generating kernel with vectorized loads strategy...
Evaluating... compiled ✓, correct ✓, speedup: 1.12x
Iteration 1: 1.12x - below target, trying block tuning...
Evaluating... compiled ✓, correct ✓, speedup: 1.31x
Iteration 2 complete. Best: 1.31x
Target reached (>= 1.3x). Saved.
```

**Example 2: Batch run with simple mode (default: --strategies=1)**
```
User: /kernel-bench level1 --session=prod_run

Response:
Initializing session "prod_run"...
Config: workers=4, strategies=1 (simple mode)
Found 100 tasks in level1.
Spawning 4 workers in parallel... [ALL spawned in single message]
  - optimizer-1: running in background
  - optimizer-2: running in background
  - optimizer-3: running in background
  - optimizer-4: running in background

Progress: 25/100 (avg 1.28x)  [4 workers, each trying 1 strategy per iteration]
Progress: 50/100 (avg 1.31x)
Progress: 75/100 (avg 1.33x)

=== Session Complete ===
Completed: 97/100 (97%)
Failed: 3
Average speedup: 1.34x
Top performer: 88_MinGPTNewGelu (2.1x)
```

**Example 3: Batch run with exploration mode (--strategies=3)**
```
User: /kernel-bench level1 --session=explore_run --strategies=3

Response:
Initializing session "explore_run"...
Config: workers=4, strategies=3 (exploration mode)
Found 100 tasks in level1.
Spawning 4 workers in parallel...
  Each worker will spawn 3 strategy sub-agents per iteration

Progress: 15/100 (avg 1.41x)  [higher speedups due to exploration]
Progress: 30/100 (avg 1.45x)
...

=== Session Complete ===
Completed: 98/100 (98%)
Average speedup: 1.47x  [better coverage from 3 strategies]
```

**Example 4: Custom worker count with exploration**
```
User: /kernel-bench level1 --session=prod_run --workers=8 --strategies=3

Response:
Initializing session "prod_run"...
Config: workers=8, strategies=3 (exploration mode)
Spawning 8 workers in parallel... [ALL 8 spawned in single message]
...
```

**Example 5: Resume**
```
User: /kernel-bench --resume prod_run

Response:
Resuming session "prod_run"...
Config: workers=4, strategies=1 (from session config)
Found 50 completed, 2 stale (cleaned), 48 pending.
Spawning 4 workers in parallel for remaining 48 tasks...
```
