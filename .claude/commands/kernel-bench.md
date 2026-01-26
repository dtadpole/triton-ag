---
name: kernel-bench
description: Run kernel benchmark optimization with multiple input styles
---

# Kernel Bench

Optimize CUDA/Triton kernels for kernel_bench tasks with crash recovery and parallel execution.

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
→ Spawns coordinator + workers for all tasks in directory

### Style 3: Full Parameters
```
/kernel-bench level1 --session=my_run --workers=4 --resume
/kernel-bench level1 my_run --workers=4
```
→ Batch mode with explicit session ID and worker count

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
4. **Detect parameters**: Contains `--session` or `--workers` → BATCH MODE with config
5. **Detect natural language**: Contains numbers + keywords ("tasks", "agents", "random") → interpret and route

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

### BATCH MODE (Coordinator-led)

When user provides directory, level name, or session parameters:

1. **Initialize or resume session**:
   - If `--resume`: Call `get_session_state(session_id)` to check progress
   - Else: Call `init_session(session_id, level, config)` to create new session

2. **Spawn Coordinator agent** via Task tool with prompt:
   ```
   You are coordinating a kernel optimization batch run.
   Session: {session_id}
   Level: {level}
   Workers: {num_workers}

   Your responsibilities:
   1. Call get_session_state() to check current progress
   2. Spawn {num_workers} Optimizer Worker agents in parallel
   3. Monitor progress periodically
   4. Generate final summary when all workers complete

   Use the `.claude/agents/kernel-bench-coordinator.md` prompt.
   ```

3. **Monitor and report progress** as coordinator provides updates

4. **Show final summary** when complete

### RESUME MODE

When user provides `--resume my_session`:

1. **Check session exists**: Call `get_session_state(my_session)`
2. **Show current state**: Display completed, in_progress, pending counts
3. **Clean stale markers**: Auto-cleanup happens in get_session_state()
4. **Continue with Coordinator**: Spawn coordinator to finish remaining tasks

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

**Example 1: Single task**
```
User: /kernel-bench level1/19_ReLU.py

Response:
Reading task... PyTorch ReLU activation.
Generating 3 strategies in parallel...
  Strategy A (vectorized loads): 1.12x
  Strategy B (fast math): 1.08x
  Strategy C (block tuning): 1.31x ← Best
Iteration 1 complete. Best: 1.31x
Target reached (>= 1.3x). Saved.
```

**Example 2: Batch run**
```
User: /kernel-bench level1 --session=prod_run --workers=4

Response:
Initializing session "prod_run"...
Found 100 tasks in level1.
Spawning coordinator...
Coordinator spawned 4 optimizer workers.

Progress: 25/100 (avg 1.28x)
Progress: 50/100 (avg 1.31x)
Progress: 75/100 (avg 1.33x)

=== Session Complete ===
Completed: 97/100 (97%)
Failed: 3
Average speedup: 1.34x
Top performer: 88_MinGPTNewGelu (2.1x)
```

**Example 3: Resume**
```
User: /kernel-bench --resume prod_run

Response:
Resuming session "prod_run"...
Found 50 completed, 2 stale (cleaned), 48 pending.
Spawning coordinator for remaining 50 tasks...
```
