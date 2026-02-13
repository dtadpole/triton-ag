---
name: kernel-bench
description: Run kernel benchmark optimization with multiple input styles
---

# Kernel Bench

Optimize CUDA/Triton kernels for kernel_bench tasks with crash recovery and **parallel execution**.

## Default Behavior: Parallel Execution

**CRITICAL: Parallel execution is the default and expected behavior.**

- **Default workers**: 4 concurrent workers (configurable via `--workers=N`)
- **Default strategies**: 1 sub-agent per task (configurable via `--strategies=N`)
- **Parallel spawning**: ALL workers MUST be spawned in a SINGLE message with multiple Task tool calls
- **Concurrent tasks**: Each worker claims and processes tasks independently in parallel

## Strategy Modes

Each optimizer worker dispatches tasks to sub-agents that run the full optimization loop (up to 10 iterations each):

| Mode | Flag | Behavior | Use Case |
|------|------|----------|----------|
| **Simple (default)** | `--strategies=1` | 1 optimizer → 1 sub-agent per task (runs full 10-iteration loop) | Fast, lower cost |
| **Exploration** | `--strategies=3` | 1 optimizer → 3 independent sub-agents per task (each runs full loop) | Better coverage, higher cost |

### Simple Mode (1:1 ratio) - DEFAULT
```
/kernel-bench level1 --session=my_run
/kernel-bench level1 --session=my_run --strategies=1
```
Each optimizer spawns ONE sub-agent per task. The sub-agent runs the full write → eval → fix loop for up to 10 iterations.
- Faster per-task completion
- Lower API cost
- Good for simple operations (element-wise, basic reductions)

### Exploration Mode (1:3 ratio)
```
/kernel-bench level1 --session=my_run --strategies=3
```
Each optimizer spawns 3 independent sub-agents per task IN PARALLEL. Each sub-agent runs its own full optimization loop with a different initial strategy. The optimizer takes the best result.
- Better chance of finding optimal kernel
- Higher API cost (3x more generations per task)
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

### Style 6: Progress Query
```
/kernel-bench progress test1
/kernel-bench progress test1 --detail
/kernel-bench progress test1 19_ReLU
```
→ Query batch progress or single task details

### Style 7: Server Configuration
```
/kernel-bench server
/kernel-bench server --port=5676
/kernel-bench server --port=8082 --provider=gpu_server
/kernel-bench server --list
```
→ View or update kbEval server configuration (always localhost via SSH tunnel)

## Parsing Logic

When invoked with `/kernel-bench [args]`, parse the input:

**⚠️ CRITICAL: For "progress" queries, you MUST run `python3 kb_score.py` - see PROGRESS MODE section below!**

1. **Detect server config**: Starts with `server` → **SERVER MODE**
2. **Detect progress query**: Starts with `progress` OR user just says "progress" → **PROGRESS MODE** (run the script!)
3. **Detect single task**: Path ends with `.py` → SINGLE TASK MODE
4. **Detect directory**: Path ends with `/` or is a level name (level1, level2, level3) → BATCH MODE
5. **Detect resume**: Contains `--resume` or starts with `resume` → RESUME MODE
6. **Detect parameters**: Contains `--session`, `--workers`, `--strategies`, or `--iterations` → BATCH MODE with config
7. **Detect natural language**: Contains numbers + keywords ("tasks", "agents", "random") → interpret and route

**Parameter defaults:**
- `--workers=4` (if not specified)
- `--strategies=1` (if not specified, simple mode)
- `--iterations=10` (if not specified, max iterations per task)
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
5. **Aggregate reflections**: Run `python3 kb_reflect.py {session_id}` to collect reflections, then spawn the learning agent (see BATCH MODE Step 6b) to update `.claude/agents/learned/`

### BATCH MODE (Parallel Workers)

When user provides directory, level name, or session parameters:

1. **Parse parameters** (with defaults):
   - `--workers=N` → N concurrent workers (default: 4)
   - `--strategies=N` → N strategies per optimizer per iteration (default: 1)
   - `--iterations=N` → max iterations per task (default: 10)
   - `--session=ID` → session identifier (default: auto-generated)
   - `--provider=X` → kbEval provider (default: "local")

2. **Initialize session with full config**:

   Build the original command string from parsed args:
   ```
   original_command = "/kernel-bench {level} --session={session_id} --workers={num_workers} --strategies={num_strategies} --iterations={max_iterations}"
   ```

   Call `init_session()` with all config params:
   ```
   init_session(
     session_id=session_id,
     level=level,
     num_workers=num_workers,
     num_strategies=num_strategies,
     max_iterations=max_iterations,
     provider=provider,
     code_type="triton",
     original_command=original_command
   )
   ```

   This stores the full config in `session_manifest.json` for resume support.

3. **Spawn workers DIRECTLY in parallel** (skip coordinator for efficiency):

   **CRITICAL: You MUST spawn ALL workers in a SINGLE message with MULTIPLE Task tool calls.**

   Example with 4 workers - send ONE message containing ALL of these Task calls:
   ```
   Task(
     subagent_type: "general-purpose",
     prompt: "You are optimizer worker 1 for session {session_id}.
              Strategies per iteration: {num_strategies}
              Max iterations per task: {max_iterations}
              Follow .claude/agents/kernel-bench-optimizer.md protocol.
              Use claim_task() to get work, process, repeat until done.",
     name: "optimizer-1",
     run_in_background: true
   )
   Task(
     subagent_type: "general-purpose",
     prompt: "You are optimizer worker 2 for session {session_id}.
              Strategies per iteration: {num_strategies}
              Max iterations per task: {max_iterations}
              Follow .claude/agents/kernel-bench-optimizer.md protocol.
              Use claim_task() to get work, process, repeat until done.",
     name: "optimizer-2",
     run_in_background: true
   )
   Task(
     subagent_type: "general-purpose",
     prompt: "You are optimizer worker 3 for session {session_id}.
              Strategies per iteration: {num_strategies}
              Max iterations per task: {max_iterations}
              Follow .claude/agents/kernel-bench-optimizer.md protocol.
              Use claim_task() to get work, process, repeat until done.",
     name: "optimizer-3",
     run_in_background: true
   )
   Task(
     subagent_type: "general-purpose",
     prompt: "You are optimizer worker 4 for session {session_id}.
              Strategies per iteration: {num_strategies}
              Max iterations per task: {max_iterations}
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

6. **Aggregate reflections** (post-batch):

   After all workers complete, run the two-step learning pipeline:

   **Step 6a: Collect reflections**
   ```bash
   python3 kb_reflect.py {session_id}
   ```
   This concatenates all `reflection.md` files into `~/.inference/claude_code_output/{session_id}/all_reflections.md`.

   **Step 6b: Spawn learning agent**
   ```
   Task(
     subagent_type: "general-purpose",
     prompt: "Read .claude/agents/kernel-bench-learner.md — it contains your full instructions.

              You are the learning agent for session '{session_id}'.
              Reflections file: ~/.inference/claude_code_output/{session_id}/all_reflections.md
              Output directory: .claude/agents/learned/

              Read ALL reflections, identify cross-cutting patterns, and produce:
              - learned/common.md (environment constraints, anti-patterns, universal techniques)
              - learned/{op_type}.md for each op type (top 3 successes + top 2 failures)

              If existing learned files exist, merge with them (keep best from both).
              Return a summary of what you wrote."
   )
   ```
   The learning agent reads all reflections (~26KB for 50 tasks) and produces distilled
   knowledge files. This replaces the old mechanical top-5-by-speedup aggregation with
   intelligent pattern identification that captures failures, anti-patterns, and cross-cutting insights.

**Key point:** The SKILL itself handles monitoring - no separate coordinator agent needed.

### RESUME MODE

When user provides `--resume my_session`:

1. **Get session state with config**: Call `get_session_state(my_session)`

   The response includes the original config:
   ```json
   {
     "session_id": "my_session",
     "level": "level1",
     "created_at": "2026-02-04T...",
     "config": {
       "original_command": "/kernel-bench level1 --session=my_session --workers=4 --strategies=3",
       "num_workers": 4,
       "num_strategies": 3,
       "provider": "local",
       "code_type": "triton"
     },
     "total": 100,
     "completed": 50,
     "pending": 50,
     ...
   }
   ```

2. **Extract config from session** (use stored values, not defaults):
   ```
   num_workers = state.config.get("num_workers", 4)
   num_strategies = state.config.get("num_strategies", 1)
   provider = state.config.get("provider", "local")
   ```

3. **Show resume info to user**:
   ```
   Resuming session "{session_id}"...
   Original command: {state.config.original_command}
   Config: workers={num_workers}, strategies={num_strategies}, provider={provider}
   Progress: {state.completed}/{state.total} completed, {state.pending} remaining
   ```

4. **Clean stale markers**: Auto-cleanup happens in `get_session_state()`

5. **Spawn workers directly**: Same as BATCH MODE step 3 - spawn N workers in parallel using the stored `num_workers` and `num_strategies`

6. **Monitor and complete**: Same as BATCH MODE steps 4-5

### PROGRESS MODE

**⚠️ MANDATORY: You MUST run the progress script - do NOT just call MCP tools directly!**

When user says "progress", "progress {session_id}", or just asks about progress:

**STEP 1 (REQUIRED): Run the progress script via Bash:**
```bash
python3 kb_score.py {session_id}
```

This script:
- Outputs a quick summary to stdout (which you display to the user)
- **Automatically generates a detailed markdown report file** in the session directory

**STEP 2: Display the script output** directly in the chat:
```
   ═══ Session: {session_id} ═══

     Total:       100
     Completed:   98
     In Progress: 2
     Pending:     0
     Failed:      0
     Avg Speedup: 2.607x

   📄 Detailed report: /path/to/session/progress_YYYYMMDD_HHMMSS.md
   ```

The script automatically generates a detailed markdown report file in the session directory with:
- Full summary table
- In-progress tasks with worker/iteration info
- All completed tasks sorted by speedup with iteration paths
- Failed tasks with error details
- Pending tasks list

**If no session_id is provided**, use the most recent session:
```bash
session=$(ls -t ~/.inference/claude_code_output/ | head -1)
python3 kb_score.py $session
```

#### Single Task Detail ("progress {session_id} {task_name}")

```bash
# Get single task progress via MCP
result = get_task_progress(session_id='{session_id}', task_name='{task_name}')
```

Format as:
```
=== Task: {task_name} ===
Status: {status} | Worker: {worker} | Iterations: {done}/{planned}

| Iter | Strategy | Compiled | Correct | Speedup | Runtime | Error |
|------|----------|----------|---------|---------|---------|-------|
(all iterations with full details)

Best: iteration {N}, {speedup}x ({strategy})
```

### SERVER MODE

When user says "server", "server --list", or provides server configuration parameters:


**Architecture Note:** The kbEval server runs on a remote GPU machine. You access it via SSH tunnel,
so from kernel-bench's perspective, **all servers are localhost** - the remote hostname is implicit
in the SSH tunnel the user sets up separately.

```
User's machine                          Remote GPU server
┌─────────────────┐                     ┌─────────────────┐
│ Claude Code     │                     │ kbEvalServer    │
│ kernel-bench    │─── SSH tunnel ────▶│ (port 5676)     │
│ localhost:5676  │                     │ GPU evaluation  │
└─────────────────┘                     └─────────────────┘
```

**Use the `kb_server.py` script for all server operations.**

#### List Available Providers (`/kernel-bench server` or `/kernel-bench server --list`)

```bash
python3 kb_server.py list
```

This displays all configured providers, their localhost URLs, and API key status.

#### Add/Update Provider (`/kernel-bench server --port=PORT [--provider=NAME]`)

```bash
python3 kb_server.py add {provider_name} http://localhost:{port} [--timeout=N]
```

Examples:
```bash
# Default provider on port 5676 (most common - SSH tunnel to single server)
python3 kb_server.py add local http://localhost:5676

# Named provider on different port (for multiple SSH tunnels)
python3 kb_server.py add gpu_server http://localhost:8082 --timeout=600
```

**Note:** The `--devices` flag is informational only. The actual GPU count is reported by the server.

#### Test Server Connection (`/kernel-bench server test [PROVIDER]`)

```bash
python3 kb_server.py test              # Test default (local)
python3 kb_server.py test gpu_server   # Test specific provider
```

#### Set API Key (`/kernel-bench server key`)

The API key is stored at `~/.keys/kbeval.api.key` and shared by all providers.

```bash
python3 kb_server.py key                    # View current key
python3 kb_server.py key --set=YOUR_KEY     # Set new key
```

If user doesn't have an API key, ask them to provide one or check with server admin.

#### SSH Tunnel Setup (`/kernel-bench server tunnel`)

SSH tunnels are managed separately by the user. This command provides a helper:

```bash
python3 kb_server.py tunnel user@remote-server:8082 --local=5676
```

This runs: `ssh -N -L 5676:localhost:8082 user@remote-server`

**Typical workflow:**
1. User opens SSH tunnel in a separate terminal: `ssh -L 5676:localhost:5676 -N devvm.server.com`
2. Add provider pointing to local tunnel port: `python3 kb_server.py add local http://localhost:5676`
3. Run kernel-bench normally - it accesses localhost:5676 which tunnels to the remote server

#### Change Port (`/kernel-bench server --port=PORT`)

When only `--port` is provided, update the default `local` provider:

```bash
python3 kb_server.py add local http://localhost:{port}
```

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

Progress: 25/100 (avg 1.28x)  [each worker dispatches tasks to sub-agents]
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
  Each worker spawns 3 independent sub-agents per task (each runs full loop)

Progress: 15/100 (avg 1.41x)  [higher speedups due to parallel exploration]
Progress: 30/100 (avg 1.45x)
...

=== Session Complete ===
Completed: 98/100 (98%)
Average speedup: 1.47x  [better coverage from 3 independent sub-agents]
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

**Example 5: Resume (config restored from session)**
```
User: /kernel-bench --resume prod_run

Response:
Resuming session "prod_run"...
Original command: /kernel-bench level1 --session=prod_run --workers=4 --strategies=3
Config restored: workers=4, strategies=3, provider=local
Progress: 50/100 completed, 2 stale (cleaned), 48 pending.
Spawning 4 workers in parallel for remaining 48 tasks...
  - optimizer-1: running in background
  - optimizer-2: running in background
  - optimizer-3: running in background
  - optimizer-4: running in background

Progress: 55/100 (avg 1.38x)
...
```

**Example 6: Query batch progress**
```
User: /kernel-bench progress test1

Response:
=== Session: test1 (level1) ===
Progress: 25/100 completed, 12 in progress, 3 failed, 60 pending
Average speedup: 1.34x

| Task                | Status      | Worker      | Iter | Best   | Last Result           |
|---------------------|-------------|-------------|------|--------|-----------------------|
| 19_ReLU             | in_progress | optimizer-3 | 2/3  | 1.12x  | ✓ compiled, ✓ correct |
| 23_Softmax          | in_progress | optimizer-7 | 1/3  | -      | ✗ compile error       |
| 88_MinGPTNewGelu    | completed   | optimizer-2 | 2/3  | 2.1x   | ✓ done                |
```

**Example 7: Query single task detail**
```
User: /kernel-bench progress test1 19_ReLU

Response:
=== Task: 19_ReLU ===
Status: in_progress
Worker: optimizer-3
Started: 2026-02-04T22:16:26
Iterations: 2/3

| Iter | Strategy          | Compiled | Correct | Speedup | Runtime  |
|------|-------------------|----------|---------|---------|----------|
| 0    | vectorized_loads  | ✓        | ✓       | 1.12x   | 0.042ms  |
| 1    | block_tuning_512  | ✓        | ✓       | 1.31x   | 0.038ms  |

Best: iteration 1, 1.31x (block_tuning_512)
```

**Example 8: Configure eval server port**
```
User: /kernel-bench server --port=8082 --provider=gpu_cluster

Response:
[Runs: python3 kb_server.py add gpu_cluster http://localhost:8082]

Updated provider 'gpu_cluster':
  URL: http://localhost:8082
  Timeout: 300s

Testing gpu_cluster (http://localhost:8082)...
  ✓ Connected - 8 GPU(s) available

Note: This assumes you have an SSH tunnel forwarding localhost:8082 to your GPU server.
To use: /kernel-bench level1 --session=test --provider=gpu_cluster
```

**Example 9: List available servers**
```
User: /kernel-bench server

Response:
[Runs: python3 kb_server.py list]

=== kbEval Server Configuration ===

Provider        URL                              Timeout
----------------------------------------------------------------------
local           http://localhost:5676            300s
gpu_cluster     http://localhost:8082            300s

Note: All providers use localhost - remote servers are accessed via SSH tunnel.

API Key: test_key...2345 (/Users/user/.keys/kbeval.api.key)
```

**Example 10: Set API key and test connection**
```
User: /kernel-bench server key --set=my_secret_key

Response:
[Runs: python3 kb_server.py key --set=my_secret_key]

API key saved to /Users/user/.keys/kbeval.api.key
```

**Example 11: SSH tunnel helper**
```
User: /kernel-bench server tunnel user@gpu-server.example.com:8082

Response:
[Runs: python3 kb_server.py tunnel user@gpu-server.example.com:8082 --local=5676]

Setting up SSH tunnel:
  Local:  localhost:5676
  Remote: user@gpu-server.example.com:8082

Command: ssh -N -L 5676:localhost:8082 user@gpu-server.example.com

Starting tunnel (Ctrl+C to stop)...

Note: After tunnel is running, kernel-bench accesses localhost:5676 to reach the remote GPU server.
```
