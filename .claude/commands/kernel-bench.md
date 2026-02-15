---
name: kernel-bench
description: Run kernel benchmark optimization with multiple input styles
---

# Kernel Bench

Optimize CUDA/Triton kernels for kernel_bench tasks with crash recovery and **parallel execution**.

## Default Behavior: Parallel Execution

**CRITICAL: Parallel execution is the default and expected behavior.**

- **Default workers**: 4 concurrent workers (configurable via `--workers=N`)
- **Default strategies**: 1 optimizer per task (configurable via `--strategies=N`)
- **Parallel spawning**: ALL workers MUST be spawned in a SINGLE message with multiple Task tool calls
- **Concurrent tasks**: Each worker claims and processes tasks independently in parallel

## Strategy Modes

Each worker dispatches tasks to optimizer agents that run the full optimization loop (up to 20 iterations each):

| Mode | Flag | Behavior | Use Case |
|------|------|----------|----------|
| **Simple (default)** | `--strategies=1` | 1 worker → 1 optimizer per task (runs full optimization loop) | Fast, lower cost |
| **Exploration** | `--strategies=3` | 1 worker → 3 independent optimizers per task (each runs full loop) | Better coverage, higher cost |

### Simple Mode (1:1 ratio) - DEFAULT
```
/kernel-bench level1 --session=my_run
/kernel-bench level1 --session=my_run --strategies=1
```
Each worker spawns ONE optimizer per task. The optimizer runs the full write → eval → fix loop for up to 20 iterations.
- Faster per-task completion
- Lower API cost
- Good for simple operations (element-wise, basic reductions)

### Exploration Mode (1:3 ratio)
```
/kernel-bench level1 --session=my_run --strategies=3
```
Each worker spawns 3 independent optimizer agents per task IN PARALLEL. Each optimizer runs its own full optimization loop with a different initial strategy. The worker takes the best result.
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
- `--iterations=20` (if not specified, max iterations per task)
- `--session={level}_{timestamp}` (if not specified)

## Execution Modes

### SINGLE TASK MODE (Interactive)

When user provides a single .py file path:

1. **Read task** via `get_task_details(task_path)`
2. **Run deep optimization loop** directly (no coordinator needed):
   - UNDERSTAND: Classify operation, retrieve similar kernels
   - STRATEGIZE: Generate 3 candidate strategies
   - PARALLEL GENERATION: Spawn 3 optimizer agents via Task tool
   - AGGREGATE: Pick best result
   - ITERATE: Up to 3 iterations
3. **Show progress** interactively to user after each iteration
4. **Save result** via `save_benchmark_result()`
5. **Aggregate reflections**: Run `python3 kb_reflect.py {session_id}` to collect reflections, then spawn the learning agent (see supervisor protocol) to update `.claude/agents/learned/`

### BATCH MODE (Supervisor-Managed)

When user provides directory, level name, or session parameters:

1. **Parse parameters** (with defaults):
   - `--workers=N` → N concurrent workers (default: 4)
   - `--strategies=N` → N strategies per worker per task (default: 1)
   - `--iterations=N` → max iterations per task (default: 20)
   - `--session=ID` → session identifier (default: auto-generated)
   - `--provider=X` → kbEval provider (default: "local")

2. **Spawn supervisor** (handles the entire session lifecycle):

   ```
   Task(
     subagent_type="general-purpose",
     description="kernel-bench supervisor",
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
   ```

3. **Control loop** (monitor → verify → correct):

   This is a unified loop that handles monitoring, stall detection, supervisor restart,
   and post-completion verification. It only exits when the session is fully verified.

   ```
   max_supervisor_attempts = 2   # original + 1 restart
   supervisor_attempts = 1

   while true:
       # ═══ MONITOR PHASE (while supervisor is running) ═══
       last_output_length = 0
       stale_reads = 0
       last_completed_count = 0
       stall_checks_without_progress = 0

       while supervisor is running:
           content = Read(output_file)
           if len(content) > last_output_length:
               Display new lines to user
               last_output_length = len(content)
               stale_reads = 0
           else:
               stale_reads += 1

           if stale_reads >= 10:
               state = get_session_state(session_id)

               if state.pending == 0 and len(state.in_progress) == 0:
                   break  # All tasks done, fall through to verify

               if state.completed == last_completed_count:
                   stall_checks_without_progress += 1
               else:
                   stall_checks_without_progress = 0
                   last_completed_count = state.completed

               if stall_checks_without_progress >= 2:
                   print("Supervisor stalled (0 progress across 2 checks).")
                   TaskStop(supervisor_task_id)  # best-effort
                   break  # Fall through to verify → restart
               else:
                   stale_reads = 0  # give supervisor time

       # ═══ VERIFY PHASE ═══
       state = get_session_state(session_id)
       session_dir = ~/.inference/claude_code_output/{session_id}/

       # Check 1: Tasks complete?
       remaining = state.pending + len(state.in_progress) + len(state.incomplete)
       if remaining > 0 and supervisor_attempts < max_supervisor_attempts:
           restart_workers = min(remaining, num_workers)
           print(f"Tasks incomplete ({remaining} remaining). Restarting supervisor with {restart_workers} workers...")

           supervisor_task = Task(
             subagent_type="general-purpose",
             description="kernel-bench supervisor (recovery)",
             prompt="Read .claude/agents/kernel-bench-supervisor.md — it contains your full protocol.
                     Mode: resume
                     Session: {session_id}
                     Workers: {restart_workers}
                     Strategies: {num_strategies}
                     Max iterations: {max_iterations}
                     Provider: {provider}",
             run_in_background=true
           )
           supervisor_attempts += 1
           continue  # Back to monitor phase for the new supervisor

       # If we get here, either all tasks are done OR we've exhausted restart attempts.
       # Report any remaining failures to user.
       if remaining > 0:
           print(f"Warning: {remaining} tasks still incomplete after {supervisor_attempts} supervisor attempts.")

       # Check 2: Reflections collected?
       # Use Bash: test -s {session_dir}/all_reflections.md
       if all_reflections.md does NOT exist or is empty:
           print("Reflections missing. Collecting...")
           Run: python3 kb_reflect.py {session_id}

       # Check 3: Learning ran?
       # Use Bash: stat -f %m .claude/agents/learned/common.md (macOS) to get mtime
       # Compare with session_manifest.json created_at timestamp
       if learned/common.md was NOT modified after session start:
           print("Learning not run. Spawning learner...")
           Task(
             subagent_type="general-purpose",
             description="kernel-bench learner (corrective)",
             prompt="Read .claude/agents/kernel-bench-learner.md — it contains your full instructions.
                     You are the learning agent for session '{session_id}'.
                     Reflections file: {session_dir}/all_reflections.md
                     Output directory: .claude/agents/learned/
                     Read ALL reflections, identify cross-cutting patterns, and produce:
                     - learned/common.md (environment constraints, anti-patterns, universal techniques)
                     - learned/{op_type}.md for each op type (top 3 successes + top 2 failures)
                     If existing learned files exist, merge with them (keep best from both).
                     Return a summary of what you wrote.",
             run_in_background=false  # BLOCKING
           )

       # Check 4: Score report generated?
       # Use Bash: ls {session_dir}/progress_*.md 2>/dev/null
       if no progress_*.md report exists in session_dir:
           print("Score report missing. Generating...")
           Run: python3 kb_score.py {session_id}

       print("All checks passed.")
       break  # Exit control loop
   ```

4. **Display final summary** from session state.

### RESUME MODE

When user provides `--resume my_session`:

1. **Get session state** to show user what's resuming:

   ```
   state = get_session_state(my_session)

   print(f"Resuming session '{my_session}'...")
   print(f"Original command: {state.config.original_command}")
   print(f"Config: workers={state.config.num_workers}, strategies={state.config.num_strategies}")
   print(f"Progress: {state.completed}/{state.total} completed, {state.pending} remaining")
   ```

2. **Extract config** from session state:
   ```
   num_workers = state.config.get("num_workers", 4)
   num_strategies = state.config.get("num_strategies", 1)
   provider = state.config.get("provider", "local")
   max_iterations = state.config.get("max_iterations", 20)  # fallback matches --iterations default above
   ```

3. **Spawn supervisor** in resume mode with scaled workers:

   ```
   remaining = state.pending + len(state.incomplete)
   resume_workers = min(remaining, num_workers)

   Task(
     subagent_type="general-purpose",
     description="kernel-bench supervisor (resume)",
     prompt="Read .claude/agents/kernel-bench-supervisor.md — it contains your full protocol.

             Mode: resume
             Session: {session_id}
             Workers: {resume_workers}
             Strategies: {num_strategies}
             Max iterations: {max_iterations}
             Provider: {provider}",
     run_in_background=true
   )
   ```

4. **Control loop**: Same as BATCH MODE step 3 — monitor, verify, correct.

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
Spawning supervisor...
[init] Session prod_run: 100 tasks, 4 workers, strategies=1
[progress] 25/100 completed, 4 active, avg 1.28x
[progress] 50/100 completed, 4 active, avg 1.31x
[progress] 75/100 completed, 4 active, avg 1.33x
[workers] All workers finished (round 1)
[evaluate] No retryable tasks
[finalize] Collecting reflections...
[finalize] Learning agent completed
[finalize] Score report generated
[complete] Session prod_run done: 97/100, avg 1.34x

Verifying post-batch outcomes...
  Session complete: 97/100 done, 0 pending, 0 in progress
  Reflections: all_reflections.md exists
  Learning: learned/common.md updated
  Score report: progress_20260214_153022.md generated
All checks passed.
```

**Example 3: Batch run with exploration mode (--strategies=3)**
```
User: /kernel-bench level1 --session=explore_run --strategies=3

Response:
Initializing session "explore_run"...
Config: workers=4, strategies=3 (exploration mode)
Found 100 tasks in level1.
Spawning supervisor...
  Each worker spawns 3 independent optimizers per task (each runs full loop)

[progress] 15/100 completed, 4 active, avg 1.41x  [higher speedups due to parallel exploration]
[progress] 30/100 completed, 4 active, avg 1.45x
...
[complete] Session explore_run done: 98/100, avg 1.47x
```

**Example 4: Custom worker count with exploration**
```
User: /kernel-bench level1 --session=prod_run --workers=8 --strategies=3

Response:
Initializing session "prod_run"...
Config: workers=8, strategies=3 (exploration mode)
Spawning supervisor with 8 workers...
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
Spawning supervisor in resume mode...
[resume] Session prod_run: 50/100 done, 48 remaining
[progress] 55/100 completed, 4 active, avg 1.38x
...
[complete] Session prod_run done: 98/100, avg 1.41x
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
| 19_ReLU             | in_progress | worker-3    | 2/3  | 1.12x  | ✓ compiled, ✓ correct |
| 23_Softmax          | in_progress | worker-7    | 1/3  | -      | ✗ compile error       |
| 88_MinGPTNewGelu    | completed   | worker-2    | 2/3  | 2.1x   | ✓ done                |
```

**Example 7: Query single task detail**
```
User: /kernel-bench progress test1 19_ReLU

Response:
=== Task: 19_ReLU ===
Status: in_progress
Worker: worker-3
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
