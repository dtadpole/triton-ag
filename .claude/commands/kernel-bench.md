---
name: kernel-bench
description: Run kernel benchmark optimization with multiple input styles
---

# Kernel Bench

Optimize CUDA/Triton kernels for kernel_bench tasks with crash recovery and **parallel execution**.

## Default Behavior: Parallel Execution

**CRITICAL: Parallel execution is the default and expected behavior.**

- **Default workers**: 15 concurrent optimizers (configurable via `--workers=N`)
- **Parallel spawning**: ALL optimizers + monitor MUST be spawned in a SINGLE message with multiple Task tool calls
- **Concurrent tasks**: Each optimizer claims and processes tasks independently via claim loop

Each optimizer runs the full optimization loop (up to 20 iterations) per task. In batch mode, optimizers run a claim loop processing multiple tasks sequentially.

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
→ Spawns **4 parallel optimizers** + monitor directly

### Style 3: Full Parameters
```
/kernel-bench level1 --session=te --workers=4 --resume
/kernel-bench level1 my_run --workers=8
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

### Style 8: Verify Session
```
/kernel-bench verify test1
/kernel-bench verify test1 --semantic
/kernel-bench verify test1 19_ReLU
```
→ Run protocol compliance verification on completed session

### Style 9: Learn from Session
```
/kernel-bench learn test1
/kernel-bench learn test1 --kernel-only
/kernel-bench learn test1 --algo-only
```
→ Run learning on completed session (kernel knowledge + algorithm improvement)

### Style 10: Chain (multi-level sequence)
```
/kernel-bench chain level1,level2,level3
/kernel-bench chain level1,level1,level2 --workers=8
```
→ Sequential multi-batch chain with learning between each batch

### Style 11: Multi-batch (same level repeated)
```
/kernel-bench level1 --batches=3
/kernel-bench level2 --batches=5 --retry=all --workers=8
```
→ Equivalent to `chain level1,level1,level1` — same level repeated N times

**Chain options:**
- `--batches=N` — number of batches (default for chain: length of level list)
- `--retry=below_target|failed|all` — which tasks to retry (default: `below_target`)
  - `below_target`: re-run tasks with speedup < 1.3x
  - `failed`: only tasks with 0x (compile/correctness/server error)
  - `all`: re-run every task
- `--workers=N` — max workers per batch (default: 15, scaled down for smaller retry sets)
- `--iterations=N` — max iterations per task (default: 20)

## Parsing Logic

When invoked with `/kernel-bench [args]`, parse the input:

**⚠️ CRITICAL: For "progress" queries, you MUST run `python3 kb_score.py` - see PROGRESS MODE section below!**

1. **Detect server config**: Starts with `server` → **SERVER MODE**
2. **Detect progress query**: Starts with `progress` OR user just says "progress" → **PROGRESS MODE** (run the script!)
3. **Detect verify**: Starts with `verify` → **VERIFY MODE** (run kb_verify.py!)
3b. **Detect learn**: Starts with `learn` → **LEARN MODE**
3c. **Detect chain**: Starts with `chain ` → **CHAIN MODE**
3d. **Detect --batches**: Contains `--batches=N` → **CHAIN MODE** (same level repeated N times)
4. **Detect single task**: Path ends with `.py` → SINGLE TASK MODE
5. **Detect directory**: Path ends with `/` or is a level name (level1, level2, level3) → BATCH MODE
6. **Detect resume**: Contains `--resume` or starts with `resume` → RESUME MODE (check for chain resume — see below)
7. **Detect parameters**: Contains `--session`, `--workers`, or `--iterations` → BATCH MODE with config
8. **Detect natural language**: Contains numbers + keywords ("tasks", "agents", "random") → interpret and route

**Chain parsing details:**
- `chain level1,level2,level3` → `levels = ["level1","level2","level3"]`, `max_batches = 3`
- `level1 --batches=3` → `levels = ["level1","level1","level1"]`, `max_batches = 3`
- Extract `--retry=below_target|failed|all` (default: `below_target`)
- Extract `--workers=N`, `--iterations=N` (same defaults as batch mode)

**Parameter defaults:**
- `--workers=15` (if not specified)
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
5. **Aggregate reflections**: Run `python3 kb_reflect.py {session_id}` to collect reflections, then spawn the learning agent to update `.claude/agents/reference/`

### BATCH MODE (Flat Spawning: Optimizers + Monitor)

When user provides directory, level name, or session parameters:

1. **Parse parameters** (with defaults):
   - `--workers=N` → N concurrent optimizers (default: 15)
   - `--iterations=N` → max iterations per task (default: 20)
   - `--session=ID` → session identifier (default: auto-generated)
   - `--provider=X` → kbEval provider (default: "local")

2. **Initialize session:**

   ```
   init_session(
     session_id=session_id,
     level=level,
     num_workers=num_workers,
     max_iterations=max_iterations,
     provider=provider,
     code_type="triton",
     original_command="/kernel-bench {original args}"
   )
   ```

3. **Spawn optimizers + monitor** (ALL in ONE message — critical for parallelism):

   ```
   # Spawn N optimizer agents — each runs a claim loop
   For i in 1..num_workers:
       Task(
         subagent_type="general-purpose",
         description="kernel-bench optimizer {i}",
         prompt="Read .claude/agents/kernel-bench-optimizer.md — it contains your full protocol,
                 rules, templates, and iteration loop instructions.

                 Also read these reference files at startup (if they exist):
                 - .claude/agents/reference/common.md
                 - .claude/agents/reference/optimizer_algorithm.md

                 You are optimizer-{i} operating in BATCH MODE.
                 Run the claim loop described in optimizer.md:
                 get_pending_tasks → claim_task → optimize → complete → repeat

                 Session: {session_id}
                 Provider: {provider}
                 Max iterations: {max_iterations}
                 Worker ID: optimizer-{i}

                 When no tasks remain, print '[optimizer-{i}] No more tasks. Exiting.' and stop.",
         run_in_background=true
       )

   # Spawn 1 monitor agent
   Task(
     subagent_type="general-purpose",
     description="kernel-bench monitor",
     prompt="Read .claude/agents/kernel-bench-monitor.md — it contains your full protocol.

             Session: {session_id}
             Total tasks: {total_tasks}
             Expected optimizers: {num_workers}

             Poll session state every ~30s and print [progress] lines.
             Exit with [monitor] ALL_DONE, LOW_ACTIVE, or STALL.",
     run_in_background=true
   )
   ```

   Save the monitor's `output_file` path for the wait loop.

4. **Wait loop** (read monitor output, recover if needed):

   Optimizers running claim loops will commonly hit context limits and exit after
   processing some tasks. The monitor detects this (LOW_ACTIVE) or detects zero
   progress (STALL). The skill spawns replacement optimizers for each recovery round.

   Recovery continues as long as each round makes forward progress. It stops when:
   - All tasks are done (ALL_DONE)
   - A round made zero progress (true stall — the problem isn't transient)
   - Max 5 recovery rounds (safety cap)

   ```
   max_rounds = 5
   round = 1
   completed_before_round = 0  # track per-round progress

   while true:
       # ─── POLL MONITOR OUTPUT ───
       while true:
           content = Read(monitor_output_file)

           # Exit signal: all tasks done
           if content contains "[monitor] ALL_DONE":
               break  # proceed to finalize

           # Exit signal: monitor detected a problem
           if content contains "[monitor] STALL" or "[monitor] LOW_ACTIVE":
               state = get_session_state(session_id)
               remaining = state.pending + len(state.in_progress) + len(state.incomplete)

               # Actually done? (race between monitor exit and last task completing)
               if remaining == 0:
                   break  # proceed to finalize

               # Did this round make progress?
               round_progress = state.completed - completed_before_round
               if round_progress == 0:
                   print(f"Round {round} made no progress. {remaining} tasks remain. Stopping.")
                   break  # true stall — don't retry

               # Hit safety cap?
               if round >= max_rounds:
                   print(f"Max recovery rounds ({max_rounds}) reached. {remaining} tasks remain.")
                   break

               # ─── SPAWN RECOVERY ───
               print(f"Round {round}: completed {round_progress} tasks, {remaining} remaining. Spawning recovery...")
               completed_before_round = state.completed
               recovery_workers = min(remaining, num_workers)
               round += 1

               # Spawn recovery_workers optimizers + new monitor (ONE message)
               # Same prompt template as step 3, with:
               #   - recovery_workers optimizer agents
               #   - monitor with expected_optimizers = recovery_workers
               # Save new monitor's output_file for next iteration of outer loop
               continue outer loop

           # Not done yet — wait before reading again
           Bash: sleep 30

       break  # Exit wait loop
   ```

   **End-to-end example (100 tasks, 4 optimizers, ~15 tasks/optimizer before context limit):**

   ```
   Round 1: Spawn 4 optimizers + monitor
     optimizer-1: 14 tasks → context full → exits
     optimizer-2: 12 tasks → exits
     optimizer-3: 15 tasks → exits
     optimizer-4: 13 tasks → exits
     Monitor: 54/100 done, 0 active, 46 pending → [monitor] LOW_ACTIVE
     Skill: round_progress=54, remaining=46 → spawn recovery

   Round 2: Spawn 4 recovery optimizers + monitor
     ~48 more tasks processed, covers remaining 46
     Monitor: 100/100 done → [monitor] ALL_DONE
     Skill: → finalize ✓
   ```

   **Worse case (flaky, ~8 tasks/optimizer):**

   ```
   Round 1: 32/100 done → LOW_ACTIVE → recovery (progress=32)
   Round 2: 64/100 done → LOW_ACTIVE → recovery (progress=32)
   Round 3: 96/100 done → LOW_ACTIVE → recovery (progress=32)
   Round 4: 100/100 done → ALL_DONE ✓
   ```

   **True stall (eval server down):**

   ```
   Round 1: 0/100 done → STALL → recovery (progress=0) → STOP
   Skill: "Round 1 made no progress. 100 tasks remain. Stopping."
   → finalize with 0 completed
   ```

5. **Finalize** (ALWAYS runs):

   ```
   session_dir = ~/.inference/claude_code_output/{session_id}/

   # 5a. Collect reflections, traces, and classify
   if all_reflections.md does NOT exist or is empty:
       Run: python3 kb_reflect.py {session_id}

   if all_algo_traces.md does NOT exist or is empty:
       Run: python3 kb_reflect.py collect_traces {session_id}

   # Classify reflections by op type for parallel learning
   Run: python3 kb_reflect.py classify {session_id}
   # Parse output to get POPULATED_OPS line (e.g., "POPULATED_OPS: matmul,conv,reduction")
   # Extract populated_op_types list from this line

   # 5b. Snapshot optimizer.md (for algorithm learning rollback)
   Bash: cp .claude/agents/kernel-bench-optimizer.md \
         {session_dir}/optimizer_snapshot.md

   # 5c. Gain write access BEFORE spawning learner agents
   # Background agents cannot prompt the user for file write permissions.
   # The skill controller (foreground) MUST gain write access upfront by
   # writing to each target directory. Do this by touching/writing a small
   # marker file to establish permission:
   #   - Write to .claude/agents/reference/ (for kernel learners)
   #   - Write to .claude/agents/kernel-bench-optimizer.md (for algorithm learner)
   # The optimizer snapshot copy above already writes to {session_dir}/.
   # For the reference directory, write a placeholder:
   Bash: echo "# write access marker" >> .claude/agents/reference/.learn_marker && \
         rm -f .claude/agents/reference/.learn_marker

   # 5d. Spawn ALL learner agents in ONE message (background)
   # Check if reference files were modified after session start
   if reference/common.md was NOT modified after session start:

       # All agents spawned in a SINGLE message for maximum parallelism:

       # Kernel learner: common patterns
       Task(
         subagent_type="general-purpose",
         description="kernel learner common",
         prompt="Read .claude/agents/kernel-bench-learner-common.md — your full instructions.

                 Session: {session_id}
                 Reflections file: {session_dir}/all_reflections.md
                 Output directory: .claude/agents/reference/
                 Read ALL reflections, identify cross-cutting patterns, and write
                 reference/common.md. Merge with existing if present.
                 IMPORTANT: Preserve the Code Templates section.
                 Return a summary of what you wrote.",
         run_in_background=true
       )

       # Kernel learner: per-op-type (one per populated op type)
       for each op_type in populated_op_types:
           Task(
             subagent_type="general-purpose",
             description="kernel learner {op_type}",
             prompt="Read .claude/agents/kernel-bench-learner-op.md — your full instructions.

                     Op type: {op_type}
                     Reflections: {session_dir}/reflections_by_op/{op_type}.md
                     Existing reference: .claude/agents/reference/{op_type}.md
                     Output: .claude/agents/reference/{op_type}.md

                     Classify entries by tier, merge with existing, write updated file.
                     IMPORTANT: Preserve the Code Templates section.
                     Return a summary of what you wrote.",
             run_in_background=true
           )

       # Algorithm learner
       Task(
         subagent_type="general-purpose",
         description="algorithm learner",
         prompt="Read .claude/agents/kernel-bench-learner-algo.md — your full instructions.

                 Session: {session_id}
                 Session directory: {session_dir}
                 Algo traces file: {session_dir}/all_algo_traces.md
                 Optimizer file: .claude/agents/kernel-bench-optimizer.md
                 Learned directory: .claude/agents/reference/

                 Analyze algorithm traces, update mutable sections of optimizer.md,
                 write algorithm_changelog.md to session directory,
                 and update reference/optimizer_algorithm.md.
                 Return a summary of what was changed.",
         run_in_background=true
       )

   # 5e. Wait for ALL learner agents to complete
   # Read each agent's output_file until complete

   # 5f. Generate score report
   if no progress_*.md report exists in session_dir:
       Run: python3 kb_score.py {session_id}
   ```

6. **Display final summary** from session state.

### RESUME MODE

When user provides `--resume my_session`:

**First, detect if this is a chain resume:**
```
chain_dir = ~/.inference/claude_code_output/{my_session}
if chain_manifest.json exists in chain_dir → CHAIN RESUME (see below)
else → SINGLE SESSION RESUME (existing logic below)
```

Also support explicit: `/kernel-bench --resume chain_XXXXXX`

#### Single Session Resume

1. **Get session state** to show user what's resuming:

   ```
   state = get_session_state(my_session)

   print(f"Resuming session '{my_session}'...")
   print(f"Original command: {state.config.original_command}")
   print(f"Config: workers={state.config.num_workers}")
   print(f"Progress: {state.completed}/{state.total} completed, {state.pending} remaining")
   ```

2. **Extract config** from session state:
   ```
   num_workers = state.config.get("num_workers", 15)
   provider = state.config.get("provider", "local")
   max_iterations = state.config.get("max_iterations", 20)
   ```

3. **Spawn optimizers + monitor** (same as BATCH MODE step 3):

   ```
   remaining = state.pending + len(state.incomplete)
   resume_workers = min(remaining, num_workers)

   # Spawn resume_workers optimizers + 1 monitor in ONE message
   # Same prompt template as BATCH MODE step 3
   ```

4. **Wait loop + Finalize**: Same as BATCH MODE steps 4-6.

#### Chain Resume

When `chain_manifest.json` exists in the session directory:

1. **Read chain manifest:**
   ```
   chain_manifest = Read chain_dir/chain_manifest.json
   config = chain_manifest["config"]
   levels = [b["level"] for b in chain_manifest["batches"]]
   max_batches = config["max_batches"]
   workers = config["workers"]
   max_iterations = config.get("iterations", 20)
   retry_mode = config["retry_mode"]
   ```

2. **Check for chain extension:**
   ```
   # If user specified --batches=N on resume, extend the chain
   if --batches=N provided:
       max_batches = len(chain_manifest["batches"]) + N
       config["max_batches"] = max_batches
       Write chain_manifest.json
   ```

3. **Find last batch and determine resume action:**
   ```
   last_batch = chain_manifest["batches"][-1]

   if last_batch["status"] == "running":
       # Mid-batch crash. Delegate to single session resume for this batch.
       Resume session last_batch["session_id"] using Single Session Resume above.
       # After session completes, update status
       Update last_batch["status"] to "tasks_done" in chain_manifest.json

   if last_batch["status"] == "tasks_done":
       # Tasks done but learning never started. Run full learning pipeline.
       Update last_batch["status"] to "learning" in chain_manifest.json
       Run finalize steps 5a-5f from BATCH MODE.
       Update last_batch["status"] to "completed" in chain_manifest.json

   elif last_batch["status"] == "learning":
       # Partial learning. Detect what finished, re-run missing.
       session_dir = ~/.inference/claude_code_output/{last_batch["session_id"]}

       # Check which artifacts exist:
       missing_learners = []
       if all_reflections.md NOT in session_dir → re-run kb_reflect.py
       if all_algo_traces.md NOT in session_dir → re-run kb_reflect.py collect_traces
       if algorithm_changelog.md NOT in session_dir → re-spawn algorithm learner
       if reference/common.md mtime < session start → re-spawn common learner
       # For each op type with reflections, check reference/{op}.md mtime

       # Re-run only missing learners
       if missing_learners:
           Spawn missing learner agents (same prompts as BATCH MODE finalize)
           Wait for completion
       Update last_batch["status"] to "completed" in chain_manifest.json

   # last_batch["status"] is now "completed"
   ```

4. **Continue chain from next batch:**
   ```
   next_batch_index = last_batch["batch_index"] + 1

   # Rebuild all_best_results from all completed batches
   for batch in chain_manifest["batches"]:
       if batch["status"] == "completed":
           batch_progress = get_batch_progress(batch["session_id"])
           for task in batch_progress["tasks"]:
               name = task["task_name"]
               speedup = task.get("best", {}).get("speedup", 0) if task.get("best") else 0
               if name not in all_best_results or speedup > all_best_results[name]:
                   all_best_results[name] = speedup

   # Re-derive levels list for remaining batches
   # If --batches was used, extend levels list to match max_batches
   # Then continue the CHAIN MODE batch loop from next_batch_index

   if next_batch_index < max_batches:
       Continue CHAIN MODE batch loop from batch_index = next_batch_index
   else:
       Print chain summary table
   ```

### PROGRESS MODE

**⚠️ MANDATORY: You MUST run the progress script - do NOT just call MCP tools directly!**

When user says "progress", "progress {session_id}", or just asks about progress:

**First, detect if this is a chain session:**
```
If session_id starts with "chain_":
    chain_dir = ~/.inference/claude_code_output/{session_id}
    if chain_manifest.json exists in chain_dir → run chain progress (see below)
```

#### Single Session Progress

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

#### Chain Progress

When session_id is a chain (starts with `chain_` and has `chain_manifest.json`):

```bash
python3 kb_score.py {chain_id}
```

The script auto-detects chain mode and produces:
- Cross-batch progression table (batch rate, cumulative rate, delta)
- Per-task cross-batch progression (speedup at each batch for every retried task)
- Tasks that crossed the 1.3x threshold across batches
- Remaining failures with structural ceiling reasons
- Final results summary

A detailed markdown report is saved to `{chain_dir}/chain_progress_{timestamp}.md`.

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

### VERIFY MODE

**Run protocol compliance checks on a completed (or in-progress) session.**

When user says "verify", "verify {session_id}", or "verify {session_id} --semantic":

**Chain detection:** If session_id starts with `chain_` or `chain_manifest.json` exists
in the session directory, run `python3 kb_verify.py --chain {session_id}` instead.
This runs 12 chain-level checks (C1-C12) in addition to per-batch task verification.

**STEP 1 (REQUIRED): Run the verification script via Bash:**
```bash
python3 kb_verify.py {session_id}
# OR for chains:
python3 kb_verify.py --chain {chain_id}
```

This script:
- Checks 17 structural compliance items per task (file existence, strategy naming convention,
  phase ordering, reflection/algo_trace content sections, speedup consistency)
- Outputs a quick summary to stdout (display to the user)
- Generates a detailed `verification_YYYYMMDD_HHMMSS.md` report in the session directory

**STEP 2: Display the script output** directly to the user.

**STEP 3 (if `--semantic` specified): Spawn semantic verifier agent:**
```
Task(
  subagent_type="general-purpose",
  description="kernel-bench verifier",
  prompt="Read .claude/agents/kernel-bench-verifier.md — it contains your full protocol.

          Session: {session_id}
          Session directory: ~/.inference/claude_code_output/{session_id}
          Verification report: (path from kb_verify.py output)

          Run semantic quality checks on all tasks marked PASS or WARN.
          Focus on: strategy diversity, diagnosis coherence, reflection quality.
          Write semantic_verification.md to the session directory.
          Return summary of findings.",
  run_in_background=false
)
```

**Single task detail:** `verify {session_id} {task_name}`
```bash
python3 kb_verify.py {session_id} {task_name}
```

**If no session_id is provided**, use the most recent session:
```bash
python3 kb_verify.py
```

### LEARN MODE

**Run learning on a completed session to update reference files and algorithm.**

When user says "learn {session_id}" or provides learning flags:

1. **Parse session_id and flags:**
   - `--kernel-only`: only run kernel knowledge learning (skip algorithm learning)
   - `--algo-only`: only run algorithm learning (skip kernel knowledge learning)

2. **Validate session:**
   ```
   state = get_session_state(session_id)
   if state.completed == 0:
       "No completed tasks in session '{session_id}'. Nothing to learn from."
       exit
   ```

3. **Collect artifacts** (if not already present):
   ```
   session_dir = ~/.inference/claude_code_output/{session_id}

   if --algo-only is NOT set:
       if all_reflections.md does NOT exist:
           Run: python3 kb_reflect.py {session_id}
       Run: python3 kb_reflect.py classify {session_id}
       # Parse output to get POPULATED_OPS line
       # Extract populated_op_types list

   if --kernel-only is NOT set:
       if all_algo_traces.md does NOT exist:
           Run: python3 kb_reflect.py collect_traces {session_id}
   ```

4. **Snapshot optimizer.md** (if algorithm learning will run):
   ```
   if --kernel-only is NOT set:
       Bash: cp .claude/agents/kernel-bench-optimizer.md \
             {session_dir}/optimizer_snapshot.md
   ```

5. **Gain write access BEFORE spawning learner agents:**

   Background agents cannot prompt the user for file write permissions.
   The skill controller (foreground) MUST gain write access upfront by
   writing to each target directory that learner agents will modify.
   Do this now — before spawning any agents:

   ```
   # Write to .claude/agents/reference/ (for kernel + algorithm learners)
   Bash: echo "# write access marker" >> .claude/agents/reference/.learn_marker && \
         rm -f .claude/agents/reference/.learn_marker

   # Write to .claude/agents/kernel-bench-optimizer.md (for algorithm learner)
   # The snapshot copy in step 4 already establishes access to {session_dir}/
   # but we also need access to the optimizer file itself:
   if --kernel-only is NOT set:
       Read then touch .claude/agents/kernel-bench-optimizer.md
       # (the algorithm learner will Edit this file)
   ```

6. **Spawn learner agents** (ALL in ONE message, background):
   ```
   agents = []

   if --algo-only is NOT set:
       # Kernel learner: common patterns
       agents.append(Task(
         subagent_type="general-purpose",
         description="kernel learner common",
         prompt="Read .claude/agents/kernel-bench-learner-common.md — your full instructions.

                 Session: {session_id}
                 Reflections file: {session_dir}/all_reflections.md
                 Output directory: .claude/agents/reference/
                 Read ALL reflections, identify cross-cutting patterns, and write
                 reference/common.md. Merge with existing if present.
                 IMPORTANT: Preserve the Code Templates section.
                 Return a summary of what you wrote.",
         run_in_background=true
       ))

       # Kernel learner: per-op-type (only for op-types with reflections)
       for each op_type in populated_op_types:
           agents.append(Task(
             subagent_type="general-purpose",
             description="kernel learner {op_type}",
             prompt="Read .claude/agents/kernel-bench-learner-op.md — your full instructions.

                     Op type: {op_type}
                     Reflections: {session_dir}/reflections_by_op/{op_type}.md
                     Existing reference: .claude/agents/reference/{op_type}.md
                     Output: .claude/agents/reference/{op_type}.md

                     Classify entries by tier, merge with existing, write updated file.
                     IMPORTANT: Preserve the Code Templates section.
                     Return a summary of what you wrote.",
             run_in_background=true
           ))

   if --kernel-only is NOT set:
       # Algorithm learner
       agents.append(Task(
         subagent_type="general-purpose",
         description="algorithm learner",
         prompt="Read .claude/agents/kernel-bench-learner-algo.md — your full instructions.

                 Session: {session_id}
                 Session directory: {session_dir}
                 Algo traces file: {session_dir}/all_algo_traces.md
                 Optimizer file: .claude/agents/kernel-bench-optimizer.md
                 Learned directory: .claude/agents/reference/

                 Analyze algorithm traces, update mutable sections of optimizer.md,
                 write algorithm_changelog.md to session directory,
                 and update reference/optimizer_algorithm.md.
                 Return a summary of what was changed.",
         run_in_background=true
       ))

   Spawn all agents in ONE message (background)
   ```

7. **Wait for all agents to complete:**
   Read each agent's output_file until all are done.

8. **Display summary:**
   ```
   Print: "Learning complete for session '{session_id}'."
   Print: "Kernel learners: {N} (common + {M} op-types)"  # if not --algo-only
   Print: "Algorithm learner: done"                         # if not --kernel-only
   Print: "Changelog: {session_dir}/algorithm_changelog.md" # if not --kernel-only
   ```

### CHAIN MODE (Multi-Batch Orchestration)

When user provides `chain level1,level2,...` or `level1 --batches=N`:

**CHAIN MODE automates the batch → learn → retry loop. Each batch spawns fresh agents
that read the latest reference files from disk. No restart needed between batches.**

#### Chain Initialization

1. **Generate chain ID and directory:**
   ```
   chain_id = "chain_{YYYYMMDD_HHMMSS}"
   chain_dir = ~/.inference/claude_code_output/{chain_id}
   Bash: mkdir -p {chain_dir}
   ```

2. **Write initial chain manifest:**
   ```json
   {
     "chain_id": "{chain_id}",
     "config": {
       "retry_mode": "{retry_mode}",
       "target_speedup": 1.3,
       "max_batches": {max_batches},
       "workers": {workers},
       "iterations": {max_iterations},
       "original_command": "/kernel-bench {original args}"
     },
     "batches": [],
     "cumulative": {
       "tasks_total": 0,
       "tasks_passing": 0,
       "cumulative_success_rate": 0,
       "avg_speedup": 0
     }
   }
   ```
   Write to `{chain_dir}/chain_manifest.json`.

3. **Initialize tracking:**
   ```
   all_best_results = {}  # task_name → best speedup across all batches
   ```

#### Batch Loop

For each `batch_index` in `range(max_batches)`:

**IMPORTANT: Be deliberately terse between batches. No verbose summaries. Re-read
chain_manifest.json from disk at each batch start — it is the ground truth, not
conversation memory. All state is computed from files.**

1. **Determine level and task set:**
   ```
   level = levels[batch_index]   # from parsed level list

   # Get all tasks for this level
   level_tasks = list_kernel_bench_tasks(level)

   # Check if this level appeared in a prior batch
   prior_sessions_for_level = [b["session_id"] for b in chain_manifest["batches"]
                                if b["level"] == level and b["status"] == "completed"]

   if not prior_sessions_for_level:
       # First appearance — run all tasks
       task_names = None
       task_count = len(level_tasks)
   else:
       # Level seen before — compute retry set
       # Union best speedups per task across all prior batches of this level
       for prior_sid in prior_sessions_for_level:
           batch_progress = get_batch_progress(prior_sid)
           for task in batch_progress["tasks"]:
               name = task["task_name"]
               speedup = task.get("best", {}).get("speedup", 0) if task.get("best") else 0
               if name not in all_best_results or speedup > all_best_results[name]:
                   all_best_results[name] = speedup

       # Filter by retry mode
       if retry_mode == "below_target":
           retry_set = [t["name"] for t in level_tasks
                       if all_best_results.get(t["name"], 0) < 1.3]
       elif retry_mode == "failed":
           retry_set = [t["name"] for t in level_tasks
                       if all_best_results.get(t["name"], 0) == 0]
       elif retry_mode == "all":
           retry_set = [t["name"] for t in level_tasks]

       if len(retry_set) == 0:
           Print: "All {level} tasks pass. Skipping batch {batch_index}."
           continue  # skip to next batch in chain

       task_names = retry_set
       task_count = len(retry_set)
   ```

2. **Convergence check** (only for same-level retries, batch_index >= 2):
   ```
   if prior_sessions_for_level and batch_index >= 2:
       prev_batch = chain_manifest["batches"][-1]
       prev_cumulative_rate = prev_batch.get("cumulative_success_rate", 0)
       current_cumulative_rate = chain_manifest["cumulative"]["cumulative_success_rate"]

       delta_pp = current_cumulative_rate - prev_cumulative_rate
       # Also check avg speedup delta (compute from all_best_results)

       if delta_pp < 0.03:
           Print: f"Converged after batch {batch_index-1} (Δ={delta_pp:.1%}pp). Stopping chain."
           break
   ```

3. **Scale workers:**
   ```
   effective_workers = min(workers, max(4, task_count // 3))
   ```

4. **Session setup:**
   ```
   session_id = "{chain_id}_b{batch_index}"
   session_dir = ~/.inference/claude_code_output/{session_id}

   init_session(
     session_id=session_id,
     level=level,
     task_names=task_names,    # null for first appearance = all tasks
     num_workers=effective_workers,
     max_iterations=max_iterations,
     provider=provider,
     code_type="triton",
     original_command=chain_manifest["config"]["original_command"]
   )
   ```

5. **Write task histories** (for batch_index >= 1, only if level was seen before):
   ```
   if prior_sessions_for_level and task_names:
       Bash: python3 kb_history.py {chain_dir} {batch_index} {session_id}
   ```

6. **Write-ahead: mark batch as running:**
   ```
   batch_entry = {
     "batch_index": batch_index,
     "session_id": session_id,
     "level": level,
     "task_count": task_count,
     "status": "running"
   }
   # Append to chain_manifest.batches, save to disk
   Read chain_manifest.json → append batch_entry → Write chain_manifest.json
   ```

7. **Run batch** (reuse BATCH MODE steps 3-4):
   ```
   Spawn effective_workers optimizers + 1 monitor in ONE message.
   Same prompt template as BATCH MODE step 3.
   Same wait loop with recovery as BATCH MODE step 4.
   ```

8. **Finalize batch** (reuse BATCH MODE step 5):
   ```
   # Write-ahead: tasks_done
   Update batch_entry status to "tasks_done" in chain_manifest.json

   # Write-ahead: learning
   Update batch_entry status to "learning" in chain_manifest.json

   # Run finalize: reflections → classify → learners (same as BATCH MODE step 5)
   # 5a-5f from BATCH MODE finalize

   # Write-ahead: completed
   Update batch_entry status to "completed" in chain_manifest.json
   ```

9. **Update cumulative stats:**
   ```
   # Get results from this batch
   batch_progress = get_batch_progress(session_id)

   # Update all_best_results with this batch's results
   for task in batch_progress["tasks"]:
       name = task["task_name"]
       speedup = task.get("best", {}).get("speedup", 0) if task.get("best") else 0
       if name not in all_best_results or speedup > all_best_results[name]:
           all_best_results[name] = speedup

   # Compute cumulative stats
   tasks_total = len(all_best_results)
   tasks_passing = sum(1 for s in all_best_results.values() if s >= 1.3)
   cumulative_success_rate = tasks_passing / tasks_total if tasks_total > 0 else 0
   all_speedups = [s for s in all_best_results.values() if s > 0]
   avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0

   # Compute this batch's stats
   batch_speedups = []
   batch_passing = 0
   for task in batch_progress["tasks"]:
       s = task.get("best", {}).get("speedup", 0) if task.get("best") else 0
       if s > 0:
           batch_speedups.append(s)
       if s >= 1.3:
           batch_passing += 1

   batch_success_rate = batch_passing / task_count if task_count > 0 else 0
   batch_avg_speedup = sum(batch_speedups) / len(batch_speedups) if batch_speedups else 0

   # Update manifest
   batch_entry["success_rate"] = round(batch_success_rate, 3)
   batch_entry["avg_speedup"] = round(batch_avg_speedup, 3)
   batch_entry["cumulative_success_rate"] = round(cumulative_success_rate, 3)
   batch_entry["completed_at"] = current_timestamp_iso

   chain_manifest["cumulative"] = {
     "tasks_total": tasks_total,
     "tasks_passing": tasks_passing,
     "cumulative_success_rate": round(cumulative_success_rate, 3),
     "avg_speedup": round(avg_speedup, 3)
   }

   # Save updated manifest
   Write chain_manifest.json
   ```

10. **Print terse batch summary:**
    ```
    Print: f"Batch {batch_index}: {batch_passing}/{task_count} passed ({batch_success_rate:.1%}), cumulative {tasks_passing}/{tasks_total} ({cumulative_success_rate:.1%})"
    ```

#### Chain Summary

After loop ends (max batches, convergence, or all pass), run the chain progress report:

```bash
python3 kb_score.py {chain_id}
```

This auto-detects chain mode and prints:
- Cross-batch progression table with cumulative rates and deltas
- Per-task cross-batch progression (speedup at each batch for retried tasks)
- Tasks that crossed the 1.3x threshold
- Remaining failures with ceiling reasons
- Final results summary

A detailed markdown report is saved to `{chain_dir}/chain_progress_{timestamp}.md`.

Display the script output directly to the user.

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
Generating kernel with vectorized loads strategy...
Evaluating... compiled ✓, correct ✓, speedup: 1.12x
Iteration 1: 1.12x - below target, trying block tuning...
Evaluating... compiled ✓, correct ✓, speedup: 1.31x
Iteration 2 complete. Best: 1.31x
Target reached (>= 1.3x). Saved.
```

**Example 2: Batch run**
```
User: /kernel-bench level1 --session=prod_run

Response:
Initializing session "prod_run"...
Config: workers=4
Found 100 tasks in level1.
Spawning 4 optimizers + monitor...
[progress] 25/100 completed, 4 active, avg 1.28x
[progress] 50/100 completed, 4 active, avg 1.31x
[progress] 75/100 completed, 3 active, avg 1.33x
[monitor] ALL_DONE 100/100 completed
[finalize] Collecting reflections...
[finalize] Learning agent completed
[finalize] Score report generated

Session prod_run complete: 97/100, avg 1.34x
All checks passed.
```

**Example 3: Custom worker count**
```
User: /kernel-bench level1 --session=prod_run --workers=8

Response:
Initializing session "prod_run"...
Config: workers=8
Spawning 8 optimizers + monitor...
...
```

**Example 4: Resume (config restored from session)**
```
User: /kernel-bench --resume prod_run

Response:
Resuming session "prod_run"...
Original command: /kernel-bench level1 --session=prod_run --workers=4
Config restored: workers=4, provider=local
Progress: 50/100 completed, 2 stale (cleaned), 48 pending.
Spawning 4 optimizers + monitor...
[progress] 55/100 completed, 4 active, avg 1.38x
...
[monitor] ALL_DONE 98/100 completed

Session prod_run complete: 98/100, avg 1.41x
```

**Example 5: Query batch progress**
```
User: /kernel-bench progress test1

Response:
=== Session: test1 (level1) ===
Progress: 25/100 completed, 12 in progress, 3 failed, 60 pending
Average speedup: 1.34x

| Task                | Status      | Worker        | Iter | Best   | Last Result           |
|---------------------|-------------|---------------|------|--------|-----------------------|
| 19_ReLU             | in_progress | optimizer-3   | 2/3  | 1.12x  | compiled, correct     |
| 23_Softmax          | in_progress | optimizer-1   | 1/3  | -      | compile error         |
| 88_MinGPTNewGelu    | completed   | optimizer-2   | 2/3  | 2.1x   | done                  |
```

**Example 6: Query single task detail**
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

**Example 7: Configure eval server port**
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

**Example 8: List available servers**
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

**Example 9: Set API key and test connection**
```
User: /kernel-bench server key --set=my_secret_key

Response:
[Runs: python3 kb_server.py key --set=my_secret_key]

API key saved to /Users/user/.keys/kbeval.api.key
```

**Example 10: SSH tunnel helper**
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

**Example 11: Verify session compliance**
```
User: /kernel-bench verify test1

Response:
[Runs: python3 kb_verify.py test1]

   ═══ Verification: test1 ═══

     Verified:    95 tasks (5 skipped)
     PASS:        72
     WARN:        18
     FAIL:        5

     Score: 72/95 PASS, 18 WARN, 5 FAIL

   ─── Top Issues ───
      5x  [FAIL] reflection.md exists and non-empty
      8x  [WARN] >=2 distinct explore strategies
      7x  [WARN] algo_trace Phase C

   Detailed report: ~/.inference/claude_code_output/test1/verification_20260214_153000.md
```

**Example 12: Verify with semantic analysis**
```
User: /kernel-bench verify test1 --semantic

Response:
[Runs: python3 kb_verify.py test1]
(mechanical results displayed)
[Spawns semantic verifier agent...]
Semantic verification complete. Report: ~/.inference/claude_code_output/test1/semantic_verification.md
```

**Example 13: Learn from session (full)**
```
User: /kernel-bench learn test1

Response:
Collecting reflections... 95 reflections collected.
Collecting algo traces... 92 traces collected.
Classifying by op type... matmul(32), conv(28), reduction(15), pointwise(10), normalization(5), other(5)
Snapshotting optimizer.md...
Spawning 8 learner agents (common + 6 op-types + algorithm)...
[waiting for agents...]
Learning complete for session 'test1'.
Kernel learners: 7 (common + 6 op-types)
Algorithm learner: done — 2 sections updated (iteration_budget_table v1→v2, exploit_decision_tree v1→v2)
Changelog: ~/.inference/claude_code_output/test1/algorithm_changelog.md
```

**Example 14: Learn algorithm only**
```
User: /kernel-bench learn test1 --algo-only

Response:
Collecting algo traces... 92 traces collected.
Snapshotting optimizer.md...
Spawning algorithm learner...
Algorithm learner: done — 1 section updated (bottleneck_diagnosis v1→v2)
Changelog: ~/.inference/claude_code_output/test1/algorithm_changelog.md
```
