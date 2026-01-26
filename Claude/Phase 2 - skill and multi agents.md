# Phase 2 - Skill and Multi-Agent Architecture

> **Extends `Phase1 - Claude code integration.md`** with advanced features for production-grade kernel optimization.

---

## 1. Goals

### 1.1 Problem Statement

Phase 1 integration works for basic tasks but lacks:
- **Ad-hoc prompts** - User must manually specify agent count, task distribution, session IDs
- **No crash recovery** - If session interrupted during 200+ task run, must restart from beginning
- **Shallow optimization** - Single-pass generation with basic iteration
- **No learning** - Each task starts fresh, doesn't leverage past successes

### 1.2 Phase 2 Goals

| Goal | Description |
|------|-------------|
| **Simple invocation** | Single `/kernel-bench` command handles all use cases |
| **Resilient execution** | Complete 200+ tasks with crash recovery and resume |
| **Deep optimization** | Multi-strategy search with self-reflection |
| **Progress visibility** | Real-time tracking, structured outputs |

### 1.3 Key Design Decisions

| Decision | Implementation |
|----------|----------------|
| **Skill with multiple input styles** | Single `/kernel-bench` skill handles all use cases |
| **MD-based agents** | Agent prompts in `.claude/agents/*.md`, spawned via Task tool |
| **Structured JSON responses** | All agent communication uses JSON to mitigate LLM non-determinism |

---

## 2. User Experience

### 2.1 Skill Input Styles

The `/kernel-bench` skill supports flexible invocation:

| Style | Example | Mode |
|-------|---------|------|
| **Single task** | `/kernel-bench level1/19_ReLU.py` | Direct optimization |
| **Directory batch** | `/kernel-bench level1/` | Coordinator + workers |
| **Full parameters** | `/kernel-bench level1 --session=my_run --workers=4` | Batch with config |
| **Natural language** | `/kernel-bench 4 random tasks from level1` | Interpreted |
| **Resume** | `/kernel-bench --resume my_run` | Continue session |

### 2.2 Example Sessions

**Single Task (interactive):**
```
User: /kernel-bench level1/19_ReLU.py

Claude: Reading task... PyTorch ReLU activation.
        Generating Triton kernel (iteration 1)...
        Evaluating... compiled ✓, correct ✓, speedup: 1.12x
        Reflecting: Memory-bound, trying vectorized loads...
        Generating (iteration 2)...
        Evaluating... compiled ✓, correct ✓, speedup: 1.31x
        Target reached. Saved to ~/.inference/claude_code_output/
```

**Batch Run (parallel workers):**
```
User: /kernel-bench level1 --session=prod_run --workers=4

Claude: Initializing session "prod_run"...
        Found 100 tasks in level1, 0 completed.
        Spawning 4 optimizer workers...

        Progress: 25/100 completed, avg speedup: 1.28x
        Progress: 50/100 completed, avg speedup: 1.31x
        ...

        === Session Complete ===
        Completed: 97/100 (97%)
        Failed: 3 (flagged for review)
        Average speedup: 1.34x
```

**Resume After Crash:**
```
User: /kernel-bench --resume prod_run

Claude: Resuming session "prod_run"...
        Found 50 completed, 2 in_progress (stale), 48 pending.
        Cleaning stale markers...
        Spawning 4 workers for remaining 50 tasks...
```

### 2.3 Structured Output

All responses use JSON for consistency:

```json
{
  "agent": "coordinator",
  "session_id": "prod_run",
  "status": { "total": 100, "completed": 97, "failed": 3, "avg_speedup": 1.34 },
  "top_performers": [{"task": "88_MinGPTNewGelu", "speedup": 2.1}]
}
```

---

## 3. Architecture

### 3.1 System Design

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CLAUDE CODE SESSION                                     │
│                                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                        MD-DEFINED AGENTS & SKILL                               │  │
│  │                                                                                │  │
│  │   .claude/skills/kernel-bench.md        .claude/agents/kernel-bench-*.md      │  │
│  │   ┌─────────────────────────────┐       ┌─────────────────────────────────┐   │  │
│  │   │  SKILL                      │       │  COORDINATOR    OPTIMIZER       │   │  │
│  │   │  • Parse input styles       │       │  (spawned via   WORKERS         │   │  │
│  │   │  • Route to mode            │──────▶│   Task tool)   (spawned by      │   │  │
│  │   │  • Spawn coordinator        │       │                 coordinator)    │   │  │
│  │   └─────────────────────────────┘       └─────────────────────────────────┘   │  │
│  │                                                       │                        │  │
│  │                            Agents call MCP tools      │                        │  │
│  │                            as instructed in MD        ▼                        │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                          │                           │
│  ┌───────────────────────────────────────────────────────┴───────────────────────┐  │
│  │                         MCP TOOLS (Python)                                     │  │
│  │                    claudeCodeKernelBenchServer.py                              │  │
│  │                                                                                │  │
│  │   STATE MANAGEMENT              KERNEL EVALUATION                              │  │
│  │   ┌──────────────────────┐      ┌──────────────────────┐                      │  │
│  │   │ • init_session()     │      │ • list_kernel_bench_ │                      │  │
│  │   │ • get_session_state()│      │   tasks()            │                      │  │
│  │   │ • get_pending_tasks()│      │ • get_task_details() │                      │  │
│  │   │ • claim_task()       │      │ • eval_kernel()      │──────────────────┐   │  │
│  │   │ • release_task()     │      │ • save_benchmark_    │                  │   │  │
│  │   │ • search_knowledge_  │      │   result()           │                  │   │  │
│  │   │   base()             │      └──────────────────────┘                  │   │  │
│  │   │ • update_knowledge_  │                                                │   │  │
│  │   │   base()             │                                                │   │  │
│  │   └──────────┬───────────┘                                                │   │  │
│  │              │                                                            │   │  │
│  └──────────────┼────────────────────────────────────────────────────────────┼───┘  │
│                 │                                                            │      │
│                 ▼                                                            │      │
│  ┌──────────────────────────────────────────┐                                │      │
│  │     LOCAL FILE-BASED STATE               │                                │      │
│  │     ~/.inference/claude_code_output/     │                                │      │
│  │                                          │                                │      │
│  │  {session_id}/                           │                                │      │
│  │  ├── session_manifest.json               │                                │      │
│  │  ├── {task}/                             │                                │      │
│  │  │   ├── .in_progress    ← atomic claim  │                                │      │
│  │  │   ├── iteration_*.py  ← kernels       │                                │      │
│  │  │   ├── iteration_*.json← eval results  │                                │      │
│  │  │   └── best_result.json← completion    │                                │      │
│  │  └── summary.json                        │                                │      │
│  │                                          │                                │      │
│  │  _knowledge_base/                        │                                │      │
│  │  ├── index.json          ← RAG index     │                                │      │
│  │  └── kernels/            ← patterns      │                                │      │
│  └──────────────────────────────────────────┘                                │      │
│                                                                              │      │
└──────────────────────────────────────────────────────────────────────────────┼──────┘
                                                                               │
                                    ═══════════════════════════════════════════╪═══════
                                                   SSH Tunnel                  │
                                    ═══════════════════════════════════════════╪═══════
                                                                               │
                                                                               ▼
                                              ┌────────────────────────────────────────┐
                                              │         REMOTE GPU SERVER              │
                                              │         kbEvalServer (stateless)       │
                                              │                                        │
                                              │  • Receives kernel code via HTTP       │
                                              │  • Compiles with torch/triton          │
                                              │  • Benchmarks on GPU                   │
                                              │  • Returns: compiled, correct, speedup │
                                              │  • No state persistence                │
                                              └────────────────────────────────────────┘
```

**Key Design Points:**

| Layer | Implementation | Responsibility |
|-------|----------------|----------------|
| **MD-Defined Agents/Skill** | `.claude/skills/*.md`, `.claude/agents/*.md` | User interface, orchestration logic, optimization strategy |
| **MCP Tools (Python)** | `claudeCodeKernelBenchServer.py` | Atomic operations, state management, HTTP calls to GPU |
| **Local File State** | `~/.inference/claude_code_output/` | Persistent state, crash recovery, progress tracking |
| **Remote GPU Server** | `kbEvalServer.py` | Stateless kernel compilation and benchmarking |

**How the layers integrate:**
1. **Skill** (MD) parses user input, decides mode, spawns coordinator
2. **Coordinator** (MD) calls MCP tools to check state, spawns workers
3. **Workers** (MD) call MCP tools to claim tasks, eval kernels, save results
4. **MCP Tools** (Python) handle atomic file ops, HTTP to GPU server
5. **Local State** (filesystem) persists progress, enables resume
6. **GPU Server** (remote) does actual compilation/benchmarking

### 3.2 Agent Hierarchy

```
                    ┌─────────────────┐
                    │   Coordinator   │  ← .claude/agents/kernel-bench-coordinator.md
                    │   (supervisor)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ↓                ↓                ↓
     ┌──────────┐     ┌──────────┐     ┌──────────┐
     │ Optimizer│     │ Optimizer│     │ Optimizer│  ← .claude/agents/kernel-bench-optimizer.md
     │ Worker 1 │     │ Worker 2 │     │ Worker N │
     └────┬─────┘     └────┬─────┘     └────┬─────┘
          │                │                │
    ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
    │ Strategy  │    │ Strategy  │    │ Strategy  │  ← .claude/agents/kernel-bench-strategy.md
    │ Sub-Agents│    │ Sub-Agents│    │ Sub-Agents│     (spawned per iteration, 3 per worker)
    │ (A, B, C) │    │ (A, B, C) │    │ (A, B, C) │
    └───────────┘    └───────────┘    └───────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   eval_kernel()  │  ← MCP tool with GPU semaphore
                  │  (HW-bounded)    │
                  └──────────────────┘
```

| Agent | Role | Key MCP Tools | GPU Access |
|-------|------|---------------|------------|
| **Coordinator** | Spawns workers, monitors progress | `get_session_state`, `get_pending_tasks` | None |
| **Optimizer Worker** | Claims tasks, orchestrates strategies | `claim_task`, `save_benchmark_result` | None |
| **Strategy Sub-Agent** | Generates ONE kernel, evaluates it | `eval_kernel()` | **Yes (semaphore-bounded)** |

**GPU resource control:** Only Strategy Sub-Agents call `eval_kernel()`. The MCP server's adaptive semaphore (set to `num_devices`) ensures at most N concurrent GPU evaluations regardless of how many sub-agents are spawned.

### 3.3 File Structure

```
triton-ag/
├── .claude/
│   ├── skills/kernel-bench.md              # Skill definition
│   ├── agents/
│   │   ├── kernel-bench-coordinator.md     # Coordinator prompt
│   │   ├── kernel-bench-optimizer.md       # Worker prompt
│   │   └── kernel-bench-strategy.md        # Strategy sub-agent prompt
│   ├── memory/
│   │   ├── kernel-strategies.md            # Optimization strategies
│   │   └── learnings.md                    # Accumulated learnings
│   └── logs/workflow-trace.md              # Progress log

~/.inference/claude_code_output/
├── _knowledge_base/                        # RAG storage
└── {session_id}/                           # Per-session state
    ├── session_manifest.json
    ├── {task_name}/
    │   ├── .in_progress                    # Claim marker
    │   ├── iteration_*.py                  # Kernels
    │   ├── best_result.json                # Completion marker
    │   └── failures.json                   # Error log
    └── summary.json
```

---

## 4. Detailed Specifications

### 4.1 State Management

#### 4.1.1 State Location

All state lives on the **local client** (Mac), not on the remote GPU server:

```
LOCAL CLIENT (Mac)                          REMOTE SERVER (GPU)
─────────────────                          ──────────────────
~/.inference/claude_code_output/           kbEvalServer
├── {session_id}/                          • Stateless
│   ├── session_manifest.json              • Only compiles/benchmarks
│   ├── {task_name}/                       • Returns results via HTTP
│   │   ├── .in_progress
│   │   ├── iteration_*.py
│   │   └── best_result.json
│   └── summary.json
```

**Why client-side state:**
- Claude Code runs locally, has direct filesystem access
- MCP tools execute in local Python process
- No network latency for state operations
- Survives GPU server restarts
- Easy to inspect with `ls`, `cat`, `jq`

#### 4.1.2 State Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `session_manifest.json` | `{session_id}/` | Session config, metadata, created_at |
| `session_manifest.json.bak` | `{session_id}/` | Backup before overwrite (RL pattern) |
| `.in_progress` | `{session_id}/{task}/` | Atomic claim marker with worker, pid, timestamp |
| `iteration_*.py` | `{session_id}/{task}/` | Generated kernel code per iteration |
| `iteration_*.json` | `{session_id}/{task}/` | Eval results per iteration |
| `best_result.json` | `{session_id}/{task}/` | Completion marker with best speedup |
| `failures.json` | `{session_id}/{task}/` | Failed attempts for retry analysis |
| `summary.json` | `{session_id}/` | Aggregated session stats |

#### 4.1.3 Task Status Detection

| Status | Detection | Next Action |
|--------|-----------|-------------|
| `pending` | Task directory doesn't exist | Can be claimed |
| `in_progress` | `.in_progress` marker exists and not stale | Skip (another worker has it) |
| `incomplete` | Has iteration files but no `best_result.json` | Can be claimed (previous attempt failed) |
| `completed` | Has `best_result.json` | Skip (already done) |

#### 4.1.4 How Coordinator Uses State

```
COORDINATOR STARTUP
───────────────────
1. Call get_session_state(session_id)
   └─► MCP tool scans ~/.inference/claude_code_output/{session_id}/
   └─► Returns: {total: 100, completed: 45, in_progress: 2, pending: 53}

2. If in_progress tasks have stale markers:
   └─► Auto-cleanup stale markers (30min timeout or dead PID)
   └─► Those tasks become "incomplete" → claimable

3. Spawn N workers with session_id
   └─► Each worker will call get_pending_tasks() independently

COORDINATOR MONITORING (periodic)
─────────────────────────────────
1. Call get_session_state(session_id) every few minutes
2. Compare progress to previous check
3. Detect stalled workers (no progress for >10 min)
4. Log to .claude/logs/workflow-trace.md

COORDINATOR FINALIZATION
────────────────────────
1. All workers report completion
2. Call get_session_state() for final counts
3. Generate summary report with metrics
4. Write summary.json
```

#### 4.1.5 How Resume Works

**Scenario:** User runs `/kernel-bench level1 --session=prod_run --workers=4`, session crashes at 50/100 tasks.

```
CRASH STATE (50 tasks done, 2 in-progress, 48 pending)
──────────────────────────────────────────────────────
~/.inference/claude_code_output/prod_run/
├── session_manifest.json           # Exists
├── 19_ReLU/
│   └── best_result.json            # ✓ Completed
├── 20_Sigmoid/
│   └── best_result.json            # ✓ Completed
... (48 more completed)
├── 69_Softmax/
│   ├── .in_progress                # ← Stale (worker crashed)
│   └── iteration_00_cuda_kernel.py # Partial work
├── 70_LayerNorm/
│   └── .in_progress                # ← Stale (worker crashed)
└── (48 task directories don't exist) # Pending

USER RESUMES
────────────
User: /kernel-bench --resume prod_run

SKILL BEHAVIOR
──────────────
1. Detects --resume flag
2. Calls get_session_state("prod_run")
3. MCP tool returns:
   {
     "total": 100,
     "completed": 50,
     "in_progress": 2,    # Stale markers detected
     "pending": 48,
     "stale_cleaned": 2   # Auto-cleaned during scan
   }

4. After stale cleanup:
   - 69_Softmax → status becomes "incomplete" (has iteration files)
   - 70_LayerNorm → status becomes "incomplete" (has marker removed)

5. Spawns coordinator with resume=true
6. Coordinator spawns 4 workers
7. Workers see 50 pending tasks (48 + 2 incomplete)
8. Workers claim and process remaining tasks
9. Session completes 100/100
```

**Key resume behaviors:**
- Session manifest preserved (config, created_at)
- Completed tasks untouched (have `best_result.json`)
- Stale markers auto-cleaned (30min timeout or dead PID)
- Incomplete tasks re-claimable (partial work preserved for context)
- New workers continue where previous workers left off

#### 4.1.6 Stale Marker Detection

Markers are considered stale when ANY of these conditions are true:

| Condition | Detection | Rationale |
|-----------|-----------|-----------|
| **Time-based** | `started_at` > 30 minutes ago | Task should complete in <30 min |
| **PID-based** | `os.kill(pid, 0)` raises OSError | Process no longer running |
| **Hostname mismatch** | `hostname` != current hostname | Different machine, can't check PID |
| **Corrupted JSON** | `json.loads()` fails | Marker file corrupted |

**Stale cleanup happens automatically** when `get_session_state()` or `get_pending_tasks()` is called.

#### 4.1.7 Atomic Task Claiming

Workers compete for tasks using atomic file creation:

```python
def claim_task(session_id, task_name, worker_id):
    marker = Path(f"~/.inference/claude_code_output/{session_id}/{task_name}/.in_progress")
    marker.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 'x' mode = exclusive create, fails if file exists
        with open(marker, 'x') as f:
            json.dump({
                "worker": worker_id,
                "started_at": datetime.now().isoformat(),
                "pid": os.getpid(),
                "hostname": socket.gethostname()
            }, f)
        return {"success": True}
    except FileExistsError:
        return {"success": False, "reason": "Already claimed"}
```

**Why this works:**
- `open(path, 'x')` is atomic on POSIX filesystems
- Two workers calling simultaneously: exactly one succeeds
- No race condition between check and create

### 4.2 Deep Optimization Loop

> **Performance is CRITICAL.** The optimization loop is where kernel quality is determined. Poor loop design → poor speedups → failed benchmarks.

#### 4.2.1 The Core Question: Sub-Agents for Tree of Thoughts?

**Option A: Sequential Strategy (Simple)**
```
Worker tries Strategy A → eval → if bad, try B → eval → if bad, try C
```
- Pros: Simple, predictable GPU usage
- Cons: Slow (strategies evaluated one-by-one), misses parallel exploration

**Option B: Parallel Strategy Sub-Agents (Better Coverage)**
```
Worker spawns 3 sub-agents in parallel:
├── Sub-Agent A: Generate kernel with Strategy A → eval_kernel()
├── Sub-Agent B: Generate kernel with Strategy B → eval_kernel()
└── Sub-Agent C: Generate kernel with Strategy C → eval_kernel()
Worker waits, picks best result
```
- Pros: Faster wall-clock, explores more strategies, finds better optima
- Cons: More GPU contention, needs resource management

**Recommendation: Option B with GPU Semaphore Bounding**

Parallel strategy exploration is worth the complexity because:
1. **Different strategies find different optima** - Strategy A might hit 1.2x, B might hit 1.5x
2. **Some strategies fail to compile** - Having backups avoids wasted iterations
3. **LLM generation is fast** - The bottleneck is GPU eval, not kernel generation
4. **Semaphore already exists** - MCP server's adaptive semaphore bounds concurrent evals

#### 4.2.2 GPU Resource Bounding

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          GPU RESOURCE FLOW                                       │
│                                                                                  │
│   WORKERS (4)                    MCP SERVER                    GPU SERVER       │
│   ───────────                    ──────────                    ──────────       │
│                                                                                  │
│   Worker 1 ─┬─ Sub-A ──► eval_kernel() ──┐                                      │
│             ├─ Sub-B ──► eval_kernel() ──┤                                      │
│             └─ Sub-C ──► eval_kernel() ──┤      ┌──────────────┐                │
│                                          │      │              │                │
│   Worker 2 ─┬─ Sub-A ──► eval_kernel() ──┼─────►│  SEMAPHORE   │───► GPU 0     │
│             ├─ Sub-B ──► eval_kernel() ──┤      │  (limit = 2) │               │
│             └─ Sub-C ──► eval_kernel() ──┤      │              │───► GPU 1     │
│                                          │      │  Queues up   │                │
│   Worker 3 ─┬─ Sub-A ──► eval_kernel() ──┤      │  to 12 reqs  │                │
│             ├─ Sub-B ──► eval_kernel() ──┤      └──────────────┘                │
│             └─ Sub-C ──► eval_kernel() ──┤                                      │
│                                          │                                      │
│   Worker 4 ─┬─ Sub-A ──► eval_kernel() ──┤                                      │
│             ├─ Sub-B ──► eval_kernel() ──┘                                      │
│             └─ Sub-C ──► eval_kernel() ──┘                                      │
│                                                                                  │
│   4 workers × 3 sub-agents = 12 potential concurrent evals                      │
│   Semaphore(2) ensures only 2 run at a time → no GPU OOM                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Mechanism:** The MCP server's adaptive semaphore (initialized from `/info` endpoint's `num_devices`) limits concurrent `eval_kernel()` calls regardless of how many sub-agents are spawned.

#### 4.2.3 Optimization Loop with Parallel Strategies

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DEEP OPTIMIZATION LOOP (Per Task)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1: UNDERSTAND (single agent, no GPU)                                     │
│  ├── Read PyTorch code via get_task_details()                                   │
│  ├── Classify: reduction | element-wise | matmul | normalization | conv | attn │
│  ├── Identify memory patterns, compute intensity, data dependencies             │
│  └── [RAG] Retrieve similar successful kernels via search_knowledge_base()     │
│                                                                                  │
│  PHASE 2: STRATEGIZE (single agent, no GPU)                                     │
│  ├── Generate 3 candidate optimization strategies based on operation type       │
│  ├── For each strategy: describe approach, expected benefit, risk               │
│  └── Output: strategies = [A, B, C]                                             │
│                                                                                  │
│  PHASE 3: PARALLEL GENERATION + EVALUATION (sub-agents, GPU-bounded)           │
│  │                                                                               │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │
│  │   │  Spawn 3 Strategy Sub-Agents via Task tool (parallel)               │    │
│  │   │                                                                      │    │
│  │   │  Sub-Agent A:                                                        │    │
│  │   │  ├── Read strategy A from prompt                                     │    │
│  │   │  ├── Generate kernel with CoT reasoning (LLM, fast)                 │    │
│  │   │  ├── Call eval_kernel() (blocks on semaphore if GPU busy)           │    │
│  │   │  └── Return: {kernel_code, compiled, correct, speedup}              │    │
│  │   │                                                                      │    │
│  │   │  Sub-Agent B: (same, in parallel)                                    │    │
│  │   │  Sub-Agent C: (same, in parallel)                                    │    │
│  │   │                                                                      │    │
│  │   │  Worker waits for all 3 sub-agents to complete                       │    │
│  │   └─────────────────────────────────────────────────────────────────────┘    │
│  │                                                                               │
│  └── Collect results: [result_A, result_B, result_C]                            │
│                                                                                  │
│  PHASE 4: AGGREGATE + REFLECT (single agent, no GPU)                            │
│  ├── Compare results, pick best by: compiled → correct → speedup               │
│  ├── Analyze why best strategy worked                                           │
│  ├── Analyze why others failed (compile error? incorrect? slow?)               │
│  └── Record learnings for future tasks                                          │
│                                                                                  │
│  PHASE 5: ITERATE OR FINALIZE                                                   │
│  │                                                                               │
│  ├── IF best_speedup >= 1.5x (excellent):                                       │
│  │   └── Save best_result.json, update knowledge base, DONE                    │
│  │                                                                               │
│  ├── IF best_speedup >= 1.3x AND iteration >= 2:                               │
│  │   └── Good enough, save and move on                                          │
│  │                                                                               │
│  ├── IF all 3 failed to compile:                                                │
│  │   └── Flag for manual review, release task                                   │
│  │                                                                               │
│  └── ELSE (need improvement):                                                   │
│      ├── Take best partial result as starting point                             │
│      ├── Generate 3 NEW strategies that address identified issues               │
│      └── Loop back to PHASE 3 (max 3 iterations)                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.4 Strategy Sub-Agent Prompt

The sub-agent receives a focused prompt for one strategy:

```markdown
# Strategy Sub-Agent

You are generating a CUDA/Triton kernel using ONE specific strategy.

## Your Strategy
{strategy_description}

## Task
{pytorch_code}

## Instructions
1. Apply ONLY this strategy to generate the kernel
2. Use Chain-of-Thought: explain each decision as you code
3. Verify memory access patterns, occupancy, arithmetic intensity
4. Call eval_kernel() with your generated code
5. Return structured result

## Output Format
{
  "strategy": "...",
  "kernel_code": "...",
  "reasoning": ["step1", "step2", ...],
  "eval_result": {compiled, correct, speedup}
}
```

#### 4.2.5 Why This Design Maximizes Performance

| Aspect | Benefit |
|--------|---------|
| **Parallel strategy exploration** | 3x more strategies tried per iteration |
| **LLM parallelism** | Sub-agents generate kernels concurrently (no GPU needed) |
| **GPU bounded by semaphore** | Never overwhelms GPU, no OOM |
| **Best-of-3 selection** | Higher chance of finding good strategy |
| **Early termination** | Stop at 1.5x, don't waste time over-optimizing |
| **Learning from failures** | Reflect phase captures insights for next iteration |

#### 4.2.6 Iteration Budget

```
Per task: up to 3 iterations × 3 strategies = 9 kernel attempts max
Per iteration: ~30 seconds GPU time (3 evals × 10s each, with semaphore queuing)
Total per task: ~90 seconds worst case, often faster with early termination
```

#### 4.2.7 Comparison: Sequential vs Parallel Strategies

| Metric | Sequential (Option A) | Parallel (Option B) |
|--------|----------------------|---------------------|
| Strategies per iteration | 1 | 3 |
| Wall-clock per iteration | 30s (1 eval) | 30s (3 evals, parallel + queue) |
| Chance of finding 1.5x+ | Lower | 3x higher |
| GPU utilization | Sparse | Dense (via semaphore) |
| Complexity | Simple | Medium |
| **Recommended for** | Quick tests | Production runs |

### 4.3 Optimization Strategies by Operation Type

| Operation | Key Strategies |
|-----------|---------------|
| Reduction | Tree reduction, warp primitives, vectorized loads |
| Element-wise | Vectorized loads, fusion, fast math |
| Matmul | Tiled + shared memory, register blocking |
| Normalization | Single-pass mean/var, Welford's algorithm |

### 4.4 Coordinator Agent Spec

**Responsibilities:**
1. Initialize session via `get_session_state()`
2. Spawn N workers via Task tool (parallel)
3. Monitor progress periodically
4. Flag repeated failures for review
5. Generate final summary with metrics

**Output:** Structured JSON with status counts, worker list, avg_speedup

### 4.5 Optimizer Worker Agent Spec

**Task Loop:**
1. Get pending tasks via `get_pending_tasks()`
2. Claim task via `claim_task()` (atomic)
3. Run deep optimization loop (5 iterations max)
4. Save best result, release task
5. Repeat until no pending tasks

**Output:** Structured JSON with task_name, iterations, best_speedup, reflections

### 4.6 MCP Tools (Phase 2 Additions)

| Tool | Purpose |
|------|---------|
| `init_session(session_id, level, config)` | Create session manifest |
| `claim_task(session_id, task_name, worker_id)` | Atomic task claiming with marker + PID |
| `release_task(session_id, task_name, error)` | Release claim, record failures |
| `get_session_state(session_id)` | Full progress view with stale cleanup |
| `get_pending_tasks(session_id, level)` | List unclaimed tasks |
| `search_knowledge_base(operation_type)` | RAG retrieval for similar kernels |
| `update_knowledge_base(task, kernel, speedup)` | Store successful patterns |

---

## 5. Implementation Plan

### 5.1 Work Items

| Phase | Item | Status |
|-------|------|--------|
| **1. Foundation** | | |
| 1.1 | Create `.claude/{skills,agents,memory,logs}/` directories | PENDING |
| 1.2 | Create `.claude/skills/kernel-bench.md` | PENDING |
| 1.3 | Create `.claude/agents/kernel-bench-coordinator.md` | PENDING |
| 1.4 | Create `.claude/agents/kernel-bench-optimizer.md` | PENDING |
| 1.5 | Create `.claude/agents/kernel-bench-strategy.md` | PENDING |
| 1.6 | Create memory templates (kernel-strategies.md, learnings.md) | PENDING |
| **2. MCP Tools** | | |
| 2.1 | `init_session()` | PENDING |
| 2.2 | `claim_task()` with atomic marker + PID | PENDING |
| 2.3 | `release_task()` with failure recording | PENDING |
| 2.4 | `get_session_state()` with stale cleanup | PENDING |
| 2.5 | `get_pending_tasks()` | PENDING |
| **3. Knowledge Base** (defer for MVP) | | |
| 3.1 | `search_knowledge_base()` | PENDING |
| 3.2 | `update_knowledge_base()` | PENDING |

### 5.2 Implementation Order

1. **Phase 1: Foundation** - Directory structure and prompt files (including strategy sub-agent)
2. **Phase 2: MCP Tools** - Focus on `claim_task()` and `get_session_state()` first
3. **Phase 3: Knowledge Base** - Can defer for MVP

---

## 6. Test-Driven Verification

### 6.1 Test Plan

| Test | Description | Validates |
|------|-------------|-----------|
| **T1: Skill Parsing** | `/kernel-bench level1/19_ReLU.py` routes to single task mode | Input style detection |
| **T2: Skill Parsing** | `/kernel-bench level1/` routes to batch mode | Directory detection |
| **T3: Skill Parsing** | `/kernel-bench --resume my_run` loads existing session | Resume detection |
| **T4: Coordinator Spawn** | Coordinator spawns N workers via Task tool | Agent hierarchy |
| **T5: Atomic Claim** | Two workers claim same task, only one succeeds | `claim_task()` atomicity |
| **T6: Stale Cleanup** | Create 35-min old marker, verify auto-cleanup | Stale detection |
| **T7: PID Cleanup** | Create marker with dead PID, verify cleanup | PID-based staleness |
| **T8: Worker Loop** | Worker claims, processes, saves, claims next | End-to-end worker |
| **T9: Resume** | Kill mid-session, resume, verify continuation | Crash recovery |
| **T10: Parallel Workers** | 4 workers, 20 tasks, no duplicates claimed | Worker concurrency |
| **T11: Structured Output** | All agent outputs parse as valid JSON | JSON format |
| **T12: Strategy Sub-Agents** | Worker spawns 3 strategy sub-agents in parallel | Parallel strategies |
| **T13: GPU Semaphore** | 12 concurrent eval_kernel() calls, only 2 run at a time | GPU bounding |
| **T14: Best-of-3 Selection** | Worker picks best result from 3 strategies | Strategy aggregation |
| **T15: Full Batch** | Complete level1 (100 tasks) with 4 workers | Production readiness |

### 6.2 Verification Commands

```bash
# T1-T3: Skill parsing (manual in Claude Code)
/kernel-bench level1/19_ReLU.py
/kernel-bench level1/
/kernel-bench --resume test_session

# T5: Atomic claim (programmatic)
python -c "
from claudeCodeKernelBenchServer import claim_task
import asyncio
r1 = asyncio.run(claim_task('test', 'task1', 'w1'))
r2 = asyncio.run(claim_task('test', 'task1', 'w2'))
assert r1['success'] and not r2['success']
print('T5 PASS')
"

# T6: Stale cleanup (create old marker, run get_session_state)
# T9: Resume (kill mid-session, re-run with --resume)
# T12: Full batch
/kernel-bench level1 --session=full_test --workers=4
```

### 6.3 Success Criteria

| Metric | Target |
|--------|--------|
| Skill input styles | All 5 styles parse correctly |
| Atomic claiming | 0 duplicate claims in parallel test |
| Stale cleanup | 100% stale markers removed |
| Resume | Picks up exactly where left off |
| Parallel workers | Wall time < sequential time / workers |
| **Strategy sub-agents** | 3 sub-agents spawn per iteration |
| **GPU semaphore** | Max 2 concurrent evals (with 2 GPUs) |
| **Best-of-3** | Worker correctly selects highest speedup |
| Full batch | >= 95% task completion rate |
| **Speedup improvement** | Parallel strategies achieve higher avg speedup than sequential |

---

## Appendix: Reliability Considerations

Based on multi-agent LLM research ([arXiv:2503.13657](https://arxiv.org/html/2503.13657v1)):

| Issue | Mitigation |
|-------|------------|
| LLM non-determinism (~15% variance) | MCP tools for atomic ops, structured JSON |
| Context loss in handoffs | File-based state, workflow-trace.md |
| Premature termination | Coordinator gates, explicit completion markers |
| Task disobedience | Clear output format, explicit constraints |

**This design works well for:** Kernel bench (independent, idempotent tasks) ✓
