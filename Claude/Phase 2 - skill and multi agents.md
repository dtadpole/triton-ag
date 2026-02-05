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
| **Parallel execution by default** | Workers and strategy sub-agents MUST be spawned in parallel using multiple Task calls in a single message |
| **Default 4 concurrent workers** | Batch mode spawns 4 workers unless overridden with `--workers=N` |
| **Skill with multiple input styles** | Single `/kernel-bench` skill handles all use cases |
| **MD-based agents** | Agent prompts in `.claude/agents/*.md`, spawned via Task tool |
| **Structured JSON responses** | All agent communication uses JSON to mitigate LLM non-determinism |

### 1.4 Parallel Execution Architecture

**CRITICAL: All parallelism is achieved by spawning multiple Task tool calls in a SINGLE message.**

```
CORRECT (parallel):
┌─────────────────────────────────────────────────────────────┐
│ One Message containing:                                      │
│   Task(optimizer-1) + Task(optimizer-2) + Task(optimizer-3) │
│   + Task(optimizer-4)                                        │
│                                                              │
│   Result: All 4 workers run concurrently                     │
└─────────────────────────────────────────────────────────────┘

WRONG (sequential):
┌───────────────────────┐
│ Message 1: Task(opt-1)│ → wait for completion
└───────────────────────┘
┌───────────────────────┐
│ Message 2: Task(opt-2)│ → wait for completion
└───────────────────────┘
... (workers run one at a time)
```

**Where parallelism applies:**
| Level | Component | Parallelism |
|-------|-----------|-------------|
| Session | Workers (optimizer agents) | N workers spawned in parallel (default: 4) |
| Task | Strategies (strategy sub-agents) | 3 strategies spawned in parallel per iteration |
| GPU | eval_kernel() calls | Bounded by semaphore (default: num_devices) |

---

## 2. User Experience

### 2.1 Skill Input Styles

The `/kernel-bench` skill supports flexible invocation:

| Style | Example | Mode |
|-------|---------|------|
| **Single task** | `/kernel-bench level1/19_ReLU.py` | Direct optimization |
| **Directory batch** | `/kernel-bench level1/` | Skill spawns parallel workers |
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

**Batch Run (parallel workers - DEFAULT BEHAVIOR):**
```
User: /kernel-bench level1 --session=prod_run

Claude: Initializing session "prod_run"...
        Found 100 tasks in level1, 0 completed.
        Spawning 4 workers in parallel (default)...
        [ALL workers spawned in single message]
          - optimizer-1: running in background
          - optimizer-2: running in background
          - optimizer-3: running in background
          - optimizer-4: running in background

        Progress: 25/100 completed, avg speedup: 1.28x
        Progress: 50/100 completed, avg speedup: 1.31x
        ...

        === Session Complete ===
        Completed: 97/100 (97%)
        Failed: 3 (flagged for review)
        Average speedup: 1.34x
```

**Batch Run with Custom Worker Count:**
```
User: /kernel-bench level1 --session=prod_run --workers=8

Claude: Initializing session "prod_run"...
        Found 100 tasks in level1, 0 completed.
        Spawning 8 workers in parallel...
        [ALL 8 workers spawned in single message]
        ...
```

**Resume After Crash:**
```
User: /kernel-bench --resume prod_run

Claude: Resuming session "prod_run"...
        Found 50 completed, 2 in_progress (stale), 48 pending.
        Cleaning stale markers...
        Spawning 4 workers in parallel for remaining 50 tasks...
```

### 2.3 Structured Output

All responses use JSON for consistency:

```json
{
  "agent": "skill",
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
│  │   │  SKILL                      │       │  OPTIMIZER WORKERS              │   │  │
│  │   │  • Parse input styles       │       │  (spawned directly by skill     │   │  │
│  │   │  • Spawn workers directly   │──────▶│   via Task tool, in parallel)   │   │  │
│  │   │  • Monitor progress         │       │                                 │   │  │
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
1. **Skill** (MD) parses user input, decides mode, spawns workers directly in parallel
2. **Workers** (MD) call MCP tools to claim tasks, eval kernels, save results
3. **MCP Tools** (Python) handle atomic file ops, HTTP to GPU server
4. **Local State** (filesystem) persists progress, enables resume
5. **GPU Server** (remote) does actual compilation/benchmarking

### 3.2 Agent Hierarchy (Parallel by Default)

```
                    ┌─────────────────┐
                    │      Skill      │  ← .claude/commands/kernel-bench.md
                    │ (parses input)  │
                    └────────┬────────┘
                             │
                             │ Batch Mode: Spawns workers DIRECTLY
                             │ (Single message with N Task calls)
                             │
            ┌────────────────┼────────────────┐────────────────┐
            ↓                ↓                ↓                ↓
     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
     │ Optimizer│     │ Optimizer│     │ Optimizer│     │ Optimizer│  ← Spawned IN PARALLEL
     │ Worker 1 │     │ Worker 2 │     │ Worker 3 │     │ Worker N │    (default N=4)
     └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
          │                │                │                │
          │ Per task:      │                │                │
          │ 3 strategies   │                │                │
          │ in parallel    │                │                │
    ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
    │ Strategy  │    │ Strategy  │    │ Strategy  │    │ Strategy  │  ← Spawned IN PARALLEL
    │ Sub-Agents│    │ Sub-Agents│    │ Sub-Agents│    │ Sub-Agents│    (3 per task iteration)
    │ (A, B, C) │    │ (A, B, C) │    │ (A, B, C) │    │ (A, B, C) │
    └───────────┘    └───────────┘    └───────────┘    └───────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │   eval_kernel()  │  ← MCP tool with GPU semaphore
                          │  (HW-bounded)    │     (max concurrent = num_devices)
                          └──────────────────┘
```

**Parallelism summary:**
- **Workers**: 4 by default, all spawned in ONE message (parallel)
- **Strategies per task**: 3 per iteration, all spawned in ONE message (parallel)
- **GPU evals**: Bounded by semaphore to prevent OOM

| Agent | Role | Parallel Spawning |
|-------|------|-------------------|
| **Optimizer Workers** | Claim tasks, orchestrate strategies | 4 workers in 1 message |
| **Strategy Sub-Agents** | Generate ONE kernel, evaluate it | 3 strategies in 1 message per iteration |

**GPU resource control:** Only Strategy Sub-Agents call `eval_kernel()`. The MCP server's adaptive semaphore (set to `num_devices`) ensures at most N concurrent GPU evaluations regardless of how many sub-agents are spawned.

### 3.3 File Structure

```
triton-ag/
├── .claude/
│   ├── skills/kernel-bench.md              # Skill definition
│   ├── agents/
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

#### 4.1.4 How Skill Uses State

```
SKILL STARTUP (Batch Mode)
──────────────────────────
1. Call get_session_state(session_id)
   └─► MCP tool scans ~/.inference/claude_code_output/{session_id}/
   └─► Returns: {total: 100, completed: 45, in_progress: 2, pending: 53}

2. If in_progress tasks have stale markers:
   └─► Auto-cleanup stale markers (30min timeout or dead PID)
   └─► Those tasks become "incomplete" → claimable

3. Spawn N workers with session_id (in parallel, single message)
   └─► Each worker will call get_pending_tasks() independently

SKILL MONITORING (periodic)
───────────────────────────
1. Call get_session_state(session_id) periodically
2. Compare progress to previous check
3. Report to user: "Progress: X/Y completed"
4. Detect stalled workers (no progress for >10 min)

SKILL FINALIZATION
──────────────────
1. All pending tasks completed (pending == 0, in_progress == 0)
2. Call get_session_state() for final counts
3. Call get_session_summary() for metrics
4. Generate summary report
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

5. Skill spawns 4 workers directly (in parallel, single message)
6. Workers see 50 pending tasks (48 + 2 incomplete)
7. Workers claim and process remaining tasks
8. Session completes 100/100
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
│   WORKERS (4, default)             MCP SERVER                    GPU SERVER     │
│   ─────────────────────            ──────────                    ──────────     │
│   [ALL spawned in ONE msg]                                                       │
│                                                                                  │
│   Worker 1 ─┬─ Sub-A ──► eval_kernel() ──┐                                      │
│   (running) ├─ Sub-B ──► eval_kernel() ──┤                                      │
│             └─ Sub-C ──► eval_kernel() ──┤      ┌──────────────┐                │
│                        [3 in ONE msg]    │      │              │                │
│   Worker 2 ─┬─ Sub-A ──► eval_kernel() ──┼─────►│  SEMAPHORE   │───► GPU 0     │
│   (running) ├─ Sub-B ──► eval_kernel() ──┤      │  (limit = 2) │               │
│             └─ Sub-C ──► eval_kernel() ──┤      │              │───► GPU 1     │
│                        [3 in ONE msg]    │      │  Queues up   │                │
│   Worker 3 ─┬─ Sub-A ──► eval_kernel() ──┤      │  to 12 reqs  │                │
│   (running) ├─ Sub-B ──► eval_kernel() ──┤      └──────────────┘                │
│             └─ Sub-C ──► eval_kernel() ──┤                                      │
│                        [3 in ONE msg]    │                                      │
│   Worker 4 ─┬─ Sub-A ──► eval_kernel() ──┤                                      │
│   (running) ├─ Sub-B ──► eval_kernel() ──┘                                      │
│             └─ Sub-C ──► eval_kernel() ──┘                                      │
│                        [3 in ONE msg]                                           │
│                                                                                  │
│   4 workers × 3 sub-agents = 12 potential concurrent evals                      │
│   Semaphore(2) ensures only 2 run at a time → no GPU OOM                        │
│                                                                                  │
│   KEY: Workers and strategies spawned in PARALLEL (single messages)             │
│        GPU access bounded by semaphore (prevents OOM)                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Mechanism:** The MCP server's adaptive semaphore (initialized from `/info` endpoint's `num_devices`) limits concurrent `eval_kernel()` calls regardless of how many sub-agents are spawned in parallel.

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

### 4.4 Worker Spawning Model (Updated)

**The skill now spawns workers DIRECTLY (no separate coordinator agent).**

This simplifies the architecture and ensures parallel execution:

```
/kernel-bench level1 --session=my_run
                │
                ▼
         ┌──────────────┐
         │    SKILL     │  Parses input, calls init_session()
         └──────┬───────┘
                │
                │ SINGLE MESSAGE with 4 Task calls
                │
    ┌───────────┼───────────┬───────────┐
    ▼           ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ │Worker 3│ │Worker 4│  ← ALL run in PARALLEL
└────────┘ └────────┘ └────────┘ └────────┘
```

**Why direct spawning?**
- Simpler architecture with fewer layers
- Skill can spawn workers directly with `run_in_background=true`
- Skill handles monitoring directly (no coordinator overhead)
- Parallel execution is guaranteed by single-message multi-Task pattern

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
| 1.1 | Create `.claude/{skills,agents,memory,logs}/` directories | ✅ DONE |
| 1.2 | Create `.claude/commands/kernel-bench.md` | ✅ DONE |
| 1.3 | Create `.claude/agents/kernel-bench-optimizer.md` | ✅ DONE |
| 1.4 | Create `.claude/agents/kernel-bench-strategy.md` | ✅ DONE |
| 1.5 | Create memory templates (kernel-strategies.md, learnings.md) | ✅ DONE |
| **2. MCP Tools** | | |
| 2.1 | `init_session()` with task filtering | ✅ DONE + TESTED |
| 2.2 | `claim_task()` with atomic marker + PID | ✅ DONE + TESTED |
| 2.3 | `release_task()` with failure recording | ✅ DONE + TESTED |
| 2.4 | `get_session_state()` with stale cleanup | ✅ DONE + TESTED |
| 2.5 | `get_pending_tasks()` | ✅ DONE + TESTED |
| **3. Parallel Execution** | | |
| 3.1 | Skill spawns workers directly (no coordinator layer) | ✅ DONE |
| 3.2 | Default 4 concurrent workers | ✅ DONE |
| 3.3 | `--workers=N` parameter support | ✅ DONE |
| 3.4 | Single-message multi-Task spawning documented | ✅ DONE |
| **4. Testing** | | |
| 4.1 | Automated tests for MCP tools (T5, T6, T7, T9, T10) | ✅ DONE |
| 4.2 | Resume flow end-to-end test | ✅ DONE |
| 4.3 | Integration tests with skill/worker agents | 🔲 PENDING |
| **5. Knowledge Base** (defer for MVP) | | |
| 5.1 | `search_knowledge_base()` | 🔲 PENDING |
| 5.2 | `update_knowledge_base()` | 🔲 PENDING |

### 5.2 Implementation Order

1. **Phase 1: Foundation** - Directory structure and prompt files ✅
2. **Phase 2: MCP Tools** - State management with tests ✅
3. **Phase 3: Parallel Execution** - Skill spawns workers directly ✅
4. **Phase 4: Integration** - Agent testing with skill/workers 🔲
5. **Phase 5: Knowledge Base** - Defer for MVP 🔲

### 5.3 Test Files

| File | Purpose |
|------|---------|
| `test_phase2_comprehensive.py` | T5, T6, T7, T9, T10 + task filtering |
| `test_phase2_resume_flow.py` | End-to-end resume scenario |
| `test_phase2_session.py` | Basic session management |

---

## 6. Test-Driven Verification

### 6.1 Automated Tests (Unit)

Run all automated tests:
```bash
python3 test_phase2_comprehensive.py
```

---

#### T5: Atomic Claim

| Field | Value |
|-------|-------|
| **What is tested** | `claim_task()` uses atomic file creation. Two workers cannot claim the same task. |
| **Expected result** | First worker succeeds, second worker gets `{success: false, reason: "already_claimed"}` |
| **Status** | ✅ PASS |

**Test steps (automated):**
1. Initialize session with 5 tasks
2. Worker-1 claims task A → expects success
3. Worker-2 claims task A → expects failure with reason "already_claimed"
4. Worker-2 claims task B → expects success (different task)

---

#### T6: Stale Cleanup (Time-based)

| Field | Value |
|-------|-------|
| **What is tested** | Markers older than 30 minutes are automatically cleaned by `get_session_state()` |
| **Expected result** | Old marker removed, task becomes claimable again |
| **Status** | ✅ PASS |

**Test steps (automated):**
1. Claim a task (creates `.in_progress` marker)
2. Modify marker's `started_at` to 35 minutes ago
3. Call `get_session_state()` → should report `stale_cleaned: 1`
4. Verify marker file deleted
5. Claim same task again → should succeed

---

#### T7: PID Cleanup (Dead Process)

| Field | Value |
|-------|-------|
| **What is tested** | Markers with non-existent PIDs are cleaned (crashed worker recovery) |
| **Expected result** | Dead PID marker removed, task becomes claimable |
| **Status** | ✅ PASS |

**Test steps (automated):**
1. Create marker with `pid: 99999999` (non-existent process)
2. Call `get_session_state()` → should report `stale_cleaned: 1`
3. Verify marker file deleted
4. Claim task → should succeed

**Why this matters:** If a worker crashes mid-task, its claimed tasks become available again without waiting 30 minutes.

---

#### T9: Resume (Crash Recovery)

| Field | Value |
|-------|-------|
| **What is tested** | Session state correctly tracks completed vs in-progress vs pending tasks |
| **Expected result** | After partial completion, resume shows exact progress |
| **Status** | ✅ PASS |

**Test steps (automated):**
1. Initialize session with 5 specific tasks
2. Complete 2 tasks (create `best_result.json`)
3. Leave 1 task in-progress (has `.in_progress` marker)
4. Leave 2 tasks untouched
5. Call `get_session_state()` → expect:
   - `total: 5`
   - `completed: 2`
   - `in_progress: 1`
   - `pending: 2`
6. Call `get_pending_tasks()` → expect 2 tasks available

---

#### T10: Parallel Workers (Concurrency)

| Field | Value |
|-------|-------|
| **What is tested** | Multiple concurrent workers claiming tasks without duplicates |
| **Expected result** | All 20 tasks claimed exactly once, zero duplicates |
| **Status** | ✅ PASS |

**Test steps (automated):**
1. Initialize session with 20 tasks
2. Spawn 4 async workers, each trying to claim up to 10 tasks
3. Collect all claims
4. Verify: unique claims = total claims (no duplicates)
5. Verify: all 20 tasks claimed
6. Verify: `get_session_state()` shows 20 in-progress

---

#### Additional: Task Filtering

| Field | Value |
|-------|-------|
| **What is tested** | `init_session(task_names=[...])` stores only specified tasks, not all from level |
| **Expected result** | Manifest contains exactly the requested tasks |
| **Status** | ✅ PASS |

**Test steps (automated):**
1. Call `init_session(task_names=["task1", "task2", "task3"])`
2. Verify manifest has `total_tasks: 3` (not 100)
3. Verify manifest `tasks` array matches input exactly
4. Call `init_session()` without `task_names` → should get all 100 level1 tasks

---

### 6.2 Manual Tests (CLI)

These tests require running Claude Code and invoking the `/kernel-bench` skill.

---

#### T1: Skill Parsing - Single Task Mode

| Field | Value |
|-------|-------|
| **What is tested** | Single `.py` file path routes to direct optimization (no workers spawned) |
| **Expected result** | Interactive optimization of one task with progress shown |
| **Status** | 🔲 PENDING |

**Prompt:**
```
/kernel-bench level1/19_ReLU.py
```

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Task read | Claude shows PyTorch code | See `class Model(torch.nn.Module)` in output | "Task not found" error |
| 2. Direct mode | No workers spawned | Claude generates kernel immediately | See "spawning workers" |
| 3. Kernel eval | `eval_kernel()` called | See output like `compiled: true, correct: true, speedup: 1.xx` | "kbEval server unavailable" error |
| 4. Iteration | Continues if speedup < 1.3x | See "Iteration 2" or "trying different approach" | Stops after first attempt regardless of speedup |
| 5. Save result | `save_benchmark_result()` called | See "Saved to ~/.inference/claude_code_output/" | No save confirmation |

**Post-test file check:**
```bash
# Verify result saved
ls ~/.inference/claude_code_output/*/19_ReLU/best_result.json
cat ~/.inference/claude_code_output/*/19_ReLU/best_result.json
# Should contain: {"speedup": 1.xx, "completed_at": "..."}
```

---

#### T2: Skill Parsing - Batch Mode

| Field | Value |
|-------|-------|
| **What is tested** | Directory path routes to batch mode with parallel workers |
| **Expected result** | Workers spawned in parallel, process tasks concurrently |
| **Status** | 🔲 PENDING |

**Prompt:**
```
/kernel-bench level1/ --session=test_batch --workers=2
```

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Session init | `init_session()` called | See "Initialized session: test_batch with N tasks" in logs | "Session not found" error |
| 2. Workers spawned | 2 workers spawned in parallel | See "spawning 2 workers" and multiple Task calls in single message | Workers spawned sequentially |
| 3. Parallel execution | Workers claim different tasks | See different task names in claim logs | Same task claimed twice |
| 4. Claims | Workers claim tasks | See `claim_task()` calls in logs | Workers process without claiming |
| 5. Progress | Periodic updates | See "Progress: X/Y completed" messages | No progress updates |

**Post-test file check:**
```bash
# Verify session created
cat ~/.inference/claude_code_output/test_batch/session_manifest.json
# Should contain: {"session_id": "test_batch", "total_tasks": 100, ...}

# Verify claims happening
ls ~/.inference/claude_code_output/test_batch/*/.in_progress 2>/dev/null | head -5
# Should show in-progress markers with different worker IDs

# Check worker IDs are different
cat ~/.inference/claude_code_output/test_batch/*/.in_progress | jq '.worker' | sort -u
# Should show multiple unique worker IDs (e.g., "worker-1", "worker-2")
```

---

#### T3: Skill Parsing - Resume Mode

| Field | Value |
|-------|-------|
| **What is tested** | `--resume` flag loads existing session and continues |
| **Expected result** | Shows completed count, continues with remaining tasks |
| **Status** | 🔲 PENDING |

**Setup:**
1. Run T2 first to create session `test_batch`
2. Interrupt with Ctrl+C after some tasks complete
3. Verify some `best_result.json` files exist

**Prompt:**
```
/kernel-bench --resume test_batch
```

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. State check | `get_session_state()` called | See "Session test_batch: X/Y complete" | "Session not found" error |
| 2. Progress shown | Reports existing progress | See "Found X completed, Y pending" or similar | Shows "0 completed" when files exist |
| 3. No re-process | Completed tasks skipped | Completed task names NOT in new claim logs | Same tasks being re-claimed |
| 4. Continues | Remaining tasks processed | New claims for pending tasks | No new work happening |

**Pre-test verification:**
```bash
# Check how many completed before resume
ls ~/.inference/claude_code_output/test_batch/*/best_result.json 2>/dev/null | wc -l
# Note this number: ___

# Check total tasks
cat ~/.inference/claude_code_output/test_batch/session_manifest.json | jq '.total_tasks'
# Note this number: ___
```

**Post-test verification:**
```bash
# Completed count should be higher than before
ls ~/.inference/claude_code_output/test_batch/*/best_result.json 2>/dev/null | wc -l
# Should be > pre-resume count

# Verify original completed files unchanged (same timestamp)
ls -la ~/.inference/claude_code_output/test_batch/*/best_result.json | head -3
# Timestamps of early files should be from T2 run, not T3 run
```

---

### 6.3 Integration Tests (Agents)

These tests validate the agent hierarchy and coordination. They require the full agent system to be working.

---

#### T4: Parallel Worker Spawning

| Field | Value |
|-------|-------|
| **What is tested** | Skill spawns N worker agents via Task tool in a SINGLE message |
| **Expected result** | N workers running in parallel, each claiming different tasks |
| **Status** | 🔲 PENDING |

**Prompt:**
```
/kernel-bench level1/ --session=test_parallel --workers=4
```

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Single message | All Task calls in one message | See 4 Task tool calls in same response | Task calls spread across multiple messages |
| 2. Background mode | Workers run in background | See `run_in_background: true` in Task calls | Workers block the skill |
| 3. Unique worker IDs | Each worker has different ID | `.in_progress` files show worker-1, worker-2, etc. | All claims show same worker ID |
| 4. Parallel claims | Multiple workers claim simultaneously | Different tasks claimed within same second | Claims happen sequentially with gaps |

**Post-test file check:**
```bash
# Check worker ID distribution
cat ~/.inference/claude_code_output/test_parallel/*/.in_progress 2>/dev/null | jq -r '.worker' | sort | uniq -c
# Should show roughly equal distribution across 4 workers
# Example: "25 optimizer-1", "25 optimizer-2", "25 optimizer-3", "25 optimizer-4"
```

---

#### T8: Worker Loop

| Field | Value |
|-------|-------|
| **What is tested** | Worker claims task → optimizes → saves → claims next → repeats |
| **Expected result** | Worker processes multiple tasks sequentially |
| **Status** | 🔲 PENDING |

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Find work | Worker calls `get_pending_tasks()` | See pending tasks query in logs | Worker starts without checking |
| 2. Claim before work | `claim_task()` before optimization | Claim happens before kernel generation | Kernel generated without claim |
| 3. Optimization | Kernel generated and evaluated | `eval_kernel()` called with result | No eval or error |
| 4. Save result | `save_benchmark_result()` called | See "Saved result" message | No save or save error |
| 5. Completion marker | `best_result.json` created | File exists after task done | Only `.in_progress` remains |
| 6. Loop continues | Worker claims next task | Multiple tasks processed by same worker | Worker stops after first task |
| 7. Graceful exit | Worker stops when done | "No pending tasks" or "Worker complete" | Crashes or hangs |

**Observation method:**
Watch a single worker's log output as it processes multiple tasks. You should see a repeating pattern:
```
[Worker-1] Claiming task: 19_ReLU
[Worker-1] Generating kernel...
[Worker-1] Evaluating... speedup: 1.25x
[Worker-1] Saved result
[Worker-1] Claiming task: 20_LeakyReLU
...
[Worker-1] No pending tasks. Worker complete.
```

---

#### T11: Structured JSON Output

| Field | Value |
|-------|-------|
| **What is tested** | Agent outputs are valid JSON for reliable parsing |
| **Expected result** | All skill/worker outputs parse as JSON |
| **Status** | 🔲 PENDING |

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Skill JSON | Status updates in JSON | `{"status": "running", "completed": 10, ...}` | Plain text like "10 tasks done" |
| 2. Worker JSON | Results in JSON | `{"task": "19_ReLU", "speedup": 1.3, ...}` | Plain text results |
| 3. No mixed output | JSON not mixed with prose | Clean JSON blocks | "Here's the result: {json}" |

**Test method:**
Capture agent output and attempt to parse:
```python
import json
# If this fails, T11 fails
result = json.loads(agent_output)
```

---

#### T12: Strategy Sub-Agents

| Field | Value |
|-------|-------|
| **What is tested** | Worker spawns 3 strategy sub-agents in parallel per iteration |
| **Expected result** | 3 different strategies tried, best one selected |
| **Status** | 🔲 PENDING |

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Strategy spawn | Worker spawns sub-agents | "Spawning 3 strategy agents" or 3 Task tool calls | Only 1 strategy or sequential |
| 2. Different approaches | Each strategy is distinct | See "vectorized", "tiled", "fused" etc. | All 3 identical |
| 3. Parallel eval | 3 `eval_kernel()` calls | 3 eval results returned | Only 1 eval |
| 4. Best selected | Highest speedup chosen | Worker says "Selected strategy X with 1.3x" | Random or first selection |

**Observation:**
```
[Worker] Spawning 3 strategy sub-agents...
[Strategy-A] Vectorized loads approach: speedup 1.15x
[Strategy-B] Tiled shared memory approach: speedup 1.32x  ← Best
[Strategy-C] Fused operations approach: speedup 1.21x
[Worker] Selected Strategy-B with speedup 1.32x
```

---

#### T13: GPU Semaphore

| Field | Value |
|-------|-------|
| **What is tested** | MCP server limits concurrent `eval_kernel()` calls to GPU count |
| **Expected result** | With N GPUs, max N evals run simultaneously |
| **Status** | 🔲 PENDING |

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Semaphore init | Server startup log | "Adaptive semaphore: N slots" | No semaphore message |
| 2. Queuing | Many evals queue up | Evals complete in batches of N | All evals run at once |
| 3. No OOM | GPU memory stable | All evals complete | CUDA OOM errors |

**Test method (requires MCP server logs):**
```bash
# Start kbEval server and watch logs
python kbEvalServer.py --device 0,1  # 2 GPUs

# In server logs, look for:
# "Adaptive semaphore: 2 slots"

# During heavy load, you should see:
# "Eval queued (2/2 slots in use)"
# "Eval started (1/2 slots in use)"
```

---

#### T14: Best-of-3 Selection

| Field | Value |
|-------|-------|
| **What is tested** | Worker correctly selects highest speedup from 3 strategy results |
| **Expected result** | Best kernel saved, others discarded |
| **Status** | 🔲 PENDING |

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Multiple results | 3 different speedups returned | See 3 distinct speedup values | Only 1 value or all same |
| 2. Max selected | Highest speedup chosen | Worker reports selecting max | Selects non-max value |
| 3. Saved correctly | `best_result.json` has max | File contains highest speedup | File contains lower speedup |

**Post-test verification:**
```bash
# Check a completed task
cat ~/.inference/claude_code_output/test_session/19_ReLU/best_result.json | jq '.speedup'
# Should match the highest of the 3 strategies shown in logs

# Check iteration files to see all attempts
ls ~/.inference/claude_code_output/test_session/19_ReLU/iteration_*.json
cat ~/.inference/claude_code_output/test_session/19_ReLU/iteration_*_eval.json | jq '.speedup'
# best_result.json speedup should be max of these
```

---

### 6.4 Production Test

#### T15: Full Batch

| Field | Value |
|-------|-------|
| **What is tested** | Complete level1 (100 tasks) with 4 workers end-to-end |
| **Expected result** | ≥95% completion rate, results saved correctly |
| **Status** | 🔲 PENDING |

**Prompt:**
```
/kernel-bench level1 --session=prod_full --workers=4
```

**Validation steps:**

| Step | What to look for | Pass if | Fail if |
|------|------------------|---------|---------|
| 1. Session init | 100 tasks initialized | "Initialized session with 100 tasks" | Wrong task count |
| 2. Workers active | 4 workers processing | See 4 worker IDs in logs | Fewer workers or single-threaded |
| 3. Progress updates | Periodic status | "Progress: 50/100" messages | No updates for >5 min |
| 4. Completion rate | ≥95 tasks complete | ≥95 `best_result.json` files | <95 completed |
| 5. Failures logged | Failed tasks recorded | `failures.json` exists for failed tasks | Failures silently dropped |
| 6. Summary accurate | Stats match reality | `summary.json` counts match file counts | Mismatched stats |

**Post-test verification:**
```bash
# 1. Check completion count (should be ≥95)
COMPLETED=$(ls ~/.inference/claude_code_output/prod_full/*/best_result.json 2>/dev/null | wc -l)
echo "Completed: $COMPLETED / 100"
# PASS if ≥95, FAIL if <95

# 2. Check for failures
FAILED=$(ls ~/.inference/claude_code_output/prod_full/*/failures.json 2>/dev/null | wc -l)
echo "Failed: $FAILED"

# 3. Verify total = completed + failed (approximately)
echo "Total accounted: $((COMPLETED + FAILED))"
# Should be close to 100

# 4. Check summary statistics
cat ~/.inference/claude_code_output/prod_full/summary.json | jq '._stats'
# Expected output:
# {
#   "total_tasks": 100,
#   "total_iterations": ...,
#   "success_count": ≥95,
#   "avg_speedup": 1.xx,
#   "last_updated": "..."
# }

# 5. Verify summary matches file count
SUMMARY_COUNT=$(cat ~/.inference/claude_code_output/prod_full/summary.json | jq '._stats.success_count')
echo "Summary says: $SUMMARY_COUNT, Files show: $COMPLETED"
# These should match

# 6. Check average speedup is reasonable
AVG_SPEEDUP=$(cat ~/.inference/claude_code_output/prod_full/summary.json | jq '._stats.avg_speedup')
echo "Average speedup: ${AVG_SPEEDUP}x"
# Should be >1.0 (faster than PyTorch baseline)

# 7. Spot check a few results
for task in 19_ReLU 23_Softmax 88_MinGPTNewGelu; do
  echo "--- $task ---"
  cat ~/.inference/claude_code_output/prod_full/$task/best_result.json 2>/dev/null | jq '{speedup, completed_at}'
done
```

**Success criteria:**
- ✓ ≥95/100 tasks have `best_result.json`
- ✓ Average speedup > 1.0x
- ✓ No unaccounted tasks (completed + failed ≈ 100)
- ✓ Summary stats match actual file counts
- ✓ Run completed without crashes

---

### 6.5 Test Summary

| Category | Tests | Passed | Pending |
|----------|-------|--------|---------|
| **Automated (Unit)** | T5, T6, T7, T9, T10, Task Filtering | 6 | 0 |
| **Manual (CLI)** | T1, T2, T3 | 0 | 3 |
| **Integration (Agents)** | T4, T8, T11, T12, T13, T14 | 0 | 6 |
| **Production** | T15 | 0 | 1 |
| **Total** | 16 | **6** | **10** |

**Run automated tests:**
```bash
python3 test_phase2_comprehensive.py   # T5, T6, T7, T9, T10 + extras
python3 test_phase2_resume_flow.py     # Detailed resume scenario
python3 test_phase2_session.py         # Basic session operations
```

---

## Appendix: Reliability Considerations

Based on multi-agent LLM research ([arXiv:2503.13657](https://arxiv.org/html/2503.13657v1)):

| Issue | Mitigation |
|-------|------------|
| LLM non-determinism (~15% variance) | MCP tools for atomic ops, structured JSON |
| Context loss in handoffs | File-based state, workflow-trace.md |
| Premature termination | Skill monitoring loop, explicit completion markers |
| Task disobedience | Clear output format, explicit constraints |

**This design works well for:** Kernel bench (independent, idempotent tasks) ✓
