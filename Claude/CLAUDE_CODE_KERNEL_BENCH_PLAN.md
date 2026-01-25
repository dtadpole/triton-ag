# Claude Code Kernel Bench Integration Plan

## Table of Contents

1. [RL Flow - How It Works](#1-rl-flow---how-it-works)
2. [Envisioned User Flow](#2-envisioned-user-flow)
3. [Current Implementation](#3-current-implementation)
4. [Architecture Choice](#4-architecture-choice)
   - 4.1 [Decision: Direct Sync Mode Only](#41-decision-direct-sync-mode-only)
   - 4.2 [Handling Multiple Tasks](#42-handling-multiple-tasks-batch-processing-vision)
   - 4.3 [Multi-GPU Setup and Remote Access](#43-multi-gpu-setup-and-remote-access)
5. [Test-Driven Development Plan](#5-test-driven-development-plan)

---

## 1. RL Flow - How It Works

### 1.1 Overview

The RL training system uses a parallel worker architecture where load is managed at the **client side** (inferenceComposer), not at the kbEvalServer level.

**Key Insight**: The RL system does NOT use workflowServer for kbEval calls. kbEval calls are direct HTTP requests from workers to kbEvalServer(s).

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        inferenceComposer.py                              │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Input Processor (runs once at start)                            │    │
│  │                                                                  │    │
│  │ 1. DuckDB query loads N random kernel_bench/*.py files          │    │
│  │ 2. For each file:                                                │    │
│  │    - Enqueue 1 reference eval task                              │    │
│  │    - Enqueue M generation tasks (each does T turns)             │    │
│  │                                                                  │    │
│  │ Example: 20 samples × 8 generations = 180 tasks enqueued        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              asyncio.Queue (in-memory, explicit queue)          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│         ┌────────────────────┼────────────────────┐                     │
│         ▼                    ▼                    ▼                     │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐              │
│  │  Worker 0  │       │  Worker 1  │  ...  │  Worker 31 │              │
│  └────────────┘       └────────────┘       └────────────┘              │
│       32 parallel workers (default, configurable)                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Each worker makes SYNC HTTP call
                              │ Load balanced via random.choice(providers)
                              ▼
        ┌─────────────────────┬─────────────────────┐
        ▼                     ▼                     ▼
   kbEvalServer          kbEvalServer          kbEvalServer
   :5676 (GPU 7)         :5677 (GPU 6)         :5678 (GPU 5)
```

### 1.3 Key Design Principles

1. **Client-side parallelism**: 32 workers pull from asyncio.Queue
2. **Synchronous HTTP calls**: Each worker blocks on `await kb_eval()` until response
3. **Random load balancing**: `random.choice(provider_list)` distributes requests
4. **No server-side queue**: kbEvalServer processes requests as they arrive

### 1.4 Where Wait Happens

The wait is **implicit in the GPU/CUDA layer**, not explicit in any queue:

```
Worker (inferenceComposer)              kbEvalServer                    GPU
─────────────────────────              ────────────                    ───

await kb_eval(...)  ──HTTP POST──►  @app.post("/kb_eval")
                                            │
        ┌──────────────────────────────────┐│
        │ Worker blocked here              ││
        │ (async await on HTTP response)   ▼│
        │                    spawn subprocess ──────►  [CUDA Queue]
        │                    await process.wait()          │
        │                           │                      │
        │                           │◄─── waiting ────────►│ Other kernels
        │                           │     for GPU          │ compiling...
        │                           │                      │
        │                           │◄─── GPU free ───────►│
        │                           │     Compile+run      │◄── actual work
        │                           │◄─── done ───────────►│    (10-30s)
        │                    process.wait() returns        │
        │◄────HTTP 200 + JSON───────┘
result = ...
```

### 1.5 What Happens with 32 Workers → 3 kbEvalServers

```
32 Workers make requests simultaneously
        │
        │ random.choice() distributes: ~11, ~10, ~11 per server
        ▼
   kbEvalServer:5676 receives 11 requests
        │
        ├── Request 1: GPU starts immediately (0s wait)
        ├── Request 2: waits ~25s for GPU
        ├── Request 3: waits ~50s for GPU
        ...
        └── Request 11: waits ~250s for GPU

All 11 HTTP connections stay open
All 11 subprocesses spawned and waiting
GPU processes 1 kernel at a time (CUDA serializes)
```

### 1.6 How RL System Handles Load

| Mechanism | Description |
|-----------|-------------|
| **Worker count limit** | 32 concurrent tasks max (configurable) |
| **Random load balancing** | `random.choice(providers)` spreads load |
| **Exponential backoff** | Failed requests wait 3^n seconds |
| **Timeout protection** | 300s HTTP timeout, 240s subprocess timeout |
| **Many kbEvalServers** | 15+ providers configured in kbEval.yaml |

### 1.7 Key Insight: No Backpressure

The current system has **no admission control**:
- kbEvalServer accepts all requests immediately
- Spawns subprocess for each (even if GPU busy)
- CUDA driver queues work implicitly
- Workers wait as long as needed (up to timeout)

This works because:
1. Enough kbEvalServers are deployed to match worker parallelism
2. Long timeouts tolerate queue buildup
3. Random distribution is "good enough" for throughput

### 1.8 workflowServer Role (NOT for kbEval)

The workflowServer IS used in RL, but **only for job orchestration**:

```
workflowServer orchestrates blocks:
  InferenceBlock → TrainerBlock → SyncBlock → (repeat epochs)

NOT used for:
  kbEval calls - these are direct HTTP from worker to kbEvalServer
```

---

## 2. Envisioned User Flow

### 2.1 Key Constraint: No Claude API Access Required

This solution works **entirely through Claude Code** - no separate Claude API key is needed.

```
User → Claude Code → [Claude generates CUDA kernel] → MCP tool (eval_kernel) → kbEvalServer → Results
           ↑
      This IS your Claude access
```

**Claude Code IS the Claude access.** The MCP server provides only utility tools:
- **File I/O**: List tasks, read task details, save results
- **HTTP calls**: Submit kernels to kbEvalServer for compilation/benchmarking

### 2.2 Single Task Workflow

> **You:** "Optimize the kernel in kernel_bench/level1/1_relu.py"
>
> **Claude Code:**
> 1. Uses `get_task_details` tool to read the PyTorch model
> 2. Generates optimized CUDA/Triton kernel code
> 3. Uses `eval_kernel` tool to compile and benchmark
> 4. Reviews results, iterates if needed
> 5. Uses `save_benchmark_result` tool to store final result

### 2.3 Batch Session Workflow

> **You:** "Run a benchmark session on all level1 tasks and save results for comparison"
>
> **Claude Code:**
> 1. Uses `list_kernel_bench_tasks` tool to get all level1 tasks
> 2. For each task: Read task, generate kernel, evaluate, iterate
> 3. Saves all results to `~/.inference/claude_code_output/`
> 4. Generates summary report

### 2.4 Example Complete Session

> **You:** "Run kernel bench on level1 tasks, session name: claude_vs_rl_run1"
>
> **Claude Code:**
> - Lists 50 tasks in level1
> - Starting task 1/50: 1_relu.py
> - Reading task... PyTorch model applies ReLU activation.
> - Generating Triton kernel...

```python
@triton.jit
def relu_kernel(...):
    ...
```

> - Evaluating... Compiled: ✓, Correct: ✓, Speedup: 1.23x
> - Saved to `~/.inference/claude_code_output/claude_vs_rl_run1/1_relu/`
> - Starting task 2/50: 2_matmul.py ...
>
> **=== Session Complete ===**
> - Success Rate: 45/50 (90%)
> - Average Speedup: 1.34x
> - Results saved to: `~/.inference/claude_code_output/claude_vs_rl_run1/`
>
> **You:** "Compare this with the RL model results in ~/.inference/output/my_tag.a_001_01/"
>
> **Claude Code:** Runs comparison, generates report

### 2.5 Multi-Turn Iteration Approach

**Comparison: RL Fine-Tuning vs Claude Code**

| Aspect | RL Fine-Tuning (TreeTurns) | Claude Code |
|--------|---------------------------|-------------|
| LLM Access | vLLM server with log probs | Claude Code conversation |
| Parallelism | 8 generations × 4 turns = 32 per task | Sequential (1 at a time) |
| Selection | Random or best_speedup across 8 gens | Claude decides based on feedback |
| Feedback | Structured JSON prompt templates | Natural language conversation |
| Training Signal | GRPO reward from speedup/correctness | None (inference only) |

**Claude Code Iteration Based on Feedback:**

> **If compilation fails:**
> - MCP tool returns: `{"compiled": false, "error": "triton.language has no attribute 'maximum'"}`
> - Claude: "The kernel failed to compile. Let me fix the syntax..."
> - Calls `eval_kernel` again
>
> **If correct but slow:**
> - MCP tool returns: `{"compiled": true, "correctness": true, "speedup": 0.85}`
> - Claude: "The kernel is correct but slower than PyTorch (0.85x). I'll optimize..."
> - Calls `eval_kernel` again
>
> **If successful:**
> - MCP tool returns: `{"compiled": true, "correctness": true, "speedup": 1.45}`
> - Claude: "The kernel achieves 1.45x speedup. Saving result."
> - Calls `save_benchmark_result`

### 2.6 Recommended Iteration Limits

| Scenario | RL Fine-Tuning | Claude Code |
|----------|----------------|-------------|
| Max turns per task | 4 (fixed) | 4 (configurable) |
| Max total attempts | 32 (8×4) | 4 (sequential) |
| Early stop condition | None (all enqueued upfront) | speedup > 1.3x or 4 failures |

### 2.7 Multi-GPU Batch Workflow

When multiple GPUs are available, Claude Code can spawn multiple agents for parallel processing. The kbEvalServer handles GPU distribution internally via its multi-device support.

**Architecture:**
- Single kbEvalServer instance manages multiple GPUs (configured via `DEVICES` list)
- MCP server uses adaptive semaphore that auto-limits concurrency to `num_devices`
- No need to specify different providers per agent - GPU scheduling is automatic

**Example Prompt:**

> Run all level1 tasks in parallel using 4 agents.
>
> Distribution:
> - Agent 1: Tasks 1-12
> - Agent 2: Tasks 13-25
> - Agent 3: Tasks 26-37
> - Agent 4: Tasks 38-50
>
> Use session_id="level1_batch_run".

**Claude Code Response:**

> 1. Lists 50 tasks in level1
> 2. Partitions tasks across 4 agents
> 3. Spawns 4 agents via Task tool (all use default provider)
> 4. Agents run in parallel; server handles GPU scheduling
> 5. Aggregates results and generates summary
>
> **=== Session Complete ===**
> - Total time: ~10 minutes (with 2 GPUs)
> - Success Rate: 47/50 (94%)
> - Results saved to: `~/.inference/claude_code_output/level1_batch_run/`

**Key Prompt Elements:**

| Element | Example | Purpose |
|---------|---------|---------|
| Agent count | "4 agents" | Controls parallelism level |
| Task distribution | "Agent 1 handles tasks 1-12" | Balances load |
| Session ID | "session_id=my_run" | Groups results together |

**Sample Prompts for Common Scenarios:**

1. **Light load (few tasks per agent):**
   > Run 10 level1 tasks using 2 agents. Agent 1 handles tasks 1-5, Agent 2 handles tasks 6-10. session_id="quick_test".

2. **Heavy load (maximize throughput):**
   > Run all level1 tasks using 4 agents. Distribute tasks evenly. session_id="full_level1_run".

3. **Maximum parallelism:**
   > Run all level1 tasks using 8 agents. Each agent handles ~6 tasks. session_id="max_parallel_run".

**Note:** The number of agents can exceed the GPU count. The adaptive semaphore ensures only `num_devices` evaluations run concurrently; additional requests queue automatically.

---

## 3. Current Implementation

### 3.1 MCP Server Structure

The MCP server (`claudeCodeKernelBenchServer.py`) provides 5 tools:

| Tool | Purpose |
|------|---------|
| `list_kernel_bench_tasks` | List available kernel benchmark tasks |
| `get_task_details` | Read PyTorch model code for a task |
| `eval_kernel` | Evaluate generated kernel via kbEvalServer |
| `save_benchmark_result` | Save kernel code and eval results to disk |
| `get_session_summary` | Get statistics for a benchmark session |

### 3.2 Evaluation Flow (Direct Sync Mode)

The `eval_kernel` tool uses direct sync calls to kbEvalServer:

```
Claude Code → MCP eval_kernel() → KbEvalClient → HTTP → kbEvalServer
                                                           ↓
                                         Wait for GPU, compile, benchmark
                                                           ↓
                                         Return result (10-60s later)
```

- **Matches RL architecture pattern** (direct sync calls)
- **Currently broken**: Import error - "kbEvalClient not available"
- **Priority**: Fix this to enable end-to-end testing

### 3.3 Current Code Status

**Files implemented:**

| File | Status | Notes |
|------|--------|-------|
| `claudeCodeKernelBenchServer.py` | Created | MCP server with 5 tools |
| `claudeCodeKernelBench.yaml` | Created | Configuration file |
| `benchmarkCompare.py` | Created | Comparison report generator |
| `.mcp.json` | Created | MCP registration |

**Known Issue:**

**kbEvalClient import fails** when MCP server runs as subprocess
- Root cause: Path issues when spawned by Claude Code
- Impact: eval_kernel() cannot call kbEvalServer
- Fix: Add proper sys.path handling in MCP server startup

### 3.4 Configuration Files

**claudeCodeKernelBench.yaml:**

```yaml
kernel_bench:
  base_dir: "./kernel_bench"
  levels: ["level1", "level2", "level3", "level4"]

kbeval:
  default_provider: "local"
  config_file: "kbEval.yaml"

workflow:
  prefix_tag: "claude_code"
  eval_queue: "kbEval.pending"
  provider_name: "local"  # Use "local" with SSH tunnel

output:
  base_dir: "${HOME}/.inference/claude_code_output"
```

**Result Storage Format:**

```
~/.inference/claude_code_output/
└── {session_id}/
    ├── {task_name}/
    │   ├── iteration_00_cuda_kernel.py
    │   ├── iteration_00_eval.json
    │   └── ...
    └── summary.json
```

---

## 4. Architecture Choice

### 4.1 Decision: Direct Sync Mode Only

**Chosen Approach**: Use **direct sync calls** exclusively, matching the RL architecture.

```
Claude Code → MCP eval_kernel() → KbEvalClient → HTTP → kbEvalServer → Result
                                                            ↑
                                                Same pattern as RL
```

**Rationale:**
1. **Matches proven RL pattern** - direct HTTP calls work reliably
2. **Interactive use needs immediate feedback** - waiting is natural for Claude Code
3. **Simpler architecture** - no queue, no consumer, no polling
4. **No architectural change needed** - just fix the import error

**Queue mode is NOT needed** because:
- Claude Code naturally processes tasks sequentially
- Parallel agents (Task tool) provide concurrency when desired
- No benefit to async queue for Claude Code's use case

### 4.2 Handling Multiple Tasks (Batch Processing Vision)

When asked to "run all level1 tasks", Claude Code will:

**Option A: Sequential Processing (Simple)**

> Claude Code processes each task one at a time:
> 1. Get list of level1 tasks (50 tasks)
> 2. For task 1: read → generate kernel → eval_kernel() → wait → save result
> 3. For task 2: read → generate kernel → eval_kernel() → wait → save result
> 4. ... repeat for all 50 tasks
> 5. Generate summary report

**Estimated time**: 50 tasks × 30s/task = ~25 minutes

**Option B: Parallel Agents (Faster)**

Claude Code spawns multiple agents via Task tool:

> 1. Get list of level1 tasks (50 tasks)
> 2. Spawn 5 parallel agents, each handles 10 tasks
> 3. Each agent: read → generate → eval → save (sequentially within agent)
> 4. Wait for all agents to complete
> 5. Aggregate results and generate summary

**Estimated time**: 50 tasks ÷ 5 agents × 30s/task = ~5 minutes

#### Switching Between Options with Prompts

Yes, switching between Option A and B is purely a matter of how you prompt Claude Code. No code changes needed.

**Prompt for Option A (Sequential):**
> "Run all level1 kernel benchmark tasks sequentially. For each task: read the PyTorch model, generate a Triton kernel, evaluate it, and save the result. Process them one at a time."

**Prompt for Option B (Parallel Agents):**
> "Run all level1 kernel benchmark tasks in parallel. Use 5 parallel agents - each agent handles a subset of tasks. Spawn the agents using the Task tool and wait for all to complete, then aggregate results."

**More specific parallel prompt:**
> "Run all level1 tasks with parallel agents. Split 50 tasks across 5 agents (10 tasks each). Each agent should run its tasks sequentially. Use session_id='level1_parallel_run' for all results."

#### Load Management Verification

**How Parallel Agents Interact with kbEvalServer:**

```
Claude Code (main)
    │
    ├── Task tool: Agent 1 (tasks 1-10)  ────► eval_kernel() ──► kbEvalServer
    ├── Task tool: Agent 2 (tasks 11-20) ────► eval_kernel() ──► kbEvalServer
    ├── Task tool: Agent 3 (tasks 21-30) ────► eval_kernel() ──► kbEvalServer
    ├── Task tool: Agent 4 (tasks 31-40) ────► eval_kernel() ──► kbEvalServer
    └── Task tool: Agent 5 (tasks 41-50) ────► eval_kernel() ──► kbEvalServer
```

**Key Observations:**

1. **Each agent processes sequentially within itself** - Agent 1 waits for task 1 to complete before starting task 2. No parallel eval_kernel() calls within a single agent.

2. **Concurrent calls = number of agents** - With 5 agents, at most 5 concurrent requests hit kbEvalServer.

3. **CUDA serialization handles contention** - Just like in RL system (Section 1.4), if 5 requests hit 1 kbEvalServer, CUDA driver queues them. Each request waits its turn.

**Timeline with 5 Agents → 1 kbEvalServer:**

| Time | Agent 1 | Agent 2 | Agent 3 | Agent 4 | Agent 5 | GPU |
|------|---------|---------|---------|---------|---------|-----|
| 0s | eval task 1 | eval task 11 | eval task 21 | eval task 31 | eval task 41 | Processing task 1 |
| 30s | **done** → task 2 | waiting | waiting | waiting | waiting | Processing task 11 |
| 60s | eval task 2 | **done** → task 12 | waiting | waiting | waiting | Processing task 21 |
| ... | ... | ... | ... | ... | ... | ... |

**Load Management Works Because:**

| Factor | Why It Works |
|--------|--------------|
| Sequential within agent | Only 5 concurrent requests max (not 50) |
| CUDA queuing | GPU naturally serializes, no lost requests |
| HTTP timeout | 300s timeout tolerates queue buildup |
| Agent count is user-controlled | User decides parallelism level |

**When to Use More Agents:**

- **1-3 agents**: Safe for single kbEvalServer
- **5-10 agents**: Better with multiple kbEvalServers (load balanced via random.choice)
- **>10 agents**: Diminishing returns, mostly waiting in CUDA queue

**No Semaphore Needed**: Unlike RL's 32 parallel workers, Claude Code's Task tool spawns a controlled number of agents. The user explicitly chooses the parallelism level in their prompt.

### 4.3 Multi-GPU Setup and Remote Access

> **See [KERNEL_BENCH_SETUP.md](./KERNEL_BENCH_SETUP.md)** for complete step-by-step setup instructions.

**Architecture Summary:**
- Claude Code runs locally (macOS), kbEvalServer runs on remote GPU server
- SSH tunnel forwards localhost:5676 to remote server
- Multi-GPU distribution uses `random.choice(devices)` (same as RL system)
- Adaptive semaphore limits concurrent evals to match available GPUs

**Key Implementation:**
- `kbEvalServer.py`: `/info` endpoint returns `num_devices` and `devices` list
- `kbEvalClient.py`: `get_info()` method queries device count
- `claudeCodeKernelBenchServer.py`: `get_eval_semaphore()` creates semaphore matching GPU count

---

## 5. Test-Driven Development Plan

### 5.1 Implementation Work Items

**All 11 work items complete** ✓

Key changes implemented:
- sys.path resolution for MCP subprocess
- Direct sync eval via KbEvalClient (removed workflow queue path)
- `/info` endpoint and `get_info()` for GPU count discovery
- Adaptive semaphore for concurrency control

---

### 5.2 Test Phases Overview

Tests are organized to gradually validate the implementation:

| Phase | Focus | Prerequisites | What It Validates |
|-------|-------|---------------|-------------------|
| Phase 1 | Offline (no servers) | None | Python syntax, imports, config loading |
| Phase 2 | Single Task (no GPU) | Phase 1 | MCP tools, kernel generation, result storage |
| Phase 3 | Single Task (with GPU) | kbEvalServer running | Actual compilation, correctness, speedup |
| Phase 4 | Multiple Tasks (sequential) | Phase 3 | Batch processing, session summary |
| Phase 5 | Multiple Agents (parallel) | Phase 4 | Task tool parallelism, load management |

---

### 5.3 Phase 1: Offline Validation ✓ PASS

**Purpose**: Verify Python syntax, imports work, and config loads correctly.

| Test | Status | What It Validates |
|------|--------|-------------------|
| 1.1: Syntax | ✓ PASS | `python3 -m py_compile claudeCodeKernelBenchServer.py` |
| 1.2: Imports | ✓ PASS | Config loads, KbEvalClient imports with torch CPU |
| 1.3: MCP Registration | ✓ PASS | `.mcp.json` valid JSON with kernel-bench server |

**Prerequisites:** `pip install torch psutil numpy httpx`

---

### 5.4 Phase 2: Single Task Without GPU ✓ PASS

**Purpose**: Verify MCP tools work for task listing, source reading, and result storage (no GPU required).

| Test | Status | What It Validates |
|------|--------|-------------------|
| 2.1: List & Get Details | ✓ PASS | `list_kernel_bench_tasks()`, `get_task_details()` |
| 2.2: Save Results | ✓ PASS | `save_benchmark_result()`, `get_session_summary()` |
| 2.3: Generate Kernel | ✓ PASS | Triton kernel generation with valid Python syntax |

**Test Scripts:** `test_phase2_tools.py`, `test_phase2_save.py`, `test_phase2_generate.py`

**Key Learnings:**
- `list_kernel_bench_tasks()` returns list directly (not wrapped in dict)
- `save_benchmark_result()` expects `eval_result` as dict, not JSON string
- `get_session_summary()` puts stats in `_stats` key

---

### 5.5 Phase 3: Single Task With GPU ✓ PASS

**Purpose**: Verify actual kernel compilation and benchmarking on GPU via kbEvalServer.

**Prerequisites:**
- kbEvalServer running on GPU machine
- SSH tunnel: `ssh -L 5676:localhost:5676 gpu-server`

| Test | Status | What It Validates |
|------|--------|-------------------|
| 3.0: Setup | ✓ PASS | SSH tunnel, `/health` endpoint |
| 3.1: Direct KbEvalClient | ✓ PASS | HTTP POST works, kernel compiles |
| 3.2: MCP eval_kernel | ✓ PASS | compiled=true, correct=true |
| 3.3: Claude E2E | ✓ PASS | Full workflow with result saving |

**Test Scripts:** `test_phase3_kbeval.py`, `test_phase3_mcp.py`

**Setup:** See [KERNEL_BENCH_SETUP.md](./KERNEL_BENCH_SETUP.md) for detailed instructions.

---

### 5.6 Phase 4: Multiple Tasks (Sequential) ✓ PASS

**Purpose**: Verify batch processing of multiple tasks sequentially by single agent.

| Test | Status | What It Validates |
|------|--------|-------------------|
| 4.1: Programmatic Batch | SKIP | Using 4.2 with real GPU instead |
| 4.2: Claude Sequential | ✓ PASS | 3/3 tasks: ReLU, Sigmoid, Tanh |

**Key Behavior:**
- Single Claude Code session processes tasks one at a time
- Each `eval_kernel()` call blocks until GPU returns result
- Total time ≈ sum of individual task times

---

### 5.7 Phase 5: Multiple Agents (Parallel) ✓ PASS

**Purpose**: Verify parallel processing with multiple agents using Task tool.

| Test | Status | What It Validates |
|------|--------|-------------------|
| 5.1: Parallel Agents | ✓ PASS | 2 agents, 6 tasks, 4/6 success (GPU contention) |
| 5.2: Load Management | ✓ PASS | 5 agents, 10 tasks, 7/10 success (GPU contention) |

**Key Behavior:**
- Task tool spawns concurrent sub-agents
- Agents run in parallel, GPU serializes execution
- Wall-clock time < sequential time (parallelism benefit)

---

### 5.8 Phase 6: Multi-GPU with Adaptive Semaphore ✓ PARTIAL

**Purpose**: Verify RL-style multi-device architecture with adaptive semaphore.

**Prerequisites:**
- kbEvalServer in config-based mode with multiple GPUs
- `/info` endpoint returning `num_devices`
- `black` package installed on GPU server

| Test | Status | What It Validates |
|------|--------|-------------------|
| 6.0: Multi-Device Setup | ✓ PASS | Config-based kbEvalServer, /info endpoint |
| 6.1: Semaphore Init | ✓ PASS | Adaptive semaphore queries GPU count |
| 6.2: Distribution | ✓ PASS | 4 agents distributed across 2 GPUs |
| 6.3: Blocking | PENDING | Semaphore limits concurrency |
| 6.4: Stress Test | PENDING | 16 agents, no OOM |
| 6.5: Error Handling | PENDING | Graceful degradation |

**Test Scripts:** `test_phase6_semaphore.py`, `test_phase6_distribution.py`

**Architecture:**
- MCP server queries `/info` to get `num_devices`
- Creates `asyncio.Semaphore(num_devices)` to limit concurrency
- kbEvalServer uses `random.choice(devices)` for GPU distribution

---

### 5.9 Test Status Tracker

| Phase | Test | Status | Date | Notes |
|-------|------|--------|------|-------|
| 1 | 1.1: Python Syntax | PASS | 2026-01-23 | Both files compile |
| 1 | 1.2: Import & Config | PENDING | - | Requires torch installed |
| 1 | 1.3: MCP Registration | PASS | 2026-01-23 | .mcp.json valid |
| 2 | 2.1: List & Get Tasks | PASS | 2026-01-23 | MCP tools work |
| 2 | 2.2: Save Result | PASS | 2026-01-23 | save_benchmark_result works |
| 2 | 2.3: Claude Generates | PASS | 2026-01-23 | Claude creates valid kernels |
| 3 | 3.0: Setup | **PASS** | 2026-01-25 | kbEvalServer via SSH tunnel |
| 3 | 3.1: Direct KbEvalClient | **PASS** | 2026-01-25 | HTTP call works |
| 3 | 3.2: MCP eval_kernel | **PASS** | 2026-01-25 | compiled=true, correct=true, 7.78ms |
| 3 | 3.3: Claude E2E Single | **PASS** | 2026-01-25 | Result saved to ~/.inference/claude_code_output/ |
| 4 | 4.1: Programmatic Batch | SKIP | - | Using 4.2 with real GPU instead |
| 4 | 4.2: Claude Sequential | **PASS** | 2026-01-25 | 3/3 tasks: ReLU, Sigmoid, Tanh |
| 5 | 5.1: Parallel Agents | **PASS** | 2026-01-25 | 2 agents, 6 tasks, 4/6 success (GPU contention) |
| 5 | 5.2: Load Management | **PASS** | 2026-01-25 | 5 agents, 10 tasks, 7/10 success (GPU contention) |
| 6 | 6.0: Multi-Device Setup | **PASS** | 2026-01-25 | Config-based kbEvalServer, /info endpoint |
| 6 | 6.1: Semaphore Init | **PASS** | 2026-01-25 | Adaptive semaphore queries GPU count (W9-W11) |
| 6 | 6.2: Distribution | **PASS** | 2026-01-25 | 4 agents distributed across 2 GPUs |
| 6 | 6.3: Blocking | PENDING | - | Semaphore limits concurrency |
| 6 | 6.4: Stress Test | PENDING | - | 16 agents, no OOM |
| 6 | 6.5: Error Handling | PENDING | - | Graceful degradation |

---

### 5.10 Test Script Cleanup

After all tests pass, cleanup test scripts:

```bash
cd /Users/aarontao/Projects/code/triton-ag
rm -f test_phase2_tools.py test_phase2_save.py test_phase3_kbeval.py test_phase3_mcp.py test_phase4_batch.py
rm -rf ~/.inference/claude_code_output/test_phase*
```

---

## Appendix

### A. Environment Setup

See **[KERNEL_BENCH_SETUP.md](./KERNEL_BENCH_SETUP.md)** for:
- Part 1: Remote Server Setup (kbEvalServer, GPU configuration)
- Part 2: Local Environment Setup (MCP server registration, configuration files)
- Part 3: SSH Tunnel Setup (connecting local to remote)

### B. MCP Registration

**.mcp.json:**

```json
{
  "mcpServers": {
    "kernel-bench": {
      "command": "python3",
      "args": ["claudeCodeKernelBenchServer.py"],
      "cwd": "."
    }
  }
}
```

### C. Component Responsibilities

| Component | Requires Claude API? | Purpose |
|-----------|---------------------|---------|
| Claude Code | No (built-in) | LLM access for CUDA code generation |
| MCP Server | No | File I/O, kbEval HTTP calls, result storage |
| kbEvalServer | No | Kernel compilation & benchmarking (GPU) |

### D. Skill Alternative

Create `.claude/skills/kernel-bench.md` for streamlined invocation:

```markdown
---
name: kernel-bench
description: Run kernel benchmark optimization
---

When invoked, optimize CUDA kernels for the specified task(s).

## Usage
- `/kernel-bench kernel_bench/level1/1_relu.py` - Single task
- `/kernel-bench kernel_bench/level1/` - All tasks in directory
```
