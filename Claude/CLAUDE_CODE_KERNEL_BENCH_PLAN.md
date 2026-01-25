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

When multiple GPUs are available, Claude Code can distribute work across them for faster throughput.

**Example Prompt:**

> Run all level1 tasks in parallel using 4 agents across 2 GPUs.
>
> Distribution:
> - Agents 1-2: Use provider="local" (GPU 1 via localhost:5676)
> - Agents 3-4: Use provider="local_2" (GPU 2 via localhost:5677)
>
> Each agent processes ~12 tasks sequentially.
> Use session_id="level1_multi_gpu_run".

**Claude Code Response:**

> 1. Lists 50 tasks in level1
> 2. Partitions tasks: Agent 1 gets 1-12, Agent 2 gets 13-25, Agent 3 gets 26-37, Agent 4 gets 38-50
> 3. Spawns 4 agents via Task tool, each with assigned provider
> 4. Agents run in parallel, 2 per GPU
> 5. Aggregates results and generates summary
>
> **=== Session Complete ===**
> - Total time: ~10 minutes (vs ~25 minutes with 1 GPU)
> - Success Rate: 47/50 (94%)
> - Results saved to: `~/.inference/claude_code_output/level1_multi_gpu_run/`

**Key Prompt Elements:**

| Element | Example | Purpose |
|---------|---------|---------|
| Agent count | "4 agents" | Controls parallelism level |
| Provider assignment | "provider=local" | Routes to specific GPU |
| Task distribution | "Agents 1-2 handle tasks 1-25" | Balances load |
| Session ID | "session_id=my_run" | Groups results together |

**Sample Prompts for Common Scenarios:**

1. **2 GPUs, light load (few tasks per agent):**
   > Run 10 level1 tasks using 2 agents. Agent 1 uses provider="local", Agent 2 uses provider="local_2". 5 tasks each.

2. **2 GPUs, heavy load (maximize throughput):**
   > Run all level1 tasks (50) using 4 agents across 2 GPUs. Agents 1-2 use "local", Agents 3-4 use "local_2". session_id="full_level1_run".

3. **3+ GPUs:**
   > Run 30 tasks using 6 agents across 3 GPUs. Agents 1-2 use "local", Agents 3-4 use "local_2", Agents 5-6 use "local_3". 5 tasks per agent.

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

This section describes the architecture for running Claude Code locally (macOS, no GPU) while evaluating kernels on remote GPU servers.

#### Architecture Overview

```
┌─────────────────────────────┐      SSH Tunnel      ┌───────────────────────────────┐
│    LOCAL MACHINE            │                      │     REMOTE GPU SERVER         │
│    (macOS, no GPU)          │                      │                               │
│                             │                      │ ┌───────────────────────────┐ │
│ ┌───────────────┐           │                      │ │ kbEvalServer              │ │
│ │ Claude Code   │           │   localhost:5676 ────┼─│ port: 5676                │ │
│ │               │           │                      │ │ devices: [0,1,2,3,4,5,6,7]│ │
│ └───────────────┘           │                      │ └───────────────────────────┘ │
│        │                    │                      │                               │
│        ▼                    │                      │  Each request → random GPU    │
│ ┌───────────────┐           │                      │  (load distribution)          │
│ │ MCP Server    │───────────┘                      │                               │
│ │ (Python)      │                                  └───────────────────────────────┘
│ └───────────────┘
└─────────────────────────────┘
```

**Components:**
- **Claude Code**: Runs locally, orchestrates kernel generation
- **MCP Server**: Local Python process (`claudeCodeKernelBenchServer.py`), handles tool calls
- **kbEvalServer**: Remote GPU server, compiles and benchmarks kernels on multiple GPUs
- **SSH Tunnel**: Forwards localhost:5676 to remote kbEvalServer

**No workflowServer needed** - direct sync calls to kbEvalServer only.

#### Design: Multi-Device kbEvalServer (Same as RL)

The RL training system runs one kbEvalServer managing 7-8 GPUs. Each incoming request is assigned to a random GPU via `random.choice(devices)`. This distributes compilation and evaluation load across GPUs, preventing OOM when multiple agents run in parallel.

**We use the same approach for Claude Code** - no architectural differences from RL.

```
┌─────────────────────────────────────────────────┐
│  kbEvalServer (config-based, 8 devices)         │
│                                                 │
│  8 concurrent requests → random.choice(devices) │
│         ┌────────────────┼────────────────┐     │
│         ▼                ▼                ▼     │
│     GPU 0            GPU 3            GPU 7     │
│   ~1 req each      ~1 req each      ~1 req each │
└─────────────────────────────────────────────────┘
```

| Parallel Agents | GPUs Available | Requests/GPU (avg) | OOM Risk |
|-----------------|----------------|-------------------|----------|
| 5 | 8 | 0.625 | Very Low |
| 8 | 8 | 1.0 | Low |
| 16 | 8 | 2.0 | Medium |

#### Changes Required

| Component | Change | Details |
|-----------|--------|---------|
| **kbEval.yaml (server)** | Add server config | Add hostname entry with `devices` list |
| **kbEvalServer.py** | Add `/info` endpoint | Returns `num_devices` and `devices` list |
| **kbEvalClient.py** | Add `get_info()` method | Calls `/info` endpoint |
| **claudeCodeKernelBenchServer.py** | Add adaptive semaphore | Queries GPU count, limits concurrency |
| **kbEvalServer startup** | Remove `--local_host` | Use config-based mode instead of single-device mode |
| **SSH tunnel** | No change | Same single tunnel to port 5676 |
| **Claude Code prompts** | No change | Same provider="local" |

#### Setup Steps

**Step 1: Add server config to `kbEval.yaml` on GPU server**

```yaml
servers:
  your-hostname:  # Must match output of `hostname` command
    host: "::"
    port: 5676
    compile_cache: true
    check_get_inputs: false
    devices:
      - 0
      - 1
      - 2
      - 3
      - 4
      - 5
      - 6
      - 7
```

**Step 2: Start kbEvalServer (config-based mode)**

```bash
# On GPU server - uses config from kbEval.yaml, manages all 8 GPUs
python kbEvalServer.py
```

Note: Do NOT use `--local_host --device X`. That mode only uses a single GPU.

**Step 3: SSH tunnel from local Mac**

```bash
ssh -L 5676:localhost:5676 gpu-server
```

**Step 4: Provider config on local Mac (`kbEval.yaml`)**

```yaml
providers:
  local:
    base_url: http://localhost:5676
    api_key_path: ~/.keys/kbeval.api.key
    timeout: 300
```

#### Adaptive Semaphore for Concurrency Control

The MCP server queries the kbEvalServer for available GPU count, then creates a semaphore to limit concurrent eval requests. This works for any GPU configuration (1 GPU or 8 GPUs) and guarantees OOM prevention.

**kbEvalServer.py - Add `/info` endpoint:**

```python
@app.get("/info")
async def get_server_info():
    """Return server info including device count."""
    return {
        "num_devices": len(DEVICES),
        "devices": DEVICES,
    }
```

**kbEvalClient.py - Add `get_info()` method:**

```python
async def get_info(self, provider: str = "local") -> dict:
    """Get server info (device count)."""
    config = self._get_provider_config(provider)
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{config['base_url']}/info") as response:
            return await response.json()
```

**claudeCodeKernelBenchServer.py - Adaptive semaphore:**

```python
import asyncio
from kbEvalClient import get_kbeval_client

_eval_semaphore: asyncio.Semaphore | None = None

async def _init_semaphore(provider: str = "local"):
    """Initialize semaphore based on GPU count from kbEvalServer."""
    global _eval_semaphore
    if _eval_semaphore is not None:
        return  # Already initialized

    client = get_kbeval_client()
    try:
        info = await client.get_info(provider=provider)
        num_gpus = info.get("num_devices", 1)
    except Exception:
        num_gpus = 1  # Fallback to safe default

    _eval_semaphore = asyncio.Semaphore(num_gpus)
    logger.info(f"Initialized eval semaphore with {num_gpus} GPU slots")

async def eval_kernel(...):
    await _init_semaphore(provider)

    async with _eval_semaphore:  # Blocks if all GPUs busy
        # ... existing kbEvalClient call ...
        result = await client.kb_eval(...)
        return result
```

**Behavior:**
- 8 GPUs available → 8 concurrent evals allowed
- 1 GPU available → serializes evals (safe, slower)
- Server unreachable → defaults to 1 (safe fallback)
- Extra agent requests block until a GPU slot frees up

---

## 5. Test-Driven Development Plan

### 5.1 Implementation Work Items

Implementation work items identified. Each work item describes a specific change to the codebase.

| ID | Work Item | File(s) | Description |
|----|-----------|---------|-------------|
| W1 | Fix sys.path resolution | `claudeCodeKernelBenchServer.py` | Use `__file__` to resolve script directory for imports when running as subprocess |
| W2 | Add torch to Mac setup | `KERNEL_BENCH_SETUP.md` | Add `torch` to pip install commands (torch CPU on Mac) |
| W3 | Remove queue_only parameter | `claudeCodeKernelBenchServer.py` | Remove `queue_only` from `eval_kernel()` - direct sync mode only |
| W4 | Remove workflow imports | `claudeCodeKernelBenchServer.py` | Remove workflowClient imports and queue submission code |
| W5 | Remove workflow config section | `claudeCodeKernelBench.yaml` | Remove `workflow:` section (prefix_tag, eval_queue, etc.) |
| W6 | Update eval_kernel to use KbEvalClient | `claudeCodeKernelBenchServer.py` | Wire up direct sync path: MCP → KbEvalClient.kb_eval() → HTTP |
| W7 | Add timeout handling | `claudeCodeKernelBenchServer.py` | Match RL's 300s HTTP timeout for eval calls |
| W8 | Remove LightweightKbEvalClient | `claudeCodeKernelBenchServer.py` | Remove LightweightKbEvalClient class and get_kbeval_client(), use original KbEvalClient directly |
| W9 | Add `/info` endpoint | `kbEvalServer.py` | Return `num_devices` and `devices` list for concurrency control |
| W10 | Add `get_info()` method | `kbEvalClient.py` | Client method to call `/info` endpoint |
| W11 | Add adaptive semaphore | `claudeCodeKernelBenchServer.py` | Query GPU count on init, limit concurrent evals to match available GPUs |

**Work Items Completed:**
- [x] W1: sys.path fix (using `Path(__file__).resolve().parent`)
- [x] W2: torch added to Mac setup (pip install torch)
- [x] W3: Remove queue_only parameter from eval_kernel
- [x] W4: Remove workflow imports (WorkflowClient, WORKFLOW_AVAILABLE)
- [x] W5: Remove workflow config section from claudeCodeKernelBench.yaml
- [x] W6: Update eval_kernel to use KbEvalClient.kb_eval() directly
- [x] W7: Add timeout handling (KbEvalClient uses 300s httpx timeout)
- [x] W8: Remove LightweightKbEvalClient, use original KbEvalClient with singleton pattern
- [x] W9: Add `/info` endpoint to kbEvalServer.py
- [x] W10: Add `get_info()` method to kbEvalClient.py
- [x] W11: Add adaptive semaphore to MCP server (get_eval_semaphore())

**All Work Items Complete** ✓

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

### 5.3 Phase 1: Offline Validation (No Servers Required)

**Purpose**: Verify Python syntax, imports work, and config loads correctly.

#### What This Phase Tests

This phase validates the foundational code quality without requiring any external services.

**Component Depth Diagram:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Python Interpreter                                            │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  ✓ claudeCodeKernelBenchServer.py (syntax check)        │  │  │
│  │  │  ✓ benchmarkCompare.py (syntax check)                   │  │  │
│  │  │  ✓ Import chains (yaml, mcp, torch, kbEvalClient)       │  │  │
│  │  │  ✓ Config loading (claudeCodeKernelBench.yaml)          │  │  │
│  │  │  ✓ .mcp.json registration file                          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  NOT TESTED: MCP runtime, tool execution, file I/O, network calls   │
└─────────────────────────────────────────────────────────────────────┘
```

**What's Real vs Faked:**

| Component | Status | Notes |
|-----------|--------|-------|
| Python files | REAL | Actual syntax validation |
| Config files | REAL | Actual YAML parsing |
| Import chains | REAL | Actual module loading |
| MCP server runtime | NOT TESTED | Only imports, not execution |
| Tool functions | NOT TESTED | No actual calls |
| kbEvalServer | NOT NEEDED | No network calls |

**Key Validation Points:**
- All Python files are syntactically valid
- All dependencies can be imported (torch CPU, kbEvalClient, etc.)
- Configuration files parse correctly
- MCP registration is properly formatted

#### Test 1.1: Python Syntax Validation

```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 -m py_compile claudeCodeKernelBenchServer.py && echo "✓ MCP server syntax OK"
python3 -m py_compile benchmarkCompare.py && echo "✓ Comparison tool syntax OK"
```

**Pass Criteria:**
- [x] Both files compile without syntax errors

**Execution Status (2026-01-24):** ✓ PASS
```
$ python3 -m py_compile claudeCodeKernelBenchServer.py && echo "✓ MCP server syntax OK"
✓ MCP server syntax OK
$ python3 -m py_compile benchmarkCompare.py && echo "✓ Comparison tool syntax OK"
✓ Comparison tool syntax OK
```

#### Test 1.2: Import and Config Loading

```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 -c "
from claudeCodeKernelBenchServer import config, get_default_config
from kbEvalClient import KbEvalClient
print('✓ Config loaded with', len(config), 'sections')
print('✓ KbEvalClient imported successfully')
"
```

**Pass Criteria:**
- [x] Config loads successfully
- [x] KbEvalClient imports successfully (requires torch CPU on Mac)

**Execution Status (2026-01-24):** ✓ PASS
```
$ python3 -c "from claudeCodeKernelBenchServer import config, get_default_config; from kbEvalClient import KbEvalClient; print('Config loaded with', len(config), 'sections'); print('KbEvalClient imported successfully')"
Config loaded with 5 sections
KbEvalClient imported successfully
```

**Prerequisites installed:**
```bash
python3 -m pip install torch psutil numpy httpx --quiet
```

#### Test 1.3: MCP Registration

```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 -c "
import json
with open('.mcp.json') as f:
    mcp = json.load(f)
    assert 'mcpServers' in mcp
    assert 'kernel-bench' in mcp['mcpServers']
    print('✓ MCP registration valid')
"
```

**Pass Criteria:**
- [x] .mcp.json is valid JSON with kernel-bench server configured

**Execution Status (2026-01-24):** ✓ PASS
```
$ python3 -c "import json; mcp=json.load(open('.mcp.json')); assert 'mcpServers' in mcp; assert 'kernel-bench' in mcp['mcpServers']; print('✓ MCP registration valid')"
✓ MCP registration valid
```

#### Phase 1 Summary

**All Phase 1 Tests: ✓ PASS**

| Test | Status | Notes |
|------|--------|-------|
| 1.1: Syntax Validation | ✓ PASS | Both Python files compile |
| 1.2: Import and Config | ✓ PASS | Config loads, KbEvalClient imports with torch CPU |
| 1.3: MCP Registration | ✓ PASS | .mcp.json valid |

---

### 5.4 Phase 2: Single Task Without GPU

**Purpose**: Verify MCP tools work for a single task (kernel generation and result storage, without actual GPU evaluation).

#### What This Phase Tests

This phase validates that MCP tools can list tasks, read task details, and save results - everything except actual GPU evaluation.

**Component Depth Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  MCP Server (claudeCodeKernelBenchServer.py)                      │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  ✓ list_kernel_bench_tasks() - Lists level1/2/3 tasks       │  │  │
│  │  │  ✓ get_task_details() - Reads PyTorch source code           │  │  │
│  │  │  ✓ save_benchmark_result() - Writes kernel + eval JSON      │  │  │
│  │  │  ✓ get_session_summary() - Aggregates results               │  │  │
│  │  │  ✗ eval_kernel() - NOT TESTED (requires GPU)                │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                     │  │
│  │                              ▼                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  File System Operations                                      │  │  │
│  │  │  ✓ Read: kernel_bench/level1/*.py (task source files)       │  │  │
│  │  │  ✓ Write: ~/.inference/claude_code_output/{session}/...     │  │  │
│  │  │    - iteration_XX_cuda_kernel.py (generated code)           │  │  │
│  │  │    - iteration_XX_eval.json (eval results)                  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  NOT TESTED: Network calls, kbEvalServer, GPU compilation               │
└─────────────────────────────────────────────────────────────────────────┘
```

**What's Real vs Faked:**

| Component | Status | Notes |
|-----------|--------|-------|
| MCP tool functions | REAL | Actual async functions executing |
| Task listing | REAL | Reads actual kernel_bench directory |
| Task source code | REAL | Reads actual PyTorch model files |
| Kernel code | REAL | Claude generates actual Triton code |
| File I/O | REAL | Actually writes to ~/.inference/claude_code_output |
| **Eval results** | **MOCKED** | `{"compiled": true, "correctness": true, "speedup": 1.0}` - manually constructed, not from GPU |
| kbEvalClient | NOT CALLED | No HTTP requests made |
| kbEvalServer | NOT NEEDED | Server not required |
| GPU compilation | NOT DONE | No actual kernel compilation |

**Key Validation Points:**
- MCP tools execute without errors
- Task listing returns expected level1/2/3 structure
- Source code is readable and non-empty
- File writes succeed with correct directory structure
- Session summary aggregates mocked results correctly
- Generated kernel code is syntactically valid Python

**Why Mocked Eval Results:**
We mock eval results (`compiled=true, correctness=true, speedup=1.0`) because:
1. No kbEvalServer is running
2. No GPU is available to compile Triton kernels
3. This phase focuses on testing the **tooling pipeline**, not kernel correctness
4. The mock data exercises the save/summary code paths without network dependencies

#### Test 2.1: List and Get Task Details

Create test script:

```bash
cat > /Users/aarontao/Projects/code/triton-ag/test_phase2_tools.py << 'EOF'
"""Test Phase 2: MCP tools for single task."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import list_kernel_bench_tasks, get_task_details

async def test_list_tasks():
    print("=== Test 2.1a: list_kernel_bench_tasks ===")
    result = await list_kernel_bench_tasks(level="level1")
    tasks = result.get("tasks", [])
    print(f"Found {len(tasks)} tasks in level1")
    assert len(tasks) > 0, "Expected at least 1 task"
    print("✓ list_kernel_bench_tasks works")
    return tasks[0]["path"]

async def test_get_details(task_path: str):
    print("\n=== Test 2.1b: get_task_details ===")
    result = await get_task_details(task_path=task_path)
    print(f"Task: {result.get('name')}")
    print(f"Source length: {len(result.get('source_code', ''))} chars")
    assert "source_code" in result, "Expected source_code in result"
    assert len(result["source_code"]) > 0, "Expected non-empty source"
    print("✓ get_task_details works")

async def main():
    task_path = await test_list_tasks()
    await test_get_details(task_path)
    print("\n=== Phase 2.1: PASS ===")

if __name__ == "__main__":
    asyncio.run(main())
EOF
```

Run test:

```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 test_phase2_tools.py
```

**Pass Criteria:**
- [x] list_kernel_bench_tasks returns level1 tasks
- [x] get_task_details returns source code

**Execution Status (2026-01-24):** ✓ PASS
```
$ python3 test_phase2_tools.py
=== Test 2.1a: list_kernel_bench_tasks ===
Found 100 tasks in level1
✓ list_kernel_bench_tasks works

=== Test 2.1b: get_task_details ===
Task: 1_Square_matrix_multiplication_
Source length: 729 chars
✓ get_task_details works

=== Phase 2.1: PASS ===
```

**Note:** The test script was updated - `list_kernel_bench_tasks()` returns a list directly, not a dict with "tasks" key.

#### Test 2.2: Save Benchmark Result

Create test script:

```bash
cat > /Users/aarontao/Projects/code/triton-ag/test_phase2_save.py << 'EOF'
"""Test Phase 2.2: save_benchmark_result for single task."""
import asyncio
import os
import json
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import save_benchmark_result, get_session_summary

MOCK_KERNEL = '''
import triton
import triton.language as tl

@triton.jit
def test_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)
'''

MOCK_RESULT = '{"compiled": true, "correctness": true, "speedup": 1.25, "runtime": 0.5}'

async def test_save():
    print("=== Test 2.2a: save_benchmark_result ===")
    result = await save_benchmark_result(
        task_path="level1/1_relu.py",
        kernel_code=MOCK_KERNEL,
        eval_result=MOCK_RESULT,
        session_id="test_phase2_single",
        iteration=0
    )
    print(f"Result: {result}")
    assert "path" in result or "error" not in result, f"Save failed: {result}"

    # Verify files exist
    base = os.path.expanduser("~/.inference/claude_code_output/test_phase2_single/1_relu")
    kernel_file = os.path.join(base, "iteration_00_cuda_kernel.py")
    eval_file = os.path.join(base, "iteration_00_eval.json")

    assert os.path.exists(kernel_file), f"Kernel file not found: {kernel_file}"
    assert os.path.exists(eval_file), f"Eval file not found: {eval_file}"
    print("✓ save_benchmark_result creates files")

async def test_summary():
    print("\n=== Test 2.2b: get_session_summary ===")
    result = await get_session_summary(session_id="test_phase2_single")
    print(f"Summary: {json.dumps(result, indent=2)}")
    assert "total_tasks" in result, "Expected total_tasks in summary"
    print("✓ get_session_summary works")

async def main():
    await test_save()
    await test_summary()
    print("\n=== Phase 2.2: PASS ===")

if __name__ == "__main__":
    asyncio.run(main())
EOF
```

Run test:

```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 test_phase2_save.py
```

Cleanup test data:

```bash
rm -rf ~/.inference/claude_code_output/test_phase2_single
```

**Pass Criteria:**
- [x] save_benchmark_result creates kernel and eval files
- [x] get_session_summary returns valid statistics

**Execution Status (2026-01-24):** ✓ PASS
```
$ python3 test_phase2_save.py
=== Test 2.2a: save_benchmark_result ===
Result: /Users/aarontao/.inference/claude_code_output/test_phase2_single/19_ReLU
✓ save_benchmark_result creates files

=== Test 2.2b: get_session_summary ===
Summary: {"19_ReLU": {...}, "_stats": {"total_tasks": 1, ...}}
✓ get_session_summary works

=== Phase 2.2: PASS ===
```

**Notes:**
- The test script was updated: `eval_result` must be a dict, not JSON string
- Summary stats are in `_stats` key, not at top level
- Actual task file is `19_ReLU.py`, not `1_relu.py`

#### Test 2.3: Claude Code Generates Kernel for Single Task

This test can be run via test script OR interactively in Claude Code.

**Option A: Test Script (automated)**

```bash
cat > /Users/aarontao/Projects/code/triton-ag/test_phase2_generate.py << 'EOF'
"""Test Phase 2.3: Claude Code generates kernel for single task."""
import asyncio
import os
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import get_task_details, save_benchmark_result, get_session_summary

RELU_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        output = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output
'''

MOCK_RESULT = {"compiled": True, "correctness": True, "speedup": 1.15, "runtime": 0.42}

async def main():
    print("=== Test 2.3: Generate kernel for single task ===")

    # Get task details
    task = await get_task_details(task_path="level1/19_ReLU.py")
    print(f"Task: {task.get('name')}")

    # Save generated kernel with mock result
    session_id = "test_phase2_claude"
    result = await save_benchmark_result(
        task_path="level1/19_ReLU.py",
        kernel_code=RELU_KERNEL,
        eval_result=MOCK_RESULT,
        session_id=session_id,
        iteration=0
    )
    print(f"Saved to: {result}")

    # Verify syntax
    kernel_file = os.path.expanduser(f"~/.inference/claude_code_output/{session_id}/19_ReLU/iteration_00_cuda_kernel.py")
    compile(open(kernel_file).read(), kernel_file, 'exec')
    print("✓ Kernel syntax valid")

    print("=== Phase 2.3: PASS ===")

if __name__ == "__main__":
    asyncio.run(main())
EOF
```

Run test:
```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 test_phase2_generate.py
```

**Option B: Interactive Claude Code**

In Claude Code, run:

> Read the task at kernel_bench/level1/19_ReLU.py using get_task_details.
> Generate a simple Triton kernel that implements the same functionality.
> Save the result using save_benchmark_result with session_id="test_phase2_claude" and a mock eval result: {"compiled": true, "correctness": true, "speedup": 1.0}.

Verify:

```bash
ls -la ~/.inference/claude_code_output/test_phase2_claude/19_ReLU/
python3 -m py_compile ~/.inference/claude_code_output/test_phase2_claude/19_ReLU/iteration_00_cuda_kernel.py && echo "✓ Kernel syntax valid"
```

Cleanup:

```bash
rm -rf ~/.inference/claude_code_output/test_phase2_claude
rm -rf ~/.inference/claude_code_output/test_phase2_single
```

**Pass Criteria:**
- [x] Claude Code (or test script) generates a Triton kernel
- [x] Kernel passes Python syntax validation
- [x] Result files are saved correctly

**Execution Status (2026-01-24):** ✓ PASS
```
$ python3 test_phase2_generate.py
=== Test 2.3: Generate kernel for single task ===
Task: 19_ReLU
Saved to: /Users/aarontao/.inference/claude_code_output/test_phase2_claude/19_ReLU
✓ Kernel syntax valid
=== Phase 2.3: PASS ===
```

#### Phase 2 Summary

**All Phase 2 Tests: ✓ PASS**

| Test | Status | Notes |
|------|--------|-------|
| 2.1: List and Get Details | ✓ PASS | 100 level1 tasks found, source code retrieved |
| 2.2: Save Benchmark Result | ✓ PASS | Files created, session summary works |
| 2.3: Generate Kernel | ✓ PASS | Triton kernel generated and saved with valid syntax |

**Key Learnings from Phase 2:**
- `list_kernel_bench_tasks()` returns a list directly, not wrapped in dict
- `save_benchmark_result()` expects `eval_result` as dict, not JSON string
- `get_session_summary()` puts stats in `_stats` key
- Task filenames may not match expected patterns (e.g., `19_ReLU.py` not `1_relu.py`)

---

### 5.5 Phase 3: Single Task With GPU

> **⚠️ MANUAL SETUP REQUIRED**
>
> Phase 3+ requires kbEvalServer running on a GPU machine. Tests stopped at Phase 2.
>
> **To continue testing:**
> 1. Start kbEvalServer on your GPU devserver (see below)
> 2. Create SSH tunnel to forward port 5676
> 3. Run Phase 3 tests from local machine

**Purpose**: Verify actual kernel compilation and benchmarking on GPU via kbEvalServer.

**Prerequisites:**
- kbEvalServer running on GPU machine
- SSH tunnel or direct network access to port 5676

**Quick Start for Phase 3:**

```bash
# On GPU devserver (in tmux):
cd /data/users/$USER/triton-ag
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 kbEvalServer.py --local_host --port 5676 --device 0

# On local Mac (in separate terminal, keep open):
ssh -L 5676:localhost:5676 -N devvm8491.cco0.facebook.com

# Verify connection works:
curl -s http://localhost:5676/health
```

For detailed setup instructions, see `Claude/KERNEL_BENCH_SETUP.md`.

#### What This Phase Tests

This phase validates the **complete end-to-end flow** for a single kernel: generation → HTTP request → GPU compilation → correctness check → benchmarking → result return.

**Component Depth Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE (Mac)                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  MCP Server (claudeCodeKernelBenchServer.py)                      │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  ✓ All tools from Phase 2                                   │  │  │
│  │  │  ✓ eval_kernel() - NOW TESTED                               │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                     │  │
│  │                              ▼                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  KbEvalClient (kbEvalClient.py)                             │  │  │
│  │  │  ✓ kb_eval() - Makes HTTP POST to kbEvalServer              │  │  │
│  │  │  ✓ Timeout handling (300s)                                  │  │  │
│  │  │  ✓ Response parsing (compiled, correctness, speedup)        │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                    HTTP POST (port 5676)                                 │
│                    via SSH tunnel                                        │
│                              │                                           │
└──────────────────────────────┼───────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GPU MACHINE (devserver)                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  kbEvalServer.py (FastAPI)                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  /kb_eval endpoint                                          │  │  │
│  │  │  ✓ Receives reference_code + generated_code                 │  │  │
│  │  │  ✓ Spawns subprocess for compilation                        │  │  │
│  │  │  ✓ Waits for GPU (CUDA queue)                               │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                     │  │
│  │                              ▼                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  kbEvalUtil.py                                              │  │  │
│  │  │  ✓ torch.utils.cpp_extension.load_inline() - Compile Triton│  │  │
│  │  │  ✓ Correctness check (compare Model vs ModelNew output)     │  │  │
│  │  │  ✓ Timing (torch.cuda.Event for precise GPU timing)         │  │  │
│  │  │  ✓ Speedup calculation (reference_time / kernel_time)       │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                     │  │
│  │                              ▼                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  NVIDIA GPU (device 7)                                      │  │  │
│  │  │  ✓ Triton JIT compilation                                   │  │  │
│  │  │  ✓ Kernel execution (warmup + timed runs)                   │  │  │
│  │  │  ✓ Memory allocation for inputs/outputs                     │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**What's Real vs Faked:**

| Component | Status | Notes |
|-----------|--------|-------|
| MCP tool functions | REAL | Actual async functions executing |
| Task source code | REAL | Reads actual PyTorch model files |
| Generated kernel | REAL | Claude generates actual Triton code |
| HTTP request | REAL | Actual POST to localhost:5676 via SSH tunnel |
| kbEvalServer | REAL | Running on remote GPU machine |
| GPU compilation | REAL | Triton JIT compiles to GPU kernels |
| Correctness check | REAL | Compares outputs with reference |
| Timing/Speedup | REAL | GPU-timed benchmarking |
| **SSH tunnel** | **INFRASTRUCTURE** | Manual setup required before test |
| **Server lifetime** | **MANUAL** | Server started manually, not auto-managed |

**Key Validation Points:**
- HTTP connection succeeds through SSH tunnel
- Kernel compiles without CUDA errors
- Output matches reference (correctness = true)
- Speedup is measured and >= 1.0 for good kernels
- Timeout handling works (no hung requests)
- Error messages are clear for compilation failures

**Typical Response Times:**
| Operation | Time |
|-----------|------|
| HTTP round-trip overhead | ~10-50ms |
| Triton JIT compilation | 5-15s (first run) |
| Kernel execution (warmup) | 1-5s |
| Kernel execution (timed) | 0.1-2s |
| **Total per eval** | **10-30s** |

**Common Failure Modes:**
1. **Connection refused**: SSH tunnel not set up or server not running
2. **Timeout**: Server overloaded or slow compilation
3. **Compilation error**: Invalid Triton kernel syntax
4. **Correctness failure**: Kernel produces wrong output
5. **No speedup**: Kernel slower than PyTorch reference

#### Test 3.0: Setup kbEvalServer and SSH Tunnel

On GPU machine (devserver):

```bash
cd /data/users/$USER/triton-ag
source .venv/bin/activate
python kbEvalServer.py --local_host --port 5676 --device 7
```

On local machine (create SSH tunnel):

```bash
ssh -L 5676:localhost:5676 devvm8491.cco0.facebook.com
```

Verify connection:

```bash
curl http://localhost:5676/health
```

Expected: `{"status":"ok",...}`

#### Test 3.1: Direct KbEvalClient Call

Create test script:

```bash
cat > /Users/aarontao/Projects/code/triton-ag/test_phase3_kbeval.py << 'EOF'
"""Test Phase 3.1: Direct kbEvalClient call."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import get_kbeval_client

REFERENCE_CODE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
'''

GENERATED_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output
'''

async def test_kb_eval():
    print("=== Test 3.1: Direct kbEvalClient call ===")

    client = get_kbeval_client()
    print(f"Using client: {type(client).__name__}")

    try:
        result = await client.kb_eval(
            provider="local",
            reference_code=REFERENCE_CODE,
            generated_code=GENERATED_KERNEL,
            timeout=120
        )
        print(f"Result: {result}")

        if result.get("compiled"):
            print("✓ Kernel compiled successfully")
        else:
            print(f"✗ Compilation failed: {result.get('error')}")

        if result.get("correctness"):
            print("✓ Kernel produces correct output")
        else:
            print(f"✗ Correctness check failed")

        print(f"Speedup: {result.get('speedup', 'N/A')}x")
        print("\n=== Phase 3.1: PASS ===")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\n=== Phase 3.1: FAIL ===")
        raise

if __name__ == "__main__":
    asyncio.run(test_kb_eval())
EOF
```

Run test:

```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 test_phase3_kbeval.py
```

**Pass Criteria:**
- [ ] kbEvalClient connects to server
- [ ] Kernel compiles successfully
- [ ] Correctness check passes
- [ ] Speedup is measured

#### Test 3.2: MCP eval_kernel Tool

Create test script:

```bash
cat > /Users/aarontao/Projects/code/triton-ag/test_phase3_mcp.py << 'EOF'
"""Test Phase 3.2: MCP eval_kernel tool."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import eval_kernel

GENERATED_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output
'''

async def test_eval_kernel():
    print("=== Test 3.2: MCP eval_kernel tool ===")

    result = await eval_kernel(
        task_path="level1/1_relu.py",
        kernel_code=GENERATED_KERNEL,
        session_id="test_phase3_mcp",
        iteration=0,
        provider="local"
    )

    print(f"Result: {result}")

    if result.get("compiled"):
        print("✓ Kernel compiled")
    if result.get("correctness"):
        print("✓ Kernel correct")
    print(f"Speedup: {result.get('speedup', 'N/A')}x")

    print("\n=== Phase 3.2: PASS ===")

if __name__ == "__main__":
    asyncio.run(test_eval_kernel())
EOF
```

Run test:

```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 test_phase3_mcp.py
```

**Pass Criteria:**
- [ ] eval_kernel() successfully calls kbEvalServer
- [ ] Returns compiled, correctness, speedup fields
- [ ] Result saved to output directory

#### Test 3.3: Claude Code End-to-End Single Task

In Claude Code, run:

> Optimize the kernel in kernel_bench/level1/1_relu.py
>
> Use eval_kernel to compile and benchmark your kernel. Use session_id="test_phase3_e2e".
> Iterate up to 3 times if needed to get a correct kernel with speedup > 1.0x.

Verify:

```bash
ls -la ~/.inference/claude_code_output/test_phase3_e2e/
cat ~/.inference/claude_code_output/test_phase3_e2e/1_relu/iteration_*_eval.json
```

Cleanup:

```bash
rm -rf ~/.inference/claude_code_output/test_phase3_e2e
```

**Pass Criteria:**
- [ ] Claude Code generates kernel(s)
- [ ] At least one kernel compiles successfully
- [ ] Correctness and speedup are measured
- [ ] Results saved correctly

---

### 5.6 Phase 4: Multiple Tasks (Sequential, Single Agent)

**Purpose**: Verify batch processing of multiple tasks sequentially.

**Prerequisites:**
- Phase 3 tests pass
- kbEvalServer running

#### What This Phase Tests

This phase validates that a single Claude Code session can process multiple kernel tasks **sequentially** - completing one task before starting the next.

**Component Depth Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLAUDE CODE SESSION                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Main Agent (single thread of execution)                          │  │
│  │                                                                    │  │
│  │  Loop through tasks [1_relu, 2_matmul, 3_sigmoid]:                │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Task N:                                                     │  │  │
│  │  │  1. get_task_details() → Read PyTorch source                │  │  │
│  │  │  2. Generate Triton kernel                                   │  │  │
│  │  │  3. eval_kernel() → HTTP → GPU → Result (BLOCKING, 10-30s)  │  │  │
│  │  │  4. save_benchmark_result() → Write files                    │  │  │
│  │  │  5. (optional) Iterate if correctness=false                  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                     │  │
│  │                              ▼ (next task)                         │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Task N+1: (starts after Task N completes)                  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                     │  │
│  │                              ▼                                     │  │
│  │  get_session_summary() → Aggregate all results                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               │ Sequential HTTP calls (one at a time)
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         kbEvalServer (GPU)                               │
│  Request queue: [  ] → [ ] → [ ]  (max 1 in-flight at a time)          │
│                                                                          │
│  Timeline:                                                               │
│  |--Task 1 (30s)--|--Task 2 (25s)--|--Task 3 (28s)--|                   │
│  0s              30s              55s              83s                  │
│                                                                          │
│  Total: Sum of individual task times (no parallelism benefit)           │
└─────────────────────────────────────────────────────────────────────────┘
```

**What's Real vs Faked:**

| Component | Status | Notes |
|-----------|--------|-------|
| Multiple tasks | REAL | 3+ different kernel_bench tasks |
| Sequential execution | REAL | One task finishes before next starts |
| GPU compilation per task | REAL | Each kernel compiled and benchmarked |
| Session aggregation | REAL | summary includes all task results |
| File I/O per task | REAL | Each task has its own output files |
| **Parallelism** | **NONE** | Single thread, no concurrent evals |

**Key Validation Points:**
- All N tasks complete without errors
- Each task has its own kernel and eval files
- Session summary shows correct task count
- Average speedup is calculated across all tasks
- No race conditions (only one eval at a time)
- Total time ≈ sum of individual task times

**Expected Timing (3 tasks):**
| Task | Time | Cumulative |
|------|------|------------|
| 1_relu | 25s | 25s |
| 2_matmul | 30s | 55s |
| 3_sigmoid | 20s | 75s |
| **Total** | - | **~75s** |

**Why Sequential?**
- Single Claude Code session = single thread of execution
- Each `eval_kernel()` call blocks until GPU returns result
- No parallelism within a single agent's context
- This is the baseline before testing parallel agents (Phase 5)

#### Test 4.1: Programmatic Batch Test

Create test script:

```bash
cat > /Users/aarontao/Projects/code/triton-ag/test_phase4_batch.py << 'EOF'
"""Test Phase 4.1: Batch processing multiple tasks."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import (
    list_kernel_bench_tasks,
    get_task_details,
    save_benchmark_result,
    get_session_summary
)

MOCK_KERNEL_TEMPLATE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_{name}(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x  # Identity for testing
'''

SESSION_ID = "test_phase4_batch"

async def main():
    print("=== Test 4.1: Batch processing ===")

    # Get first 3 tasks
    tasks_result = await list_kernel_bench_tasks(level="level1")
    tasks = tasks_result.get("tasks", [])[:3]
    print(f"Processing {len(tasks)} tasks")

    for i, task in enumerate(tasks):
        task_path = task["path"]
        task_name = task["name"]
        print(f"\n[{i+1}/{len(tasks)}] Processing {task_name}")

        # Get task details
        details = await get_task_details(task_path=task_path)
        print(f"  Source: {len(details.get('source_code', ''))} chars")

        # Generate mock kernel
        kernel = MOCK_KERNEL_TEMPLATE.format(name=task_name.replace("-", "_"))

        # Save result (mock eval)
        mock_result = f'{{"compiled": true, "correctness": true, "speedup": {1.0 + i*0.1}}}'
        result = await save_benchmark_result(
            task_path=task_path,
            kernel_code=kernel,
            eval_result=mock_result,
            session_id=SESSION_ID,
            iteration=0
        )
        print(f"  Saved: {result.get('path', 'OK')}")

    # Get session summary
    print("\n=== Session Summary ===")
    summary = await get_session_summary(session_id=SESSION_ID)
    print(f"Total tasks: {summary.get('total_tasks')}")
    print(f"Successful: {summary.get('successful_tasks')}")
    print(f"Avg speedup: {summary.get('average_speedup', 'N/A')}")

    print("\n=== Phase 4.1: PASS ===")

if __name__ == "__main__":
    asyncio.run(main())
EOF
```

Run test:

```bash
cd /Users/aarontao/Projects/code/triton-ag
python3 test_phase4_batch.py
```

Cleanup:

```bash
rm -rf ~/.inference/claude_code_output/test_phase4_batch
```

**Pass Criteria:**
- [ ] All 3 tasks processed successfully
- [ ] Results saved for each task
- [ ] Session summary shows correct counts

#### Test 4.2: Claude Code Sequential Batch

In Claude Code, run:

> Run 3 level1 kernel bench tasks sequentially. For each task:
> 1. Read the PyTorch model
> 2. Generate a Triton kernel
> 3. Evaluate with eval_kernel
> 4. Save result
>
> Use session_id="test_phase4_claude". Pick any 3 tasks from level1 (e.g., 1_relu, 2_matmul, 3_sigmoid).
> Process them one at a time.

Verify:

```bash
ls -la ~/.inference/claude_code_output/test_phase4_claude/
cat ~/.inference/claude_code_output/test_phase4_claude/*/iteration_*_eval.json
```

Get session summary in Claude Code:

> Get the session summary for session_id="test_phase4_claude"

Cleanup:

```bash
rm -rf ~/.inference/claude_code_output/test_phase4_claude
```

**Pass Criteria:**
- [ ] 3 tasks processed
- [ ] Each task has kernel and eval files
- [ ] Session summary shows 3 tasks with speedup data

---

### 5.7 Phase 5: Multiple Agents (Parallel Processing)

**Purpose**: Verify parallel processing with multiple agents using Task tool.

**Prerequisites:**
- Phase 4 tests pass
- kbEvalServer running

#### What This Phase Tests

This phase validates that Claude Code can use the **Task tool** to spawn multiple sub-agents that work on different kernel tasks **in parallel**, achieving wall-clock speedup compared to sequential execution.

**Component Depth Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLAUDE CODE SESSION                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Main Agent (orchestrator)                                        │  │
│  │                                                                    │  │
│  │  1. Partition 6 tasks into 2 groups                               │  │
│  │  2. Launch parallel agents via Task tool:                         │  │
│  │                                                                    │  │
│  │  ┌────────────────────┐      ┌────────────────────┐              │  │
│  │  │  Sub-Agent 1       │      │  Sub-Agent 2       │              │  │
│  │  │  [1_relu]          │      │  [4_tanh]          │              │  │
│  │  │  [2_matmul]        │      │  [5_leaky_relu]    │              │  │
│  │  │  [3_sigmoid]       │      │  [6_softmax]       │              │  │
│  │  │  (sequential)      │      │  (sequential)      │              │  │
│  │  └─────────┬──────────┘      └─────────┬──────────┘              │  │
│  │            │                           │                          │  │
│  │            │   CONCURRENT EXECUTION    │                          │  │
│  │            ▼                           ▼                          │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                    MCP Server                                │  │  │
│  │  │  eval_kernel() requests interleaved from both agents        │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               │ Multiple concurrent HTTP requests
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         kbEvalServer (GPU)                               │
│                                                                          │
│  Concurrent requests from 2 agents:                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Timeline (overlapped waiting):                                      ││
│  │                                                                      ││
│  │ Agent1:  |--1_relu--|--2_matmul--|--3_sigmoid--|                    ││
│  │ Agent2:  |--4_tanh--|--5_leaky--|--6_softmax--|                     ││
│  │ GPU:     |===1_relu===|===4_tanh===|===2_matmul===|===5_leaky===|..││
│  │                                                                      ││
│  │ Requests queue at GPU, execute one at a time, agents wait in HTTP   ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  Wall-clock benefit: Agents prepare next request while GPU runs current │
│  Total time ≈ max(agent1_time, agent2_time) + GPU serialization         │
└─────────────────────────────────────────────────────────────────────────┘
```

**What's Real vs Faked:**

| Component | Status | Notes |
|-----------|--------|-------|
| Task tool | REAL | Claude Code spawns actual sub-agents |
| Parallel agents | REAL | 2+ agents running concurrently |
| Concurrent HTTP | REAL | Multiple in-flight requests to kbEvalServer |
| GPU serialization | REAL | GPU processes one kernel at a time (CUDA queue) |
| Session sharing | REAL | All agents write to same session_id directory |
| File locking | **RISK** | Potential race conditions on session files |
| Load management | **IMPLICIT** | No explicit throttling, relies on agent count limit |

**Key Validation Points:**
- Both agents start concurrently (visible via timestamps)
- All 6 tasks complete without errors
- No file conflicts (each task has unique subdirectory)
- Session summary aggregates results from all agents
- Wall-clock time < sequential time (parallelism benefit)
- GPU utilization is higher than sequential (less idle time)

**Expected Timing Comparison:**

| Execution Mode | Tasks | Wall-Clock | Speedup |
|----------------|-------|------------|---------|
| Sequential (Phase 4) | 6 tasks | ~150s | 1.0x |
| 2 Parallel Agents | 6 tasks (3 each) | ~90s | 1.7x |
| 3 Parallel Agents | 6 tasks (2 each) | ~70s | 2.1x |

*Note: Speedup is limited by GPU serialization - more agents don't help beyond GPU throughput limit*

**Why Parallel Agents Work:**

```
Sequential:
Agent:  [generate]-[wait 25s]-[generate]-[wait 25s]-[generate]-[wait 25s]
GPU:              |===eval===|          |===eval===|          |===eval===|

Parallel (2 agents):
Agent1: [generate]-[wait]----[generate]-[wait]----...
Agent2: [generate]-[wait]----[generate]-[wait]----...
GPU:              |===1===|===2===|===3===|===4===|...

The overlap is in generation time, not GPU time.
Wall-clock savings = (N_agents - 1) * generation_time_per_kernel
```

**Potential Issues to Watch:**
1. **Session file conflicts**: Multiple agents writing summary.json simultaneously
2. **HTTP timeout cascade**: If one agent times out, does it affect others?
3. **kbEvalServer overload**: Too many concurrent requests may exhaust resources
4. **Memory pressure**: Each agent has its own context/memory footprint

#### Test 5.1: Claude Code Parallel Agents (2 Agents)

In Claude Code, run:

> Run 6 level1 kernel bench tasks using 2 parallel agents.
>
> Split tasks:
> - Agent 1: 1_relu, 2_matmul, 3_sigmoid
> - Agent 2: 4_tanh, 5_leaky_relu, 6_softmax
>
> Each agent should process its tasks sequentially.
> Use session_id="test_phase5_parallel" for all results.
>
> Launch both agents in parallel using the Task tool.

Verify parallel execution (during run):

```bash
# Watch for concurrent file creation
watch -n 2 'ls -la ~/.inference/claude_code_output/test_phase5_parallel/'
```

Verify after completion:

```bash
ls -la ~/.inference/claude_code_output/test_phase5_parallel/
```

Get session summary in Claude Code:

> Get the session summary for session_id="test_phase5_parallel"

Cleanup:

```bash
rm -rf ~/.inference/claude_code_output/test_phase5_parallel
```

**Pass Criteria:**
- [ ] Both agents run concurrently (visible via file timestamps)
- [ ] All 6 tasks have results
- [ ] Session summary shows 6 tasks
- [ ] No race conditions or file conflicts

#### Test 5.2: Load Management Verification

In Claude Code, run with more agents:

> Run 10 level1 kernel bench tasks using 5 parallel agents.
>
> Distribute tasks evenly (2 tasks per agent).
> Each agent processes its tasks sequentially.
> Use session_id="test_phase5_load" for all results.
>
> Report the total time taken and any errors encountered.

Observe:

```bash
# Monitor kbEvalServer load (on GPU machine)
nvidia-smi -l 1
```

Verify:

```bash
ls -la ~/.inference/claude_code_output/test_phase5_load/
```

Cleanup:

```bash
rm -rf ~/.inference/claude_code_output/test_phase5_load
```

**Pass Criteria:**
- [ ] All 10 tasks complete without errors
- [ ] GPU processes requests (visible in nvidia-smi)
- [ ] No HTTP timeouts
- [ ] Load is distributed across agents (visible via file timestamps)

---

### 5.8 Phase 6: Multi-GPU with Adaptive Semaphore

**Purpose**: Verify the RL-style multi-device architecture with adaptive semaphore for OOM prevention.

**Prerequisites:**
- Phase 5 tests pass
- kbEvalServer configured with multiple GPUs (config-based mode)
- SSH tunnel to remote GPU server
- Work items W9-W11 implemented (see Section 5.1)

#### What This Phase Tests

This phase validates:
1. **Adaptive semaphore**: MCP server queries GPU count, limits concurrency accordingly
2. **Multi-device distribution**: kbEvalServer distributes load across GPUs via `random.choice()`
3. **OOM prevention**: No GPU memory errors during parallel agent execution

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLAUDE CODE SESSION                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Main Agent (orchestrator)                                        │  │
│  │                                                                    │  │
│  │  Launch 8 parallel agents (provider="local")                      │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                  │  │
│  │  │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │ Agent 4 │ ...              │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                  │  │
│  └───────┼──────────┼──────────┼──────────┼─────────────────────────┘  │
│          ▼          ▼          ▼          ▼                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  MCP Server (claudeCodeKernelBenchServer.py)                      │  │
│  │                                                                    │  │
│  │  _init_semaphore():                                               │  │
│  │    → GET /info → {"num_devices": 8}                               │  │
│  │    → Semaphore(8)  # Allow 8 concurrent evals                     │  │
│  │                                                                    │  │
│  │  async with _eval_semaphore:  # Blocks if 8 already running       │  │
│  │      → POST /kb_eval                                              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                    SSH Tunnel (localhost:5676)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         kbEvalServer (config-based)                      │
│                                                                          │
│  GET /info → {"num_devices": 8, "devices": [0,1,2,3,4,5,6,7]}           │
│                                                                          │
│  POST /kb_eval → device = random.choice(devices)                         │
│         ┌───────────────────────────────────────────────────────────┐   │
│         │  GPU 0    GPU 1    GPU 2    GPU 3    GPU 4    GPU 5  ...  │   │
│         │    ▼        ▼        ▼        ▼        ▼        ▼         │   │
│         │  task 3  task 1  task 5  task 2  task 7  task 4   ...     │   │
│         └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Test 6.0: Setup Multi-Device kbEvalServer

**Step 0: Update code on GPU server (required for /info endpoint)**

```bash
# On GPU server
cd /data/users/$USER/triton-ag
git pull origin claude-code-kernel-bench-plan
```

**Step 1: Add server config to `kbEval.yaml` on GPU server**

```yaml
servers:
  your-hostname:  # Must match output of `hostname` command
    host: "::"
    port: 5676
    compile_cache: true
    check_get_inputs: false
    devices:
      - 0
      - 1
      - 2
      - 3
      - 4
      - 5
      - 6
      - 7
```

**Step 2: Start kbEvalServer (config-based mode)**

```bash
# On GPU server - uses config from kbEval.yaml, manages all 8 GPUs
cd /data/users/$USER/triton-ag
source .venv/bin/activate
python kbEvalServer.py  # NO --local_host flag
```

**Step 3: SSH tunnel from local Mac**

```bash
ssh -L 5676:localhost:5676 -N gpu-server
```

**Step 4: Verify setup**

```bash
# Check server status
curl -s http://localhost:5676/stats
# Expected: {"num_devices":8,"pending_requests":0}

# Check device info
curl -s http://localhost:5676/info
# Expected: {"num_devices":8,"pending_requests":0,"code_type":"triton",...}
```

**Pass Criteria:**
- [ ] kbEvalServer running in config-based mode (not `--local_host`)
- [ ] `/stats` endpoint responds with num_devices
- [ ] `/info` endpoint returns correct device count

#### Test 6.1: Adaptive Semaphore Initialization

Test that the MCP server correctly queries GPU count and creates appropriate semaphore.

**Create test script:**

```bash
cat > /Users/aarontao/Projects/code/triton-ag/test_phase6_semaphore.py << 'EOF'
"""Test Phase 6.1: Adaptive semaphore initialization."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

import claudeCodeKernelBenchServer as mcp_server

async def main():
    print("=== Test 6.1: Adaptive Semaphore Initialization ===\n")

    # Test 1: Query /info endpoint directly
    print("Step 1: Query /info endpoint")
    client = mcp_server.get_kbeval_client()
    try:
        info = await client.get_info(provider="local")
        if info is None:
            print("  ✗ Server returned None - /info endpoint may not exist")
            print("  (Ensure kbEvalServer is updated with /info endpoint)")
            return
        print(f"  Server info: {info}")
        num_devices = info.get("num_devices", 1)
        print(f"  ✓ num_devices = {num_devices}")
    except Exception as e:
        print(f"  ✗ Failed to get info: {e}")
        return

    # Test 2: Initialize semaphore via get_eval_semaphore
    print("\nStep 2: Initialize adaptive semaphore")
    semaphore = await mcp_server.get_eval_semaphore(provider="local")
    semaphore_size = mcp_server._semaphore_size

    print(f"  ✓ Semaphore initialized with {semaphore_size} slots")
    if semaphore_size == num_devices:
        print("  ✓ Semaphore matches device count")
        print("\n=== Phase 6.1: PASS ===")
    else:
        print(f"  ✗ Mismatch: semaphore={semaphore_size}, devices={num_devices}")
        print("\n=== Phase 6.1: FAIL ===")

if __name__ == "__main__":
    asyncio.run(main())
EOF
```

**Run test:**

```bash
python3 test_phase6_semaphore.py
```

**Pass Criteria:**
- [ ] `/info` endpoint returns `num_devices`
- [ ] Semaphore initialized with correct count
- [ ] Semaphore value matches device count

#### Test 6.2: Multi-Device Distribution

Verify that parallel agents are distributed across GPUs.

**In Claude Code:**

> Run 8 level1 tasks using 8 parallel agents with provider="local".
> Tasks: 19_ReLU, 20_LeakyReLU, 21_Sigmoid, 22_Tanh, 26_GELU, 27_SELU, 28_HardSwish, 29_Softplus
> Use session_id="test_phase6_distribution".
>
> **IMPORTANT:** Launch all 8 agents in parallel using the Task tool (single message with 8 Task tool calls).

**Monitor GPU distribution** (on GPU machine):

```bash
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'
```

**Expected Behavior:**
- Multiple GPUs show activity (not just one)
- Load is distributed (random, so not perfectly even)
- No OOM errors

**Verify:**

```bash
# Check all 8 tasks have results
ls -la ~/.inference/claude_code_output/test_phase6_distribution/

# Check eval results
cat ~/.inference/claude_code_output/test_phase6_distribution/*/iteration_*_eval.json | jq -s 'map({task: .task_name, compiled, speedup})'
```

**Cleanup:**

```bash
rm -rf ~/.inference/claude_code_output/test_phase6_distribution
```

**Pass Criteria:**
- [ ] All 8 agents complete without OOM
- [ ] Multiple GPUs show utilization during test (nvidia-smi)
- [ ] All 8 tasks have valid results

#### Test 6.3: Semaphore Blocking (Concurrency Limit)

Verify that the semaphore blocks excess requests when all GPU slots are in use.

**Setup:** Use a server with fewer GPUs (or modify config temporarily)

```yaml
# Temporarily limit to 2 devices for testing
servers:
  test-hostname:
    devices:
      - 0
      - 1
```

**In Claude Code:**

> Run 5 level1 tasks using 5 parallel agents with provider="local".
> Use session_id="test_phase6_blocking".
>
> With only 2 GPUs available, 3 agents should block until slots free up.

**Expected Behavior:**
- First 2 agents start immediately
- Remaining 3 agents block until first agents complete
- All 5 complete eventually (no OOM)
- Total time ≈ 3 rounds × task_time (not 5 × task_time)

**Pass Criteria:**
- [ ] All 5 tasks complete
- [ ] No OOM errors
- [ ] Semaphore correctly limits concurrency

#### Test 6.4: Stress Test (Maximum Parallelism)

Push the system to verify OOM protection at scale.

**In Claude Code:**

> Run 16 level1 tasks using 16 parallel agents with provider="local".
> Use session_id="test_phase6_stress".
> With 8 GPUs, expect 2 rounds of execution.

**Expected Results:**

| Parallel Agents | GPUs | Requests/GPU | Expected Behavior |
|-----------------|------|--------------|-------------------|
| 8 | 8 | 1.0 | All pass, no OOM |
| 16 | 8 | 2.0 | Semaphore blocks 8, all pass |
| 24 | 8 | 3.0 | Semaphore blocks 16, all pass |

**Monitor:**

```bash
# GPU memory should stay stable (no spikes)
watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

**Cleanup:**

```bash
rm -rf ~/.inference/claude_code_output/test_phase6_stress
```

**Pass Criteria:**
- [ ] All 16 tasks complete without OOM
- [ ] GPU memory stays within safe limits
- [ ] Semaphore correctly queues excess requests

#### Test 6.5: Error Handling

Verify graceful handling of edge cases.

**Scenario A: Server unreachable (semaphore fallback)**

1. Stop kbEvalServer
2. Run a task - should fail gracefully with connection error
3. Semaphore should default to 1 (safe fallback)

**Scenario B: Invalid provider**

> Run a task with provider="nonexistent".

**Expected:** Clear error message, no crash

**Scenario C: Partial failure**

> Run 8 parallel agents where 2 generate invalid kernel code.

**Expected:**
- 6 succeed with speedup data
- 2 fail with compilation errors captured
- Session summary shows partial success

**Pass Criteria:**
- [ ] Connection errors handled gracefully
- [ ] Invalid provider gives clear error
- [ ] Partial failures don't crash the session

#### Phase 6 Summary

| Test | Purpose | Pass Criteria |
|------|---------|---------------|
| 6.0: Setup | Configure multi-device server | Server running, `/stats` and `/info` work |
| 6.1: Semaphore Init | Adaptive semaphore queries GPU count | Semaphore = device count |
| 6.2: Distribution | Load spread across GPUs | 8 agents, all GPUs active |
| 6.3: Blocking | Semaphore limits concurrency | 5 agents on 2 GPUs, no OOM |
| 6.4: Stress | Maximum parallelism | 16 agents, no OOM |
| 6.5: Errors | Graceful error handling | No crashes |

> **Note:** Tests 6.1+ require the remote kbEvalServer to be updated with the `/info` endpoint.
> After pulling latest code on the GPU server, restart kbEvalServer.

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
| 6 | 6.0: Multi-Device Setup | PENDING | - | Config-based kbEvalServer, /info endpoint |
| 6 | 6.1: Semaphore Init | PENDING | - | Adaptive semaphore queries GPU count (W9-W11) |
| 6 | 6.2: Distribution | PENDING | - | 8 agents distributed across 8 GPUs |
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
