# Claude Code Kernel Bench Integration Plan

## Goal
Create a solution that allows Claude Code to:
1. Trigger kernel benchmark tasks
2. Generate CUDA/Triton kernel code
3. Integrate with the inference eval queue
4. Generate benchmark results comparable to RL-tuned models

## Key Constraint: No Claude API Access Required

This solution works **entirely through Claude Code** - no separate Claude API key is needed.

### How It Works

```
User → Claude Code → [Claude generates CUDA kernel] → MCP tool (eval_kernel) → kbEvalServer → Results
           ↑
      This IS your Claude access
```

**Claude Code IS the Claude access.** The MCP server provides only utility tools:
- **File I/O**: List tasks, read task details, save results
- **HTTP calls**: Submit kernels to kbEvalServer for compilation/benchmarking

**What generates the CUDA code?** Claude (via Claude Code conversation). When you ask Claude Code to optimize a kernel, Claude generates the code directly in the conversation - no external LLM API call.

### Contrast with Existing Agent System

The existing `agent_kernel_coder.py` DOES require API access (OpenAI, DeepSeek, or local vLLM) because it uses the OpenAI Agents SDK to make LLM calls programmatically. This solution avoids that dependency entirely.

## User Workflow

Once the MCP server is registered, you interact with Claude Code naturally:

**Setup (one-time):**
```bash
# 1. Ensure kbEval server is running on a GPU machine
python kbEvalServer.py --local_host --port 5676 --device 0

# 2. Register MCP server in Claude Code config (~/.claude/mcp.json or project .mcp.json)
# See Configuration section below
```

**Usage - Single Task:**
```
You: "Optimize the kernel in kernel_bench/level1/1_relu.py"

Claude Code:
  1. Uses `get_task_details` tool to read the PyTorch model
  2. Generates optimized CUDA/Triton kernel code
  3. Uses `eval_kernel` tool to compile and benchmark
  4. Reviews results, iterates if needed
  5. Uses `save_benchmark_result` tool to store final result
```

**Usage - Batch Session:**
```
You: "Run a benchmark session on all level1 tasks and save results for comparison"

Claude Code:
  1. Uses `list_kernel_bench_tasks` tool to get all level1 tasks
  2. For each task:
     - Read task, generate kernel, evaluate, iterate
  3. Saves all results to ~/.inference/claude_code_output/
  4. Generates summary report
```

**Example Complete Session:**
```
You: "Run kernel bench on level1 tasks, session name: claude_vs_rl_run1"

Claude Code:
  [Lists 50 tasks in level1]
  [Starting task 1/50: 1_relu.py]

  Reading task... PyTorch model applies ReLU activation.
  Generating Triton kernel...

  ```python
  @triton.jit
  def relu_kernel(...):
      ...
  ```

  Evaluating... Compiled: ✓, Correct: ✓, Speedup: 1.23x
  Saved to ~/.inference/claude_code_output/claude_vs_rl_run1/1_relu/

  [Starting task 2/50: 2_matmul.py]
  ...

  === Session Complete ===
  Success Rate: 45/50 (90%)
  Average Speedup: 1.34x
  Results saved to: ~/.inference/claude_code_output/claude_vs_rl_run1/

You: "Compare this with the RL model results in ~/.inference/output/my_tag.a_001_01/"

Claude Code:
  [Runs comparison, generates report]
```

### How Claude Code Knows Task Locations

In a new session, Claude Code needs to know where kernel_bench tasks are. Three mechanisms ensure this:

**1. MCP Server Configuration (Primary)**

The MCP server has a config file specifying the kernel_bench base path:

```yaml
# claudeCodeKernelBench.yaml
kernel_bench:
  base_dir: "${HOME}/KernelBench/KernelBench"  # or project-relative path
  levels: ["level1", "level2", "level3"]
```

When `list_kernel_bench_tasks` is called, it reads from this configured location.

**2. Project CLAUDE.md (Context)**

The project's CLAUDE.md already documents the structure:
```markdown
## Data Directories
- `kernel_bench/level1/`, `level2/`, `level3/` - Benchmark tasks
```

Claude Code reads CLAUDE.md at session start, providing context.

**3. Explicit Path Override**

Users can always specify explicit paths:
```
You: "Run benchmark on tasks in /path/to/my/kernel_bench/level1/"
```

**Resolution Order:**
1. User-specified path (highest priority)
2. MCP server config (`claudeCodeKernelBench.yaml`)
3. CLAUDE.md documented paths (context)
4. Default: `~/KernelBench/KernelBench` (fallback)

## Prompt Structure and Multi-Turn Iteration

### Comparison: RL Fine-Tuning vs Claude Code

| Aspect | RL Fine-Tuning (TreeTurns) | Claude Code |
|--------|---------------------------|-------------|
| LLM Access | vLLM server with log probs | Claude Code conversation |
| Parallelism | 8 generations × 4 turns = 32 per task | Sequential (1 at a time) |
| Selection | Random or best_speedup across 8 gens | Claude decides based on feedback |
| Feedback | Structured JSON prompt templates | Natural language conversation |
| Training Signal | GRPO reward from speedup/correctness | None (inference only) |

### RL Fine-Tuning Multi-Turn Structure

The RL training uses **TreeTurns** - a tree-structured generation approach:

```
Task (Reference Code)
  │
  └─→ Turn 0: 8 Parallel Generations (fresh start)
      ├─ Gen 00_t00 → eval → {compiled, correctness, speedup}
      ├─ Gen 01_t00 → eval → {compiled, correctness, speedup}
      └─ ... (8 total)

  └─→ [SELECTION: Pick best or random from turn 0]

  └─→ Turn 1: 8 Parallel Generations (refine selected)
      ├─ Gen 00_t01: Previous code + feedback → improved code
      └─ ... (8 total, all starting from same selected turn 0 code)

  └─→ [SELECTION: Pick best from turn 1]

  └─→ Turn 2, Turn 3: Similar pattern...
```

**Key parameters:**
- `num_generations: 8` - 8 independent attempts per turn
- `num_turns_per_generation: 4` - Up to 4 refinement turns
- `selection_strategy: "random"` or `"best_speedup"`

### RL Prompt Templates

The RL system uses three distinct user prompts based on evaluation state:

**1. Initial Generation (`user_prompt.init`)**
```
Following is the reference PyTorch code, implement the complete new module
with Triton (no testing code, no init code).

Reference code:
```python
{reference_code}
```

Reference code evaluation:
```json
{reference_eval}  // {"runtime": 0.5, "compiled": true, "correctness": true}
```
```

**2. Fix Errors (`user_prompt.fix`)** - when compiled=false OR correctness=false
```
Your code did not compile or run correctly. Please fix the code and return
the correct code.

Generated code evaluation:
```json
{generated_eval}  // {"compiled": false, "error": "..."}
```
```

**3. Optimize Performance (`user_prompt.perf`)** - when correct but needs speedup
```
Compare to the reference code performance evaluation and improve your code
performance by optimizing the code.

Generated code evaluation:
```json
{generated_eval}  // {"compiled": true, "correctness": true, "speedup": 0.8}
```
```

### Claude Code Multi-Turn Approach

For Claude Code, we mirror the same feedback loop but in a conversational format:

**Turn 0: Initial Generation**
```
Claude Code reads task via get_task_details(), then Claude generates:

User: "Optimize the kernel in kernel_bench/level1/1_relu.py"

Claude: [Reads task details]
        "The task is a simple ReLU activation. I'll generate a Triton kernel:

        ```python
        @triton.jit
        def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = tl.maximum(x, 0)
            tl.store(out_ptr + offsets, out, mask=mask)

        class ModelNew(nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                n = x.numel()
                relu_kernel[(n // 1024,)](x, out, n, BLOCK_SIZE=1024)
                return out
        ```"

Claude: [Calls eval_kernel MCP tool]
```

**Turn 1+: Iteration Based on Feedback**

If compilation fails:
```
[MCP tool returns: {"compiled": false, "error": "triton.language has no attribute 'maximum'"}]

Claude: "The kernel failed to compile. The error indicates I used wrong syntax.
        Let me fix it - Triton uses tl.where() for conditional operations:

        ```python
        out = tl.where(x > 0, x, 0.0)
        ```"

Claude: [Calls eval_kernel again]
```

If correct but slow:
```
[MCP tool returns: {"compiled": true, "correctness": true, "speedup": 0.85}]

Claude: "The kernel is correct but slower than PyTorch (0.85x speedup).
        I'll optimize by increasing block size and using vectorized loads:

        ```python
        # Use larger blocks and coalesced memory access
        BLOCK_SIZE=4096
        # Add prefetching hints...
        ```"

Claude: [Calls eval_kernel again]
```

If successful:
```
[MCP tool returns: {"compiled": true, "correctness": true, "speedup": 1.45}]

Claude: "The kernel achieves 1.45x speedup. Saving result."

Claude: [Calls save_benchmark_result MCP tool]
```

### Iteration Flow Diagram

```
                    RL Fine-Tuning                    Claude Code
                    ──────────────                    ───────────

Start               8 parallel generations           1 generation
                    from system prompt               from conversation
                           │                               │
                           ▼                               ▼
Evaluate            8 parallel kbEval calls          1 kbEval call
                           │                               │
                           ▼                               ▼
Feedback            JSON template selection:         Natural language:
                    - compiled=false → fix           Claude interprets error
                    - correct=false → fix            and decides approach
                    - correct=true → perf
                           │                               │
                           ▼                               ▼
Selection           Pick 1 of 8 to continue          N/A (only 1 path)
                    (random or best_speedup)
                           │                               │
                           ▼                               ▼
Next Turn           8 parallel generations           1 generation
                    all from selected code           from conversation
                           │                               │
                    ───────┴───────────────────────────────┴───────
                                        │
                                        ▼
                              Repeat for N turns
                              (RL: 4 turns, Claude: until success/limit)
```

### Key Differences

**1. Parallelism vs Sequential**
- RL: Explores 8 paths simultaneously, selects best
- Claude: Single path, but Claude can reason about multiple approaches

**2. Feedback Format**
- RL: Structured JSON triggers template selection (`fix` vs `perf`)
- Claude: Natural language allows richer reasoning ("the error suggests...")

**3. Exploration Strategy**
- RL: Breadth-first (8 × 4 = 32 total attempts)
- Claude: Depth-first (iterate on single path, but smarter per-step)

**4. Training Signal**
- RL: Collects log probs, computes rewards, trains model
- Claude: No training, but leverages Claude's reasoning capabilities

### MCP Tool Feedback Structure

The `eval_kernel` tool returns the same fields used by RL training:

```json
{
  "compiled": true,
  "correctness": true,
  "runtime": 0.234,
  "speedup": 1.45,
  "runtime_stats": {
    "mean": 0.234,
    "std": 0.012,
    "min": 0.220,
    "max": 0.251
  },
  "error": null,
  "error_type": null
}
```

Claude interprets this and decides:
- `compiled=false` → Read error message, fix syntax/logic
- `correctness=false` → Debug numerical issues, check edge cases
- `speedup < 1.0` → Optimize memory access, parallelism, block sizes
- `speedup >= target` → Save and move to next task

### Recommended Iteration Limits

| Scenario | RL Fine-Tuning | Claude Code |
|----------|----------------|-------------|
| Max turns per task | 4 (fixed) | 4 (configurable) |
| Max total attempts | 32 (8×4) | 4 (sequential) |
| Early stop condition | None (all enqueued upfront) | speedup > 1.3x or 4 failures |

## Architecture Decision

**Chosen Approach: MCP Server**

Create an MCP server (`claudeCodeKernelBenchServer.py`) that Claude Code can use directly. This leverages:
- Existing `kbEvalClient.py` for kernel evaluation
- Existing `workflowClient.py` for queue integration
- Same result format as RL training runs for direct comparison

## Implementation Overview

### Existing Components (No Changes Needed)

| Component | File | Purpose |
|-----------|------|---------|
| Workflow Server | `workflowServer.py` | Queue management, already exists |
| Workflow Client | `workflowClient.py` | Queue submission API |
| kbEval Server | `kbEvalServer.py` | Kernel compilation/benchmarking |
| kbEval Client | `kbEvalClient.py` | HTTP client for kbEval |
| KernelExecResult | `kbEvalUtil.py` | Result data model |
| Logger | `logger.py` | Logging utilities |

### New Components to Create

| Component | File | Purpose |
|-----------|------|---------|
| MCP Server | `claudeCodeKernelBenchServer.py` | MCP tools for Claude Code |
| MCP Config | `claudeCodeKernelBench.yaml` | Configuration for MCP server |
| Comparison Tool | `benchmarkCompare.py` | Compare Claude vs RL results |
| MCP Registration | `.mcp.json` | Register MCP server in project |

## Implementation Steps

### Step 1: Create MCP Server
**File: `claudeCodeKernelBenchServer.py`**

```python
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent
import yaml

from kbEvalClient import KBEvalClient
from workflowClient import WorkflowClient
from logger import logger

# Load configuration
def load_config(config_path: str = "claudeCodeKernelBench.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

server = Server("kernel-bench")
config = load_config()
kb_client = KBEvalClient()
workflow_client = WorkflowClient()

@server.tool()
async def list_kernel_bench_tasks(level: str = None) -> list[dict]:
    """
    List available kernel benchmark tasks.

    Args:
        level: Filter by level (e.g., "level1", "level2"). If None, list all.

    Returns:
        List of task info dicts with path, name, level
    """
    base_dir = Path(os.path.expanduser(config["kernel_bench"]["base_dir"]))
    levels = [level] if level else config["kernel_bench"]["levels"]

    tasks = []
    for lvl in levels:
        level_dir = base_dir / lvl
        if level_dir.exists():
            for task_file in sorted(level_dir.glob("*.py")):
                if not task_file.name.startswith("_"):
                    tasks.append({
                        "path": str(task_file),
                        "name": task_file.stem,
                        "level": lvl
                    })
    return tasks

@server.tool()
async def get_task_details(task_path: str) -> dict:
    """
    Get PyTorch model code and input specifications for a task.

    Args:
        task_path: Path to the task file (e.g., "kernel_bench/level1/1_relu.py")

    Returns:
        Dict with source_code, model_class, input_specs
    """
    task_path = Path(os.path.expanduser(task_path))
    if not task_path.exists():
        # Try resolving relative to base_dir
        base_dir = Path(os.path.expanduser(config["kernel_bench"]["base_dir"]))
        task_path = base_dir / task_path

    source_code = task_path.read_text()

    return {
        "path": str(task_path),
        "source_code": source_code,
        "name": task_path.stem
    }

@server.tool()
async def eval_kernel(
    task_path: str,
    kernel_code: str,
    session_id: str,
    iteration: int = 0,
    provider: str = "local",
    queue_only: bool = False
) -> dict:
    """
    Evaluate generated kernel against reference PyTorch implementation.

    Args:
        task_path: Path to the original task file
        kernel_code: Generated CUDA/Triton kernel code
        session_id: Session identifier for grouping results
        iteration: Iteration number within the task
        provider: kbEval provider (default: "local")
        queue_only: If True, submit to queue without waiting (for offline testing)

    Returns:
        KernelExecResult dict with compiled, correctness, runtime, speedup
    """
    if queue_only:
        # Submit to workflow queue without waiting for result
        queue_name = config.get("workflow", {}).get("eval_queue", "kbEval.pending")
        prefix_tag = config.get("workflow", {}).get("prefix_tag", "claude_code")

        work_item = {
            "task_path": task_path,
            "kernel_code": kernel_code,
            "session_id": session_id,
            "iteration": iteration,
            "submitted_at": datetime.now().isoformat(),
            "status": "pending"
        }

        await workflow_client.enqueue(prefix_tag, queue_name, work_item)
        return {"status": "queued", "queue": queue_name, "work_item": work_item}

    # Normal mode: call kbEval and wait for result
    result = await kb_client.kb_eval(
        task_path=task_path,
        kernel_code=kernel_code,
        provider=provider
    )

    return result.model_dump()

@server.tool()
async def save_benchmark_result(
    task_path: str,
    kernel_code: str,
    eval_result: dict,
    session_id: str,
    iteration: int = 0
) -> str:
    """
    Save benchmark result in format compatible with RL training output.

    Args:
        task_path: Path to the original task file
        kernel_code: Generated kernel code
        eval_result: Evaluation result from eval_kernel
        session_id: Session identifier
        iteration: Iteration number

    Returns:
        Path to saved result directory
    """
    output_base = Path(os.path.expanduser(config["output"]["base_dir"]))
    task_name = Path(task_path).stem

    session_dir = output_base / session_id / task_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save kernel code
    kernel_file = session_dir / f"iteration_{iteration:02d}_cuda_kernel.py"
    kernel_file.write_text(kernel_code)

    # Save eval result
    eval_file = session_dir / f"iteration_{iteration:02d}_eval.json"
    eval_result["model"] = "claude-code"
    eval_result["timestamp"] = datetime.now().isoformat()
    eval_file.write_text(json.dumps(eval_result, indent=2))

    # Update summary
    summary_file = output_base / session_id / "summary.json"
    summary = {}
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())

    if task_name not in summary:
        summary[task_name] = {"iterations": []}
    summary[task_name]["iterations"].append({
        "iteration": iteration,
        "compiled": eval_result.get("compiled", False),
        "correctness": eval_result.get("correctness", False),
        "speedup": eval_result.get("speedup", 0)
    })
    summary_file.write_text(json.dumps(summary, indent=2))

    return str(session_dir)

if __name__ == "__main__":
    import mcp
    mcp.run(server)
```

### Step 2: Create MCP Configuration
**File: `claudeCodeKernelBench.yaml`**

```yaml
# Claude Code Kernel Bench MCP Server Configuration

kernel_bench:
  # Base directory for kernel benchmark tasks
  base_dir: "${HOME}/KernelBench/KernelBench"
  levels: ["level1", "level2", "level3"]

kbeval:
  # kbEval server settings (uses kbEval.yaml for provider details)
  default_provider: "local"

workflow:
  # Workflow server settings for queue operations
  prefix_tag: "claude_code"
  eval_queue: "kbEval.pending"
  # Uses workflow.yaml for server connection details

output:
  # Output directory for Claude Code results
  base_dir: "${HOME}/.inference/claude_code_output"
```

### Step 3: Create Comparison Tool
**File: `benchmarkCompare.py`**

```python
#!/usr/bin/env python
"""
Compare Claude Code kernel benchmark results with RL-trained model results.
"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from logger import logger

def load_results(result_dir: Path) -> dict:
    """Load all eval results from a result directory."""
    results = {}
    for task_dir in result_dir.iterdir():
        if task_dir.is_dir() and task_dir.name != "summary.json":
            task_name = task_dir.name
            iterations = []
            for eval_file in sorted(task_dir.glob("*_eval.json")):
                with open(eval_file) as f:
                    iterations.append(json.load(f))
            if iterations:
                results[task_name] = iterations
    return results

def compute_metrics(results: dict) -> dict:
    """Compute aggregate metrics from results."""
    total_tasks = len(results)
    compiled_count = 0
    correct_count = 0
    speedups = []
    iterations_to_success = []

    for task_name, iterations in results.items():
        # Check best result across iterations
        best_correct = False
        best_speedup = 0
        success_iteration = None

        for i, result in enumerate(iterations):
            if result.get("compiled", False):
                compiled_count += 1
                if result.get("correctness", False):
                    if not best_correct:
                        success_iteration = i
                    best_correct = True
                    best_speedup = max(best_speedup, result.get("speedup", 0))

        if best_correct:
            correct_count += 1
            speedups.append(best_speedup)
            if success_iteration is not None:
                iterations_to_success.append(success_iteration + 1)

    return {
        "total_tasks": total_tasks,
        "compiled_rate": compiled_count / total_tasks if total_tasks else 0,
        "success_rate": correct_count / total_tasks if total_tasks else 0,
        "avg_speedup": sum(speedups) / len(speedups) if speedups else 0,
        "max_speedup": max(speedups) if speedups else 0,
        "avg_iterations_to_success": sum(iterations_to_success) / len(iterations_to_success) if iterations_to_success else 0
    }

def compare(claude_dir: Path, rl_dir: Path) -> dict:
    """Compare Claude Code results with RL model results."""
    claude_results = load_results(claude_dir)
    rl_results = load_results(rl_dir)

    claude_metrics = compute_metrics(claude_results)
    rl_metrics = compute_metrics(rl_results)

    # Per-task comparison
    all_tasks = set(claude_results.keys()) | set(rl_results.keys())
    task_comparison = {}

    for task in all_tasks:
        claude_best = 0
        rl_best = 0

        if task in claude_results:
            for r in claude_results[task]:
                if r.get("correctness"):
                    claude_best = max(claude_best, r.get("speedup", 0))

        if task in rl_results:
            for r in rl_results[task]:
                if r.get("correctness"):
                    rl_best = max(rl_best, r.get("speedup", 0))

        task_comparison[task] = {
            "claude_speedup": claude_best,
            "rl_speedup": rl_best,
            "winner": "claude" if claude_best > rl_best else "rl" if rl_best > claude_best else "tie"
        }

    return {
        "claude_metrics": claude_metrics,
        "rl_metrics": rl_metrics,
        "task_comparison": task_comparison,
        "summary": {
            "claude_wins": sum(1 for t in task_comparison.values() if t["winner"] == "claude"),
            "rl_wins": sum(1 for t in task_comparison.values() if t["winner"] == "rl"),
            "ties": sum(1 for t in task_comparison.values() if t["winner"] == "tie")
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Compare Claude Code vs RL model benchmark results")
    parser.add_argument("--claude-dir", required=True, help="Claude Code results directory")
    parser.add_argument("--rl-dir", required=True, help="RL model results directory")
    parser.add_argument("--output", default="comparison_report.json", help="Output report file")
    args = parser.parse_args()

    claude_dir = Path(os.path.expanduser(args.claude_dir))
    rl_dir = Path(os.path.expanduser(args.rl_dir))

    report = compare(claude_dir, rl_dir)

    with open(args.output, "w") as f:
        json.dump(report, indent=2, fp=f)

    # Print summary
    print(f"\n=== Comparison Report ===")
    print(f"Claude Code: {report['claude_metrics']['success_rate']:.1%} success, {report['claude_metrics']['avg_speedup']:.2f}x avg speedup")
    print(f"RL Model:    {report['rl_metrics']['success_rate']:.1%} success, {report['rl_metrics']['avg_speedup']:.2f}x avg speedup")
    print(f"\nHead-to-head: Claude wins {report['summary']['claude_wins']}, RL wins {report['summary']['rl_wins']}, Ties {report['summary']['ties']}")
    print(f"\nFull report saved to: {args.output}")

if __name__ == "__main__":
    main()
```

### Step 4: Create MCP Registration
**File: `.mcp.json` (project root)**

```json
{
  "mcpServers": {
    "kernel-bench": {
      "command": "python",
      "args": ["claudeCodeKernelBenchServer.py"],
      "cwd": "."
    }
  }
}
```

### Step 5: Result Storage Format
**Directory: `~/.inference/claude_code_output/`**

Match existing RL output format for comparison:
```
~/.inference/claude_code_output/
└── claude_code_{session_id}/
    ├── {task_name}/
    │   ├── iteration_00_cuda_kernel.py
    │   ├── iteration_00_eval.json
    │   ├── iteration_01_cuda_kernel.py
    │   └── iteration_01_eval.json
    └── summary.json
```

Each `*_eval.json` contains:
```json
{
  "compiled": true,
  "correctness": true,
  "runtime": 0.234,
  "speedup": 1.45,
  "metadata": {...},
  "model": "claude-opus-4-5-20251101",
  "timestamp": "2025-01-23T..."
}
```

## Critical Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `claudeCodeKernelBenchServer.py` | Create | MCP server for Claude Code |
| `claudeCodeKernelBench.yaml` | Create | Configuration for MCP server |
| `benchmarkCompare.py` | Create | Comparison report generator |
| `.mcp.json` | Modify | Register MCP server for project |

## Dependencies (Existing Code to Reuse)

- `kbEvalClient.py`: `kb_eval()`, `kb_eval_ref()` for kernel evaluation
- `kbEvalUtil.py`: `KernelExecResult` model
- `util.py`: Path constants (`INFERENCE_DIR`, etc.)
- `logger.py`: Logging

## Verification Plan

### Test 0: Offline Integration Test (No kbEvalServer Required)

Test the Claude Code integration without a running kbEvalServer. This validates the generation and queue submission logic independently.

**Prerequisites:**
- MCP server registered in Claude Code
- Workflow server running (for queue operations)
- kbEvalServer NOT required

**Test Steps:**

```bash
# 1. Start workflow server (for queue submission)
python workflowServer.py --host :: --port 8488
```

**In Claude Code session:**
```
You: "Run offline integration test: generate kernels for kernel_bench/level1/1_relu.py
     and kernel_bench/level1/2_matmul.py, submit to kbEval queue, but don't wait for results"
```

**Verify:**

1. **Kernel Generation Works:**
   ```bash
   # Check generated kernel files exist
   ls ~/.inference/claude_code_output/test_session/1_relu/iteration_*_cuda_kernel.py
   ls ~/.inference/claude_code_output/test_session/2_matmul/iteration_*_cuda_kernel.py

   # Verify kernel code is valid Python/Triton syntax
   python -m py_compile ~/.inference/claude_code_output/test_session/1_relu/iteration_00_cuda_kernel.py
   ```

2. **Batch Iteration Works:**
   ```bash
   # Check both tasks were processed
   ls ~/.inference/claude_code_output/test_session/
   # Should show: 1_relu/  2_matmul/  summary.json

   # Verify summary.json has entries for both tasks
   cat ~/.inference/claude_code_output/test_session/summary.json
   ```

3. **Queue Submission Works:**
   ```bash
   # Check workflow server queue has pending items
   curl http://localhost:8488/queue/qsize/{prefix_tag}/kbEval.pending

   # Peek at queue to verify work item structure
   curl http://localhost:8488/queue/peek/{prefix_tag}/kbEval.pending
   ```

**Expected Queue Item Structure:**
```json
{
  "task_path": "kernel_bench/level1/1_relu.py",
  "kernel_code": "...",
  "session_id": "test_session",
  "iteration": 0,
  "submitted_at": "2025-01-23T...",
  "status": "pending"
}
```

**MCP Tool Additions for Offline Mode:**

Add `--dry-run` / `--queue-only` mode to `eval_kernel` tool:
```python
@server.tool()
async def eval_kernel(
    task_path: str,
    kernel_code: str,
    provider: str = "local",
    queue_only: bool = False  # Submit to queue without waiting for result
) -> dict:
    if queue_only:
        # Submit to workflow queue, return immediately
        await submit_to_queue(task_path, kernel_code)
        return {"status": "queued", "queue": "kbEval.pending"}
    else:
        # Normal: call kbEvalClient and wait for result
        return await kbeval_client.kb_eval(...)
```

### Test 1: Unit Test MCP Tools
```bash
# Start MCP server
python claudeCodeKernelBenchServer.py

# Test via MCP client or direct HTTP
```

### Test 2: Integration Test via Claude Code
- Add MCP server to Claude Code config
- Use Claude Code to generate and evaluate a simple kernel (level1/1_relu.py)
- Verify results saved correctly

### Test 3: Comparison Test
- Run Claude Code on level1 tasks
- Compare against existing RL output
- Generate comparison report

### Test 4: End-to-End Workflow
```bash
# Ensure kbEval server is running
python kbEvalServer.py --local_host --port 5676 --device 0

# Run Claude Code session on benchmark tasks
# (Claude Code will use MCP tools)

# Generate comparison
python benchmarkCompare.py --claude-dir ... --rl-dir ...
```

## Configuration

Add to `mcp.json`:
```json
{
  "mcpServers": {
    "kernel_bench": {
      "command": "python",
      "args": ["claudeCodeKernelBenchServer.py"],
      "cwd": "/path/to/triton-ag"
    }
  }
}
```

## Notes

- **No Claude API key required** - Claude Code provides the LLM access; MCP tools are just utilities
- Requires kbEval server running for kernel compilation/benchmarking
- Results use same JSON schema as RL training for direct comparison
- MCP server approach provides native Claude Code integration

## Component Responsibilities

| Component | Requires Claude API? | Purpose |
|-----------|---------------------|---------|
| Claude Code | No (built-in) | LLM access for CUDA code generation |
| MCP Server | No | File I/O, kbEval HTTP calls, result storage |
| kbEvalServer | No | Kernel compilation & benchmarking (GPU) |

---

## Appendix

### Alternative: Skill (Slash Command)

Create a `/kernel-bench` skill for streamlined invocation:

**Setup:**
Create `.claude/skills/kernel-bench.md`:
```markdown
---
name: kernel-bench
description: Run kernel benchmark optimization
---

# Kernel Bench Skill

When invoked, optimize CUDA kernels for the specified task(s).

## Usage
- `/kernel-bench kernel_bench/level1/1_relu.py` - Single task
- `/kernel-bench kernel_bench/level1/` - All tasks in directory
- `/kernel-bench --session my_session` - Named session for comparison

## Workflow
1. List or identify target task(s)
2. For each task:
   a. Read PyTorch model using `get_task_details` MCP tool
   b. Generate optimized CUDA/Triton kernel
   c. Evaluate using `eval_kernel` MCP tool
   d. If not correct or slow, iterate (up to 4 attempts)
   e. Save result using `save_benchmark_result` MCP tool
3. Generate summary with success rate and speedups
```

**Usage:**
```
You: /kernel-bench kernel_bench/level1/

Claude Code executes the skill workflow automatically.
```

### Alternative: CLI Direct (No MCP)

For scripting or non-interactive use, create a CLI wrapper `claudeCodeKernelBench.py`:

```bash
# List available tasks
python claudeCodeKernelBench.py list --level level1

# Get task details (for manual inspection)
python claudeCodeKernelBench.py get kernel_bench/level1/1_relu.py

# Evaluate a kernel you've already written
python claudeCodeKernelBench.py eval kernel_bench/level1/1_relu.py \
  --kernel-file my_kernel.py \
  --session-id my_session

# Run full benchmark session
python claudeCodeKernelBench.py session --tasks kernel_bench/level1/ \
  --output-dir ~/.inference/claude_code_output/
```

This approach is useful for:
- Scripted/automated benchmarking
- Integration with CI/CD pipelines
- Non-interactive environments
