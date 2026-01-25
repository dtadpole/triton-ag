#!/usr/bin/env python
"""
MCP Server for Claude Code Kernel Bench Integration.

This server provides tools for Claude Code to:
1. List available kernel benchmark tasks
2. Get task details (PyTorch model code)
3. Evaluate generated kernels via kbEvalServer
4. Save benchmark results for comparison with RL training outputs

Usage:
    python claudeCodeKernelBenchServer.py

Configure in .mcp.json:
    {
        "mcpServers": {
            "kernel-bench": {
                "command": "python",
                "args": ["claudeCodeKernelBenchServer.py"],
                "cwd": "/path/to/triton-ag"
            }
        }
    }
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import logging

# Ensure the project directory is in sys.path for local module imports
# This is needed when the MCP server runs as a subprocess from Claude Code
if '.' not in sys.path:
    sys.path.insert(0, '.')

# Optional yaml import - fall back to defaults if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Optional project logger - fall back to standard logging
try:
    from logger import logger
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

# These imports are optional for standalone testing
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None

try:
    from kbEvalClient import KbEvalClient
    KBEVAL_AVAILABLE = True
except ImportError:
    KBEVAL_AVAILABLE = False
    KbEvalClient = None

try:
    from workflowClient import WorkflowClient
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    WorkflowClient = None


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        "kernel_bench": {
            "base_dir": "./kernel_bench",
            "levels": ["level1", "level2", "level3", "level4"]
        },
        "kbeval": {
            "default_provider": "local",
            "config_file": "kbEval.yaml"
        },
        "workflow": {
            "prefix_tag": "claude_code",
            "eval_queue": "kbEval.pending",
            "config_file": "workflow.yaml",
            "provider_name": "local"
        },
        "output": {
            "base_dir": os.path.expanduser("~/.inference/claude_code_output")
        },
        "iteration": {
            "max_turns": 4,
            "early_stop_speedup": 1.3
        }
    }


def load_config(config_path: str = "claudeCodeKernelBench.yaml") -> dict:
    """Load MCP server configuration."""
    config_path = Path(config_path)

    # Return defaults if config doesn't exist or yaml not available
    if not config_path.exists() or not YAML_AVAILABLE:
        if config_path.exists() and not YAML_AVAILABLE:
            logger.warning(f"YAML not available, using default config (install PyYAML to use {config_path})")
        return get_default_config()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Expand environment variables in paths
    if "kernel_bench" in config:
        config["kernel_bench"]["base_dir"] = os.path.expanduser(
            os.path.expandvars(config["kernel_bench"].get("base_dir", "~/KernelBench/KernelBench"))
        )
    if "output" in config:
        config["output"]["base_dir"] = os.path.expanduser(
            os.path.expandvars(config["output"].get("base_dir", "~/.inference/claude_code_output"))
        )

    return config


# Initialize config first (always works)
config = load_config()

# Initialize MCP server only if available
if MCP_AVAILABLE:
    mcp = FastMCP("kernel-bench")
    # Use the actual decorator
    def mcp_tool():
        return mcp.tool()
else:
    mcp = None
    logger.warning("MCP not available - install 'mcp' package for MCP server functionality")
    # No-op decorator for when MCP is not available
    def mcp_tool():
        def decorator(func):
            return func
        return decorator


@mcp_tool()
async def list_kernel_bench_tasks(level: str = None) -> list[dict]:
    """
    List available kernel benchmark tasks.

    Args:
        level: Filter by level (e.g., "level1", "level2", "level3").
               If None, lists all levels.

    Returns:
        List of task info dicts with path, name, and level.

    Example:
        >>> await list_kernel_bench_tasks("level1")
        [{"path": "/path/to/1_relu.py", "name": "1_relu", "level": "level1"}, ...]
    """
    base_dir = Path(config["kernel_bench"]["base_dir"])
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
        else:
            logger.warning(f"Level directory not found: {level_dir}")

    logger.info(f"[kernel-bench] Listed {len(tasks)} tasks from levels: {levels}")
    return tasks


@mcp_tool()
async def get_task_details(task_path: str) -> dict:
    """
    Get PyTorch model code and specifications for a benchmark task.

    Args:
        task_path: Path to the task file (absolute or relative to kernel_bench base).
                   Examples:
                   - "kernel_bench/level1/1_relu.py"
                   - "level1/1_relu.py"
                   - "/full/path/to/1_relu.py"

    Returns:
        Dict with:
        - path: Resolved absolute path
        - source_code: Full Python source code of the task
        - name: Task name (filename without .py)

    The source_code contains the PyTorch Model class that needs to be
    optimized with a CUDA/Triton implementation in ModelNew.
    """
    task_path = Path(os.path.expanduser(task_path))

    # Try to resolve relative paths
    if not task_path.exists():
        # Try relative to base_dir
        base_dir = Path(config["kernel_bench"]["base_dir"])
        candidate = base_dir / task_path
        if candidate.exists():
            task_path = candidate
        else:
            # Try with kernel_bench prefix stripped if present
            path_str = str(task_path)
            if path_str.startswith("kernel_bench/"):
                candidate = base_dir / path_str[len("kernel_bench/"):]
                if candidate.exists():
                    task_path = candidate

    if not task_path.exists():
        return {
            "error": f"Task file not found: {task_path}",
            "tried_paths": [str(task_path)],
            "base_dir": config["kernel_bench"]["base_dir"]
        }

    source_code = task_path.read_text()

    logger.info(f"[kernel-bench] Read task: {task_path.name} ({len(source_code)} chars)")

    return {
        "path": str(task_path),
        "source_code": source_code,
        "name": task_path.stem
    }


@mcp_tool()
async def eval_kernel(
    task_path: str,
    kernel_code: str,
    session_id: str = "default",
    iteration: int = 0,
    provider: str = "local",
    queue_only: bool = False
) -> dict:
    """
    Evaluate generated kernel against reference PyTorch implementation.

    Args:
        task_path: Path to the original task file (contains reference Model)
        kernel_code: Generated CUDA/Triton kernel code (must define ModelNew)
        session_id: Session identifier for grouping results
        iteration: Iteration number within the task (0, 1, 2, ...)
        provider: kbEval provider from kbEval.yaml (default: "local")
        queue_only: If True, submit to workflow queue without waiting for result.
                    Useful for offline testing when kbEvalServer is not running.

    Returns:
        If queue_only=False:
            KernelExecResult dict with:
            - compiled: bool - Did the kernel compile?
            - correctness: bool - Does output match reference?
            - runtime: float - Kernel execution time in ms
            - speedup: float - Speedup vs reference PyTorch
            - error: str|None - Error message if failed

        If queue_only=True:
            {"status": "queued", "queue": queue_name, "work_item": {...}}

    Example:
        >>> result = await eval_kernel(
        ...     "level1/1_relu.py",
        ...     "import triton\\n@triton.jit\\ndef relu_kernel(...): ...",
        ...     session_id="my_session",
        ...     iteration=0
        ... )
        >>> print(result["speedup"])
        1.45
    """
    # Resolve task path
    task_path_obj = Path(os.path.expanduser(task_path))
    if not task_path_obj.exists():
        base_dir = Path(config["kernel_bench"]["base_dir"])
        candidate = base_dir / task_path
        if candidate.exists():
            task_path_obj = candidate

    task_name = task_path_obj.stem if task_path_obj.exists() else Path(task_path).stem

    if queue_only:
        # Submit to workflow queue without waiting for result
        queue_name = config.get("workflow", {}).get("eval_queue", "kbEval.pending")
        prefix_tag = config.get("workflow", {}).get("prefix_tag", "claude_code")

        work_item = {
            "task_path": str(task_path),
            "kernel_code": kernel_code,
            "session_id": session_id,
            "iteration": iteration,
            "submitted_at": datetime.now().isoformat(),
            "status": "pending"
        }

        if not WORKFLOW_AVAILABLE:
            logger.warning(f"[kernel-bench] WorkflowClient not available - cannot queue work item")
            return {
                "status": "queue_error",
                "error": "WorkflowClient not available - install workflowClient module",
                "work_item": work_item
            }

        try:
            workflow_client = WorkflowClient(
                prefix_tag=prefix_tag,
                provider_name=config.get("workflow", {}).get("provider_name", "local")
            )
            await workflow_client.enqueue(queue_name, work_item, create_queue=True)
            logger.info(f"[kernel-bench] Queued {task_name} iteration {iteration} to {queue_name}")
            return {"status": "queued", "queue": queue_name, "work_item": work_item}
        except Exception as e:
            logger.warning(f"[kernel-bench] Queue submission failed: {e}")
            return {
                "status": "queue_error",
                "error": str(e),
                "work_item": work_item
            }

    # Normal mode: call kbEval and wait for result
    if not KBEVAL_AVAILABLE:
        return {
            "compiled": False,
            "correctness": False,
            "error": "KbEvalClient not available - install kbEvalClient module",
            "runtime": 0,
            "speedup": 0
        }

    try:
        # Read reference code
        if not task_path_obj.exists():
            return {
                "compiled": False,
                "correctness": False,
                "error": f"Task file not found: {task_path}",
                "runtime": 0,
                "speedup": 0
            }

        reference_code = task_path_obj.read_text()

        kb_client = KbEvalClient(config_file=config.get("kbeval", {}).get("config_file", "kbEval.yaml"))

        result = await kb_client.kb_eval(
            provider=provider,
            reference_code=reference_code,
            generated_code=kernel_code,
            run_tag=f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_tag="claude_code",
            task_tag=task_name,
            eval_tag=f"iter_{iteration:02d}",
            code_type="cuda"
        )

        if result is None:
            return {
                "compiled": False,
                "correctness": False,
                "error": "kbEval server unavailable or returned None",
                "runtime": 0,
                "speedup": 0
            }

        logger.info(f"[kernel-bench] Eval {task_name} iter {iteration}: "
                   f"compiled={result.get('compiled')}, correct={result.get('correctness')}, "
                   f"speedup={result.get('speedup', 0):.2f}x")

        return result

    except Exception as e:
        logger.error(f"[kernel-bench] Eval error: {e}")
        return {
            "compiled": False,
            "correctness": False,
            "error": str(e),
            "runtime": 0,
            "speedup": 0
        }


@mcp_tool()
async def save_benchmark_result(
    task_path: str,
    kernel_code: str,
    eval_result: dict,
    session_id: str,
    iteration: int = 0
) -> str:
    """
    Save benchmark result in format compatible with RL training output.

    This stores results in ~/.inference/claude_code_output/ using the same
    structure as RL training outputs, enabling direct comparison.

    Args:
        task_path: Path to the original task file
        kernel_code: Generated kernel code to save
        eval_result: Evaluation result from eval_kernel
        session_id: Session identifier (becomes subdirectory name)
        iteration: Iteration number

    Returns:
        Path to the saved result directory.

    Directory structure:
        ~/.inference/claude_code_output/
        └── {session_id}/
            ├── {task_name}/
            │   ├── iteration_00_cuda_kernel.py
            │   ├── iteration_00_eval.json
            │   └── ...
            └── summary.json
    """
    output_base = Path(config["output"]["base_dir"])
    task_name = Path(task_path).stem

    session_dir = output_base / session_id / task_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save kernel code
    kernel_file = session_dir / f"iteration_{iteration:02d}_cuda_kernel.py"
    kernel_file.write_text(kernel_code)

    # Save eval result with metadata
    eval_file = session_dir / f"iteration_{iteration:02d}_eval.json"
    eval_result_copy = eval_result.copy()
    eval_result_copy["model"] = "claude-code"
    eval_result_copy["timestamp"] = datetime.now().isoformat()
    eval_result_copy["task_name"] = task_name
    eval_result_copy["iteration"] = iteration
    eval_file.write_text(json.dumps(eval_result_copy, indent=2))

    # Update summary
    summary_file = output_base / session_id / "summary.json"
    summary = {}
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            summary = {}

    if task_name not in summary:
        summary[task_name] = {"iterations": []}

    summary[task_name]["iterations"].append({
        "iteration": iteration,
        "compiled": eval_result.get("compiled", False),
        "correctness": eval_result.get("correctness", False),
        "speedup": eval_result.get("speedup", 0),
        "timestamp": datetime.now().isoformat()
    })

    # Update overall stats
    all_iterations = []
    for task_data in summary.values():
        if isinstance(task_data, dict) and "iterations" in task_data:
            all_iterations.extend(task_data["iterations"])

    summary["_stats"] = {
        "total_tasks": len([k for k in summary.keys() if not k.startswith("_")]),
        "total_iterations": len(all_iterations),
        "success_count": sum(1 for i in all_iterations if i.get("correctness")),
        "avg_speedup": sum(i.get("speedup", 0) for i in all_iterations if i.get("correctness")) /
                       max(1, sum(1 for i in all_iterations if i.get("correctness"))),
        "last_updated": datetime.now().isoformat()
    }

    summary_file.write_text(json.dumps(summary, indent=2))

    logger.info(f"[kernel-bench] Saved result: {session_dir}")

    return str(session_dir)


@mcp_tool()
async def get_session_summary(session_id: str) -> dict:
    """
    Get summary statistics for a benchmark session.

    Args:
        session_id: Session identifier

    Returns:
        Dict with session statistics including success rate, speedups, etc.
    """
    output_base = Path(config["output"]["base_dir"])
    summary_file = output_base / session_id / "summary.json"

    if not summary_file.exists():
        return {"error": f"Session not found: {session_id}"}

    try:
        summary = json.loads(summary_file.read_text())
        return summary
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    if MCP_AVAILABLE and mcp is not None:
        mcp.run()
    else:
        print("ERROR: MCP package not available. Install with: pip install mcp")
        print("For standalone testing, import this module and call functions directly.")
        exit(1)
