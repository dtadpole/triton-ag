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
# Use __file__ to get the actual script directory, not just '.'
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

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

# Try to import full kbEvalClient (may fail if torch not installed)
try:
    from kbEvalClient import KbEvalClient
    KBEVAL_AVAILABLE = True
    KBEVAL_IMPORT_ERROR = None
except ImportError as e:
    KBEVAL_AVAILABLE = False
    KbEvalClient = None
    KBEVAL_IMPORT_ERROR = str(e)

# Singleton kbEval client instance
_kbeval_client = None

# Adaptive semaphore for concurrency control based on GPU count
_eval_semaphore = None
# Default semaphore size - can be overridden via KBEVAL_CONCURRENCY env var
# Set to 4 as a reasonable default for multi-GPU servers
_semaphore_size = int(os.environ.get("KBEVAL_CONCURRENCY", "4"))

# Cache for reference runtimes to avoid repeated measurements
# Key: task_path (resolved absolute path), Value: ref_runtime in ms
_ref_runtime_cache: dict[str, float] = {}

# Iteration counter per (session_id, task_name) for auto-tracking
# Key: (session_id, task_name), Value: next iteration number
_iteration_counter: dict[tuple[str, str], int] = {}

# Best result tracker per (session_id, task_name) for auto-completion
# Key: (session_id, task_name), Value: {"speedup": float, "iteration": int, "strategy": str}
_best_result_tracker: dict[tuple[str, str], dict] = {}

# Default completion thresholds
DEFAULT_TARGET_SPEEDUP = 1.5
DEFAULT_MAX_ITERATIONS = 3


def get_kbeval_client(config_file: str = "kbEval.yaml"):
    """Get the kbEval client singleton."""
    global _kbeval_client
    if _kbeval_client is None:
        if KBEVAL_AVAILABLE:
            _kbeval_client = KbEvalClient(config_file=config_file)
            logger.info("[kernel-bench] KbEvalClient initialized")
        else:
            raise RuntimeError(
                f"KbEvalClient not available: {KBEVAL_IMPORT_ERROR}\n"
                "Install required dependencies: pip install torch httpx"
            )
    return _kbeval_client


async def get_eval_semaphore(provider: str = "local") -> asyncio.Semaphore:
    """Get or create adaptive semaphore based on server GPU count.

    Queries the kbEval server's /info endpoint to determine num_devices,
    then creates a semaphore with that capacity. Caches the semaphore
    for subsequent calls. Falls back to /stats if /info is not available.
    """
    global _eval_semaphore, _semaphore_size

    if _eval_semaphore is not None:
        return _eval_semaphore

    # Try to get server info for adaptive sizing
    try:
        kb_client = get_kbeval_client()
        info = await kb_client.get_info(provider)
        if info and "num_devices" in info:
            _semaphore_size = max(1, info["num_devices"])
            logger.info(f"[kernel-bench] Adaptive semaphore: {_semaphore_size} slots (from /info)")
        else:
            # Fall back to /stats endpoint
            logger.info("[kernel-bench] /info not available, trying /stats fallback...")
            try:
                import httpx
                provider_config = kb_client._provider_config_from_yaml(provider)
                base_url = provider_config.get('base_url', 'http://localhost:5676')
                async with httpx.AsyncClient(timeout=10.0, trust_env=False) as http_client:
                    response = await http_client.get(f"{base_url}/stats")
                    if response.status_code == 200:
                        stats = response.json()
                        _semaphore_size = max(1, stats.get("num_devices", 1))
                        logger.info(f"[kernel-bench] Adaptive semaphore: {_semaphore_size} slots (from /stats)")
            except Exception as stats_err:
                logger.warning(f"[kernel-bench] /stats fallback also failed: {stats_err}, using default {_semaphore_size}")
    except Exception as e:
        logger.warning(f"[kernel-bench] Could not get server info, using default semaphore size {_semaphore_size}: {e}")

    _eval_semaphore = asyncio.Semaphore(_semaphore_size)
    logger.info(f"[kernel-bench] Initialized eval semaphore with {_semaphore_size} concurrent slots")
    return _eval_semaphore


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


async def _auto_save_result(
    task_path: str,
    kernel_code: str,
    eval_result: dict,
    session_id: str,
    iteration: int
) -> str:
    """
    Internal helper to auto-save evaluation results.
    Called automatically by eval_kernel after each evaluation.
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
        "error": eval_result.get("error"),
        "timestamp": datetime.now().isoformat()
    })

    # Update overall stats
    all_iterations = []
    for task_data in summary.values():
        if isinstance(task_data, dict) and "iterations" in task_data:
            all_iterations.extend(task_data["iterations"])

    successful_iters = [i for i in all_iterations if i.get("correctness")]
    summary["_stats"] = {
        "total_tasks": len([k for k in summary.keys() if not k.startswith("_")]),
        "total_iterations": len(all_iterations),
        "success_count": len(successful_iters),
        "fail_count": len(all_iterations) - len(successful_iters),
        "avg_speedup": sum(i.get("speedup", 0) for i in successful_iters) /
                       max(1, len(successful_iters)),
        "last_updated": datetime.now().isoformat()
    }

    summary_file.write_text(json.dumps(summary, indent=2))

    logger.debug(f"[kernel-bench] Auto-saved iter {iteration} for {task_name}")

    return str(session_dir)


async def _auto_update_progress(
    session_id: str,
    task_name: str,
    iteration: int,
    result: dict,
    code_type: str = "triton"
) -> None:
    """
    Internal helper to auto-update progress tracking and auto-complete tasks.

    Called automatically by eval_kernel after each evaluation to:
    1. Update progress.json for real-time visibility
    2. Track best result so far
    3. Auto-complete when target reached or max iterations exhausted
    """
    global _best_result_tracker

    compiled = result.get("compiled", False)
    correct = result.get("correctness", False)
    speedup = result.get("speedup", 0)
    runtime = result.get("runtime", 0)
    error = result.get("error")

    # Update progress.json for real-time tracking
    try:
        await update_task_progress(
            session_id=session_id,
            task_name=task_name,
            iteration=iteration,
            strategy=code_type,  # Use code_type as strategy identifier
            compiled=compiled,
            correct=correct,
            speedup=speedup,
            runtime_ms=runtime,
            error=error
        )
    except Exception as e:
        logger.warning(f"[kernel-bench] Failed to update progress for {task_name}: {e}")

    # Track best result for this task
    tracker_key = (session_id, task_name)
    current_best = _best_result_tracker.get(tracker_key)

    if compiled and correct and speedup > 0:
        if current_best is None or speedup > current_best.get("speedup", 0):
            _best_result_tracker[tracker_key] = {
                "speedup": speedup,
                "iteration": iteration,
                "strategy": code_type
            }
            current_best = _best_result_tracker[tracker_key]

    # Check for auto-completion conditions
    should_complete = False
    completion_reason = None

    # Condition 1: Target speedup reached (1.5x by default)
    if speedup >= DEFAULT_TARGET_SPEEDUP:
        should_complete = True
        completion_reason = f"target_reached ({speedup:.2f}x >= {DEFAULT_TARGET_SPEEDUP}x)"

    # Condition 2: Max iterations reached (iteration is 0-indexed, so >= 2 means 3 iterations)
    elif iteration >= DEFAULT_MAX_ITERATIONS - 1:
        should_complete = True
        completion_reason = f"max_iterations ({iteration + 1} >= {DEFAULT_MAX_ITERATIONS})"

    # Auto-complete if conditions met
    if should_complete and current_best:
        try:
            await complete_task_progress(
                session_id=session_id,
                task_name=task_name,
                final_speedup=current_best["speedup"],
                final_iteration=current_best["iteration"],
                final_strategy=current_best["strategy"]
            )
            logger.info(f"[kernel-bench] Auto-completed {task_name}: {current_best['speedup']:.2f}x "
                       f"({completion_reason})")

            # Clean up tracker
            if tracker_key in _best_result_tracker:
                del _best_result_tracker[tracker_key]

        except Exception as e:
            logger.warning(f"[kernel-bench] Failed to auto-complete {task_name}: {e}")
    elif should_complete and not current_best:
        # All iterations failed - mark as completed with 0 speedup
        try:
            await complete_task_progress(
                session_id=session_id,
                task_name=task_name,
                final_speedup=0,
                final_iteration=iteration,
                final_strategy="failed"
            )
            logger.info(f"[kernel-bench] Auto-completed {task_name} as failed "
                       f"({completion_reason}, no successful iteration)")
        except Exception as e:
            logger.warning(f"[kernel-bench] Failed to auto-complete failed task {task_name}: {e}")


@mcp_tool()
async def eval_kernel(
    task_path: str,
    kernel_code: str,
    session_id: str = "default",
    iteration: int = None,
    provider: str = "local",
    code_type: str = "triton"
) -> dict:
    """
    Evaluate generated kernel against reference PyTorch implementation.

    Results are automatically saved after each evaluation. Iteration numbers
    are auto-tracked per (session_id, task_name) - you don't need to pass them.

    **Auto-progress tracking**: Each eval automatically updates progress.json
    for real-time visibility via get_batch_progress().

    **Auto-completion**: Tasks are automatically marked complete when:
    - Speedup >= 1.5x (target reached), OR
    - 3 iterations completed (max iterations)

    Args:
        task_path: Path to the original task file (contains reference Model)
        kernel_code: Generated CUDA/Triton kernel code (must define ModelNew)
        session_id: Session identifier for grouping results
        iteration: Optional iteration override (auto-tracked if not provided)
        provider: kbEval provider from kbEval.yaml (default: "local")
        code_type: Type of kernel code - "triton" (default) or "cuda"

    Returns:
        KernelExecResult dict with:
        - compiled: bool - Did the kernel compile?
        - correctness: bool - Does output match reference?
        - runtime: float - Kernel execution time in ms
        - speedup: float - Speedup vs reference PyTorch
        - iteration: int - The iteration number used
        - error: str|None - Error message if failed

    Example:
        >>> result = await eval_kernel(
        ...     "level1/1_relu.py",
        ...     "import triton\\n@triton.jit\\ndef relu_kernel(...): ...",
        ...     session_id="my_session"
        ... )
        >>> print(result["speedup"], result["iteration"])
        1.45 0
    """
    # Resolve task path
    task_path_obj = Path(os.path.expanduser(task_path))
    if not task_path_obj.exists():
        base_dir = Path(config["kernel_bench"]["base_dir"])
        candidate = base_dir / task_path
        if candidate.exists():
            task_path_obj = candidate

    task_name = task_path_obj.stem if task_path_obj.exists() else Path(task_path).stem

    # Auto-track iteration number per (session_id, task_name)
    iter_key = (session_id, task_name)
    if iteration is not None:
        # Use provided iteration (for backwards compatibility)
        current_iteration = iteration
    else:
        # Auto-increment
        current_iteration = _iteration_counter.get(iter_key, 0)

    # Always increment the counter for next call
    _iteration_counter[iter_key] = current_iteration + 1

    # Call kbEval and wait for result
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

        kb_client = get_kbeval_client(config_file=config.get("kbeval", {}).get("config_file", "kbEval.yaml"))

        # Get reference runtime (cached per task to avoid repeated measurements)
        task_key = str(task_path_obj.resolve())
        ref_runtime = _ref_runtime_cache.get(task_key)

        if ref_runtime is None:
            # Measure reference runtime first
            semaphore = await get_eval_semaphore(provider)
            async with semaphore:
                ref_result = await kb_client.kb_eval_ref(
                    provider=provider,
                    reference_code=reference_code,
                    run_tag=f"{session_id}_ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_tag="claude_code",
                    task_tag=task_name,
                )
            if ref_result and ref_result.get("compiled") and ref_result.get("runtime", 0) > 0:
                ref_runtime = ref_result["runtime"]
                _ref_runtime_cache[task_key] = ref_runtime
                logger.info(f"[kernel-bench] Cached reference runtime for {task_name}: {ref_runtime:.3f}ms")
            else:
                logger.warning(f"[kernel-bench] Failed to get reference runtime for {task_name}")
                ref_runtime = 0

        # Use adaptive semaphore for concurrency control
        semaphore = await get_eval_semaphore(provider)
        async with semaphore:
            result = await kb_client.kb_eval(
                provider=provider,
                reference_code=reference_code,
                generated_code=kernel_code,
                run_tag=f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_tag="claude_code",
                task_tag=task_name,
                eval_tag=f"iter_{current_iteration:02d}",
                code_type=code_type
            )

        if result is None:
            return {
                "compiled": False,
                "correctness": False,
                "error": "kbEval server unavailable or returned None",
                "runtime": 0,
                "speedup": 0
            }

        # Compute speedup from reference runtime and kernel runtime
        kernel_runtime = result.get("runtime", 0)
        if ref_runtime and ref_runtime > 0 and kernel_runtime and kernel_runtime > 0:
            speedup = ref_runtime / kernel_runtime
        else:
            speedup = 0

        # Add speedup, ref_runtime, and iteration to result
        result["speedup"] = round(speedup, 3)
        result["ref_runtime"] = ref_runtime
        result["iteration"] = current_iteration

        logger.info(f"[kernel-bench] Eval {task_name} iter {current_iteration}: "
                   f"compiled={result.get('compiled')}, correct={result.get('correctness')}, "
                   f"speedup={speedup:.2f}x (ref={ref_runtime:.3f}ms, kernel={kernel_runtime:.3f}ms)")

        # Auto-save the result (captures all iterations, success or failure)
        await _auto_save_result(
            task_path=str(task_path_obj),
            kernel_code=kernel_code,
            eval_result=result,
            session_id=session_id,
            iteration=current_iteration
        )

        # Auto-update progress tracking for real-time visibility
        await _auto_update_progress(
            session_id=session_id,
            task_name=task_name,
            iteration=current_iteration,
            result=result,
            code_type=code_type
        )

        return result

    except Exception as e:
        logger.error(f"[kernel-bench] Eval error for {task_name} iter {current_iteration}: {e}")
        error_result = {
            "compiled": False,
            "correctness": False,
            "error": str(e),
            "runtime": 0,
            "speedup": 0,
            "iteration": current_iteration
        }

        # Auto-save even failed results
        try:
            await _auto_save_result(
                task_path=str(task_path_obj),
                kernel_code=kernel_code,
                eval_result=error_result,
                session_id=session_id,
                iteration=current_iteration
            )
            # Also update progress for failed results
            await _auto_update_progress(
                session_id=session_id,
                task_name=task_name,
                iteration=current_iteration,
                result=error_result,
                code_type=code_type
            )
        except Exception as save_err:
            logger.warning(f"[kernel-bench] Failed to auto-save error result: {save_err}")

        return error_result


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

    NOTE: This is now OPTIONAL - eval_kernel() auto-saves all results.
    Use this only if you need to save additional/custom results.

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


# ============================================================================
# Phase 2: Session and Task Management Tools
# ============================================================================

import socket

# Stale marker timeout in seconds (30 minutes)
STALE_TIMEOUT_SECONDS = 30 * 60


def _is_marker_stale(marker_data: dict) -> bool:
    """Check if an in-progress marker is stale.

    A marker is stale if ANY of these conditions are true:
    - started_at > 30 minutes ago
    - PID is not running (only checked if hostname matches)
    - hostname doesn't match current host (can't verify PID)
    - marker data is corrupted/missing fields
    """
    try:
        started_at = datetime.fromisoformat(marker_data.get("started_at", ""))
        age_seconds = (datetime.now() - started_at).total_seconds()

        # Time-based staleness
        if age_seconds > STALE_TIMEOUT_SECONDS:
            return True

        # PID-based staleness (only if same host)
        marker_hostname = marker_data.get("hostname", "")
        current_hostname = socket.gethostname()

        if marker_hostname == current_hostname:
            pid = marker_data.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)  # Check if process exists
                except OSError:
                    return True  # Process not running

        return False

    except Exception:
        # Corrupted marker data
        return True


def _clean_stale_marker(marker_path: Path) -> bool:
    """Remove a stale marker file. Returns True if cleaned."""
    try:
        marker_path.unlink()
        logger.info(f"[kernel-bench] Cleaned stale marker: {marker_path}")
        return True
    except Exception as e:
        logger.warning(f"[kernel-bench] Failed to clean marker {marker_path}: {e}")
        return False


@mcp_tool()
async def init_session(
    session_id: str,
    level: str,
    task_names: list = None,
    config_override: dict = None,
    # Explicit config params for resume support
    num_workers: int = None,
    num_strategies: int = None,
    provider: str = None,
    code_type: str = None,
    original_command: str = None
) -> dict:
    """
    Initialize a new kernel bench session.

    Creates session directory and manifest file. If session already exists,
    returns existing session info without overwriting.

    Args:
        session_id: Unique session identifier
        level: Kernel bench level (e.g., "level1", "level2")
        task_names: Optional list of specific task names to include.
                    If provided, only these tasks will be tracked.
                    If None, all tasks from the level are included.
        config_override: Optional config overrides dict (legacy, prefer explicit params)
        num_workers: Number of parallel workers (for resume)
        num_strategies: Strategy mode (1=simple, 3=exploration) (for resume)
        provider: kbEval provider (for resume)
        code_type: "triton" or "cuda" (for resume)
        original_command: The original command that started the session (for resume)

    Returns:
        Dict with session info: session_id, level, created_at, status, config
    """
    output_base = Path(config["output"]["base_dir"])
    session_dir = output_base / session_id
    manifest_file = session_dir / "session_manifest.json"

    # Check if session already exists
    if manifest_file.exists():
        try:
            existing = json.loads(manifest_file.read_text())
            existing["status"] = "existing"
            logger.info(f"[kernel-bench] Session already exists: {session_id}")
            return existing
        except json.JSONDecodeError:
            pass  # Corrupted manifest, recreate

    # Create new session
    session_dir.mkdir(parents=True, exist_ok=True)

    # Get task list - use provided task_names or fetch all from level
    if task_names:
        # Use specific tasks provided by caller
        task_list = task_names
        logger.info(f"[kernel-bench] Using {len(task_list)} specified tasks")
    else:
        # Fetch all tasks from the level
        tasks = await list_kernel_bench_tasks(level=level)
        task_list = [t["name"] for t in tasks]

    # Build config from explicit params, falling back to config_override
    session_config = config_override.copy() if config_override else {}

    # Explicit params override config_override
    if num_workers is not None:
        session_config["num_workers"] = num_workers
    if num_strategies is not None:
        session_config["num_strategies"] = num_strategies
    if provider is not None:
        session_config["provider"] = provider
    if code_type is not None:
        session_config["code_type"] = code_type
    if original_command is not None:
        session_config["original_command"] = original_command

    manifest = {
        "session_id": session_id,
        "level": level,
        "created_at": datetime.now().isoformat(),
        "total_tasks": len(task_list),
        "tasks": task_list,
        "config": session_config,
        "status": "initialized"
    }

    manifest_file.write_text(json.dumps(manifest, indent=2))

    logger.info(f"[kernel-bench] Initialized session: {session_id} with {len(task_list)} tasks, config: {session_config}")

    return manifest


@mcp_tool()
async def claim_task(
    session_id: str,
    task_name: str,
    worker_id: str
) -> dict:
    """
    Atomically claim a task for processing.

    Uses exclusive file creation to prevent race conditions between workers.
    Creates an .in_progress marker with worker info.

    Args:
        session_id: Session identifier
        task_name: Name of the task to claim
        worker_id: Identifier of the claiming worker

    Returns:
        Dict with success status and claim info
    """
    output_base = Path(config["output"]["base_dir"])
    task_dir = output_base / session_id / task_name
    marker_file = task_dir / ".in_progress"

    # Create task directory if needed
    task_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    if (task_dir / "best_result.json").exists():
        return {
            "success": False,
            "reason": "already_completed",
            "task_name": task_name
        }

    # Check for existing marker
    if marker_file.exists():
        try:
            marker_data = json.loads(marker_file.read_text())

            # Check if stale
            if _is_marker_stale(marker_data):
                _clean_stale_marker(marker_file)
            else:
                return {
                    "success": False,
                    "reason": "already_claimed",
                    "claimed_by": marker_data.get("worker"),
                    "task_name": task_name
                }
        except json.JSONDecodeError:
            # Corrupted marker, clean it
            _clean_stale_marker(marker_file)

    # Try to claim with exclusive create
    marker_data = {
        "worker": worker_id,
        "started_at": datetime.now().isoformat(),
        "pid": os.getpid(),
        "hostname": socket.gethostname()
    }

    try:
        # 'x' mode = exclusive create, fails if file exists
        with open(marker_file, 'x') as f:
            json.dump(marker_data, f)

        logger.info(f"[kernel-bench] Worker {worker_id} claimed task: {task_name}")

        return {
            "success": True,
            "task_name": task_name,
            "worker_id": worker_id,
            "started_at": marker_data["started_at"]
        }

    except FileExistsError:
        # Another worker claimed it between our check and create
        return {
            "success": False,
            "reason": "race_condition",
            "task_name": task_name
        }


@mcp_tool()
async def release_task(
    session_id: str,
    task_name: str,
    error: str = None
) -> dict:
    """
    Release a claimed task, optionally recording an error.

    Removes the .in_progress marker. If error is provided, logs it to
    failures.json for analysis.

    Args:
        session_id: Session identifier
        task_name: Name of the task to release
        error: Optional error message if task failed

    Returns:
        Dict with release status
    """
    output_base = Path(config["output"]["base_dir"])
    task_dir = output_base / session_id / task_name
    marker_file = task_dir / ".in_progress"

    # Remove marker
    if marker_file.exists():
        try:
            marker_file.unlink()
            logger.info(f"[kernel-bench] Released task: {task_name}")
        except Exception as e:
            logger.warning(f"[kernel-bench] Failed to remove marker: {e}")

    # Record error if provided
    if error:
        failures_file = task_dir / "failures.json"
        failures = []

        if failures_file.exists():
            try:
                failures = json.loads(failures_file.read_text())
            except json.JSONDecodeError:
                failures = []

        failures.append({
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

        failures_file.write_text(json.dumps(failures, indent=2))
        logger.info(f"[kernel-bench] Recorded failure for {task_name}: {error[:50]}...")

    return {
        "success": True,
        "task_name": task_name,
        "error_recorded": error is not None
    }


@mcp_tool()
async def get_session_state(session_id: str) -> dict:
    """
    Get full session state with progress counts.

    Scans the session directory to determine task status:
    - pending: No task directory exists
    - in_progress: Has .in_progress marker (and not stale)
    - incomplete: Has iteration files but no best_result.json
    - completed: Has best_result.json

    Automatically cleans stale markers during scan.

    Args:
        session_id: Session identifier

    Returns:
        Dict with total, completed, in_progress, pending counts and task lists
    """
    output_base = Path(config["output"]["base_dir"])
    session_dir = output_base / session_id
    manifest_file = session_dir / "session_manifest.json"

    if not manifest_file.exists():
        return {"error": f"Session not found: {session_id}"}

    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return {"error": f"Corrupted manifest for session: {session_id}"}

    all_tasks = manifest.get("tasks", [])

    completed = []
    in_progress = []
    incomplete = []
    pending = []
    stale_cleaned = 0

    for task_name in all_tasks:
        task_dir = session_dir / task_name

        if not task_dir.exists():
            pending.append(task_name)
            continue

        # Check for completion marker
        if (task_dir / "best_result.json").exists():
            completed.append(task_name)
            continue

        # Check for in-progress marker
        marker_file = task_dir / ".in_progress"
        if marker_file.exists():
            try:
                marker_data = json.loads(marker_file.read_text())
                if _is_marker_stale(marker_data):
                    _clean_stale_marker(marker_file)
                    stale_cleaned += 1
                    # Falls through to incomplete check
                else:
                    in_progress.append({
                        "task": task_name,
                        "worker": marker_data.get("worker"),
                        "started_at": marker_data.get("started_at")
                    })
                    continue
            except json.JSONDecodeError:
                _clean_stale_marker(marker_file)
                stale_cleaned += 1

        # Check for partial work (has iteration files but not complete)
        iteration_files = list(task_dir.glob("iteration_*_cuda_kernel.py"))
        if iteration_files:
            incomplete.append(task_name)
        else:
            pending.append(task_name)

    # Calculate statistics
    completed_speedups = []
    for task_name in completed:
        result_file = session_dir / task_name / "best_result.json"
        if result_file.exists():
            try:
                result = json.loads(result_file.read_text())
                if result.get("speedup"):
                    completed_speedups.append(result["speedup"])
            except json.JSONDecodeError:
                pass

    avg_speedup = sum(completed_speedups) / len(completed_speedups) if completed_speedups else 0

    result = {
        "session_id": session_id,
        "level": manifest.get("level"),
        "created_at": manifest.get("created_at"),
        "config": manifest.get("config", {}),  # Include config for resume
        "total": len(all_tasks),
        "completed": len(completed),
        "in_progress": len(in_progress),
        "incomplete": len(incomplete),
        "pending": len(pending),
        "stale_cleaned": stale_cleaned,
        "avg_speedup": round(avg_speedup, 3),
        "completed_tasks": completed,
        "in_progress_tasks": in_progress,
        "incomplete_tasks": incomplete,
        "pending_tasks": pending[:20]  # Limit for readability
    }

    if len(pending) > 20:
        result["pending_tasks_truncated"] = True
        result["pending_count"] = len(pending)

    logger.info(f"[kernel-bench] Session {session_id}: "
               f"{len(completed)}/{len(all_tasks)} complete, "
               f"{len(in_progress)} in-progress, {stale_cleaned} stale cleaned")

    return result


@mcp_tool()
async def get_pending_tasks(
    session_id: str,
    limit: int = 10
) -> dict:
    """
    Get list of tasks available for claiming.

    Returns tasks that are:
    - Not completed (no best_result.json)
    - Not in progress (no valid .in_progress marker)

    Includes both pending (never started) and incomplete (partial work).

    Args:
        session_id: Session identifier
        limit: Maximum number of tasks to return

    Returns:
        Dict with list of claimable tasks and their paths
    """
    # Get full state (not using get_session_state to avoid truncation)
    output_base = Path(config["output"]["base_dir"])
    session_dir = output_base / session_id
    manifest_file = session_dir / "session_manifest.json"

    if not manifest_file.exists():
        return {"error": f"Session not found: {session_id}"}

    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return {"error": f"Corrupted manifest for session: {session_id}"}

    all_tasks = manifest.get("tasks", [])
    level = manifest.get("level", "level1")

    claimable = []

    for task_name in all_tasks:
        task_dir = session_dir / task_name

        # Skip completed tasks
        if task_dir.exists() and (task_dir / "best_result.json").exists():
            continue

        # Skip in-progress tasks (with valid markers)
        marker_file = task_dir / ".in_progress"
        if marker_file.exists():
            try:
                marker_data = json.loads(marker_file.read_text())
                if not _is_marker_stale(marker_data):
                    continue  # Valid in-progress, skip
                # Stale marker - will be cleaned, task is claimable
                _clean_stale_marker(marker_file)
            except json.JSONDecodeError:
                _clean_stale_marker(marker_file)

        # Determine status
        if task_dir.exists():
            iteration_files = list(task_dir.glob("iteration_*_cuda_kernel.py"))
            status = "incomplete" if iteration_files else "pending"
        else:
            status = "pending"

        claimable.append({
            "task_name": task_name,
            "status": status,
            "task_path": f"{level}/{task_name}.py"
        })

    # Sort by name for consistent ordering
    claimable.sort(key=lambda x: x["task_name"])

    return {
        "session_id": session_id,
        "available_count": len(claimable),
        "tasks": claimable[:limit],
        "truncated": len(claimable) > limit
    }


# ============================================================================
# Phase 2.5: Real-time Progress Tracking Tools
# ============================================================================


@mcp_tool()
async def update_task_progress(
    session_id: str,
    task_name: str,
    iteration: int,
    strategy: str,
    compiled: bool,
    correct: bool,
    speedup: float = 0.0,
    runtime_ms: float = 0.0,
    error: str = None
) -> dict:
    """
    Update task progress after each eval_kernel call.

    Workers should call this after every eval_kernel() to enable real-time
    progress tracking. Updates the progress.json file in the task directory.

    Args:
        session_id: Session identifier
        task_name: Name of the task being optimized
        iteration: Current iteration number (0-indexed)
        strategy: Name of the strategy used (e.g., "vectorized_loads", "block_tuning_512")
        compiled: Whether the kernel compiled successfully
        correct: Whether the kernel produced correct output
        speedup: Speedup vs reference (0 if not measured)
        runtime_ms: Kernel runtime in milliseconds
        error: Error message if compilation or correctness failed

    Returns:
        Dict with update status and current best result
    """
    output_base = Path(config["output"]["base_dir"])
    task_dir = output_base / session_id / task_name
    progress_file = task_dir / "progress.json"

    task_dir.mkdir(parents=True, exist_ok=True)

    # Load existing progress or create new
    progress = {
        "task_name": task_name,
        "worker_id": None,
        "status": "in_progress",
        "started_at": None,
        "completed_at": None,
        "config": {
            "max_iterations": 3,
            "strategies_per_iteration": 1
        },
        "iterations": [],
        "best": None
    }

    if progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text())
        except json.JSONDecodeError:
            pass

    # Get worker info from in_progress marker if available
    marker_file = task_dir / ".in_progress"
    if marker_file.exists():
        try:
            marker_data = json.loads(marker_file.read_text())
            progress["worker_id"] = marker_data.get("worker")
            if not progress["started_at"]:
                progress["started_at"] = marker_data.get("started_at")
        except json.JSONDecodeError:
            pass

    # Add iteration result
    iteration_data = {
        "iteration": iteration,
        "strategy": strategy,
        "compiled": compiled,
        "correct": correct,
        "speedup": round(speedup, 3) if speedup else 0,
        "runtime_ms": round(runtime_ms, 4) if runtime_ms else 0,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }

    # Update or append iteration
    existing_idx = None
    for i, it in enumerate(progress["iterations"]):
        if it["iteration"] == iteration:
            existing_idx = i
            break

    if existing_idx is not None:
        progress["iterations"][existing_idx] = iteration_data
    else:
        progress["iterations"].append(iteration_data)

    # Update best result
    successful_iters = [it for it in progress["iterations"]
                       if it.get("compiled") and it.get("correct") and it.get("speedup", 0) > 0]
    if successful_iters:
        best_iter = max(successful_iters, key=lambda x: x.get("speedup", 0))
        progress["best"] = {
            "iteration": best_iter["iteration"],
            "strategy": best_iter["strategy"],
            "speedup": best_iter["speedup"]
        }

    # Save progress
    progress_file.write_text(json.dumps(progress, indent=2))

    logger.info(f"[kernel-bench] Progress update: {task_name} iter {iteration} "
               f"({strategy}) - compiled={compiled}, correct={correct}, speedup={speedup:.2f}x")

    return {
        "success": True,
        "task_name": task_name,
        "iteration": iteration,
        "best": progress["best"]
    }


@mcp_tool()
async def complete_task_progress(
    session_id: str,
    task_name: str,
    final_speedup: float,
    final_iteration: int,
    final_strategy: str
) -> dict:
    """
    Mark a task as completed and save the best result.

    Workers should call this after optimization is done (either target reached
    or max iterations exhausted). Creates best_result.json and updates progress.json.

    Args:
        session_id: Session identifier
        task_name: Name of the completed task
        final_speedup: Best achieved speedup
        final_iteration: Iteration that achieved best result
        final_strategy: Strategy that achieved best result

    Returns:
        Dict with completion status
    """
    output_base = Path(config["output"]["base_dir"])
    task_dir = output_base / session_id / task_name
    progress_file = task_dir / "progress.json"
    best_result_file = task_dir / "best_result.json"
    marker_file = task_dir / ".in_progress"

    # Update progress.json
    progress = {}
    if progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text())
        except json.JSONDecodeError:
            pass

    progress["status"] = "completed"
    progress["completed_at"] = datetime.now().isoformat()
    progress["best"] = {
        "iteration": final_iteration,
        "strategy": final_strategy,
        "speedup": round(final_speedup, 3)
    }

    progress_file.write_text(json.dumps(progress, indent=2))

    # Create best_result.json (used by get_session_state for completion detection)
    best_result = {
        "task_name": task_name,
        "speedup": round(final_speedup, 3),
        "iteration": final_iteration,
        "strategy": final_strategy,
        "completed_at": datetime.now().isoformat()
    }
    best_result_file.write_text(json.dumps(best_result, indent=2))

    # Remove in_progress marker
    if marker_file.exists():
        try:
            marker_file.unlink()
        except Exception:
            pass

    logger.info(f"[kernel-bench] Task completed: {task_name} - {final_speedup:.2f}x "
               f"(iter {final_iteration}, {final_strategy})")

    return {
        "success": True,
        "task_name": task_name,
        "final_speedup": final_speedup,
        "final_iteration": final_iteration,
        "final_strategy": final_strategy
    }


@mcp_tool()
async def get_task_progress(
    session_id: str,
    task_name: str
) -> dict:
    """
    Get detailed progress for a single task.

    Returns all iteration results, current status, and best result.

    Args:
        session_id: Session identifier
        task_name: Name of the task

    Returns:
        Dict with full task progress including all iterations
    """
    output_base = Path(config["output"]["base_dir"])
    task_dir = output_base / session_id / task_name
    progress_file = task_dir / "progress.json"

    if not task_dir.exists():
        return {
            "task_name": task_name,
            "status": "pending",
            "iterations_planned": 3,
            "iterations_done": 0,
            "iterations": [],
            "best": None
        }

    # Check for completion
    if (task_dir / "best_result.json").exists():
        try:
            best_result = json.loads((task_dir / "best_result.json").read_text())
        except json.JSONDecodeError:
            best_result = {}

        progress = {}
        if progress_file.exists():
            try:
                progress = json.loads(progress_file.read_text())
            except json.JSONDecodeError:
                pass

        return {
            "task_name": task_name,
            "status": "completed",
            "worker_id": progress.get("worker_id"),
            "started_at": progress.get("started_at"),
            "completed_at": progress.get("completed_at") or best_result.get("completed_at"),
            "iterations_planned": progress.get("config", {}).get("max_iterations", 3),
            "iterations_done": len(progress.get("iterations", [])),
            "iterations": progress.get("iterations", []),
            "best": {
                "iteration": best_result.get("iteration"),
                "strategy": best_result.get("strategy"),
                "speedup": best_result.get("speedup")
            }
        }

    # Check if in progress
    marker_file = task_dir / ".in_progress"
    worker_id = None
    started_at = None

    if marker_file.exists():
        try:
            marker_data = json.loads(marker_file.read_text())
            if not _is_marker_stale(marker_data):
                worker_id = marker_data.get("worker")
                started_at = marker_data.get("started_at")
        except json.JSONDecodeError:
            pass

    # Load progress file
    progress = {}
    if progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text())
        except json.JSONDecodeError:
            pass

    iterations = progress.get("iterations", [])

    # Determine status
    if worker_id:
        status = "in_progress"
    elif iterations:
        status = "incomplete"
    else:
        status = "pending"

    return {
        "task_name": task_name,
        "status": status,
        "worker_id": worker_id or progress.get("worker_id"),
        "started_at": started_at or progress.get("started_at"),
        "completed_at": None,
        "iterations_planned": progress.get("config", {}).get("max_iterations", 3),
        "iterations_done": len(iterations),
        "iterations": iterations,
        "best": progress.get("best")
    }


@mcp_tool()
async def get_batch_progress(
    session_id: str,
    include_iterations: bool = False
) -> dict:
    """
    Get progress summary for all tasks in a session.

    Provides a high-level overview of batch progress with per-task status
    and iteration counts. Use include_iterations=True for detailed iteration
    data per task (larger response).

    Args:
        session_id: Session identifier
        include_iterations: If True, include full iteration details per task

    Returns:
        Dict with summary stats and per-task progress
    """
    output_base = Path(config["output"]["base_dir"])
    session_dir = output_base / session_id
    manifest_file = session_dir / "session_manifest.json"

    if not manifest_file.exists():
        return {"error": f"Session not found: {session_id}"}

    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return {"error": f"Corrupted manifest for session: {session_id}"}

    all_tasks = manifest.get("tasks", [])

    # Collect task progress
    tasks_progress = []
    completed_count = 0
    in_progress_count = 0
    failed_count = 0
    pending_count = 0
    all_speedups = []

    for task_name in all_tasks:
        task_dir = session_dir / task_name
        progress_file = task_dir / "progress.json"

        task_info = {
            "name": task_name,
            "status": "pending",
            "worker": None,
            "iterations_planned": 3,
            "iterations_done": 0,
            "best_speedup": None,
            "last_iteration": None
        }

        if not task_dir.exists():
            pending_count += 1
            tasks_progress.append(task_info)
            continue

        # Load progress
        progress = {}
        if progress_file.exists():
            try:
                progress = json.loads(progress_file.read_text())
            except json.JSONDecodeError:
                pass

        iterations = progress.get("iterations", [])
        task_info["worker"] = progress.get("worker_id")
        task_info["iterations_planned"] = progress.get("config", {}).get("max_iterations", 3)
        task_info["iterations_done"] = len(iterations)

        if include_iterations:
            task_info["iterations"] = iterations

        if iterations:
            task_info["last_iteration"] = {
                "compiled": iterations[-1].get("compiled"),
                "correct": iterations[-1].get("correct"),
                "speedup": iterations[-1].get("speedup"),
                "error": iterations[-1].get("error")
            }

        best = progress.get("best")
        if best:
            task_info["best_speedup"] = best.get("speedup")

        # Determine status
        if (task_dir / "best_result.json").exists():
            task_info["status"] = "completed"
            completed_count += 1
            if task_info["best_speedup"]:
                all_speedups.append(task_info["best_speedup"])
        else:
            # Check in-progress marker
            marker_file = task_dir / ".in_progress"
            if marker_file.exists():
                try:
                    marker_data = json.loads(marker_file.read_text())
                    if not _is_marker_stale(marker_data):
                        task_info["status"] = "in_progress"
                        task_info["worker"] = marker_data.get("worker")
                        in_progress_count += 1
                    else:
                        # Stale - check for failures
                        if iterations and all(not it.get("correct") for it in iterations):
                            task_info["status"] = "failed"
                            failed_count += 1
                        else:
                            task_info["status"] = "incomplete"
                            pending_count += 1  # Incomplete counts as pending for work
                except json.JSONDecodeError:
                    task_info["status"] = "incomplete"
                    pending_count += 1
            elif iterations:
                # Has work but no marker - incomplete
                if all(not it.get("correct") for it in iterations) and len(iterations) >= 3:
                    task_info["status"] = "failed"
                    failed_count += 1
                else:
                    task_info["status"] = "incomplete"
                    pending_count += 1
            else:
                pending_count += 1

        tasks_progress.append(task_info)

    # Sort by status priority: in_progress > incomplete > pending > failed > completed
    status_order = {"in_progress": 0, "incomplete": 1, "pending": 2, "failed": 3, "completed": 4}
    tasks_progress.sort(key=lambda x: (status_order.get(x["status"], 5), x["name"]))

    avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0

    return {
        "session_id": session_id,
        "level": manifest.get("level"),
        "summary": {
            "total": len(all_tasks),
            "completed": completed_count,
            "in_progress": in_progress_count,
            "failed": failed_count,
            "pending": pending_count,
            "avg_speedup": round(avg_speedup, 3)
        },
        "tasks": tasks_progress
    }


if __name__ == "__main__":
    if MCP_AVAILABLE and mcp is not None:
        mcp.run()
    else:
        print("ERROR: MCP package not available. Install with: pip install mcp")
        print("For standalone testing, import this module and call functions directly.")
        exit(1)
