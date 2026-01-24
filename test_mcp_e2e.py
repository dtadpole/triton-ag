#!/usr/bin/env python3
"""
Test script for MCP server End-to-End flow (Test 3).

This script tests:
1. list_kernel_bench_tasks and get_task_details (from Test 2)
2. eval_kernel with queue_only=True
3. save_benchmark_result
4. get_session_summary

Usage:
    # Without workflow server (tests file operations only):
    python3 test_mcp_e2e.py

    # With workflow server running:
    python3 test_mcp_e2e.py --with-queue
"""
import asyncio
import argparse
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, '.')

from claudeCodeKernelBenchServer import (
    list_kernel_bench_tasks,
    get_task_details,
    eval_kernel,
    save_benchmark_result,
    get_session_summary,
    config
)

# Sample generated kernel code for testing
SAMPLE_KERNEL_CODE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def sample_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * x  # Simple square operation
    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        sample_kernel[grid](x, out, n, BLOCK_SIZE=1024)
        return out
'''


async def test_list_and_get_tasks():
    """Test list_kernel_bench_tasks and get_task_details (from Test 2)."""
    print("\n=== Test: list_kernel_bench_tasks ===")
    tasks = await list_kernel_bench_tasks("level1")
    print(f"Found {len(tasks)} tasks in level1")
    if len(tasks) == 0:
        print("FAIL: No tasks found")
        return None
    print(f"First task: {tasks[0]['name']}")

    first_task = tasks[0]['name'] + ".py"
    print(f"\n=== Test: get_task_details ({first_task}) ===")
    result = await get_task_details(f"level1/{first_task}")
    if "error" in result:
        print(f"FAIL: {result['error']}")
        return None
    print(f"Task: {result['name']}")
    print(f"Source length: {len(result['source_code'])} chars")

    return tasks[0]


async def test_eval_kernel_queue_only(task_path: str, test_with_queue: bool):
    """Test eval_kernel with queue_only=True."""
    print("\n=== Test: eval_kernel (queue_only=True) ===")

    result = await eval_kernel(
        task_path=task_path,
        kernel_code=SAMPLE_KERNEL_CODE,
        session_id="test_e2e_flow",
        iteration=0,
        provider="local",
        queue_only=True
    )

    if "status" in result:
        if result["status"] == "queued":
            print(f"SUCCESS: Queued to {result['queue']}")
            print(f"Work item submitted_at: {result['work_item']['submitted_at']}")
            return True
        elif result["status"] == "queue_error":
            if test_with_queue:
                print(f"FAIL: Queue error - {result['error']}")
                return False
            else:
                print(f"EXPECTED: Queue error (workflow server not running) - {result['error']}")
                print("This is OK for Test 3 without --with-queue flag")
                return True  # Expected when workflow server not running

    print(f"UNEXPECTED result: {result}")
    return False


async def test_save_benchmark_result(task_path: str, output_dir: str):
    """Test save_benchmark_result."""
    print("\n=== Test: save_benchmark_result ===")

    # Create a mock eval result
    mock_eval_result = {
        "compiled": True,
        "correctness": True,
        "runtime": 0.234,
        "speedup": 1.45
    }

    result_path = await save_benchmark_result(
        task_path=task_path,
        kernel_code=SAMPLE_KERNEL_CODE,
        eval_result=mock_eval_result,
        session_id="test_e2e_flow",
        iteration=0
    )

    print(f"Saved to: {result_path}")

    # Verify files exist
    result_dir = Path(result_path)
    kernel_file = result_dir / "iteration_00_cuda_kernel.py"
    eval_file = result_dir / "iteration_00_eval.json"

    if not kernel_file.exists():
        print(f"FAIL: Kernel file not found: {kernel_file}")
        return False
    if not eval_file.exists():
        print(f"FAIL: Eval file not found: {eval_file}")
        return False

    # Verify kernel file is valid Python
    try:
        import py_compile
        py_compile.compile(str(kernel_file), doraise=True)
        print(f"Kernel file syntax: OK")
    except py_compile.PyCompileError as e:
        print(f"FAIL: Kernel file has syntax error: {e}")
        return False

    # Verify eval file is valid JSON with expected fields
    try:
        eval_data = json.loads(eval_file.read_text())
        required_fields = ["compiled", "correctness", "runtime", "speedup", "model", "timestamp"]
        missing = [f for f in required_fields if f not in eval_data]
        if missing:
            print(f"FAIL: Eval JSON missing fields: {missing}")
            return False
        print(f"Eval file structure: OK (model={eval_data['model']})")
    except json.JSONDecodeError as e:
        print(f"FAIL: Eval file is not valid JSON: {e}")
        return False

    return True


async def test_get_session_summary():
    """Test get_session_summary."""
    print("\n=== Test: get_session_summary ===")

    summary = await get_session_summary("test_e2e_flow")

    if "error" in summary:
        print(f"FAIL: {summary['error']}")
        return False

    # Check for expected structure
    if "_stats" not in summary:
        print("FAIL: Summary missing _stats section")
        return False

    stats = summary["_stats"]
    print(f"Session stats:")
    print(f"  Total tasks: {stats.get('total_tasks', 0)}")
    print(f"  Total iterations: {stats.get('total_iterations', 0)}")
    print(f"  Success count: {stats.get('success_count', 0)}")
    print(f"  Avg speedup: {stats.get('avg_speedup', 0):.2f}x")

    return True


async def main(test_with_queue: bool):
    print(f"Config base_dir: {config['kernel_bench']['base_dir']}")
    print(f"Output base_dir: {config['output']['base_dir']}")

    # Ensure output directory exists
    output_base = Path(os.path.expanduser(config['output']['base_dir']))
    output_base.mkdir(parents=True, exist_ok=True)

    # Test 1: List and get tasks
    first_task = await test_list_and_get_tasks()
    if first_task is None:
        print("\n=== Test 3: FAIL (task listing failed) ===")
        return 1

    task_path = f"level1/{first_task['name']}.py"

    # Test 2: eval_kernel with queue_only
    if not await test_eval_kernel_queue_only(task_path, test_with_queue):
        print("\n=== Test 3: FAIL (eval_kernel queue_only failed) ===")
        return 1

    # Test 3: save_benchmark_result
    if not await test_save_benchmark_result(task_path, str(output_base)):
        print("\n=== Test 3: FAIL (save_benchmark_result failed) ===")
        return 1

    # Test 4: get_session_summary
    if not await test_get_session_summary():
        print("\n=== Test 3: FAIL (get_session_summary failed) ===")
        return 1

    print("\n=== Test 3: PASS ===")
    print("\nNote: To fully verify queue submission, run with --with-queue flag")
    print("and ensure workflow server is running on port 8488")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCP E2E flow")
    parser.add_argument("--with-queue", action="store_true",
                       help="Require workflow server to be running")
    args = parser.parse_args()

    exit(asyncio.run(main(args.with_queue)))
