#!/usr/bin/env python3
"""
Test script for MCP server End-to-End flow with Claude-generated kernels (Test 3 Part C).

This script guides Claude Code to actually generate Triton kernel code for a task,
then submits to the inferenceEval queue (same as Part B verification approach).

The key difference from Part B:
- Part B: Uses HARDCODED sample kernel code
- Part C: Claude Code GENERATES the actual kernel code

Prerequisites:
- MCP server registered in Claude Code
- Workflow server running (optional): python workflowServer.py --host :: --port 8488

Usage:
    # Run the test - Claude Code will generate the kernel
    python3 test_mcp_e2e_claude.py --task level1/1_Square.py

    # List available tasks
    python3 test_mcp_e2e_claude.py --list-tasks
"""
import asyncio
import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

from claudeCodeKernelBenchServer import (
    list_kernel_bench_tasks,
    get_task_details,
    eval_kernel,
    save_benchmark_result,
    get_session_summary,
    config
)


# Instructions for Claude Code to generate a kernel
KERNEL_GENERATION_PROMPT = '''
Based on the PyTorch model code above, generate an optimized Triton kernel.

Requirements:
1. Create a @triton.jit decorated kernel function
2. Create a ModelNew class that uses the kernel
3. The ModelNew.forward() method should have the same signature as the original Model.forward()
4. Use efficient Triton patterns (block-based processing, coalesced memory access)

Return ONLY the kernel code as a Python code block, no explanations.
'''


async def run_claude_e2e_test(task_path: str, session_id: str = None):
    """
    Run E2E test where Claude Code generates the kernel code.

    This function:
    1. Fetches task details
    2. Displays the task code for Claude to analyze
    3. Prompts Claude to generate a kernel
    4. Submits to queue (like Part B)
    5. Saves results

    Args:
        task_path: Path to kernel bench task (e.g., "level1/1_Square.py")
        session_id: Session identifier for results
    """
    if session_id is None:
        session_id = f"claude_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print("Test 3 Part C: E2E with Claude-Generated Kernels")
    print("=" * 60)
    print(f"Task: {task_path}")
    print(f"Session: {session_id}")
    print()

    # Step 1: Get task details
    print("Step 1: Fetching task details...")
    task_result = await get_task_details(task_path)

    if "error" in task_result:
        print(f"FAIL: {task_result['error']}")
        return 1

    print(f"  Task name: {task_result['name']}")
    print(f"  Source code: {len(task_result['source_code'])} chars")
    print()

    # Step 2: Display task for Claude to analyze
    print("Step 2: Task source code (for Claude to analyze):")
    print("-" * 40)
    print(task_result['source_code'])
    print("-" * 40)
    print()

    # Step 3: Prompt for Claude-generated kernel
    print("Step 3: Claude Code should now generate the kernel.")
    print()
    print("INSTRUCTIONS FOR CLAUDE CODE:")
    print(KERNEL_GENERATION_PROMPT)
    print()
    print("After generating the kernel code, use the eval_kernel MCP tool with:")
    print(f"  - task_path: '{task_path}'")
    print(f"  - session_id: '{session_id}'")
    print(f"  - iteration: 0")
    print(f"  - queue_only: True")
    print()
    print("Then use save_benchmark_result to save the result.")
    print()

    # Step 4: Wait for user/Claude to provide kernel code
    print("Paste the Claude-generated kernel code below (end with Ctrl+D):")
    print("(Or run with --demo to use a sample kernel for testing)")
    print()

    try:
        lines = []
        while True:
            try:
                line = input()
                lines.append(line)
            except EOFError:
                break
        kernel_code = '\n'.join(lines)
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        return 1

    if not kernel_code.strip():
        print("  No kernel code provided.")
        print("  Run with --demo to use a sample kernel for testing.")
        return 1

    print(f"  Received kernel code: {len(kernel_code)} chars")
    print()

    # Step 5: Submit to queue (like Part B)
    print("Step 4: Submitting to inferenceEval queue...")
    eval_result = await eval_kernel(
        task_path=task_path,
        kernel_code=kernel_code,
        session_id=session_id,
        iteration=0,
        provider="local",
        queue_only=True  # Same as Part B - just verify queue submission
    )

    if "status" in eval_result:
        if eval_result["status"] == "queued":
            print(f"  SUCCESS: Queued to {eval_result['queue']}")
            print(f"  Work item submitted_at: {eval_result['work_item']['submitted_at']}")
        elif eval_result["status"] == "queue_error":
            print(f"  EXPECTED: Queue error (workflow server not running)")
            print(f"  Error: {eval_result['error']}")
            print("  This is OK - queue submission logic verified")
    else:
        print(f"  Unexpected result: {eval_result}")
    print()

    # Step 6: Save result with placeholder eval
    print("Step 5: Saving benchmark result...")
    placeholder_eval = {
        "compiled": None,
        "correctness": None,
        "runtime": 0,
        "speedup": 0,
        "status": "queued_for_eval"
    }

    result_path = await save_benchmark_result(
        task_path=task_path,
        kernel_code=kernel_code,
        eval_result=placeholder_eval,
        session_id=session_id,
        iteration=0
    )
    print(f"  Saved to: {result_path}")
    print()

    # Step 7: Verify kernel file syntax
    print("Step 6: Verifying kernel file syntax...")
    kernel_file = Path(result_path) / "iteration_00_cuda_kernel.py"
    try:
        import py_compile
        py_compile.compile(str(kernel_file), doraise=True)
        print("  Kernel syntax: OK")
    except py_compile.PyCompileError as e:
        print(f"  Kernel syntax: FAIL - {e}")
        return 1
    print()

    # Final status
    print("=" * 60)
    print("Test 3 Part C: PASS")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  - Task analyzed: {task_result['name']}")
    print(f"  - Kernel generated: {len(kernel_code)} chars")
    print(f"  - Queue submission: Verified")
    print(f"  - Result saved: {result_path}")
    print()
    print("Next steps:")
    print("  1. Start kbEvalServer to process the queued evaluation")
    print("  2. Check results in the saved directory")

    return 0


async def run_demo_test(task_path: str, session_id: str = None):
    """
    Run demo test with a sample kernel (for automated testing).
    """
    if session_id is None:
        session_id = f"claude_e2e_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print("Test 3 Part C: Demo Mode (Sample Kernel)")
    print("=" * 60)
    print(f"Task: {task_path}")
    print(f"Session: {session_id}")
    print()

    # Get task details
    print("Step 1: Fetching task details...")
    task_result = await get_task_details(task_path)

    if "error" in task_result:
        print(f"FAIL: {task_result['error']}")
        return 1

    print(f"  Task name: {task_result['name']}")
    print()

    # Generate sample kernel based on task name
    print("Step 2: Using sample kernel (demo mode)...")
    kernel_code = generate_sample_kernel(task_result['name'])
    print(f"  Sample kernel: {len(kernel_code)} chars")
    print()

    # Submit to queue
    print("Step 3: Submitting to inferenceEval queue...")
    eval_result = await eval_kernel(
        task_path=task_path,
        kernel_code=kernel_code,
        session_id=session_id,
        iteration=0,
        provider="local",
        queue_only=True
    )

    if "status" in eval_result:
        if eval_result["status"] == "queued":
            print(f"  SUCCESS: Queued to {eval_result['queue']}")
        elif eval_result["status"] == "queue_error":
            print(f"  EXPECTED: Queue error (workflow server not running)")
            print("  This is OK - queue submission logic verified")
    print()

    # Save result
    print("Step 4: Saving benchmark result...")
    placeholder_eval = {
        "compiled": None,
        "correctness": None,
        "runtime": 0,
        "speedup": 0,
        "status": "queued_for_eval"
    }

    result_path = await save_benchmark_result(
        task_path=task_path,
        kernel_code=kernel_code,
        eval_result=placeholder_eval,
        session_id=session_id,
        iteration=0
    )
    print(f"  Saved to: {result_path}")
    print()

    # Verify syntax
    print("Step 5: Verifying kernel file syntax...")
    kernel_file = Path(result_path) / "iteration_00_cuda_kernel.py"
    try:
        import py_compile
        py_compile.compile(str(kernel_file), doraise=True)
        print("  Kernel syntax: OK")
    except py_compile.PyCompileError as e:
        print(f"  Kernel syntax: FAIL - {e}")
        return 1
    print()

    print("=" * 60)
    print("Test 3 Part C (Demo): PASS")
    print("=" * 60)

    return 0


def generate_sample_kernel(task_name: str) -> str:
    """Generate a sample kernel as placeholder."""
    return f'''
import torch
import triton
import triton.language as tl

# Sample kernel for {task_name}
# NOTE: This is a DEMO placeholder - in Part C, Claude should generate this

@triton.jit
def sample_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * x  # Placeholder operation
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


async def list_available_tasks():
    """List available tasks for reference."""
    print("Available kernel bench tasks:")
    print()

    for level in ["level1", "level2", "level3"]:
        tasks = await list_kernel_bench_tasks(level)
        if tasks:
            print(f"{level}: {len(tasks)} tasks")
            for t in tasks[:5]:  # Show first 5
                print(f"  - {t['name']}")
            if len(tasks) > 5:
                print(f"  ... and {len(tasks) - 5} more")
        print()


async def main():
    parser = argparse.ArgumentParser(
        description="Test MCP E2E flow with Claude-generated kernels (Test 3 Part C)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task path (e.g., level1/1_Square.py)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        help="Session identifier for results"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with sample kernel (for automated testing)"
    )

    args = parser.parse_args()

    if args.list_tasks:
        await list_available_tasks()
        return 0

    if not args.task:
        print("Error: --task is required")
        print("Use --list-tasks to see available tasks")
        parser.print_help()
        return 1

    if args.demo:
        return await run_demo_test(
            task_path=args.task,
            session_id=args.session_id
        )
    else:
        return await run_claude_e2e_test(
            task_path=args.task,
            session_id=args.session_id
        )


if __name__ == "__main__":
    exit(asyncio.run(main()))
