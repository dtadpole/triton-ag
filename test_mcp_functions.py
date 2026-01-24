#!/usr/bin/env python3
"""
Test script for MCP server functions (Test 2 Part A).

Usage:
    python3 test_mcp_functions.py
"""
import asyncio
import sys
sys.path.insert(0, '.')

from claudeCodeKernelBenchServer import list_kernel_bench_tasks, get_task_details, config

async def main():
    print(f"Config base_dir: {config['kernel_bench']['base_dir']}")

    # Test list_kernel_bench_tasks
    print("\n=== Test list_kernel_bench_tasks ===")
    tasks = await list_kernel_bench_tasks("level1")
    print(f"Found {len(tasks)} tasks in level1")
    if len(tasks) == 0:
        print("FAIL: No tasks found")
        return 1
    print("First 5 tasks:")
    for t in tasks[:5]:
        print(f"  - {t['name']}")

    # Test get_task_details with first task
    first_task = tasks[0]['name'] + ".py"
    print(f"\n=== Test get_task_details ({first_task}) ===")
    result = await get_task_details(f"level1/{first_task}")
    if "error" in result:
        print(f"FAIL: {result['error']}")
        return 1
    print(f"Task: {result['name']}")
    print(f"Source length: {len(result['source_code'])} chars")
    print("First 5 lines:")
    for line in result['source_code'].split('\n')[:5]:
        print(f"  {line}")

    print("\n=== Test 2 Part A: PASS ===")
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
