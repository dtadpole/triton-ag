"""Test Phase 2: MCP tools for single task."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import list_kernel_bench_tasks, get_task_details

async def test_list_tasks():
    print("=== Test 2.1a: list_kernel_bench_tasks ===")
    tasks = await list_kernel_bench_tasks(level="level1")
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
