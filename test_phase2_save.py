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

# Use dict instead of JSON string
MOCK_RESULT = {"compiled": True, "correctness": True, "speedup": 1.25, "runtime": 0.5}

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
    # Stats are in _stats key
    assert "_stats" in result, "Expected _stats in summary"
    assert result["_stats"]["total_tasks"] > 0, "Expected at least 1 task"
    print("✓ get_session_summary works")

async def main():
    await test_save()
    await test_summary()
    print("\n=== Phase 2.2: PASS ===")

if __name__ == "__main__":
    asyncio.run(main())
