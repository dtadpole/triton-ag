"""Test Phase 2.3: Claude Code generates kernel for single task."""
import asyncio
import os
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import get_task_details, save_benchmark_result, get_session_summary

# Triton ReLU kernel - matches the 19_ReLU.py task
RELU_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for ReLU activation."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    """Optimized ReLU using Triton kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_elements = x.numel()

        # Choose block size
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        relu_kernel[grid](
            x, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

def get_inputs():
    return [torch.randn(4096, 393216, device='cuda')]

def get_init_inputs():
    return []
'''

# Mock eval result (simulates what kbEvalServer would return)
MOCK_RESULT = {
    "compiled": True,
    "correctness": True,
    "speedup": 1.15,
    "runtime": 0.42,
    "metadata": {"note": "Mock result - not from real GPU evaluation"}
}

async def test_generate_and_save():
    print("=== Test 2.3: Claude Code generates kernel for single task ===")

    # Step 1: Get task details to confirm we're targeting the right task
    print("\n1. Getting task details for 19_ReLU.py...")
    task = await get_task_details(task_path="level1/19_ReLU.py")
    print(f"   Task name: {task.get('name')}")
    print(f"   Source length: {len(task.get('source_code', ''))} chars")

    # Step 2: Save the generated kernel with mock result
    print("\n2. Saving generated kernel with mock eval result...")
    session_id = "test_phase2_claude"
    result = await save_benchmark_result(
        task_path="level1/19_ReLU.py",
        kernel_code=RELU_KERNEL,
        eval_result=MOCK_RESULT,
        session_id=session_id,
        iteration=0
    )
    print(f"   Save result: {result}")

    # Step 3: Verify files exist
    print("\n3. Verifying saved files...")
    base = os.path.expanduser(f"~/.inference/claude_code_output/{session_id}/19_ReLU")
    kernel_file = os.path.join(base, "iteration_00_cuda_kernel.py")
    eval_file = os.path.join(base, "iteration_00_eval.json")

    assert os.path.exists(kernel_file), f"Kernel file not found: {kernel_file}"
    assert os.path.exists(eval_file), f"Eval file not found: {eval_file}"
    print(f"   ✓ Kernel file: {kernel_file}")
    print(f"   ✓ Eval file: {eval_file}")

    # Step 4: Verify kernel is syntactically valid Python
    print("\n4. Verifying kernel syntax...")
    with open(kernel_file, 'r') as f:
        kernel_content = f.read()
    compile(kernel_content, kernel_file, 'exec')
    print("   ✓ Kernel is syntactically valid Python")

    # Step 5: Get session summary
    print("\n5. Getting session summary...")
    summary = await get_session_summary(session_id=session_id)
    print(f"   Total tasks: {summary.get('_stats', {}).get('total_tasks', 0)}")
    print(f"   Success rate: {summary.get('_stats', {}).get('success_rate', 0):.1%}")

    print("\n✓ Test 2.3: PASS - Claude Code can generate and save kernels")

async def main():
    await test_generate_and_save()
    print("\n=== Phase 2.3: PASS ===")

if __name__ == "__main__":
    asyncio.run(main())
