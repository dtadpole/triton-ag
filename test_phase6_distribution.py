"""Test Phase 6.2: Multi-device distribution (scaled for 2 GPUs)."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

import claudeCodeKernelBenchServer as mcp_server

# Simple Triton kernel for quick testing
TRITON_KERNEL = '''
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

# Task list (level1 activation functions)
TASKS = [
    "level1/19_ReLU.py",
    "level1/20_LeakyReLU.py",
    "level1/21_Sigmoid.py",
    "level1/22_Tanh.py",
]

async def run_single_eval(task_path: str, iteration: int = 0) -> dict:
    """Run a single kernel evaluation."""
    print(f"  Starting: {task_path}")
    result = await mcp_server.eval_kernel(
        task_path=task_path,
        kernel_code=TRITON_KERNEL,
        session_id="test_phase6_distribution",
        iteration=iteration,
        provider="local",
        code_type="triton"
    )
    status = "✓" if result.get("compiled") else "✗"
    speedup = result.get("speedup", 0)
    if not result.get("compiled"):
        # Check metadata for error
        metadata = result.get("metadata", {})
        error = metadata.get("error", result.get("error", ""))
        if error:
            print(f"  {status} {task_path}: {str(error)[:100]}")
        else:
            print(f"  {status} {task_path}: compiled=False, metadata={metadata}")
    else:
        print(f"  {status} Completed: {task_path} (speedup={speedup:.2f}x)")
    return result

async def main():
    print("=== Test 6.2: Multi-Device Distribution (2 GPUs, 4 tasks) ===\n")

    # Step 1: Verify semaphore size
    print("Step 1: Check semaphore configuration")
    semaphore = await mcp_server.get_eval_semaphore(provider="local")
    print(f"  Semaphore size: {mcp_server._semaphore_size}")

    # Step 2: Launch parallel evaluations
    print(f"\nStep 2: Launch {len(TASKS)} parallel evaluations")
    print("  (Semaphore will limit to 2 concurrent evaluations)")

    import time
    start_time = time.time()

    # Launch all tasks in parallel
    tasks = [run_single_eval(task) for task in TASKS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")

    # Step 3: Summarize results
    print("\nStep 3: Results summary")
    success_count = 0
    for task_path, result in zip(TASKS, results):
        if isinstance(result, Exception):
            print(f"  ✗ {task_path}: Exception - {result}")
        elif result.get("compiled"):
            success_count += 1
            print(f"  ✓ {task_path}: speedup={result.get('speedup', 0):.2f}x")
        else:
            error = result.get('error', 'Unknown error')
            print(f"  ✗ {task_path}: {error[:100]}")

    print(f"\nSuccess: {success_count}/{len(TASKS)}")

    if success_count == len(TASKS):
        print("\n=== Phase 6.2: PASS ===")
    else:
        print("\n=== Phase 6.2: PARTIAL (some tasks failed) ===")

if __name__ == "__main__":
    asyncio.run(main())
