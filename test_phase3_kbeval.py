"""Test Phase 3.1: Direct kbEvalClient call."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import get_kbeval_client

REFERENCE_CODE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
'''

GENERATED_KERNEL = '''
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

async def test_kb_eval():
    print("=== Test 3.1: Direct kbEvalClient call ===")

    client = get_kbeval_client()
    print(f"Using client: {type(client).__name__}")

    try:
        result = await client.kb_eval(
            provider="local",
            reference_code=REFERENCE_CODE,
            generated_code=GENERATED_KERNEL,
            code_type="triton",
            run_tag="test_phase3",
            model_tag="claude_code_test",
            task_tag="relu_test",
            eval_tag="iter_00",
        )
        print(f"Result: {result}")

        if result.get("compiled"):
            print("✓ Kernel compiled successfully")
        else:
            print(f"✗ Compilation failed: {result.get('error')}")

        if result.get("correctness"):
            print("✓ Kernel produces correct output")
        else:
            print(f"✗ Correctness check failed")

        speedup = result.get("speedup", 0)
        print(f"Speedup: {speedup:.2f}x")

        if result.get("compiled") and result.get("correctness"):
            print("\n=== Phase 3.1: PASS ===")
        else:
            print("\n=== Phase 3.1: FAIL ===")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\n=== Phase 3.1: FAIL ===")
        raise

if __name__ == "__main__":
    asyncio.run(test_kb_eval())
