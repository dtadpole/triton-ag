import asyncio

from agents import (
    Agent,
    function_tool,
    OpenAIResponsesModel,
    Runner,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

set_tracing_disabled(True)

EXAMPLE_CODE = '''
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
'''
REFERENCE_CODE = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        #
        # Simple model that performs a single square matrix multiplication (C = A * B)
        #
        def __init__(self):
            super(Model, self).__init__()
        
        def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            
            # Performs the matrix multiplication.
            # Args:
            #     A (torch.Tensor): Input matrix A of shape (N, N).
            #     B (torch.Tensor): Input matrix B of shape (N, N).

            # Returns:
            #     torch.Tensor: Output matrix C of shape (N, N).                    
            return torch.matmul(A, B)

    N = 2048 * 2

    def get_inputs():
        A = torch.rand(N, N)
        B = torch.rand(N, N)
        return [A, B]

    def get_init_inputs():
        return []  # No special initialization inputs needed
"""

prompts = """ Write an optimized CUDA kernel for square matrix multiplication.  The reference code is mentioned below. The kernel should compile and be as fast as possible. See the 
reference code below:

```
{reference_code}
```

Also here is an example of the kernel code I expect you to write (this kernel does a simple addition, I expect you to write kernel for the reference_code mentioned above, this is just an example):

```python
{example_code}
```
"""


async def main():

    agent = Agent(
        name="Assistant",
        instructions="You are an expert coder with experience in CUDA kernels.  You understand tilings, parallelism, precision, numerical stability, and other advanced concepts in the context of CUDA and GPU programming. Consider all possible optimization techniques. (e.g. shared memory, coalesced access, occupancy tuning, block size, optimization, grid stride loops, loop unrolling, kernel fusion, vectorized loads, bank conflict avoidance, warp primitives, arithmetic intenstiy, etc.)",
        model=OpenAIResponsesModel(
            # model="openai/gpt-oss-120b",
            model="hub/models--openai--gpt-oss-120b/snapshots/bc75b44b8a2a116a0e4c6659bcd1b7969885f423",
            openai_client=AsyncOpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
            ),
        ),
        # tools=[get_weather],
    )
    result = await Runner.run(
        agent,
        input=prompts.format(example_code=EXAMPLE_CODE, reference_code=REFERENCE_CODE),
    )
    print(result.final_output)
    print(result)

    # prompt = """
    # Write a bash script that takes a matrix represented as a string with
    # format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.
    # """

    # from openai import OpenAI

    # client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    # response = client.responses.create(
    #     model="hub/models--openai--gpt-oss-120b/snapshots/bc75b44b8a2a116a0e4c6659bcd1b7969885f423",
    #     reasoning={"effort": "medium"},
    #     input=[{"role": "user", "content": prompt}],
    # )

    # print(response.output_text)


if __name__ == "__main__":
    asyncio.run(main())
