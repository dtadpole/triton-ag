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
    #     reasoning={"effort": "low"},
    #     # reasoning={"effort": "medium"},
    #     # reasoning={"effort": "high"},
    #     input=[{"role": "user", "content": prompt}],
    #     max_output_tokens=3000,
    # )

    # print(response.status)
    # print("\n\n\n")
    # print(response.output_text)
    # print("\n\n\n\n")
    # print(response)


if __name__ == "__main__":
    asyncio.run(main())


# import os

# import torch
# from vllm import LLM, SamplingParams


# def detect_model_type(model_path):
#     """Detect if model is quantized and what type"""
#     if os.path.isdir(model_path):
#         # Check for quantized model files
#         files = os.listdir(model_path)
#         if any("awq" in f.lower() for f in files):
#             return "awq"
#         elif any("gptq" in f.lower() for f in files):
#             return "gptq"
#         elif any("ggml" in f.lower() or "gguf" in f.lower() for f in files):
#             return "ggml"
#         elif any("int4" in f.lower() or "int8" in f.lower() for f in files):
#             return "int_quantized"

#     # Check model name for quantization hints
#     model_name = model_path.lower()
#     if "awq" in model_name:
#         return "awq"
#     elif "gptq" in model_name:
#         return "gptq"
#     elif "ggml" in model_name or "gguf" in model_name:
#         return "ggml"

#     return "full_precision"


# def run_batch_inference(
#     prompts, max_model_length, max_new_tokens, group_size, model_path
# ):

#     # Detect model type
#     # model_type = detect_model_type(model_path)
#     # print(f"Detected model type: {model_type}")
#     model_type = "gpt-oss-120b"

#     # Configure based on model type
#     if model_type == "gpt-oss-120b":
#         # OSS 120B model
#         llm = LLM(
#             model="hub/models--openai--gpt-oss-120b/snapshots/bc75b44b8a2a116a0e4c6659bcd1b7969885f423",
#             tensor_parallel_size=1,
#             dtype=torch.bfloat16,
#             trust_remote_code=True,
#             max_model_len=max_model_length,
#             max_num_seqs=64,
#             gpu_memory_utilization=0.9,
#         )

#     elif model_type == "awq":
#         # AWQ quantized model
#         llm = LLM(
#             model=model_path,
#             quantization="awq",
#             tensor_parallel_size=1,
#             dtype=torch.float16,  # AWQ typically uses float16
#             trust_remote_code=True,
#             max_model_len=max_model_length,
#             max_num_seqs=64,
#             gpu_memory_utilization=0.95,  # Can use more memory with quantized models
#         )
#     elif model_type == "gptq":
#         # GPTQ quantized model
#         llm = LLM(
#             model=model_path,
#             quantization="gptq",
#             tensor_parallel_size=1,
#             dtype=torch.float16,
#             trust_remote_code=True,
#             max_model_len=max_model_length,
#             max_num_seqs=64,
#             gpu_memory_utilization=0.95,
#         )
#     elif model_type == "ggml":
#         print("GGML/GGUF models are not directly supported by vLLM")
#         print("Consider using llama.cpp or converting to HF format")
#         return
#     else:
#         # Full precision model
#         llm = LLM(
#             model=model_path,
#             tensor_parallel_size=1,
#             dtype=torch.float16,  # Use float16 for memory efficiency
#             # dtype=torch.bfloat16,  # Alternative: use bfloat16 if supported
#             # dtype="auto",          # Let vLLM decide the best dtype
#             trust_remote_code=True,
#             max_model_len=max_model_length,
#             gpu_memory_utilization=0.9,  # Use less memory for full precision
#         )

#     # Define sampling parameters
#     sampling_params = SamplingParams(
#         temperature=0.7, max_tokens=max_new_tokens, n=group_size
#     )

#     # Run batch inference
#     print("Running batch inference...")
#     outputs = llm.generate(prompts, sampling_params)

#     output_dict = {}

#     # Process and display results
#     for i, output in enumerate(outputs):
#         prompt = output.prompt
#         output_dict[prompt] = []
#         for o_ in output.outputs:
#             output_dict[prompt].append(o_.text)

#     return output_dict


# if __name__ == "__main__":
#     # Run the basic batch inference example
#     prompts = ["hello world!"]
#     result = run_batch_inference(prompts, 2048, 1024, 4, "gpt-oss-120b")
#     print(result)
