import os
from datetime import datetime
import traceback
from typing import Union
import asyncio
import argparse
from agents import (
    Agent,
    Runner,
    RunConfig,
    trace,
    function_tool,
    RunResult,
)
from agents.mcp import MCPServerStdio
from util import load_agent_model, init_logging, get_next_run_folder, get_run_hooks, log_result_items
from logger import logger


AGENT_NAME = "kernel_coder"

KERNEL_CODER_SYSTEM_PROMPT = """
You are an expert coder with experience in CUDA kernels.  You understand tilings, parallelism,
precision, numerical stability, and other advanced concepts in the context of CUDA and GPU programming. Consider all possible optimization techniques. (e.g. shared memory, coalesced access, occupancy tuning, block size, optimization, grid stride loops, loop unrolling, kernel fusion, vectorized loads, bank conflict avoidance, warp primitives, arithmetic intenstiy, etc.)

parent_dir: `{workspace_dir}`
current_wd: `{workspace_dir}/current`

1. Analyze the request to understand the task scope
2. All the relevant environments has already been setup
3. Check the `{workspace_dir}/current` folder and subfolders for Python files (ending with `.py`) to understand the current code structure
4. Implement specific code for the given task, do not change anything else

When generating code, always follow these instructions:
- Implement all functionalities (functions and modules) in a single file in the current working directory (not subfolder).
- You may create your own test cases to verify intermediate results, but the official verification will need to use `kb_eval_iteration` tool.
- When creating your own test cases, always write them under `tests` subfolder (under the current working directory), with filename ends with `_test.py`
"""

KERNEL_CODER_NEXT_PROMPT = """
You will iteratively improve a CUDA kernel for a given PyTorch code, up to and including iteration `{max_iterations}`.

Within each iteration, ensure that you have completed each and every step of the following:
-- Generate the CUDA kernel code.
-- Evaluate the correctness and performance of the generated CUDA kernel using `kb_eval_iteration` tool.
-- Recap the changes for the current iteration in a few sentences and upload the iteration recap using `kb_upload_iteration` tool.
Ensure that you have completed the current iteration, including uploading the iteration recap, before starting the next iteration.

Recap examples:
-- "Iteration 1: Implemented the basic matrix multiplication kernel."
-- "Iteration 2: Added loop unrolling (UNROLL_FACTOR=4). Runtime: 4.79 ms (slight regression). Correctness: passed."
-- "Iteration 3: Attempted register blocking optimization with 4x4 register tiles, but encountered correctness issues. Max difference: 202.18, indicating significant numerical errors. Need to fix indexing and memory access patterns."

Start the next iteration if and only if the current iteration has fully completed, ensure to include uploading the iteration recap using `kb_upload_iteration` tool.  Repeat new iteration and keep improving performance of the kernel code until you have reached the maximum iterations allowed, up to and including iteration `{max_iterations}` but do not exceed maximum iterations of `{max_iterations}`. for example, if you have `{max_iterations}` iterations, you will run the following sequence from <START> to <END>:

<START>
-> [Iteration 1: generate kernel code] -> [Iteration 1: evaluate correctness and performance] -> [Iteration 1: recap and upload]
-> [Iteration 2: generate kernel code] -> [Iteration 2: evaluate correctness and performance] -> [Iteration 2: recap and upload]
-> ...
-> [Iteration {max_iterations}: generate kernel code] -> [Iteration {max_iterations}: evaluate correctness and performance] -> [Iteration {max_iterations}: recap and upload]
-> <END>

Task: {task}

model_tag: `{model_tag}` 
task_tag: `{task_tag}`
time_tag: `{time_tag}`
rollout_id: `{rollout_id}`

eval_tag: the `eval_tag` is `rollout_id + iteration_number`. iteration number starts from 1 and increases by 1 for each iteration. e.g. 
-- for iteration 1, your eval_tag is '{rollout_id}_i01'
-- for iteration 3, your eval_tag is '{rollout_id}_i03'
-- for iteration 10, your eval_tag is '{rollout_id}_i10'

**GENERATE CODE**

Write full generated kernel in a single file as {workspace_dir}/current/`eval_tag`_cuda_kernel.py. Generate a new file for each iteration.  Keep improving performance of the kernel code.

Replace pytorch operators in the given module with raw CUDA kernels, optimizing for performance on NVIDIA architecture.

Use torch.utils.cpp_extension.load_inline and name your optimized output module ModelNew.

You're NOT allowed to use torch.nn (except for Parameter, containers, and init).

The input and output have to be on CUDA device. Your answer must be the complete new module (no testing code, no other code): it will be evaluated and you will be given feedback on its correctness and speedup so you can keep iterating to maximize the speedup.

Here's an example:

```python
{example_code}
```

**EVALUATE CODE**

Did you encounter error when running `kb_eval_iteration` validation?
Based on the error information, what's your next action?

Choose the most efficient path forward:
1. Do you understand the error? Can you fix the error easily?
2. If not sure why the error happened, can you create debug test cases to check each intermediate result step by step, and fix the code at each individual step?
3. If you have passed all the intermediate test cases, verify using the `kb_eval_iteration` tool.
4. Keep improving performance of the kernel code with more iterations, up to and including iteration {max_iterations}.
5. Stop the task after you have reached the maximum iterations allowed, do not exceed maximum iterations of `{max_iterations}`.
6. Immediately stop if you have exceeded maximum iterations of `{max_iterations}`.

Be concise in your reasoning (think concisely), select the appropriate tool or action.
"""

"""
In each iteration, use `kb_eval_iteration` tool to evaluate the correctness and performance of generated CUDA kernel.
At each iteration, summarize your changes in a few sentences, and use `kb_upload_iteration` tool to upload
the summary.  Always generate a summary and upload use `kb_upload_iteration` tool at each and every single iteration
step, regardless of whether `kb_eval_iteration` has error(s).  If `kb_eval_iteration` has error(s), upload the summary with the error information.  If `kb_eval_iteration` has no error(s), you should also summarize the changes in a few sentences and use `kb_upload_iteration` tool to upload the iterationsummary.
"""

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

# this is the main function that will be called by the Runner
async def run_kernel_coder(workspace_dir: str, task: str, provider: Union[str, None] = None, model_name: Union[str, None] = None, rollout_id: int = 1):

    logger.info(f"Running [{AGENT_NAME}] [{workspace_dir}] with task: {task}")

    model, model_settings, run_config, model_config = load_agent_model(AGENT_NAME, provider, model_name)

    MODEL_TAG = os.path.join(AGENT_NAME,
                             f"{provider or model_config['provider']}",
                             f"{model_name or model_config['model']}")

    TASK_TAG = os.path.join(os.path.basename(os.path.dirname(task)),
                            os.path.basename(task))

    checkpoint_server = MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "--with", "mcp", "mcp", "run", "checkpointServer.py"],
        }
    )
    async with checkpoint_server as ckpts:
        result = await ckpts.call_tool(
            tool_name="init_workspace_folder",
            arguments={
                "workspace_folder": workspace_dir,
                "reference_pytorch_code": task,
                "environ_vars": {
                    "provider": provider,
                    "model_name": model_name,
                },
                "include_verifier": False,
            },
        )
        logger.info(result)

        file_server = MCPServerStdio(
            params={
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    os.path.join(workspace_dir, "current"),
                ],
            },
            client_session_timeout_seconds=10,
        )
        # sequential_thinking_server = MCPServerStdio(
        #     params={
        #         "command": "npx",
        #         "args": [
        #             "-y",
        #             "@modelcontextprotocol/server-sequential-thinking",
        #         ],
        #     },
        #     client_session_timeout_seconds=120,
        # )
        kb_eval_iteration_server = MCPServerStdio(
            params={
                "command": "uv",
                "args": ["run", "--with", "mcp", "mcp", "run", "kbEvalMCPServer.py"],
            },
            client_session_timeout_seconds=480,
        )
        async with file_server as fs, kb_eval_iteration_server as kbs: #, sequential_thinking_server as sqs:
            try:
                kernel_bench = Agent(
                    model=model,
                    name="kernel_bench",
                    instructions=KERNEL_CODER_SYSTEM_PROMPT.format(
                        workspace_dir=workspace_dir
                    ),
                    mcp_servers=[fs, kbs, ckpts], #, sqs],
                )
                prompt = KERNEL_CODER_NEXT_PROMPT.format(
                    task="""Implement CUDA Kernel (forward pass only) for the given PyTorch code in
                    `pytorch_reference.py`, iteratively improving performance of the kernel code.
                    """,
                    workspace_dir=workspace_dir,
                    model_tag=MODEL_TAG,
                    task_tag=TASK_TAG,
                    time_tag=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    rollout_id=f"r{rollout_id:02d}",
                    max_iterations=args.max_iterations,
                    example_code=EXAMPLE_CODE,
                )

                run_hooks = get_run_hooks()

                with trace("Kernel Coder"):
                    result = await Runner.run(
                        kernel_bench,
                        input=prompt,
                        max_turns=run_config['max_turns'] if 'max_turns' in run_config else 50,
                        hooks=run_hooks,
                        run_config=RunConfig(
                            model_settings=model_settings,
                        ),
                    )
                    log_result_items(result, f"{AGENT_NAME}_r{rollout_id:02d}", MODEL_TAG, TASK_TAG, workspace_dir)
                    logger.info(result.final_output)
                    return result.final_output
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error running Kernel Coder: {e}")
                return f"Error running Kernel Coder: {e}"
            finally:
                logger.info(f"Agent [{AGENT_NAME}] completed!")


async def main(args):
    # check if the task is a file name
    if not os.path.exists(args.task):
        logger.error(f"Task file {args.task} does not exist")
        exit(1)
    init_logging(AGENT_NAME)

    if args.workspace_dir and os.path.exists(args.workspace_dir):
        workspace_dir = args.workspace_dir
        logger.info(f"Working directory: {workspace_dir}")
    else:
        workspace_dir = get_next_run_folder()
        logger.info(f"Working directory: {workspace_dir}")

    print('='*50)
    print(f"Running [{AGENT_NAME}] [{workspace_dir}] with task: {args.task}")
    print('='*50)

    tasks = []
    for rollout_id in range(1, args.total_rollouts + 1):
        tasks.append(run_kernel_coder(workspace_dir,
                                      args.task,
                                      args.provider,
                                      args.model_name,
                                      rollout_id=rollout_id))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
        # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--workspace-dir", type=str, default="")
    parser.add_argument("-p", "--provider", type=str, default=None)
    parser.add_argument("-m", "--model-name", type=str, default=None)
    parser.add_argument("-e", "--max-iterations", type=int, default=4)
    parser.add_argument("-r", "--total-rollouts", type=int, default=8)
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="./kernel_bench/level1/1_Square_matrix_multiplication_.py",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
