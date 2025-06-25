import argparse
import asyncio
import os
from datetime import datetime
from typing import Union

from agents import Agent, function_tool, RunConfig, Runner, RunResult, trace
from agents.mcp import MCPServerStdio
from logger import logger
from pydantic import Field
from pydantic.json_schema import to_jsonable_python
from util import (
    get_next_run_folder,
    get_run_hooks,
    init_logging,
    load_agent_model,
    log_result_items,
)

AGENT_NAME = "triton_coder"

TRITON_CODER_SYSTEM_PROMPT = """
You are an expert coder with experience in Triton kernels.  You understand tilings, parallelism,
precision, numerical stability, and other concepts in the context of Triton and GPU programming.

parent_dir: `{workspace_dir}`
current_wd: `{workspace_dir}/current`

1. Analyze the request to understand the task scope
2. All the relevant environments has already been setup
3. Check the `{workspace_dir}/current` folder and subfolders for Python files (ending with `.py`) to understand the current code structure
4. Implement specific code for the given task, do not change anything else

When generating code, always follow these instructions:
- Implement all functionalities (functions and modules) in a single file in the current working directory (not subfolder).
- You may create your own test cases to verify intermediate results, but the official verification will need to use `kb_eval_iteration_triton` tool.
- When creating your own test cases, always write them under `tests` subfolder (under the current working directory), with filename ends with `_test.py`
"""

TRITON_CODER_NEXT_PROMPT = """
You will iteratively improve a Triton kernel for a given PyTorch code, up to and including iteration `{max_iterations}`.

Within each iteration, ensure that you have completed each and every step of the following:
-- Generate the Triton kernel code.
-- Evaluate the correctness and performance of the generated Triton kernel using `kb_eval_iteration_triton` tool.
-- Recap the changes for the current iteration in a few sentences and upload the iteration recap using `kb_upload_iteration` tool.
Ensure that you have completed the current iteration, including uploading the iteration recap, before starting the next iteration.

Be concise in your reasoning, select the appropriate tool or action.

Start the next iteration if and only if the current iteration has completed, ensure to upload the iteration recap using `kb_upload_iteration` tool before starting the next iteration.  Repeat new iteration and keep improving performance of the kernel code until you have reached the maximum iterations allowed, up to and including iteration `{max_iterations}` but do not exceed maximum iterations of `{max_iterations}`. for example, if you have `{max_iterations}` iterations, you will run the following sequence from <|START|> to <|END|> :

<|START|>
-> [Iteration 1: generate kernel code] -> [Iteration 1: evaluate correctness and performance] -> [Iteration 1: recap and upload]
-> [Iteration 2: generate kernel code] -> [Iteration 2: evaluate correctness and performance] -> [Iteration 2: recap and upload]
-> ...
-> [Iteration {max_iterations}: generate kernel code] -> [Iteration {max_iterations}: evaluate correctness and performance] -> [Iteration {max_iterations}: recap and upload]
-> <|END|>

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

Write full generated kernel in a single file as {workspace_dir}/current/`eval_tag`_triton_kernel.py. Generate a new file for each iteration.  Keep improving performance of the kernel code.

Replace pytorch operators in the given module with raw triton kernels, optimizing for performance on NVIDIA architecture.

You're NOT allowed to use torch.nn (except for Parameter, containers, and init).

The input and output have to be on CUDA device. Your answer must be the complete new module (no testing code, no other code): it will be evaluated and you will be given feedback on its correctness and speedup so you can keep iterating to maximize the speedup.

Here's an example:

```python
{example_code}
```

**EVALUATE CODE**

Did you encounter error when running `kb_eval_iteration_triton` validation?
Based on the error information, what's your next action?

Choose the most efficient path forward:
1. Do you understand the error? Can you fix the error easily?
2. If not sure why the error happened, can you create debug test cases to check each intermediate result step by step, and fix the code at each individual step?
3. If you have passed the intermediate test cases, verify using the `kb_eval_iteration_triton` tool.  If you tried multiple times but still failed, record the error, upload the error information using `kb_upload_iteration_triton` tool, and continue to the next iteration.
4. Keep improving performance of the kernel code with more iterations, up to and including iteration {max_iterations}.
5. Stop the task after you have reached the maximum iterations allowed, do not exceed maximum iterations of `{max_iterations}`.
6. Immediately stop if you have exceeded maximum iterations of `{max_iterations}`.

Be concise in your reasoning (think concisely), select the appropriate tool or action.

"""

EXAMPLE_CODE = '''
import torch
import torch.nn as nn

import triton
import triton.language as tl

# DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 2,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 2,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 2,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "waves_per_eu": 3,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        ACTIVATION=activation,  #
    )
    return c


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul = matmul

    def forward(self, a, b):
        return self.matmul(a, b)
'''


@function_tool(
    name_override=AGENT_NAME,
    description_override="Triton Coder is an expert with experience in Triton kernels.  It will implement specific code for the given task, which can be either a single module or a single kernel function.  It can also add functionality to existing code.  It will verify correctness of the code before returning (it won't perform any benchmarks)",
)
async def triton_coder(
    workspace_dir: str = Field(
        ...,
        description="The working directory",
    ),
    task: str = Field(
        ...,
        description="The task to implement.  Please provide clear and concise task description",
    ),
):
    return await run_triton_coder(workspace_dir, task)


# this is the main function that will be called by the Runner
async def run_triton_coder(
    workspace_dir: str,
    task: str,
    provider: Union[str, None] = None,
    model_name: Union[str, None] = None,
    rollout_id: int = 1,
):

    logger.info(f"Running [{AGENT_NAME}] [{workspace_dir}] with task: {task}")

    model, model_settings, run_config, model_config = load_agent_model(
        AGENT_NAME, provider, model_name
    )

    TASK_NAME = os.path.join(
        AGENT_NAME, os.path.basename(os.path.dirname(task)), os.path.basename(task)
    )

    TASK_TAG = os.path.join(
        os.path.basename(os.path.dirname(task)), os.path.basename(task)
    )

    MODEL_TAG = (
        f"{provider or model_config['provider']}_{model_name or model_config['model']}"
    )

    print("start checkpoint_server")
    checkpoint_server = MCPServerStdio(
        params={
            "command": "python",
            "args": ["checkpointServer.py"],
        },
        client_session_timeout_seconds=10,
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

        current_path = os.path.join(workspace_dir, "current")
        os.makedirs(current_path, exist_ok=True)
        file_server = MCPServerStdio(
            params={
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    current_path,
                ],
            }
        )
        code_run_server = MCPServerStdio(
            # params={
            #     "command": "uv",
            #     "args": ["run", "--with", "mcp", "mcp", "run", "codeRunServer.py"],
            # },
            params={
                "command": "python",
                "args": ["codeRunServer.py"],
            },
            client_session_timeout_seconds=480,
        )
        async with file_server as fs, code_run_server as crs:
            try:
                triton_coder = Agent(
                    model=model,
                    name="triton_coder",
                    instructions=TRITON_CODER_SYSTEM_PROMPT.format(
                        workspace_dir=workspace_dir
                    ),
                    mcp_servers=[fs, ckpts, crs],
                )

                prompt = TRITON_CODER_NEXT_PROMPT.format(
                    task="""Implement Triton Kernel (forward pass only) for the given PyTorch code in
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

                with trace("Triton Coder"):
                    result = await Runner.run(
                        triton_coder,
                        input=prompt,
                        max_turns=(
                            run_config["max_turns"] if "max_turns" in run_config else 50
                        ),
                        hooks=run_hooks,
                        run_config=RunConfig(
                            model_settings=model_settings,
                        ),
                    )
                    # collect a list of all the tools
                    tools = [
                        tool for tool in triton_coder.tools if isinstance(tool, Tool)
                    ]
                    for mcp_server in triton_coder.mcp_servers:
                        for tool in mcp_server._tools_list:
                            tools.append(tool)
                    # log result items
                    log_result_items(
                        tools,
                        result,
                        f"{AGENT_NAME}",
                        MODEL_TAG,
                        TASK_TAG,
                        workspace_dir,
                    )
                    logger.info(result.final_output)
                    return result.final_output
            except Exception as e:
                logger.error(f"Error running Triton Coder: {e}")
                return f"Error running Triton Coder: {e}"
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

    print("=" * 50)
    print(f"Running [{AGENT_NAME}] [{workspace_dir}] with task: {args.task}")
    print("=" * 50)

    tasks = []
    for rollout_id in range(1, args.total_rollouts + 1):
        tasks.append(
            run_triton_coder(
                workspace_dir,
                args.task,
                args.provider,
                args.model_name,
                rollout_id=rollout_id,
            )
        )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--workspace-dir", type=str, default="")
    parser.add_argument("-p", "--provider", type=str, default=None)
    parser.add_argument("-m", "--model-name", type=str, default=None)
    parser.add_argument("-i", "--max-iterations", type=int, default=4)
    parser.add_argument("-r", "--total-rollouts", type=int, default=1)
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="./kernel_bench/level1/1_Square_matrix_multiplication_.py",
    )
    args = parser.parse_args()

    init_logging(AGENT_NAME)

    if args.workspace_dir and os.path.exists(args.workspace_dir):
        workspace_dir = args.workspace_dir
        logger.info(f"Working directory: {workspace_dir}")
    else:
        workspace_dir = get_next_run_folder()
        logger.info(f"Working directory: {workspace_dir}")

    asyncio.run(main(args))
