import os
import shutil
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
from pydantic import Field
from pydantic.json_schema import to_jsonable_python



AGENT_NAME = "kernel_bench"

KERNEL_BENCH_SYSTEM_PROMPT = """
You are an expert coder with experience in Triton kernels.  You understand tilings, parallelism,
precision, numerical stability, and other concepts in the context of Triton and GPU programming.

Workspace directory: {workspace_dir}

1. Analyze the request to understand the task scope
2. All the relevant environments has already been setup
3. Check the `{workspace_dir}/current` folder and subfolders for Python files (ending with `.py`) to understand the current code structure
4. Implement specific code for the given task, do not change anything else

When generating code, always follow these instructions:
- Implement all key functionalities (functions and modules) in a single file directly in the working directory (not subfolder), check if such file already exists, if so, modify it, otherwise create a new one
- Always use the provided functions in `verifier/correctness.py` to verify correctness, ensure to run corresponding test function for correctness verification. (Do not run benchmark, only verify correctness)
- Do not change any existing code in the `verifier` subfolder, do not add any new code in the `verifier` subfolder
- You may create your own test cases to verify intermediate results, but the final and official verification will need to be done using the provided function.
- When creating test cases, write them in subfolder under `tests`, with filename ends with `_test.py` (not in the main working directory)
"""

KERNEL_BENCH_NEXT_PROMPT = """
Your task is to implement a single Module in Triton or a single kernel function in Triton.

Task: {task}

Implement the task step by step, minimize changes while working on the current step, merge or replace existing code if necessary.

Did you encounter error when running the final verification?
Based on the error information, what's your next action?

Choose the most efficient path forward:
1. Do you understand the error? Can you fix the error easily?
2. If not sure why the error happened, can you create debug test cases to check each intermediate result step by step, and fix the code at each individual step?
3. If you have passed all the intermediate results, verify again using the final and official verification.
4. If the final and official verification has passed and task is complete, save the `{workspace_dir}/current` folder as a checkpoint, and stop the task.
5. If the final and official verification fails repeatedly, restore from the last checkpoint to `{workspace_dir}/current` folder and try again.
6. Print the generated Triton kernel code in the final output.

Be concise in your reasoning, select the appropriate tool or action.
"""


# this is the main function that will be called by the Runner
async def run_kernel_bench(workspace_dir: str, task: str):

    logger.info(f"Running [{AGENT_NAME}] [{workspace_dir}] with task: {task}")

    model, model_settings, run_config, model_config = load_agent_model(AGENT_NAME)

    TASK_NAME = os.path.join(AGENT_NAME, 
                             os.path.basename(os.path.dirname(task)),
                             os.path.basename(task))

    MODEL_TAG = f"{model_config['provider']}_{model_config['model']}"

    checkpoint_server = MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "--with", "mcp", "mcp", "run", "checkpointServer.py"],
        }
    )
    async with checkpoint_server as cs:
        result = await cs.call_tool(
            tool_name="init_workspace_folder",
            arguments={
                "workspace_folder": workspace_dir,
                "reference_pytorch_code": task,
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
            }
        )
        code_run_server = MCPServerStdio(
            params={
                "command": "uv",
                "args": ["run", "--with", "mcp", "mcp", "run", "codeRunServer.py"],
            },
            client_session_timeout_seconds=120,
        )
        async with file_server as fs, code_run_server as crs:
            try:
                kernel_bench = Agent(
                    model=model,
                    name="kernel_bench",
                    instructions=KERNEL_BENCH_SYSTEM_PROMPT.format(
                        workspace_dir=workspace_dir
                    ),
                    mcp_servers=[fs, cs, crs],
                )
                prompt = KERNEL_BENCH_NEXT_PROMPT.format(
                    task="Implement Triton Kernel (forward pass only) for the given PyTorch code as in `pytorch_reference.py`",
                    workspace_dir=workspace_dir
                )

                run_hooks = get_run_hooks()

                with trace("Kernel Bench"):
                    result = await Runner.run(
                        kernel_bench,
                        input=prompt,
                        max_turns=run_config['max_turns'] if 'max_turns' in run_config else 50,
                        hooks=run_hooks,
                        run_config=RunConfig(
                            model_settings=model_settings,
                        ),
                    )
                    log_result_items(result, TASK_NAME, MODEL_TAG, workspace_dir)
                    logger.info(result.final_output)
                    return result.final_output
            except Exception as e:
                logger.error(f"Error running Kernel Bench: {e}")
                return f"Error running Kernel Bench: {e}"
            finally:
                logger.info(f"Agent [{AGENT_NAME}] completed!")



if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--workspace-dir", type=str, default="")
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="./kernel-bench/level1/1_Square_matrix_multiplication_.py",
    )
    args = parser.parse_args()

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

    asyncio.run(run_kernel_bench(workspace_dir, args.task))
