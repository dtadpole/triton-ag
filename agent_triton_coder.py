import os
import asyncio
import argparse
from agents import (
    Agent,
    Runner,
    RunConfig,
    trace,
    function_tool,
)
from agents.mcp import MCPServerStdio
from util import load_agent_model, init_logging, prepare_next_run_folder, get_run_hooks
from logger import logger
from pydantic import Field

AGENT_NAME = "triton_coder"

TRITON_CODER_SYSTEM_PROMPT = """
You are an expert coder with experience in Triton kernels.  You understand tilings, parallelism,
precision, numerical stability, and other concepts in the context of Triton and GPU programming.

Working directory: {working_dir}

1. Analyze the request to understand the task scope
2. All the relevant environments has already been setup
3. Check the working directory and subfolders for Python files (ending with `.py`) to understand the current code structure
4. Implement specific code for the given task, do not change anything else in the working directory

When generating code, always follow these instructions:
- Implement all key functionalities (functions and modules) in a single file directly in the working directory (not subfolder), check if such file already exists, if so, modify it, otherwise create a new one
- Always use the provided functions in `verifier/correctness.py` to verify correctness, ensure to run corresponding test function for correctness verification. (Do not run benchmark, only verify correctness)
- Do not change any existing code in the `verifier` subfolder, do not add any new code in the `verifier` subfolder
- You may create your own test cases to verify intermediate results, but the final and official verification will need to be done using the provided function.
- When creating test cases, write them in subfolder under `tests`, with filename ends with `_test.py` (not in the main working directory)
"""

TRITON_CODER_NEXT_PROMPT = """
Your task is to implement a single Module in Triton or a single kernel function in Triton.

Task: {task}

Implement the task step by step, minimize changes while working on the current step.

Did you encounter error when running the final verification?
Based on the error information, what's your next action?

Choose the most efficient path forward:
1. Do you understand the error? Can you fix the error immediately?
2. If not sure why the error happened, can you create test cases to verify intermediate results and fix the code step by step?
3. If you have fixed the error and verified intermediate results, verify again using the final and official verification.
4. Finish the task if the final and official verification has passed.

Be concise in your reasoning, select the appropriate tool or action.
"""


@function_tool(
    name_override=AGENT_NAME,
    description_override="Triton Coder is an expert with experience in Triton kernels.  It will implement specific code for the given task, it can be either a single module or a single kernel function.  It can also add functionality to existing code.  It will verify correctness of the code before returning it (and won't perform any benchmarks)",
)
async def triton_coder(
    working_dir: str = Field(
        ...,
        description="The working directory",
    ),
    task: str = Field(
        ...,
        description="The task to implement.  Please provide a detailed and specific task description",
    ),
):
    return await run_triton_coder(working_dir, task)


# this is the main function that will be called by the Runner
async def run_triton_coder(working_dir: str, task: str):

    logger.info(f"Running [{AGENT_NAME}] [{working_dir}] with task: {task}")

    model, model_settings = load_agent_model(AGENT_NAME)

    file_server = MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", working_dir],
        }
    )
    code_run_server = MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "--with", "mcp", "mcp", "run", "codeRunServer.py"],
        }
    )
    async with file_server as fs, code_run_server as crs:
        try:
            triton_coder = Agent(
                model=model,
                name="triton_coder",
                instructions=TRITON_CODER_SYSTEM_PROMPT.format(working_dir=working_dir),
                mcp_servers=[fs, crs],
            )
            prompt = TRITON_CODER_NEXT_PROMPT.format(task=task)

            run_hooks = get_run_hooks()

            with trace("Triton Coder"):
                result = await Runner.run(
                    triton_coder,
                    input=prompt,
                    max_turns=50,
                    hooks=run_hooks,
                    run_config=RunConfig(
                        model_settings=model_settings,
                    ),
                )
                logger.info(result.final_output)
                return result.final_output
        except Exception as e:
            logger.error(f"Error running Triton Coder: {e}")
            return f"Error running Triton Coder: {e}"
        finally:
            logger.info(f"Agent [{AGENT_NAME}] completed!")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--working-dir", type=str, default="")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="Implement Triton kernel for the forward pass of nn.Linear, use autotune for the tiling parameters",
    )
    args = parser.parse_args()

    init_logging(AGENT_NAME)

    if args.working_dir and os.path.exists(args.working_dir):
        working_dir = args.working_dir
        logger.info(f"Working directory: {working_dir}")
    else:
        working_dir = prepare_next_run_folder()
        logger.info(f"Working directory: {working_dir}")

    asyncio.run(run_triton_coder(working_dir, args.input))
