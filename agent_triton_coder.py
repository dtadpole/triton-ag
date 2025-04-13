import os
import json
import asyncio
import argparse
from agents import (
    Agent,
    Runner,
    RunHooks,
    RunConfig,
    trace,
)
from agents.mcp import MCPServerStdio
from util import load_model, init_logging, prepare_next_run_folder
from logger import logger

TRITON_CODER_SYSTEM_PROMPT = """
You are an expert coder with experience in Triton kernels. You understand tilings, parallelism,
numerical precision, and other concepts in the context of Triton and GPU programming.

Working directory: {working_dir}

1. Analyze the request to understand the task scope
2. All the relevant environments has already been setup
3. Check the working directory and subfolders for Python files (ending with `.py`) to understand the current code structure
4. Implement specific code for the given task
5. Use the provided function in `verifier/correctness.py` to verify correctness

When generating code, always follow these instructions:
- Keep the main functionality in a single file in the working directory (not subfolder), check if such file already exists, if so, modify it, otherwise create a new one
- Always use the provided functions in `verifier/correctness.py` to verify correctness, ensure you have run corresponding test function to verify correctness
- You may create your own test cases to verify intermediate results, but the final and official verification will need to be done using the provided function. When create your own test cases, write them in subfolder under `tests`, with filename ends with `_test.py`
- If the final and official verification fails, fix the code and verify again, repeat the process until it passes
- If the final and official verification passes, finish the task

Be concise in your reasoning, then select the appropriate tool or action.
"""

TRITON_CODER_NEXT_PROMPT = """
Your task is to implement a single Module in Triton or a single kernel function in Triton.

Task: {task}

Did you encounter error when running the final verification?
Based on the error information, what's your next action?

Choose the most efficient path forward:
1. Do you understand the error? Can you fix the error immediately?
2. If not sure why the error happened, can you create test cases to verify intermediate results?
3. If you have fixed the error and verified intermediate results, verify again using the final and official verification.
4. If the final and official verification passes, finish the task.

Be concise in your reasoning, then select the appropriate tool or action.
"""


async def main(args):
    model, model_settings = load_model(args.provider, args.model)
    if args.working_dir and os.path.exists(args.working_dir):
        run_folder = args.working_dir
    else:
        run_folder = prepare_next_run_folder()
    print(f"Run folder: {run_folder}")
    file_server = MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", run_folder],
        }
    )
    code_run_server = MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "--with", "mcp", "mcp", "run", "codeRunServer.py"],
        }
    )
    await file_server.__aenter__()
    await code_run_server.__aenter__()
    try:
        triton_coder = Agent(
            model=model,
            name="triton_coder",
            instructions=TRITON_CODER_SYSTEM_PROMPT.format(working_dir=run_folder),
            mcp_servers=[file_server, code_run_server],
        )
        prompt = TRITON_CODER_NEXT_PROMPT.format(task=args.input)

        run_hooks = RunHooks()

        async def on_tool_start(context, agent, tool):
            logger.info(f"Agent [{agent.name}] Tool [{tool.name}] started")

        async def on_tool_end(context, agent, tool, result):
            logger.info(
                f"Agent [{agent.name}] Tool [{tool.name}] ended with result:\n{result}\n"
            )

        run_hooks.on_tool_start = on_tool_start
        run_hooks.on_tool_end = on_tool_end

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
            print(result.final_output)
    finally:
        try:
            await file_server.__aexit__()
            await code_run_server.__aexit__()
        except Exception as e:
            logger.error(f"Error exiting servers: {e}")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--provider", type=str, default="anthropic")
    parser.add_argument("-m", "--model", type=str, default="claude-3.7")
    parser.add_argument("-w", "--working-dir", type=str, default="")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="Implement triton kernel for the Forward pass of nn.Linear, use autotune for the tiling parameters",
    )
    args = parser.parse_args()

    init_logging(args)
    asyncio.run(main(args))
