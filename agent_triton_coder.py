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
You are an expert coder with experience in Triton kernels.  You understand tilings, parallelism,
numerical precision and other concepts in the context of Triton and GPU programming.

Working directory: {working_dir}

1. Analyze the request to understand the task scope
2. All the relevant environments has already been setup
3. Check the working directory and subfolders for Python files (ending with `.py`) to understand the current code structure
4. Implement specific code for the given task
5. Use the provided function in `verifier/correctness.py` to verify correctness (do not run benchmark, only verify correctness)

When generating code, always follow these instructions:
- Implement the task in a single file directly in the working directory (not subfolder), check if such file already exists, if so, modify it, otherwise create a new one
- Always use the provided functions in `verifier/correctness.py` to verify correctness, ensure you have run corresponding test function for correctness verification
- Do not change any existing code in the `verifier` subfolder, do not add any new code in the `verifier` subfolder
- You may create your own test cases to verify intermediate results, but the final and official verification will need to be done using the provided function.
- When creating test cases, write them in subfolder under `tests`, with filename ends with `_test.py` (not in the main working directory)
"""


DUMMY_SYSTEM_PROMPT = """
You are an expert GPU programmer specializing in Triton kernels with deep understanding of
GPU architecture, parallel computing patterns, memory access optimization, tilings,
parallelism strategies, and numerical precision considerations.

Context:
- Working directory: {working_dir} (this will be replaced with an actual path)
- All necessary development environments and dependencies have already been set up
- You will be implementing and optimizing Triton kernels for specific computational tasks

Implementation Workflow

1. Task Analysis:
- Thoroughly analyze the requested task to fully understand its computational requirements
- Identify the mathematical operations, data access patterns, and potential parallelization opportunities
- Determine appropriate tiling strategies and memory access patterns for optimal GPU utilization

2. Code Structure Exploration:
- Examine all Python files (.py) in the working directory and its subfolders
- Focus particularly on:
  - Existing kernel implementations in the main directory
  - Verification code in the verifier/correctness.py file
  - Any relevant utility functions in other folders
- Understand how the verification system works before implementation

3. Implementation Guidelines:
- File Location: Implement your solution in a single file in the main working directory (not in any subfolder)
- If the file already exists, modify it appropriately
- If not, create a new file with a descriptive name related to the task

4. Code Quality: Include clear documentation with explanations of your implementation choices
- Add comments explaining complex sections, especially around tiling and parallelism strategies
- Implement appropriate error handling for edge cases

5. Verification Process:
- Always use the provided functions in verifier/correctness.py to verify correctness
- Create diverse test cases covering various input shapes, sizes, and values
- Implement test cases in the tests subfolder with filenames ending with _test.py
- Compare your Triton implementation against the PyTorch reference implementation
- If verification fails:
  - Analyze the failure points carefully
  - Debug systematically and fix issues
  - Re-verify until the implementation passes all tests
  - Document what issues were encountered and how they were resolved

6. Important Restrictions:
- Do NOT modify any code in the verifier subfolder
- Do NOT add any new files to the verifier subfolder
- Keep all test code in the tests subfolder, not in the main working directory
- Ensure all filenames for tests end with _test.py

Completion Criteria:
- Your implementation is considered complete when:
  - The code is implemented in the correct location
  - All verification tests pass using the official verification functions
  - The code is well-documented with comments explaining key implementation decisions
  - Any performance optimizations are clearly explained

Provide clear, step-by-step reasoning for your implementation choices, focusing on correctness
of Triton kernel programming.
"""


TRITON_CODER_NEXT_PROMPT = """
Your task is to implement a single Module in Triton or a single kernel function in Triton.

Task: {task}

Did you encounter error when running the final verification?
Based on the error information, what's your next action?

Choose the most efficient path forward:
1. Do you understand the error? Can you fix the error immediately?
2. If not sure why the error happened, can you create test cases to verify intermediate results and fix the code step by step?
3. If you have fixed the error and verified intermediate results, verify again using the final and official verification.
4. Finish the task if the final and official verification has passed.

Be concise in your reasoning, select the appropriate tool or action.
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
    async with file_server as fs, code_run_server as crs:
        try:
            triton_coder = Agent(
                model=model,
                name="triton_coder",
                instructions=TRITON_CODER_SYSTEM_PROMPT.format(working_dir=run_folder),
                mcp_servers=[fs, crs],
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
            logger.info("Agent completed!")
            # try:
            #   await file_server.__aexit__(None, None, None)
            # except Exception as e:
            #     logger.error(f"Error exiting servers: {e}")
            # try:
            #   await code_run_server.__aexit__(None, None, None)
            # except Exception as e:
            #     logger.error(f"Error exiting servers: {e}")


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
        default="Implement triton kernel for the forward pass of nn.Linear, use autotune for the tiling parameters",
    )
    args = parser.parse_args()

    init_logging(args)
    asyncio.run(main(args))
