import os
import json
import asyncio
import argparse
import time
from agents import (
    Agent,
    Runner,
    RunHooks,
    RunConfig,
    trace,
)
from agents.mcp import MCPServer, MCPServerStdio
import mlflow
from util import load_model


import logging

from logger import logger
from pydantic import BaseModel, Field
from typing import List, Any
from dataclasses import dataclass
from pydantic.json_schema import to_jsonable_python

TRITON_CODER_SYSTEM_PROMPT = """
You are an expert coder experienced in Triton kernels.  You understand tilings, parallelism, and other concepts in the context of Triton pragramming.

1. Analysize the request to understand the task scope
2. Create specific code to implement the task
3. Use the provided code in benchmark.py to verify correctness
4. Use `finish` to conclude immediately when you are done

Available tools will vary by task but may include:
- `generate_code`: Create, update, and track plans (commands: create, update, mark_step, etc.)
- `fix_code`: End the task when complete
- `test_code`: Test the code and verify correctness

Break tasks into logical steps with clear outcomes. Avoid excessive detail or sub-steps.
Think about dependencies and verification methods.
Know when to conclude - don't continue thinking once objectives are met.
"""

TRITON_CODER_NEXT_PROMPT = """
Task: {task}

Based on the current state, what's your next action?
Choose the most efficient path forward:
1. Is the plan sufficient, or does it need refinement?
2. Can you execute the next step immediately?
3. Is the task complete? If so, use `finish` right away.

Be concise in your reasoning, then select the appropriate tool or action.
"""


async def main(args):
    model, model_settings = load_model(args.provider, args.model)
    run_folder = find_next_run_folder()
    print(f"Run folder: {run_folder}")
    file_server = MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", run_folder],
        }
    )
    plan_server = MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "--with", "mcp", "mcp", "run", "planServer.py"],
        }
    )
    code_run_server = MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "--with", "mcp", "mcp", "run", "codeRunServer.py"],
        }
    )
    await file_server.__aenter__()
    await plan_server.__aenter__()
    await code_run_server.__aenter__()
    try:
        designer = Agent(
            model=model,
            name="designer",
            instructions=DESIGNER_SYSTEM_PROMPT,
            mcp_servers=[file_server, code_run_server, plan_server],
        )
        prompt = DESIGNER_NEXT_PROMPT.format(goal=args.input)

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
                designer,
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
            await plan_server.__aexit__()
            await code_run_server.__aexit__()
        except Exception as e:
            logger.error(f"Error exiting servers: {e}")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--provider", type=str, default="anthropic")
    parser.add_argument("-m", "--model", type=str, default="claude-3.7")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="Generate triton kernel for nn.linear, no bias, both forward and backward, compare to PyTorch implementation, verify correctness, benchmark performance",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
