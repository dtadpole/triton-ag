import os
import json
import asyncio
import argparse
from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    RunHooks,
    FunctionTool,
    RunConfig,
    trace,
    enable_verbose_stdout_logging,
    set_trace_processors,
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

PLANNER_SYSTEM_PROMPT = """
You are an expert Planning Agent tasked with solving problems efficiently through structured plans.

1. Analyze requests to understand the task scope
2. Create a clear, detailed, andactionable plan that makes meaningful progress with the `planning` tool
3. After each step of changing code, test and verify correctness using available tools, fix code until it is correct
4. Track progress and adapt plans when necessary
5. Use `finish` to conclude immediately when the task is complete

Available tools will vary by task but may include:
- `planning`: Create, update, and track plans (commands: create, update, mark_step, etc.)
- `finish`: End the task when complete
Break tasks into logical steps with clear outcomes. Avoid excessive detail or sub-steps.
Think about dependencies and verification methods.
Know when to conclude - don't continue thinking once objectives are met.
"""

PLANNING_NEXT_PROMPT = """
Goal: {goal}

Based on the current state, what's your next action?
Choose the most efficient path forward:
1. Is the plan sufficient, or does it need refinement?
2. Can you execute the next step immediately?
3. Is the task complete? If so, use `finish` right away.

Be concise in your reasoning, then select the appropriate tool or action.
"""


# function to find next available folder starting with _run_<number>
def find_next_run_folder():
    i = 0
    while os.path.exists(os.path.join(os.getcwd(), f"_run_{i:03d}")):
        i += 1
    os.makedirs(os.path.join(os.getcwd(), f"_run_{i:03d}"), exist_ok=True)
    return os.path.join(os.getcwd(), f"_run_{i:03d}")


async def main(args):
    model, model_settings = load_model(args.provider, args.model)
    run_folder = find_next_run_folder()
    file_server = MCPServerStdio(
        params={
            "command": "npx",
            "args": [ "-y", "@modelcontextprotocol/server-filesystem", run_folder ],
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
        programmer = Agent(
            model=model,
            name="programmer",
            instructions=PLANNER_SYSTEM_PROMPT,
            mcp_servers=[file_server, code_run_server, plan_server],
        )
        prompt = PLANNING_NEXT_PROMPT.format(goal=args.input)

        run_hooks = RunHooks()

        async def on_tool_start(context, agent, tool):
            logger.info(f"Agent [{agent.name}] Tool [{tool.name}] started")

        async def on_tool_end(context, agent, tool, result):
            logger.info(
                f"Agent [{agent.name}] Tool [{tool.name}] ended with result:\n{result}\n"
            )

        run_hooks.on_tool_start = on_tool_start
        run_hooks.on_tool_end = on_tool_end

        result = await Runner.run(
            programmer,
            input=prompt,
            max_turns=50,
            hooks=run_hooks,
            run_config=RunConfig(
                model_settings=model_settings,
            ),
        )
        print(result.final_output)
    finally:
        await file_server.__aexit__()
        await plan_server.__aexit__()
        await code_run_server.__aexit__()


if __name__ == "__main__":
    enable_verbose_stdout_logging()
    stdout_logger = logging.getLogger("openai.agents")
    stdout_logger.setLevel(logging.INFO)
    stdout_logger.addHandler(logging.StreamHandler())

    # mlflow.openai.autolog()
    # mlflow.set_tracking_uri("http://localhost:5050")
    # mlflow.set_experiment("OpenAI Agent")

    # weave.init("openai-agents")
    # set_trace_processors([WeaveTracingProcessor()])

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
