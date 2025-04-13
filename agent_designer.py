import os
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


DESIGNER_SYSTEM_PROMPT = """
You are an expert Design Agent tasked with designing a system to solve a problem.

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

DESIGNER_NEXT_PROMPT = """
Goal: {goal}

Based on the current state, what's your next action?
Choose the most efficient path forward:
1. Is the plan sufficient, or does it need refinement?
2. Can you execute the next step immediately?
3. Is the task complete? If so, use `finish` right away.

Be concise in your reasoning, then select the appropriate tool or action.
"""


async def main(args):
    model, model_settings = load_model(args.provider, args.model)
    run_folder = prepare_next_run_folder()
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

        with trace("Agent Designer"):
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
            await file_server.cleanup()
        except Exception as e:
            logger.error(f"Error exiting servers: {e}")
        try:
            await plan_server.cleanup()
        except Exception as e:
            logger.error(f"Error exiting servers: {e}")
        try:
            await code_run_server.cleanup()
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

    init_logging(args)
    asyncio.run(main(args))
