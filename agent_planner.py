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
from agent_triton_coder import triton_coder
from util import load_agent_model, init_logging, get_next_run_folder, get_run_hooks
from logger import logger
from pydantic import Field

AGENT_NAME = "planner"

PLANNER_SYSTEM_PROMPT = """
You are an expert Planner tasked with planning a system to solve a problem.

Working directory: {working_dir}

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

PLANNER_NEXT_PROMPT = """
Goal: {goal}

Based on the current state, what's your next action?
Choose the most efficient path forward:
1. Is the plan sufficient, or does it need refinement?
2. Can you execute the next step immediately?
3. Is the task complete? If so, use `finish` right away.

Be concise in your reasoning, then select the appropriate tool or action.
"""


@function_tool(
    name_override=AGENT_NAME,
    description_override="Planner is an expert in planning.  It will break down the given task into smaller steps and find the best way to achieve the goal.  Once the plan is complete, it will use available tools to implement the plan.",
)
async def planner(
    working_dir: str = Field(..., description="The working directory"),
    goal: str = Field(..., description="The goal to achieve"),
):
    return await run_planner(goal, working_dir)


# this is the main function that will be called by the Runner
async def run_planner(goal: str, working_dir: str):

    logger.info(f"Running [{AGENT_NAME}] [{working_dir}] with goal: {goal}")

    model, model_settings = load_agent_model(AGENT_NAME)
    plan_server = MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "--with", "mcp", "mcp", "run", "planServer.py"],
        }
    )
    async with plan_server as ps:
        try:
            planner = Agent(
                model=model,
                name=AGENT_NAME,
                instructions=PLANNER_SYSTEM_PROMPT.format(working_dir=working_dir),
                tools=[triton_coder],
                mcp_servers=[ps],
            )
            prompt = PLANNER_NEXT_PROMPT.format(goal=goal)

            run_hooks = get_run_hooks()

            with trace(f"Agent [{AGENT_NAME}]"):
                result = await Runner.run(
                    planner,
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
            logger.error(f"Error running Planner: {e}")
            return f"Error running Planner: {e}"
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
        default="Generate triton kernel for nn.linear, no bias, both forward and backward, compare to PyTorch implementation, verify correctness, do _NOT_ benchmark performance",
    )
    args = parser.parse_args()

    init_logging(AGENT_NAME)

    if args.working_dir and os.path.exists(args.working_dir):
        working_dir = args.working_dir
        logger.info(f"Working directory: {working_dir}")
    else:
        working_dir = get_next_run_folder()
        logger.info(f"Working directory: {working_dir}")

    asyncio.run(run_planner(args.input, working_dir))
