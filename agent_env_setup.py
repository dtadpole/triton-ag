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
from agentUtil import load_agent_model, init_logging, get_next_run_folder, get_run_hooks
from logger import logger
from pydantic import Field

AGENT_NAME = "env_setup"

ENV_SETUP_SYSTEM_PROMPT = """
You are an expert in environment setup.  You understand the dependencies between different environments and how to setup them.

Workspace directory: {workspace_dir}
"""

ENV_SETUP_NEXT_PROMPT = """
Your task is to setup the environment for the given task.

Task: {task}

1. Analyze the request to understand the task scope
2. Check the required environments are properly setup
3. If any environment is missing, use `uv pip install` to install the required packages
4. Verify the installation by running the test cases that import the packages
5. If verification has passed and task is complete, save the `{workspace_dir}/current` folder as a checkpoint, and stop the task.

Be concise in your reasoning, select the appropriate tool or action.
"""


@function_tool(
    name_override=AGENT_NAME,
    description_override="Env Setup is an expert in environment setup",
)
async def env_setup(
    workspace_dir: str = Field(
        ...,
        description="The working directory",
    ),
    task: str = Field(
        ...,
        description="The task to check the environment setup",
    ),
):
    return await run_env_setup(workspace_dir, task)


# this is the main function that will be called by the Runner
async def run_env_setup(workspace_dir: str, task: str):

    logger.info(f"Running [{AGENT_NAME}] [{workspace_dir}] with task: {task}")

    model, model_settings = load_agent_model(AGENT_NAME)

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
            },
        )
        logger.info(result)

        code_run_server = MCPServerStdio(
            params={
                "command": "uv",
                "args": ["run", "--with", "mcp", "mcp", "run", "codeRunServer.py"],
            }
        )
        async with code_run_server as crs:
            try:
                env_setup = Agent(
                    model=model,
                    name="env_setup",
                    instructions=ENV_SETUP_SYSTEM_PROMPT.format(
                        workspace_dir=workspace_dir
                    ),
                    mcp_servers=[cs, crs],
                )
                prompt = ENV_SETUP_NEXT_PROMPT.format(
                    task=task, workspace_dir=workspace_dir
                )

                run_hooks = get_run_hooks()

                with trace("Env Setup"):
                    result = await Runner.run(
                        env_setup,
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
                logger.error(f"Error running Env Setup: {e}")
                return f"Error running Env Setup: {e}"
            finally:
                logger.info(f"Agent [{AGENT_NAME}] completed!")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--workspace-dir", type=str, default="")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="Setup environment to run Triton code and compare with PyTorch implementation",
    )
    args = parser.parse_args()

    init_logging(AGENT_NAME)

    if args.workspace_dir and os.path.exists(args.workspace_dir):
        workspace_dir = args.workspace_dir
        logger.info(f"Working directory: {workspace_dir}")
    else:
        workspace_dir = get_next_run_folder()
        logger.info(f"Working directory: {workspace_dir}")

    asyncio.run(run_env_setup(workspace_dir, args.input))
