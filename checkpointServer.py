import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from util import is_subfolder

server = FastMCP("checkpoint")


@server.tool(
    name="init_workspace_folder",
    description="Initialize a workspace folder",
)
async def init_workspace_folder(
    workspace_folder: str = Field(
        ..., description="The workspace directory for checkpointing"
    ),
    reference_pytorch_code: str = Field(..., description="The reference PyTorch code"),
    environ_vars: dict = Field({}, description="Environment variables to set"),
    include_verifier: bool = Field(
        False, description="Whether to include the verifier folder"
    ),
):
    try:
        print("init_workspace_folder")
        if not is_subfolder(parent_folder=os.getcwd(), child_folder=workspace_folder):
            raise ValueError(
                f"Workspace folder {workspace_folder} is not a subfolder of cwd {os.getcwd()}"
            )
            # if the workspace folder does not exist, create it
        init_checkpoint_folder = os.path.join(workspace_folder, f"{0:02d}.checkpoint")
        if not os.path.exists(init_checkpoint_folder):
            os.makedirs(init_checkpoint_folder, exist_ok=True)
            if include_verifier:
                # create verifier folder in the new run folder
                shutil.copytree(
                    os.path.join(os.getcwd(), "verifier"),
                    os.path.join(init_checkpoint_folder, "verifier"),
                )
        print("finished making init_checkpoint_folder")
        # copy task file to workspace_dir as "pytorch_reference.py"
        shutil.copy(
            reference_pytorch_code,
            os.path.join(init_checkpoint_folder, "pytorch_reference.py"),
        )
        # write environ_vars to a file in the init_checkpoint_folder
        with open(os.path.join(init_checkpoint_folder, "environ_vars.json"), "w") as f:
            json.dump(environ_vars, f)
        # restore from the latest checkpoint
        await restore_last_checkpoint(
            workspace_folder=workspace_folder,
            target_folder=os.path.join(workspace_folder, "current"),
        )
        return f"Workspace folder {workspace_folder} initialized"
    except Exception as e:
        return f"Error initializing workspace folder: {e}"


@server.tool(
    name="save_checkpoint",
    description="Save a checkpoint from the source directory",
)
async def save_checkpoint(
    workspace_folder: str = Field(
        ..., description="The workspace directory for checkpointing"
    ),
    source_folder: str = Field(
        ..., description="The source folder to be checkpointed (preserved)"
    ),
) -> str:
    try:
        if not is_subfolder(parent_folder=os.getcwd(), child_folder=workspace_folder):
            raise ValueError(
                f"Workspace folder {workspace_folder} is not a subfolder of cwd {os.getcwd()}"
            )
        if not is_subfolder(parent_folder=workspace_folder, child_folder=source_folder):
            raise ValueError(
                f"Source folder {source_folder} is not a subfolder of {workspace_folder}"
            )
        # find next checkpoint folder
        i = 0
        while os.path.exists(os.path.join(workspace_folder, f"{i:02d}.checkpoint")):
            i += 1
        next_checkpoint_folder = os.path.join(workspace_folder, f"{i:02d}.checkpoint")
        # copy everything from source_folder to next_checkpoint_folder
        shutil.copytree(source_folder, next_checkpoint_folder)
        # return the next_checkpoint_folder
        return f"Checkpoint created as {next_checkpoint_folder}"
    except Exception as e:
        return f"Error creating checkpoint: {e}"


@server.tool(
    name="restore_checkpoint",
    description="Restore a checkpoint to the target directory",
)
async def restore_last_checkpoint(
    workspace_folder: str = Field(
        ..., description="The workspace directory for checkpointing"
    ),
    target_folder: str = Field(
        ..., description="The target folder to restore the checkpoint to"
    ),
) -> str:
    try:
        if not is_subfolder(parent_folder=os.getcwd(), child_folder=workspace_folder):
            raise ValueError(
                f"Workspace folder {workspace_folder} is not a subfolder of cwd {os.getcwd()}"
            )
        if not is_subfolder(parent_folder=workspace_folder, child_folder=target_folder):
            raise ValueError(
                f"Target folder {target_folder} is not a subfolder of {workspace_folder}"
            )
        # check that target_folder is not a checkpoint folder
        if target_folder.endswith(".checkpoint"):
            raise ValueError(
                f"Target folder {target_folder} cannot be a checkpoint folder"
            )
        # find last checkpoint folder
        i = 0
        while os.path.exists(os.path.join(workspace_folder, f"{i:02d}.checkpoint")):
            i += 1
        if i == 0:
            raise ValueError(f"No checkpoint folders found in {workspace_folder}")
        i -= 1
        last_checkpoint_folder = os.path.join(workspace_folder, f"{i:02d}.checkpoint")
        # check if target_folder is a subfolder of parent_folder
        # remove target_folder if it exists
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        # rename last_checkpoint_folder to target_folder
        shutil.copytree(last_checkpoint_folder, target_folder)
        return f"Checkpoint restored to {target_folder}"
    except Exception as e:
        return f"Error restoring checkpoint: {e}"


if __name__ == "__main__":
    server.run(transport="stdio")
