from mcp.server.fastmcp import FastMCP
import subprocess
import asyncio
from dataclasses import dataclass
from pydantic import Field
from datetime import datetime
import os
from util import is_subfolder

server = FastMCP("codeRun")


@dataclass
class RunResult:
    returncode: int = Field(
        ...,
        description="The return code of the command",
    )
    stdout: str = Field(
        ...,
        description="The stdout of the command",
    )
    stderr: str = Field(
        ...,
        description="The stderr of the command",
    )


@server.tool(
    name="run_code",
    description="Run a command in the given working directory",
)
async def run_code(
    wd: str = Field(..., description="The working directory to run the command in"),
    cmd: str = Field(..., description="The command to run"),
    tag: str = Field(..., description="The tag for this run", pattern=r"^[a-z_]+$"),
) -> RunResult:
    if not is_subfolder(parent_folder=os.getcwd(), child_folder=wd):
        raise ValueError(
            f"Working directory {wd} is not a subfolder of cwd {os.getcwd()}"
        )
    command = f"cd {wd} && {cmd}"
    print(command)

    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    process = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print(f"Command '{command}' executed successfully.")
        print("Output:", stdout.decode())
    else:
        print(f"Command '{command}' failed with exit code {process.returncode}.")
        print("Error:", stderr.decode())

    rc_file = f"{wd}/code_{time_prefix}_{tag}.rc"
    stdout_file = f"{wd}/code_{time_prefix}_{tag}.stdout"
    stderr_file = f"{wd}/code_{time_prefix}_{tag}.stderr"

    with open(rc_file, "w") as f:
        f.write(f"{process.returncode}\n")
    with open(stdout_file, "w") as f:
        f.write(stdout.decode())
    with open(stderr_file, "w") as f:
        f.write(stderr.decode())

    return RunResult(
        returncode=process.returncode,
        stdout=stdout.decode(),
        stderr=stderr.decode(),
    )


if __name__ == "__main__":
    server.run(transport="stdio")
