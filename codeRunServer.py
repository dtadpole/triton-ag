from mcp.server.fastmcp import FastMCP
import subprocess
import asyncio
from dataclasses import dataclass

server = FastMCP("codeRun")


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str


@server.tool(
    name="echo",
    description="Echo a message",
)
def echo(msg: str) -> str:
    return msg


@server.tool(
    name="run_code",
    description="Run a command in the given working directory",
)
async def run_code(wd: str, cmd: str) -> RunResult:
    command = f"cd {wd} && {cmd}"
    print(command)

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

    return RunResult(
        returncode=process.returncode,
        stdout=stdout.decode(),
        stderr=stderr.decode(),
    )


if __name__ == "__main__":
    server.run(transport="stdio")
