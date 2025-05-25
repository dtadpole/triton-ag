from mcp.server.fastmcp import FastMCP
import asyncio
from dataclasses import dataclass
from pydantic import Field
from datetime import datetime
import os
from util import is_subfolder
from logger import logger
from kernel_bench.src.eval import KernelExecResult, eval_kernel_against_ref

server = FastMCP("kbEval")


# def eval_kernel_against_ref(
#     original_model_src: str,
#     custom_model_src: str,
#     seed_num: int = 42,
#     num_correct_trials: int = 1,
#     num_perf_trials: int = 10,
#     verbose: bool = False,
#     measure_performance: bool = False,
#     build_dir: os.PathLike = None,
#     device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
# ) -> KernelExecResult:


@server.tool(
    name="kernel_bench_eval",
    description="Run kernel bench evaluation",
)
async def kb_eval(
    wd: str,
    tag: str,
    reference_model_src: str,
    generated_model_src: str,
    generated_summary: str,
) -> KernelExecResult:
    if not is_subfolder(parent_folder=os.getcwd(), child_folder=wd):
        raise ValueError(
            f"Working directory {wd} is not a subfolder of cwd {os.getcwd()}"
        )

    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # spawn a new process and run the eval_kernel_against_ref function
    process = await asyncio.create_subprocess_shell(
        f"python -m kernel_bench.src.eval {reference_model_src} {generated_model_src} {generated_summary}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    process = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        logger.info(f"Command '{command}' executed successfully.")
        logger.info("Output:", stdout.decode())
    else:
        logger.info(f"Command '{command}' failed with exit code {process.returncode}.")
        logger.info("Error:", stderr.decode())

    rc_file = f"{wd}/code_{tag}_{time_prefix}.rc"
    stdout_file = f"{wd}/code_{tag}_{time_prefix}.stdout"
    stderr_file = f"{wd}/code_{tag}_{time_prefix}.stderr"

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
