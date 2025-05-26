from mcp.server.fastmcp import FastMCP
import asyncio
import json
from datetime import datetime
import os
from util import is_subfolder
from logger import logger
from kernel_bench.src.eval import KernelExecResult
import boto3
from pydantic import Field

server = FastMCP("kbEval")


@server.tool(
    name="kb_eval",
    description="Run kernel bench evaluation",
)
async def kb_eval(
    wd: str = Field(..., description="The working directory"),
    eval_tag: str = Field(..., description="The tag of the evaluation"),
    reference_code_path: str = Field(..., description="The path to the reference code"),
    generated_code_path: str = Field(..., description="The path to the generated code"),
) -> KernelExecResult:
    if not is_subfolder(parent_folder=os.getcwd(), child_folder=wd):
        raise ValueError(
            f"Working directory {wd} is not a subfolder of cwd {os.getcwd()}"
        )

    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    command = f"python kbEvalCli.py --wd {wd} --tag {eval_tag}_{time_prefix} --reference_code {reference_code_path} --generated_code {generated_code_path}"

    # spawn a new process and run the eval_kernel_against_ref function
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        logger.info(f"Command '{command}' executed successfully.")
        logger.info("Output:", stdout.decode())
    else:
        logger.info(f"Command '{command}' failed with exit code {process.returncode}.")
        logger.info("Error:", stderr.decode())

    rc_file = f"{wd}/kbeval_{eval_tag}_{time_prefix}.rc"
    stdout_file = f"{wd}/kbeval_{eval_tag}_{time_prefix}.stdout"
    stderr_file = f"{wd}/kbeval_{eval_tag}_{time_prefix}.stderr"

    with open(rc_file, "w") as f:
        f.write(f"{process.returncode}\n")
    with open(stdout_file, "w") as f:
        f.write(stdout.decode())
    with open(stderr_file, "w") as f:
        f.write(stderr.decode())

    # read json from {wd}/kbeval_{tag}_{time_prefix}.json
    with open(f"{wd}/kbeval_{eval_tag}_{time_prefix}.json", "r") as f:
        result = KernelExecResult.model_validate_json(f.read())

    return result


@server.tool(
    name="kb_upload_summary",
    description="Upload the summary of the evaluation to s3",
)
async def kb_upload_summary(
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    wd: str = Field(..., description="The working directory"),
    eval_tag: str = Field(..., description="The tag of the evaluation"),
    reference_code_path: str = Field(..., description="The path to the reference code"),
    generated_code_path: str = Field(..., description="The path to the generated code"),
    generated_summary: str = Field(..., description="The summary of the generated code"),
    result: KernelExecResult = Field(..., description="The result from kb_eval"),
) -> str:
    try:
        # read reference code
        with open(reference_code_path, "r") as f:
            reference_code = f.read()
        # read generated code
        with open(generated_code_path, "r") as f:
            generated_code = f.read()

        time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        # generate summary of this iteration, and write to disk as json
        summary = {
            "eval_tag": eval_tag,
            "time_prefix": time_prefix,
            "reference_code": reference_code,
            "generated_code": generated_code,
            "generated_summary": generated_summary,
            "result": result.model_dump(),
        }

        with open(f"{wd}/kbeval_{eval_tag}_{time_prefix}.summary.json", "w") as f:
            f.write(json.dumps(summary, indent=4))

        # push to aws s3
        s3_client = boto3.client("s3")
        upload_key = f"kbeval/{model_tag}/{task_tag}/{eval_tag}_{time_prefix}.summary.json"
        s3_client.put_object(Bucket="agent-xyz",
                            Key=upload_key,
                            Body=json.dumps(summary, indent=4))

        return f"uploaded to s3://agent-xyz/{upload_key}"
    except Exception as e:
        logger.error(f"Error uploading summary to s3: {e}")
        raise e

if __name__ == "__main__":
    server.run(transport="stdio")
