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
    current_wd: str = Field(..., description="The current working directory"),
    eval_tag: str = Field(..., description="The tag of the evaluation"),
    reference_code_filename: str = Field(..., description="The filename of the reference code"),
    generated_code_filename: str = Field(..., description="The filename of the generated code"),
) -> KernelExecResult:
    if not is_subfolder(parent_folder=os.getcwd(), child_folder=current_wd):
        raise ValueError(
            f"Working directory {current_wd} is not a subfolder of cwd {os.getcwd()}"
        )

    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    command = f"python kbEvalCli.py --wd {current_wd} --tag {eval_tag}_{time_prefix} --reference_code {reference_code_filename} --generated_code {generated_code_filename}"

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

    rc_file = f"{current_wd}/kbeval_{eval_tag}_{time_prefix}.rc"
    stdout_file = f"{current_wd}/kbeval_{eval_tag}_{time_prefix}.stdout"
    stderr_file = f"{current_wd}/kbeval_{eval_tag}_{time_prefix}.stderr"

    with open(rc_file, "w") as f:
        f.write(f"{process.returncode}\n")
    with open(stdout_file, "w") as f:
        f.write(stdout.decode())
    with open(stderr_file, "w") as f:
        f.write(stderr.decode())

    # read json from {wd}/kbeval_{tag}.json
    with open(f"{current_wd}/kbeval_{eval_tag}_{time_prefix}.json", "r") as f:
        result = KernelExecResult.model_validate_json(f.read())

    return result


@server.tool(
    name="kb_upload_summary",
    description="Upload the summary of the evaluation",
)
async def kb_upload_summary(
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    current_wd: str = Field(..., description="The current working directory"),
    eval_tag: str = Field(..., description="The tag of the evaluation"),
    result: KernelExecResult = Field(..., description="The result from kb_eval"),
    reference_code_filename: str = Field(..., description="The filename of the reference code"),
    generated_code_filename: str = Field(..., description="The filename of the generated code"),
    generated_summary: str = Field(..., description="The summary of the generated code"),
) -> str:
    try:
        logger.info(f"Uploading summary to s3: {model_tag}, {task_tag}, {eval_tag}")
        logger.info(f"Current working directory: {current_wd}")
        logger.info(f"Result: {result.model_dump()}")
        logger.info(f"Reference code filename: {reference_code_filename}")
        logger.info(f"Generated code filename: {generated_code_filename}")
        logger.info(f"Generated summary: {generated_summary}")

        # read reference code
        with open(os.path.join(current_wd, reference_code_filename), "r") as f:
            reference_code = f.read()
        # read generated code
        with open(os.path.join(current_wd, generated_code_filename), "r") as f:
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

        with open(f"{current_wd}/kbeval_{eval_tag}_{time_prefix}.summary.json", "w") as f:
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
