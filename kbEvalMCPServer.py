import asyncio
import json
import os
import time
import traceback
from datetime import datetime
import os
from util import is_subfolder
from logger import logger
from kbEvalClient import KbEvalClient
from kbEvalTest.kbeval import KernelExecResult
import boto3
import requests
import yaml
from kbEvalTest.kbeval import KernelExecResult
from logger import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from util import is_devserver, is_subfolder

server = FastMCP("kbEval")

yaml_config = yaml.load(open(os.path.join(os.path.dirname(__file__), "kbEval.yaml"), "r"), Loader=yaml.FullLoader)
kb_eval_client = KbEvalClient()
server_stats_and_key = {}
server_last_refresh_time = 0
api_key_mapping = {}


def upload_recap_to_s3(current_wd: str,
                         run_tag: str,
                         model_tag: str,
                         task_tag: str,
                         eval_tag: str,
                         result: dict,
                         reference_code: str,
                         generated_code: str,
                         recap: str = "",
    ):
    summary = {
        "run_tag": run_tag,
        "model_tag": model_tag,
        "task_tag": task_tag,
        "eval_tag": eval_tag,
        "summary": recap,
        "result": result,
        "reference_code": reference_code,
        "generated_code": generated_code,
    }

    try:
        with open(f"{current_wd}/kbeval_{eval_tag}_{run_tag}.summary.json", "w") as f:
            f.write(json.dumps(summary, indent=4))
    except Exception as e:
        logger.error(f"Error writing recap to file: {e}")
        logger.error(traceback.format_exc())

    upload_key = f"kbeval/{run_tag}/{model_tag}/{task_tag}/{eval_tag}.summary.json"

    # push to aws s3
    try:
        s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket="agent-xyz", Key=upload_key, Body=json.dumps(summary, indent=4)
        )
        return f"uploaded to s3://agent-xyz/{upload_key}"
    except Exception as e:
        logger.error(f"Error pushing to s3: {e}")
        logger.error(traceback.format_exc())
        return f"Error pushing to s3: {e}"


@server.tool(
    name="kb_eval_reference",
    description="Run kernel bench evaluation for the reference code",
)
async def kb_eval_reference(
    current_wd: str = Field(..., description="The current working directory"),
    run_tag: str = Field(..., description="The tag of the run"),
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    reference_code_filename: str = Field(..., description="The filename of the reference code"),
) -> KernelExecResult:
    try:
        if not is_subfolder(parent_folder=os.getcwd(), child_folder=current_wd):
            raise ValueError(
                f"Working directory {current_wd} is not a subfolder of cwd {os.getcwd()}"
            )

        # read reference code
        with open(os.path.join(current_wd, reference_code_filename), "r") as f:
            reference_code = f.read()

        # send request to remote server
        result = await kb_eval_client.kb_eval_ref(run_tag=run_tag, model_tag=model_tag, task_tag=task_tag, reference_code=reference_code)

        if result is None:
            raise Exception("Failed to evaluate reference code")

        # write response to file
        response_json = result.model_dump()
        if "metadata" not in response_json:
            response_json["metadata"] = {}
        response_json["metadata"] = response_json["metadata"] | {
            "run_tag": run_tag,
            "model_tag": model_tag,
            "task_tag": task_tag,
        }
        with open(f"{current_wd}/kbeval_reference_{run_tag}.result.json", "w") as f:
            f.write(json.dumps(response_json, indent=4))
        logger.info(
            f"Response from remote server: {json.dumps(response_json, indent=4)}"
        )

        if is_devserver() is False:
            # upload recap to s3
            upload_recap_to_s3(
                current_wd,
                model_tag,
                task_tag,
                "reference",
                run_tag,
                response_json,
                reference_code,
                "",
                recap="Benchmark the reference code",
            )

        return KernelExecResult.model_validate_json(json.dumps(response_json))

    except Exception as e:
        logger.error(f"Error in kb_eval: {e}")
        logger.error(traceback.format_exc())
        raise e


@server.tool(
    name="kb_eval_dspy",
    description="Run kernel bench evaluation and return the evaluation result",
)
async def kb_eval_dspy(
    rationale: str = Field(..., description="The rationale for the generated code"),
    current_wd: str = Field(..., description="The current working directory"),
    run_tag: str = Field(..., description="The tag of the run"),
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    eval_tag: str = Field(..., description="The tag of the iteration"),
    reference_code_filename: str = Field(
        ..., description="The filename of the reference code"
    ),
    generated_code_filename: str = Field(
        ..., description="The filename of the generated code"
    ),
) -> KernelExecResult:
    try:
        if not is_subfolder(parent_folder=os.getcwd(), child_folder=current_wd):
            raise ValueError(
                f"Working directory {current_wd} is not a subfolder of cwd {os.getcwd()}"
            )

        # read reference code
        with open(os.path.join(current_wd, reference_code_filename), "r") as f:
            reference_code = f.read()
        # read generated code
        with open(os.path.join(current_wd, generated_code_filename), "r") as f:
            generated_code = f.read()

        # send request to remote server
        result = await kb_eval_client.kb_eval(run_tag=run_tag, model_tag=model_tag, task_tag=task_tag, eval_tag=eval_tag, reference_code=reference_code, generated_code=generated_code)

        if result is None:
            raise Exception("Failed to evaluate generated code")

        response_json = result.model_dump()
        if "metadata" not in response_json:
            response_json["metadata"] = {}
        response_json["metadata"] = response_json["metadata"] | {
            "run_tag": run_tag,
            "model_tag": model_tag,
            "task_tag": task_tag,
            "eval_tag": eval_tag,
        }
        result_filename = f"{current_wd}/kbeval_{eval_tag}_{run_tag}.result.json"
        with open(result_filename, "w") as f:
            f.write(json.dumps(response_json, indent=4))
        logger.info(
            f"Response from remote server: {json.dumps(response_json, indent=4)}"
        )

        if is_devserver() is False:
            # upload recap to s3
            upload_recap_to_s3(
                current_wd,
                model_tag,
                task_tag,
                eval_tag,
                run_tag,
                response_json,
                reference_code,
                generated_code,
                recap=rationale,
            )

        return KernelExecResult.model_validate_json(json.dumps(response_json))

    except Exception as e:
        logger.error(f"Error in kb_eval_dspy: {e}")
        logger.error(traceback.format_exc())
        raise e


@server.tool(
    name="kb_eval_iteration",
    description="Run kernel bench evaluation for a specific iteration",
)
async def kb_eval_iteration(
    current_wd: str = Field(..., description="The current working directory"),
    run_tag: str = Field(..., description="The tag of the run"),
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    eval_tag: str = Field(..., description="The tag of the iteration"),
    reference_code_filename: str = Field(
        ..., description="The filename of the reference code"
    ),
    generated_code_filename: str = Field(
        ..., description="The filename of the generated code"
    ),
) -> KernelExecResult:
    try:
        if not is_subfolder(parent_folder=os.getcwd(), child_folder=current_wd):
            raise ValueError(
                f"Working directory {current_wd} is not a subfolder of cwd {os.getcwd()}"
            )

        # read reference code
        with open(os.path.join(current_wd, reference_code_filename), "r") as f:
            reference_code = f.read()
        # read generated code
        with open(os.path.join(current_wd, generated_code_filename), "r") as f:
            generated_code = f.read()

        # send request to remote server
        result = await kb_eval_client.kb_eval(run_tag=run_tag, model_tag=model_tag, task_tag=task_tag, eval_tag=eval_tag, reference_code=reference_code, generated_code=generated_code)

        if result is None:
            raise Exception("Failed to evaluate generated code")

        response_json = result.model_dump()
        if "metadata" not in response_json:
            response_json["metadata"] = {}
        response_json["metadata"] = response_json["metadata"] | {
            "run_tag": run_tag,
            "model_tag": model_tag,
            "task_tag": task_tag,
            "eval_tag": eval_tag,
        }
        with open(f"{current_wd}/kbeval_{eval_tag}_{run_tag}.result.json", "w") as f:
            f.write(json.dumps(response_json, indent=4))
        logger.info(
            f"Response from remote server: {json.dumps(response_json, indent=4)}"
        )

        if is_devserver() is False:
            # upload recap to s3
            upload_recap_to_s3(
                current_wd,
                model_tag,
                task_tag,
                eval_tag,
                run_tag,
                response_json,
                reference_code,
                generated_code,
                recap="",
            )

        return KernelExecResult.model_validate_json(json.dumps(response_json))

    except Exception as e:
        logger.error(f"Error in kb_eval: {e}")
        logger.error(traceback.format_exc())
        raise e


@server.tool(
    name="kb_upload_iteration",
    description="Upload the iteration recap for a specific iteration of the kernel generation and evaluation",
)
async def kb_upload_iteration(
    current_wd: str = Field(..., description="The current working directory"),
    run_tag: str = Field(..., description="The tag of the run"),
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    eval_tag: str = Field(..., description="The tag of the iteration"),
    reference_code_filename: str = Field(
        ..., description="The filename of the reference code"
    ),
    generated_code_filename: str = Field(
        ..., description="The filename of the generated code"
    ),
    recap: str = Field(..., description="The recap of the iteration"),
) -> str:
    try:
        logger.info(f"Uploading recap to s3: {model_tag}, {task_tag}, {eval_tag}")
        logger.info(f"Generated recap: {recap}")

        # read reference code
        with open(os.path.join(current_wd, reference_code_filename), "r") as f:
            reference_code = f.read()
        # read generated code
        with open(os.path.join(current_wd, generated_code_filename), "r") as f:
            generated_code = f.read()
        # read response from file
        with open(f"{current_wd}/kbeval_{eval_tag}_{run_tag}.result.json", "r") as f:
            result = KernelExecResult.model_validate_json(f.read())

        if is_devserver() is False:
            msg = upload_recap_to_s3(
                current_wd,
                model_tag,
                task_tag,
                eval_tag,
                run_tag,
                result.model_dump(),
                reference_code,
                generated_code,
                recap,
            )
        else:
            msg = "no s3 access on devserver"
        return msg
    except Exception as e:
        logger.error(f"Error uploading recap to s3: {e}")
        logger.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    logger.info(f"starting kbEvalMCPServer")
    server.run(transport="stdio")
    logger.info(f"kbEvalMCPServer started")
