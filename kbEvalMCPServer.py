import asyncio
import json
import os
import time
import traceback
from datetime import datetime

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
server_stats_and_key = {}
server_last_refresh_time = 0
api_key_mapping = {}

def get_server_stats_and_key():
    global server_stats_and_key, server_last_refresh_time, api_key_mapping
    time_now = time.time()
    if len(server_stats_and_key) == 0 or time_now - server_last_refresh_time > 10:
        for server in yaml_config["kbEvalClient"]["servers"]:
            response = requests.get(f"{server['url']}/stats")
            # read api_key from file if not already in api_key_mapping
            if server["api_key"] not in api_key_mapping:
                # expand ${HOME} to os.path.expanduser("~")
                api_key_filepath = server["api_key"].replace("${HOME}", os.path.expanduser("~"))
                with open(api_key_filepath, "r") as f:
                    api_key_mapping[server["api_key"]] = f.read().strip()
            # return stats and api_key
            server_stats_and_key[server["url"]] = {
                "stats": response.json(),
                "api_key": api_key_mapping[server["api_key"]],
            }
        server_last_refresh_time = time_now
    return server_stats_and_key

def pick_server_and_key():
    stats_and_key = get_server_stats_and_key()
    min_avg_load = float("inf")
    min_avg_load_server = None
    for server in stats_and_key:
        if stats_and_key[server]["stats"]["pending_requests"] / stats_and_key[server]["stats"]["num_devices"] < min_avg_load:
            min_avg_load = stats_and_key[server]["stats"]["pending_requests"] / stats_and_key[server]["stats"]["num_devices"]
            min_avg_load_server = server
    return min_avg_load_server, server_stats_and_key[min_avg_load_server]["api_key"]


def upload_recap_to_s3(
    current_wd: str,
    model_tag: str,
    task_tag: str,
    eval_tag: str,
    time_tag: str,
    result: dict,
    reference_code: str,
    generated_code: str,
    recap: str = "",
):
    summary = {
        "model_tag": model_tag,
        "task_tag": task_tag,
        "eval_tag": eval_tag,
        "time_tag": time_tag,
        "summary": recap,
        "result": result,
        "reference_code": reference_code,
        "generated_code": generated_code,
    }

    try:
        with open(f"{current_wd}/kbeval_{eval_tag}_{time_tag}.summary.json", "w") as f:
            f.write(json.dumps(summary, indent=4))
    except Exception as e:
        logger.error(f"Error writing recap to file: {e}")
        logger.error(traceback.format_exc())

    upload_key = f"kbeval/{model_tag}/{task_tag}/{eval_tag}_{time_tag}.summary.json"

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
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    time_tag: str = Field(..., description="The tag of the time"),
    reference_code_filename: str = Field(
        ..., description="The filename of the reference code"
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
        # connect to remote server
        server_url, api_key = pick_server_and_key()

        # send request to remote server
        # logger.info(f"Sending request to remote server {server_url}")
        response = requests.post(f"{server_url}/kb_eval_ref", data=json.dumps({
            "model_tag": model_tag,
            "task_tag": task_tag,
            "time_tag": time_tag,
            "reference_code": reference_code,
        }), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})

        # write response to file
        response_json = json.loads(response.text)
        if "metadata" not in response_json:
            response_json["metadata"] = {}
        response_json["metadata"] = response_json["metadata"] | {
            "model_tag": model_tag,
            "task_tag": task_tag,
            "time_tag": time_tag,
        }
        with open(f"{current_wd}/kbeval_reference_{time_tag}.result.json", "w") as f:
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
                time_tag,
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
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    time_tag: str = Field(..., description="The tag of the time"),
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

        # connect to remote server
        server_url, api_key = pick_server_and_key()

        # send request to remote server
        # logger.info(f"Sending request to remote server {server_url}")
        response = requests.post(f"{server_url}/kb_eval", data=json.dumps({
            "model_tag": model_tag,
            "task_tag": task_tag,
            "eval_tag": eval_tag,
            "time_tag": time_tag,
            "reference_code": reference_code,
            "generated_code": generated_code,
        }), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})

        # write response to file
        response_json = json.loads(response.text)
        if "metadata" not in response_json:
            response_json["metadata"] = {}
        response_json["metadata"] = response_json["metadata"] | {
            "model_tag": model_tag,
            "task_tag": task_tag,
            "eval_tag": eval_tag,
            "time_tag": time_tag,
        }
        result_filename = f"{current_wd}/kbeval_{eval_tag}_{time_tag}.result.json"
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
                time_tag,
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
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    time_tag: str = Field(..., description="The tag of the time"),
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

        # connect to remote server
        server_url, api_key = pick_server_and_key()

        # send request to remote server
        # logger.info(f"Sending request to remote server {server_url}")
        response = requests.post(f"{server_url}/kb_eval", data=json.dumps({
            "model_tag": model_tag,
            "task_tag": task_tag,
            "eval_tag": eval_tag,
            "time_tag": time_tag,
            "reference_code": reference_code,
            "generated_code": generated_code,
        }), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})

        # write response to file
        response_json = json.loads(response.text)
        if "metadata" not in response_json:
            response_json["metadata"] = {}
        response_json["metadata"] = response_json["metadata"] | {
            "model_tag": model_tag,
            "task_tag": task_tag,
            "eval_tag": eval_tag,
            "time_tag": time_tag,
        }
        with open(f"{current_wd}/kbeval_{eval_tag}_{time_tag}.result.json", "w") as f:
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
                time_tag,
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
    model_tag: str = Field(..., description="The tag of the model"),
    task_tag: str = Field(..., description="The tag of the task"),
    time_tag: str = Field(..., description="The tag of the time"),
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
        with open(f"{current_wd}/kbeval_{eval_tag}_{time_tag}.result.json", "r") as f:
            result = KernelExecResult.model_validate_json(f.read())

        if is_devserver() is False:
            msg = upload_recap_to_s3(
                current_wd,
                model_tag,
                task_tag,
                eval_tag,
                time_tag,
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
