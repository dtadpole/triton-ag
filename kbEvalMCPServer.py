from mcp.server.fastmcp import FastMCP
import asyncio
import time
import json
import yaml
import requests
import traceback
from datetime import datetime
import os
from util import is_subfolder
from logger import logger
from kbEvalTest.kbeval import KernelExecResult
import boto3
from pydantic import Field

server = FastMCP("kbEval")

yaml_config = yaml.load(open(os.path.join(os.path.dirname(__file__), "kbEval.yaml"), "r"), Loader=yaml.FullLoader)
server_stats = {}
server_last_refresh_time = 0

def get_server_stats():
    global server_stats, server_last_refresh_time
    time_now = time.time()
    if len(server_stats) == 0 or time_now - server_last_refresh_time > 10:
        for server in yaml_config["kbEvalClient"]["servers"]:
            response = requests.get(f"{server['url']}/stats")
            server_stats[server["url"]] = response.json()
        server_last_refresh_time = time_now
    return server_stats

def pick_server():
    stats = get_server_stats()
    min_avg_load = float("inf")
    min_avg_load_server = None
    for server in stats:
        if stats[server]["pending_requests"] / stats[server]["num_devices"] < min_avg_load:
            min_avg_load = stats[server]["pending_requests"] / stats[server]["num_devices"]
            min_avg_load_server = server
    return min_avg_load_server

def upload_recap_to_s3(current_wd: str,
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
        s3_client.put_object(Bucket="agent-xyz",
                            Key=upload_key,
                            Body=json.dumps(summary, indent=4))
        return f"uploaded to s3://agent-xyz/{upload_key}"
    except Exception as e:
        logger.error(f"Error pushing to s3: {e}")
        logger.error(traceback.format_exc())
        return f"Error pushing to s3: {e}"


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
    reference_code_filename: str = Field(..., description="The filename of the reference code"),
    generated_code_filename: str = Field(..., description="The filename of the generated code"),
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
        server_url = pick_server()

        # send request to remote server
        # logger.info(f"Sending request to remote server {server_url}")
        response = requests.post(f"{server_url}/kb_eval", data=json.dumps({
            "model_tag": model_tag,
            "task_tag": task_tag,
            "eval_tag": eval_tag,
            "time_tag": time_tag,
            "reference_code": reference_code,
            "generated_code": generated_code,
        }), headers={"Content-Type": "application/json"})

        # write response to file
        response_json = json.loads(response.text)
        response_json["tags"] = {
            "model_tag": model_tag,
            "task_tag": task_tag,
            "eval_tag": eval_tag,
            "time_tag": time_tag,
        }
        with open(f"{current_wd}/kbeval_{eval_tag}_{time_tag}.result.json", "w") as f:
            f.write(json.dumps(response_json, indent=4))
        logger.info(f"Response from remote server: {json.dumps(response_json, indent=4)}")

        # upload recap to s3
        upload_recap_to_s3(current_wd,
                             model_tag,
                             task_tag,
                             eval_tag,
                             time_tag,
                             response_json,
                             reference_code,
                             generated_code,
                             recap="")

        return KernelExecResult.model_validate_json(response.text)
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
    reference_code_filename: str = Field(..., description="The filename of the reference code"),
    generated_code_filename: str = Field(..., description="The filename of the generated code"),
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
            
        msg = upload_recap_to_s3(current_wd,
                             model_tag,
                             task_tag,
                             eval_tag,
                             time_tag,
                             result.model_dump(),
                             reference_code,
                             generated_code,
                             recap)
        return msg
    except Exception as e:
        logger.error(f"Error uploading recap to s3: {e}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    logger.info(f"starting kbEvalMCPServer")
    server.run(transport="stdio")
    logger.info(f"kbEvalMCPServer started")
