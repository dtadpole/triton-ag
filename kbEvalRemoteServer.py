import argparse
import asyncio
import concurrent.futures
import json
import os
import sys
import traceback
from datetime import datetime

import boto3
import torch
import yaml
from fastapi import Body, FastAPI

from kbEvalTest.kbeval import (
    get_timing_stats,
    graceful_eval_cleanup,
    KernelExecResult,
    load_custom_model,
    load_original_model_and_inputs,
    run_and_check_correctness,
    set_seed,
    time_execution_with_cuda_event,
)
from logger import logger
from pydantic import BaseModel, Field


async def get_with_timeout(queue, timeout):
    try:
        item = await asyncio.wait_for(queue.get(), timeout)
        return item
    except asyncio.TimeoutError:
        return None  # Or raise an exception, or handle it as needed


async def read_stream(stream, prefix: str, is_error: bool = False):
    """Read from a stream and print each line with a prefix."""
    while True:
        line = await stream.readline()
        if not line:
            break
        # Decode bytes to string and strip newline
        output = line.decode("utf-8").rstrip()
        if is_error:
            logger.error(f"[{prefix}] {output}")
        else:
            logger.info(f"[{prefix}] {output}")


KB_EVAL_DIR = os.path.join(os.path.expanduser("~"), ".kbeval")

# Create app
app = FastAPI()

request_counter = 0
request_counter_lock = asyncio.Lock()

devices = []


async def get_pending_task_count():
    all_tasks = asyncio.all_tasks()
    return len(set(all_tasks))


@app.get("/stats")
async def stats():
    global request_counter
    return {
        "num_devices": len(devices),
        "pending_requests": request_counter,
    }


@app.post("/kb_eval_ref")
async def kb_eval_ref(
    model_tag: str = Body(...),
    task_tag: str = Body(...),
    time_tag: str = Body(...),
    reference_code: str = Body(...),
) -> KernelExecResult:
    global request_counter, request_counter_lock, devices

    logger.info(f"kb_eval_ref: {model_tag}, {task_tag}, {time_tag}, {reference_code}")

    try:
        async with request_counter_lock:
            request_counter += 1

        # temp_dir is {HOME}/.kbeval/{model_tag}/{task_tag}/{eval_tag}/{time_tag}
        temp_dir = os.path.join(KB_EVAL_DIR, model_tag, task_tag, time_tag)
        os.makedirs(temp_dir, exist_ok=True)

        reference_file_path = os.path.join(temp_dir, f"reference_code.py")
        with open(reference_file_path, "w") as f:
            f.write(reference_code)

        # pre-compile the reference code
        command = f"python kbEvalCli.py --wd {temp_dir} --model_tag {model_tag} --task_tag {task_tag} --time_tag {time_tag} --reference_code {reference_file_path} --measure_reference --device-list {','.join([str(device) for device in devices])}"
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        logger.info(f"[KB Eval] [reference] START ====================")
        logger.info(f"[KB Eval] [reference] command: {command}")

        # Create tasks to read stdout and stderr concurrently
        stdout_task = asyncio.create_task(
            read_stream(process.stdout, "reference", is_error=False)
        )
        stderr_task = asyncio.create_task(
            read_stream(process.stderr, "reference", is_error=True)
        )

        # Wait for the process to complete
        return_code = await process.wait()

        # Wait for all output to be processed
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        if process.returncode != 0:
            logger.error(f"[KB Eval] [reference] return code: {process.returncode}")
        else:
            logger.info(f"[KB Eval] [reference] return code: {process.returncode}")

        logger.info(f"[KB Eval] [reference] END ====================")

        # read the result from {temp_dir}/kbeval_{eval_tag}.json
        result_json_path = os.path.join(temp_dir, f"reference_kbeval.json")
        with open(result_json_path, "r") as f:
            result_text = f.read()

        result = KernelExecResult.model_validate_json(result_text)

        return result
    finally:
        async with request_counter_lock:
            request_counter -= 1
            if request_counter < 0:
                logger.error(
                    f"Request counter is negative: {request_counter}, resetting to 0"
                )
                request_counter = 0


@app.post("/kb_eval")
async def kb_eval(
    model_tag: str = Body(...),
    task_tag: str = Body(...),
    eval_tag: str = Body(...),
    time_tag: str = Body(...),
    reference_code: str = Body(...),
    generated_code: str = Body(...),
) -> KernelExecResult:
    global request_counter, request_counter_lock, devices

    try:
        async with request_counter_lock:
            request_counter += 1

        # temp_dir is {HOME}/.kbeval/{model_tag}/{task_tag}/{eval_tag}/{time_tag}
        temp_dir = os.path.join(KB_EVAL_DIR, model_tag, task_tag, time_tag, eval_tag)
        os.makedirs(temp_dir, exist_ok=True)

        reference_file_path = os.path.join(temp_dir, f"reference_code.py")
        with open(reference_file_path, "w") as f:
            f.write(reference_code)

        generated_file_path = os.path.join(temp_dir, f"generated_code.py")
        with open(generated_file_path, "w") as f:
            f.write(generated_code)

        # parser.add_argument("--wd", type=str, default="./kbEvalTest")
        # parser.add_argument("--model_tag", type=str, default="model_tag")
        # parser.add_argument("--task_tag", type=str, default="task_tag")
        # parser.add_argument("--eval_tag", type=str, default="eval_tag")
        # parser.add_argument("--time_tag", type=str, default="time_tag")
        # parser.add_argument("--reference_code", type=str, default="elemAddRef.py")
        # parser.add_argument("--generated_code", type=str, default="elemAddCuda.py")

        # pre-compile the generated code
        command = f"python kbEvalCli.py --wd {temp_dir} --model_tag {model_tag} --task_tag {task_tag} --eval_tag {eval_tag} --time_tag {time_tag} --reference_code {reference_file_path} --generated_code {generated_file_path} --device-list {','.join([str(device) for device in devices])}"
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        logger.info(f"[KB Eval] [{eval_tag}] START ====================")
        logger.info(f"[KB Eval] [{eval_tag}] command: {command}")

        # Create tasks to read stdout and stderr concurrently
        stdout_task = asyncio.create_task(
            read_stream(process.stdout, eval_tag, is_error=False)
        )
        stderr_task = asyncio.create_task(
            read_stream(process.stderr, eval_tag, is_error=True)
        )

        # Wait for the process to complete
        return_code = await process.wait()

        # Wait for all output to be processed
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        if process.returncode != 0:
            logger.error(f"[KB Eval] [{eval_tag}] return code: {process.returncode}")
        else:
            logger.info(f"[KB Eval] [{eval_tag}] return code: {process.returncode}")

        logger.info(f"[KB Eval] [{eval_tag}] END ====================")

        # read the result from {temp_dir}/kbeval_{eval_tag}.json
        result_json_path = os.path.join(temp_dir, f"{eval_tag}_kbeval.json")
        with open(result_json_path, "r") as f:
            result_text = f.read()

        result = KernelExecResult.model_validate_json(result_text)

        return result
    finally:
        async with request_counter_lock:
            request_counter -= 1
            if request_counter < 0:
                logger.error(
                    f"Request counter is negative: {request_counter}, resetting to 0"
                )
                request_counter = 0


@app.post("/kb_eval_triton")
async def kb_eval_triton(
    model_tag: str = Body(...),
    task_tag: str = Body(...),
    eval_tag: str = Body(...),
    time_tag: str = Body(...),
    reference_code: str = Body(...),
    generated_code: str = Body(...),
) -> KernelExecResult:
    global request_counter, request_counter_lock, devices

    try:
        async with request_counter_lock:
            request_counter += 1

        # temp_dir is {HOME}/.kbeval/{model_tag}/{task_tag}/{eval_tag}/{time_tag}
        temp_dir = os.path.join(KB_EVAL_DIR, model_tag, task_tag, time_tag, eval_tag)
        os.makedirs(temp_dir, exist_ok=True)

        reference_file_path = os.path.join(temp_dir, f"reference_code.py")
        with open(reference_file_path, "w") as f:
            f.write(reference_code)

        generated_file_path = os.path.join(temp_dir, f"generated_code.py")
        with open(generated_file_path, "w") as f:
            f.write(generated_code)

        # parser.add_argument("--wd", type=str, default="./kbEvalTest")
        # parser.add_argument("--model_tag", type=str, default="model_tag")
        # parser.add_argument("--task_tag", type=str, default="task_tag")
        # parser.add_argument("--eval_tag", type=str, default="eval_tag")
        # parser.add_argument("--time_tag", type=str, default="time_tag")
        # parser.add_argument("--reference_code", type=str, default="elemAddRef.py")
        # parser.add_argument("--generated_code", type=str, default="elemAddCuda.py")

        # pre-compile the generated code
        command = f"python kbEvalCli.py --wd {temp_dir} --model_tag {model_tag} --task_tag {task_tag} --eval_tag {eval_tag} --time_tag {time_tag} --reference_code {reference_file_path} --generated_code {generated_file_path} --device-list {','.join([str(device) for device in devices])}"
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        logger.info(f"[KB Eval] [{eval_tag}] START ====================")
        logger.info(f"[KB Eval] [{eval_tag}] command: {command}")

        # Create tasks to read stdout and stderr concurrently
        stdout_task = asyncio.create_task(
            read_stream(process.stdout, eval_tag, is_error=False)
        )
        stderr_task = asyncio.create_task(
            read_stream(process.stderr, eval_tag, is_error=True)
        )

        # Wait for the process to complete
        return_code = await process.wait()

        # Wait for all output to be processed
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        if process.returncode != 0:
            logger.error(f"[KB Eval] [{eval_tag}] return code: {process.returncode}")
        else:
            logger.info(f"[KB Eval] [{eval_tag}] return code: {process.returncode}")

        logger.info(f"[KB Eval] [{eval_tag}] END ====================")

        # read the result from {temp_dir}/kbeval_{eval_tag}.json
        result_json_path = os.path.join(temp_dir, f"{eval_tag}_kbeval.json")
        with open(result_json_path, "r") as f:
            result_text = f.read()

        result = KernelExecResult.model_validate_json(result_text)

        return result
    finally:
        async with request_counter_lock:
            request_counter -= 1
            if request_counter < 0:
                logger.error(
                    f"Request counter is negative: {request_counter}, resetting to 0"
                )
                request_counter = 0


if __name__ == "__main__":
    # read kbEval.yaml
    with open("kbEval.yaml", "r") as f:
        kbEval_config = yaml.load(f, Loader=yaml.FullLoader)

    import socket

    hostname = socket.gethostname()
    # if hostname is not in kbEval_config["kbEvalRemoteServer"], use "one"
    if hostname not in kbEval_config["kbEvalRemoteServer"]:
        logger.warning(
            f"Hostname {hostname} not found in kbEval.yaml, using 'one' as default"
        )
        hostname = "one"
    # if hostname not in kbEval_config["kbEvalRemoteServer"]:
    #     logger.error(f"Hostname {hostname} not found in kbEval.yaml")
    #     exit(1)

    host = kbEval_config["kbEvalRemoteServer"][hostname]["host"]
    port = kbEval_config["kbEvalRemoteServer"][hostname]["port"]

    devices = [int(d) for d in kbEval_config["kbEvalRemoteServer"][hostname]["devices"]]
    logger.info(f"Running on [{hostname}:{port}] with devices: {devices}")

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))
    asyncio.run(server.serve())
