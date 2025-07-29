import argparse
import asyncio
import signal
import concurrent.futures
import json
import os
import sys
import time
import traceback
import json
import argparse
import asyncio
import yaml
import uuid
from fastapi import FastAPI, Body, HTTPException, Header, Depends, Request
from kbEvalTest.kbeval import KernelExecResult
from logger import logger
from pydantic import BaseModel, Field

KB_EVAL_TOKEN = None

CURR_ERROR_COUNT = 0
MAX_ERROR_COUNT = 10
START_TIME = time.time()
MAX_RUN_TIME = 2 * 3600 # restart periods in seconds

KB_EVAL_DIR = os.path.join(os.path.expanduser("~"), ".kbeval")

# Create app
app = FastAPI()

request_counter = 0
request_counter_lock = asyncio.Lock()

DEVICES = []


# Authentication function
def verify_token(authorization: str = Header(None)):
    """Simple token verification"""
    expected_token = KB_EVAL_TOKEN
    if not expected_token:
        raise HTTPException(status_code=500, detail="Server authentication not configured")

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format. Use 'Bearer <token>'")

    token = authorization[7:].strip()  # Remove "Bearer " prefix, and strip whitespace
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token")

    return True


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

async def check_return_code(process: asyncio.subprocess.Process):
    while True:
        try:
            return_code = await process.wait(timeout=1)
            if return_code is not None:
                logger.info(f"Child process [{process.pid}] completed with return code: {return_code}")
                return
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Error checking return code of child process [{process.pid}]: {e}")
        finally:
            await asyncio.sleep(1)

async def check_disconnect_and_kill_child_process(request: Request, process: asyncio.subprocess.Process):
    while True:
        try:
            if await request.is_disconnected():
                logger.error(f"Client disconnected, terminating child process [{process.pid}]")
                # process.terminate()
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    logger.error(f"Child process [{process.pid}] termination timed out, killing it")
                    # process.kill()
                    os.killpg(process.pid, signal.SIGKILL)
                    # return
                except Exception as e:
                    logger.error(f"Error killing child process [{process.pid}]: {e}")
                # return
            elif process.returncode is not None:
                logger.info(f"Child process [{process.pid}] completed with return code: {process.returncode}")
                return
        except Exception as e:
            logger.error(f"Error checking disconnect status: {e}")
            return
        finally:
            await asyncio.sleep(1)


async def get_pending_task_count():
    all_tasks = asyncio.all_tasks()
    return len(set(all_tasks))


@app.get("/stats")
async def stats():
    global request_counter
    return {
        "num_devices": len(DEVICES),
        "pending_requests": request_counter,
    }


@app.post("/kb_eval_ref")
async def kb_eval_ref(
    run_tag: str = Body(...),
    model_tag: str = Body(...),
    task_tag: str = Body(...),
    reference_code: str = Body(...),
    authenticated: bool = Depends(verify_token)
) -> KernelExecResult:
    global request_counter, request_counter_lock, DEVICES

    # logger.info(f"kb_eval_ref: {run_tag}, {model_tag}, {task_tag}, {reference_code}")

    try:
        async with request_counter_lock:
            request_counter += 1

        # temp_dir is {HOME}/.kbeval/{model_tag}/{task_tag}/{eval_tag}/{time_tag}
        temp_dir = os.path.join(KB_EVAL_DIR, run_tag, model_tag, task_tag)
        os.makedirs(temp_dir, exist_ok=True)

        reference_file_path = os.path.join(temp_dir, f"reference_code.py")
        with open(reference_file_path, "w") as f:
            f.write(reference_code)

        # pre-compile the reference code
        command = f"python kbEvalCli.py --wd {temp_dir} --run_tag {run_tag} --model_tag {model_tag} --task_tag {task_tag} --reference_code {reference_file_path} --measure_reference --device-list {','.join([str(device) for device in DEVICES])}"
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
            read_stream(
                process.stderr, "reference", is_error=True
            )  # seems taking warning message as error message
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

    except Exception as e:
        global CURR_ERROR_COUNT
        CURR_ERROR_COUNT += 1
        result = KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={
                "processing_error": str(e),
            },
            runtime=-1.0,
        )
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
    request: Request, # injected by fastapi
    run_tag: str = Body(...),
    model_tag: str = Body(...),
    task_tag: str = Body(...),
    eval_tag: str = Body(...),
    reference_code: str = Body(...),
    generated_code: str = Body(...),
    authenticated: bool = Depends(verify_token)
) -> KernelExecResult:
    global request_counter, request_counter_lock, DEVICES

    try:
        async with request_counter_lock:
            request_counter += 1

        # temp_dir is {HOME}/.kbeval/{run_tag}/{model_tag}/{task_tag}/{eval_tag}
        temp_dir = os.path.join(KB_EVAL_DIR, run_tag, model_tag, task_tag, eval_tag)
        os.makedirs(temp_dir, exist_ok=True)

        reference_file_path = os.path.join(temp_dir, f"reference_code.py")
        with open(reference_file_path, "w") as f:
            f.write(reference_code)

        generated_file_path = os.path.join(temp_dir, f"generated_code.py")
        with open(generated_file_path, "w") as f:
            f.write(generated_code)

        # pre-compile the generated code
        command = f"python kbEvalCli.py --wd {temp_dir} --run_tag {run_tag} --model_tag {model_tag} --task_tag {task_tag} --eval_tag {eval_tag} --reference_code {reference_file_path} --generated_code {generated_file_path} --device-list {','.join([str(device) for device in DEVICES])}"
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
        check_return_code_task = asyncio.create_task(check_return_code(process))
        check_disconnect_task = asyncio.create_task(check_disconnect_and_kill_child_process(request, process))

        # Wait for all output to be processed
        await asyncio.gather(stdout_task, stderr_task, check_return_code_task, check_disconnect_task, return_exceptions=True)
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

    except Exception as e:
        global CURR_ERROR_COUNT
        CURR_ERROR_COUNT += 1
        result = KernelExecResult(
            compiled=False,
            correctness=False,
            runtime=-1.0,
        )
        return result

    finally:
        async with request_counter_lock:
            request_counter -= 1
            if request_counter < 0:
                logger.error(
                    f"Request counter is negative: {request_counter}, resetting to 0"
                )
                request_counter = 0


async def _check_total_error_count():
    global CURR_ERROR_COUNT, MAX_ERROR_COUNT, START_TIME

    print_interval = 10
    check_interval = 3 # seconds
    counter = 0
    while True:
        try:
            counter += 1
            # check if elapsed time is greater than MAX_RUN_TIME
            CURR_TIME = time.time()
            ELAPSED_TIME = CURR_TIME - START_TIME
            if ELAPSED_TIME > MAX_RUN_TIME:
                # add an star emoji
                logger.error(f"⭐ Elapsed time is greater than {MAX_RUN_TIME/3600:.2f} hours, exiting... [parent process will restart]")
                loop = asyncio.get_event_loop()
                loop.stop()
                exit(1)
            if CURR_ERROR_COUNT > MAX_ERROR_COUNT:
                logger.error(f"❌ Total error count is greater than {MAX_ERROR_COUNT}, exiting")
                loop = asyncio.get_event_loop()
                loop.stop()
                exit(1)
            elif CURR_ERROR_COUNT > 0 and counter % print_interval == 0:
                logger.warning(f"⚠️ Total error count is {CURR_ERROR_COUNT}, continuing...")
        finally:
            await asyncio.sleep(check_interval)


async def main(args):

    #read kbEval.yaml
    with open("kbEval.yaml", "r") as f:
        kbEval_config = yaml.load(f, Loader=yaml.FullLoader)
        # use file emoji
        logger.info(f"📁 [kbEvalServer] Config file kbEval.yaml loaded")

    #########################################################
    # get hostname and host, port from kbEval.yaml
    import socket

    hostname = socket.gethostname()
    # if hostname is not in kbEval_config["kbEvalRemoteServer"], use "one"
    if hostname not in kbEval_config["kbEvalRemoteServer"]:
        logger.warning(
            f"Hostname {hostname} not found in kbEval.yaml, using 'one' as default"
        )
        hostname = "one"

    global DEVICES

    if args.local_host:
        host = "0.0.0.0"
        port = args.port
        DEVICES = [args.device]
    else:
        host = kbEval_config["kbEvalRemoteServer"][hostname]["host"]
        port = kbEval_config["kbEvalRemoteServer"][hostname]["port"]
        DEVICES = [int(d) for d in kbEval_config["kbEvalRemoteServer"][hostname]["devices"]]

    logger.info(f"Running on [{hostname}:{port}] with devices: {DEVICES}")

    #########################################################
    # get api_key from kbEval_config["kbEvalRemoteServer"]["common"]["api_key"]
    if "common" not in kbEval_config["kbEvalRemoteServer"]:
        logger.error("[kbEvalRemoteServer] [common] not found in kbEval.yaml")
        exit(1)
    if "api_key" not in kbEval_config["kbEvalRemoteServer"]["common"]:
        logger.error(f"[kbEvalRemoteServer] [api_key] not found in kbEval.yaml [{kbEval_config['kbEvalRemoteServer']['common']}]")
        exit(1)
    api_key_filepath = kbEval_config["kbEvalRemoteServer"]["common"]["api_key"]
    # read file from api_key, replace ${HOME} with os.path.expanduser("~") in api_key_filepath
    api_key_filepath = api_key_filepath.replace("${HOME}", os.path.expanduser("~"))
    if not os.path.exists(api_key_filepath):
        # create the file, and write a random string to it
        with open(api_key_filepath, "w") as f:
            api_key = str(uuid.uuid4())
            f.write(api_key)
            # add emoji to beginning and end of the string
            logger.info(f"🔑 [kbEvalRemoteServer] API key [{api_key}] created and saved to [{api_key_filepath}]")
    # now read in the api_key
    with open(api_key_filepath, "r") as f:
        global KB_EVAL_TOKEN
        KB_EVAL_TOKEN = f.read().strip()
        logger.info(f"[kbEvalRemoteServer] KB_EVAL_TOKEN loaded from [{api_key_filepath}]")
    #########################################################

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, workers=args.workers))

    # run server and check total error count in parallel
    # need running event loop to run the tasks
    tasks = [
        asyncio.create_task(_check_total_error_count()),
        asyncio.create_task(server.serve()),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_host", action="store_true")
    parser.add_argument("--port",  type=int, default=8088)
    parser.add_argument("--workers",  type=int, default=32)
    parser.add_argument("--device",  type=str, default='4')
    args = parser.parse_args()
    asyncio.run(main(args))
