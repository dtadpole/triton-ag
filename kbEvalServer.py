import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import signal
import sys
import time
import traceback
import uuid
from datetime import datetime

import uvicorn
import wandb
import yaml
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from gzipMiddleware import GunzipRequestMiddleware
from kbEvalUtil import KernelExecResult, KbEvalResult, on_process_timeout
from logger import logger
from pydantic import BaseModel, Field
from util import KB_EVAL_DIR

KB_EVAL_TOKEN = None

TOTAL_REQUEST_COUNTER = 0
TOTAL_ERROR_COUNTER = 0
MAX_ERROR_COUNT = 50
START_TIME = time.time()
MAX_RUN_TIME = 4 * 3600  # restart periods in seconds
COMPILE_CACHE = False
CHECK_GET_INPUTS = True

# Create app
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
app.add_middleware(GunzipRequestMiddleware)  # now all routes accept gzip bodies

parallel_request_counter = 0
parallel_request_counter_lock = asyncio.Lock()

DEVICES = []

MAX_TIMEOUT_SECONDS = 270  # 4.5 minutes

# Cache hit/miss tracking
CACHE_HIT_THRESHOLD = 20  # seconds - if command completes within this, it's a cache hit
CACHE_MISS_THRESHOLD = (
    30  # seconds - if command takes longer than this, it's a cache miss
)
TOTAL_CACHE_HITS = 0
TOTAL_CACHE_MISSES = 0
TOTAL_CACHE_UNCLEAR = 0  # requests between 20-30 seconds


# Authentication function
def verify_token(authorization: str = Header(None)):
    """Simple token verification"""
    expected_token = KB_EVAL_TOKEN
    if not expected_token:
        raise HTTPException(
            status_code=500, detail="Server authentication not configured"
        )

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid authorization format. Use 'Bearer <token>'"
        )

    token = authorization[7:].strip()  # Remove "Bearer " prefix, and strip whitespace
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token")

    return True


# W&B loggers
wandb_loggers = {}  # {prefix_tag: wandb.Run}


def _setup_wandb_logging(prefix_tag: str = "auto", model_tag: str = "local_qwen3-32b"):
    """Setup logging and tracking"""
    if prefix_tag.startswith("auto"):
        return None

    key = f"{prefix_tag}_{model_tag}"
    if key in wandb_loggers:
        return wandb_loggers[key]
    # Group by wandb log in the same way as in the shared folder
    model_tag = model_tag.replace("/", "_")
    wandb_run = wandb.init(
        # entity="code-gen",
        project=f"kb_eval",
        id=f"{prefix_tag}_{model_tag}",
        name=f"{prefix_tag}_{model_tag}_{datetime.now().strftime('%m%d')}",
        resume="allow",
        reinit="create_new",
        settings=wandb.Settings(init_timeout=15),
    )
    wandb_loggers[key] = wandb_run
    logger.info(f"📊 W&B logging enabled for [{key}]")
    return wandb_run


async def get_with_timeout(queue, timeout):
    try:
        item = await asyncio.wait_for(queue.get(), timeout)
        return item
    except asyncio.TimeoutError:
        return None  # Or raise an exception, or handle it as needed


async def read_stream(stream, work_dir: str, eval_tag: str, is_error: bool = False):
    """Read from a stream and print each line with a prefix."""
    while True:
        line = await stream.readline()
        if not line:
            break
        # Decode bytes to string and strip newline
        output = line.decode("utf-8").rstrip()
        if is_error:
            # logger.error(f"[{prefix}] {output}")
            # append to {work_dir}/{prefix).stderr
            with open(os.path.join(work_dir, f"{eval_tag}.stderr"), "a") as f:
                f.write(output + "\n")
        else:
            # logger.info(f"[{prefix}] {output}")
            # append to {work_dir}/{prefix}.stdout
            with open(os.path.join(work_dir, f"{eval_tag}.stdout"), "a") as f:
                f.write(output + "\n")


async def check_return_code(process: asyncio.subprocess.Process):
    start_time = time.time()
    while True:
        try:
            return_code = await process.wait()
            if return_code is not None:
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Child process [{process.pid}] completed with return code: {return_code} in [{elapsed_time:.2f}s]"
                )
                return
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(
                f"Error checking return code of child process [{process.pid}]: {e}"
            )
        finally:
            await asyncio.sleep(1)


async def check_disconnect_and_kill_child_process(
    request: Request, process: asyncio.subprocess.Process
):
    start_time = time.time()
    while True:
        try:
            if process.returncode is not None:
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Child process [{process.pid}] completed with return code: {process.returncode} in [{elapsed_time:.2f}s]"
                )
                return
            elif request._is_disconnected or await request.is_disconnected():
                logger.error(
                    f"Client disconnected, terminating child process [{process.pid}]"
                )
                process.terminate()
                # os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    logger.error(
                        f"Child process [{process.pid}] termination timed out, killing it"
                    )
                    process.kill()
                    # os.killpg(process.pid, signal.SIGKILL)
                    # return
                except Exception as e:
                    logger.error(
                        f"Error killing child process [{process.pid}]: [{type(e)}]: {e}"
                    )
                # return
        except ProcessLookupError:
            logger.error(f"Child process [{process.pid}] not found, exiting")
            return
        except Exception as e:
            logger.error(f"Error checking disconnect status: [{type(e)}]: {e}")
        finally:
            await asyncio.sleep(1)


async def get_pending_task_count():
    all_tasks = asyncio.all_tasks()
    return len(set(all_tasks))


@app.middleware("http")
async def log_preheader_crashes(request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.error("❌ Crashed before sending headers")
        raise


@app.get("/stats")
async def stats():
    global parallel_request_counter
    return {
        "num_devices": len(DEVICES),
        "pending_requests": parallel_request_counter,
    }


async def _run_eval(
    request: Request,
    temp_dir: str,
    eval_tag: str,
    command: str,
) -> KernelExecResult:
    """Run a single kbEvalCli subprocess and return its KernelExecResult."""
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )

    logger.info(f"[KB Eval] [{eval_tag}] START ====================")
    logger.info(f"[KB Eval] [{eval_tag}] command: {command}")

    stdout_task = asyncio.create_task(
        read_stream(process.stdout, temp_dir, eval_tag, is_error=False)
    )
    stderr_task = asyncio.create_task(
        read_stream(process.stderr, temp_dir, eval_tag, is_error=True)
    )
    check_return_code_task = asyncio.create_task(check_return_code(process))
    check_disconnect_task = asyncio.create_task(
        check_disconnect_and_kill_child_process(request, process)
    )

    await asyncio.gather(
        stdout_task,
        stderr_task,
        check_return_code_task,
        check_disconnect_task,
        return_exceptions=True,
    )
    if process.returncode != 0:
        logger.error(f"[KB Eval] [{eval_tag}] return code: {process.returncode}")
    else:
        logger.info(f"[KB Eval] [{eval_tag}] return code: {process.returncode}")

    result_json_path = os.path.join(temp_dir, f"{eval_tag}_kbeval.json")
    if not os.path.exists(result_json_path):
        raise FileNotFoundError(
            f"[KB Eval] [{eval_tag}] result file missing: {result_json_path}"
        )

    with open(result_json_path, "r") as f:
        result_json = json.load(f)
        logger.info(
            f"[KB Eval] [{eval_tag}] result: {json.dumps(result_json, indent=4)}"
        )

    logger.info(f"[KB Eval] [{eval_tag}] END ====================")
    return KernelExecResult.model_validate(result_json)


@app.post("/kb_eval")
async def kb_eval(
    request: Request,
    run_tag: str = Body(...),
    model_tag: str = Body(...),
    task_tag: str = Body(...),
    reference_code: str = Body(...),
    generated_code: str = Body(...),
    eval_tag: str = Body(default="eval"),
    code_type: str = Body(default="cuda"),
    authenticated: bool = Depends(verify_token),
) -> KbEvalResult:
    global TOTAL_REQUEST_COUNTER, TOTAL_ERROR_COUNTER, parallel_request_counter, parallel_request_counter_lock, DEVICES

    wandb_run = None
    start_time = time.time()

    try:
        async with parallel_request_counter_lock:
            parallel_request_counter += 1
            TOTAL_REQUEST_COUNTER += 1

        prefix_tag = re.sub(r"_\d*_\d*$", "", run_tag)
        wandb_run = _setup_wandb_logging(prefix_tag, model_tag)

        temp_dir = os.path.join(KB_EVAL_DIR, run_tag, model_tag, task_tag, eval_tag)
        os.makedirs(temp_dir, exist_ok=True)

        reference_file_path = os.path.join(temp_dir, "reference_code.py")
        with open(reference_file_path, "w") as f:
            f.write(reference_code)

        generated_file_path = os.path.join(temp_dir, "generated_code.py")
        with open(generated_file_path, "w") as f:
            f.write(generated_code)

        device_list = ",".join(str(d) for d in DEVICES)

        if COMPILE_CACHE:
            cache_tag = "--use_cuda_cache"
        else:
            cache_tag = ""
        if not CHECK_GET_INPUTS:
            check_get_inputs_tag = "--not_check_get_inputs"
        else:
            check_get_inputs_tag = ""

        # --- Phase 1: Reference benchmark ---
        ref_eval_tag = f"{eval_tag}_ref"
        ref_command = (
            f"timeout --foreground --signal=SIGTERM --kill-after=5s {MAX_TIMEOUT_SECONDS}s "
            f"python kbEvalCli.py --wd {temp_dir} --run_tag {run_tag} --model_tag {model_tag} "
            f"--task_tag {task_tag} --eval_tag {ref_eval_tag} --reference_code reference_code.py "
            f"--measure_reference --device-list {device_list} --code_type pytorch --quiet"
        )

        ref_start = time.time()
        ref_result = await _run_eval(request, temp_dir, ref_eval_tag, ref_command)
        ref_elapsed = time.time() - ref_start

        # --- Phase 2: Generated code evaluation ---
        gen_eval_tag = f"{eval_tag}_gen"
        gen_command = (
            f"timeout --foreground --signal=SIGTERM --kill-after=5s {MAX_TIMEOUT_SECONDS}s "
            f"python kbEvalCli.py --wd {temp_dir} --run_tag {run_tag} --model_tag {model_tag} "
            f"--task_tag {task_tag} --eval_tag {gen_eval_tag} --reference_code reference_code.py "
            f"--generated_code generated_code.py --device-list {device_list} "
            f"--code_type {code_type} --quiet {cache_tag} {check_get_inputs_tag}"
        )

        gen_start = time.time()
        gen_result = await _run_eval(request, temp_dir, gen_eval_tag, gen_command)
        gen_elapsed = time.time() - gen_start

        # --- Compute speedup ---
        total_elapsed = time.time() - start_time
        speedup = -1.0
        if (
            ref_result.runtime > 0
            and gen_result.runtime > 0
            and ref_result.correctness
            and gen_result.correctness
        ):
            speedup = ref_result.runtime / gen_result.runtime

        result = KbEvalResult(
            ref_compiled=ref_result.compiled,
            ref_correctness=ref_result.correctness,
            ref_runtime=ref_result.runtime,
            ref_elapsed_time=ref_elapsed,
            gen_compiled=gen_result.compiled,
            gen_correctness=gen_result.correctness,
            gen_runtime=gen_result.runtime,
            gen_elapsed_time=gen_elapsed,
            speedup=speedup,
            total_elapsed_time=total_elapsed,
            ref_metadata=ref_result.metadata,
            gen_metadata=gen_result.metadata,
            ref_runtime_stats=ref_result.runtime_stats,
            gen_runtime_stats=gen_result.runtime_stats,
        )

        # Cache hit/miss tracking
        global TOTAL_CACHE_HITS, TOTAL_CACHE_MISSES, TOTAL_CACHE_UNCLEAR
        cache_status = "unclear"
        if gen_elapsed <= CACHE_HIT_THRESHOLD:
            TOTAL_CACHE_HITS += 1
            cache_status = "hit"
        elif gen_elapsed >= CACHE_MISS_THRESHOLD:
            TOTAL_CACHE_MISSES += 1
            cache_status = "miss"
        else:
            TOTAL_CACHE_UNCLEAR += 1

        total_cache_requests = TOTAL_CACHE_HITS + TOTAL_CACHE_MISSES + TOTAL_CACHE_UNCLEAR
        cache_hit_rate = (TOTAL_CACHE_HITS / total_cache_requests) if total_cache_requests > 0 else 0

        metrics = {
            "health/completion": 1,
            "health/parallel_requests": parallel_request_counter,
            "health/error_counter": TOTAL_ERROR_COUNTER,
            "health/request_counter": TOTAL_REQUEST_COUNTER,
            "metrics/ref_compiled": 1 if ref_result.compiled else 0,
            "metrics/ref_correctness": 1 if ref_result.correctness else 0,
            "metrics/ref_runtime": max(ref_result.runtime, 0),
            "metrics/gen_compiled": 1 if gen_result.compiled else 0,
            "metrics/gen_correctness": 1 if gen_result.correctness else 0,
            "metrics/gen_runtime": max(gen_result.runtime, 0),
            "metrics/speedup": speedup if speedup > 0 else 0,
            "metrics/total_elapsed_time": total_elapsed,
            "cache/hit_rate": cache_hit_rate,
            "cache/total_hits": TOTAL_CACHE_HITS,
            "cache/total_misses": TOTAL_CACHE_MISSES,
            "cache/total_unclear": TOTAL_CACHE_UNCLEAR,
            f"{task_tag}/healthiness": 1,
            f"{task_tag}/ref_compiled": 1 if ref_result.compiled else 0,
            f"{task_tag}/gen_compiled": 1 if gen_result.compiled else 0,
            f"{task_tag}/ref_correctness": 1 if ref_result.correctness else 0,
            f"{task_tag}/gen_correctness": 1 if gen_result.correctness else 0,
            f"{task_tag}/speedup": speedup if speedup > 0 else 0,
            f"{task_tag}/elapsed_time": total_elapsed,
        }
        if wandb_run:
            wandb_run.log(metrics)

        return result

    except Exception as e:
        TOTAL_ERROR_COUNTER += 1
        elapsed_time = time.time() - start_time
        traceback.print_exc()
        logger.error(f"❌ [KB Eval] error: {type(e).__name__}: {str(e)}")
        result = KbEvalResult(
            total_elapsed_time=elapsed_time,
            ref_metadata={"processing_error": str(e), "retriable": True},
            gen_metadata={"processing_error": str(e), "retriable": True},
        )

        metrics = {
            "health/completion": 0,
            "health/parallel_requests": parallel_request_counter,
            "health/error_counter": TOTAL_ERROR_COUNTER,
            "health/request_counter": TOTAL_REQUEST_COUNTER,
            f"{task_tag}/healthiness": 0,
            f"{task_tag}/elapsed_time": elapsed_time,
        }
        if wandb_run:
            wandb_run.log(metrics)
        return result

    finally:
        async with parallel_request_counter_lock:
            parallel_request_counter -= 1
            if parallel_request_counter < 0:
                logger.error(
                    f"Request counter is negative: {parallel_request_counter}, resetting to 0"
                )
                parallel_request_counter = 0


async def graceful_exit():
    # Cancel all running tasks
    tasks = [task for task in asyncio.all_tasks() if not task.done()]
    for task in tasks:
        task.cancel()

    # Wait for cancellation to complete
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    sys.exit(1)


async def _check_total_error_count():
    global TOTAL_ERROR_COUNTER, MAX_ERROR_COUNT, START_TIME

    print_interval = 10
    check_interval = 3  # seconds
    counter = 0
    while True:
        try:
            counter += 1
            # check if elapsed time is greater than MAX_RUN_TIME
            CURR_TIME = time.time()
            ELAPSED_TIME = CURR_TIME - START_TIME
            if ELAPSED_TIME > MAX_RUN_TIME:
                # add an star emoji
                logger.error(
                    f"⭐ Elapsed time [{ELAPSED_TIME:.2f}s] is greater than {MAX_RUN_TIME/3600:.2f} hours, exiting... [parent process will restart]"
                )
                # loop = asyncio.get_event_loop()
                # loop.stop()
                for key, wandb_run in wandb_loggers.items():
                    wandb_run.finish()
                await graceful_exit()
            if TOTAL_ERROR_COUNTER > MAX_ERROR_COUNT:
                logger.error(
                    f"❌ Total error count [{TOTAL_ERROR_COUNTER}] is greater than {MAX_ERROR_COUNT}!"
                )
                # loop = asyncio.get_event_loop()
                # loop.stop()
                for key, wandb_run in wandb_loggers.items():
                    wandb_run.finish()
                await graceful_exit()
            elif TOTAL_ERROR_COUNTER > 0 and counter % print_interval == 0:
                logger.warning(
                    f"⚠️ Total error count is {TOTAL_ERROR_COUNTER}, continuing..."
                )
        finally:
            await asyncio.sleep(check_interval)


async def main(args):

    global MAX_TIMEOUT_SECONDS, COMPILE_CACHE, CHECK_GET_INPUTS
    MAX_TIMEOUT_SECONDS = args.max_timeout_seconds

    # read kbEval.yaml
    with open("kbEval.yaml", "r") as f:
        kbEval_config = yaml.load(f, Loader=yaml.FullLoader)
        # use file emoji
        logger.info(f"📁 [kbEvalServer] Config file kbEval.yaml loaded")

    #########################################################
    # get hostname and host, port from kbEval.yaml
    import socket

    hostname = socket.gethostname()
    # if hostname is not in kbEval_config["kbEvalRemoteServer"], use "one"
    if hostname not in kbEval_config["servers"]:
        logger.warning(
            f"Hostname {hostname} not found in kbEval.yaml, using 'one' as default"
        )
        hostname = "one"

    global DEVICES

    if args.local_host:
        host = "localhost"
        port = args.port
        DEVICES = [args.device]
    else:
        host = kbEval_config["servers"][hostname]["host"]
        port = kbEval_config["servers"][hostname]["port"]
        DEVICES = [int(d) for d in kbEval_config["servers"][hostname]["devices"]]

    logger.info(f"Running on [{hostname}:{port}] with devices: {DEVICES}")

    COMPILE_CACHE = bool(kbEval_config["servers"][hostname].get("compile_cache", False))
    logger.info(f"Compile cache is {COMPILE_CACHE}")

    CHECK_GET_INPUTS = bool(kbEval_config["servers"][hostname].get("check_get_inputs", True))
    logger.info(f"Check get_inputs is {CHECK_GET_INPUTS}")

    #########################################################
    # get api_key from kbEval_config["kbEvalRemoteServer"]["common"]["api_key"]
    if "common" not in kbEval_config["servers"]:
        logger.error("[kbEvalServer] [common] not found in kbEval.yaml")
        exit(1)
    if "api_key_path" not in kbEval_config["servers"]["common"]:
        logger.error(
            f"[kbEvalServer] [api_key_path] not found in kbEval.yaml [{kbEval_config['servers']['common']}]"
        )
        exit(1)
    api_key_filepath = kbEval_config["servers"]["common"]["api_key_path"]
    # read file from api_key, replace ${HOME} with os.path.expanduser("~") in api_key_filepath
    api_key_filepath = os.path.expanduser(api_key_filepath)
    if not os.path.exists(api_key_filepath):
        # create the file, and write a random string to it
        with open(api_key_filepath, "w") as f:
            api_key = str(uuid.uuid4())
            f.write(api_key)
            # add emoji to beginning and end of the string
            logger.info(
                f"🔑 [kbEvalServer] API key [{api_key}] created and saved to [{api_key_filepath}]"
            )
    # now read in the api_key
    with open(api_key_filepath, "r") as f:
        global KB_EVAL_TOKEN
        KB_EVAL_TOKEN = f.read().strip()
        logger.info(f"[kbEvalServer] KB_EVAL_TOKEN loaded from [{api_key_filepath}]")
    #########################################################

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, workers=args.workers)
    )
    logger.info(
        f"🚀 [kbEvalServer] Starting server on {host}:{port}... with {args.workers} workers"
    )

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
    parser.add_argument("--port", type=int, default=8456)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--device", type=str, default="4")
    parser.add_argument("--max_timeout_seconds", type=int, default=240)
    args = parser.parse_args()

    asyncio.run(main(args))
