import os
import sys
import traceback
import json
import argparse
import asyncio
from datetime import datetime
import boto3
import torch
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

from kbEvalTest.kbeval import KernelExecResult, eval_kernel_against_ref, set_seed, graceful_eval_cleanup, run_and_check_correctness, time_execution_with_cuda_event, get_timing_stats, load_original_model_and_inputs, load_custom_model
from logger import logger


async def get_with_timeout(queue, timeout):
    try:
        item = await asyncio.wait_for(queue.get(), timeout)
        return item
    except asyncio.TimeoutError:
        return None  # Or raise an exception, or handle it as needed

KB_EVAL_DIR = os.path.join(os.path.expanduser("~"), ".kbeval")

# Create app
app = FastAPI()

eval_tasks = []
eval_queue = asyncio.Queue()
result_queue = {}

class EvalKernelRequest:
    model_tag: str
    task_tag: str
    eval_tag: str
    time_tag: str
    Model: torch.nn.Module
    get_init_inputs: callable
    get_inputs: callable
    ModelNew: torch.nn.Module
    metadata: dict
    context: dict

    def __init__(self, model_tag: str, task_tag: str, eval_tag: str, time_tag: str, Model: torch.nn.Module, get_init_inputs: callable, get_inputs: callable, ModelNew: torch.nn.Module, metadata: dict, context: dict):
        self.model_tag = model_tag
        self.task_tag = task_tag
        self.eval_tag = eval_tag
        self.time_tag = time_tag
        self.Model = Model
        self.get_init_inputs = get_init_inputs
        self.get_inputs = get_inputs
        self.ModelNew = ModelNew
        self.metadata = metadata
        self.context = context


async def get_pending_task_count():
    all_tasks = asyncio.all_tasks()
    return len(set(all_tasks))

@app.get("/stats")
async def stats():
    global eval_queue, result_queue
    return {
        "num_eval_tasks": len(eval_tasks),
        "pending_eval_requests": len(result_queue),
        "pending_tasks": await get_pending_task_count(),
    }

@app.post("/kb_eval")
async def kb_eval(
    model_tag: str = Body(...),
    task_tag: str = Body(...),
    eval_tag: str = Body(...),
    time_tag: str = Body(...),
    reference_code: str = Body(...),
    generated_code: str = Body(...),
) -> KernelExecResult:
    # temp_dir is {HOME}/.kbeval/{model_tag}/{task_tag}
    temp_dir = os.path.join(KB_EVAL_DIR, model_tag, task_tag)
    os.makedirs(temp_dir, exist_ok=True)
    
    with open(os.path.join(temp_dir, f"{eval_tag}_{time_tag}_reference_code.py"), "w") as f:
        f.write(reference_code)
    with open(os.path.join(temp_dir, f"{eval_tag}_{time_tag}_generated_code.py"), "w") as f:
        f.write(generated_code)

    result = await compile_and_eval_kernel(
        model_tag,
        task_tag,
        eval_tag,
        time_tag,
        reference_code,
        generated_code
    )

    if result is None:
        result = KernelExecResult(
            compiled=False, correctness=False, metadata={"error": "Unknown error"}
        )

    # check if there is compilation error
    if 'compilation_error' in result.metadata:
        # print exception and stack trace
        exception = result.metadata['compilation_error']
        exception_traceback_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        # print to stderr
        logger.error(exception_traceback_str)
        result.metadata['compilation_error'] = exception_traceback_str

    # check if there is runtime error
    if 'runtime_error' in result.metadata:
        # print exception and stack trace
        exception = result.metadata['runtime_error']
        exception_traceback_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        # print to stderr
        logger.error(exception_traceback_str)
        result.metadata['runtime_error'] = exception_traceback_str

    # write result to {temp_dir}/kbeval_{eval_tag}.json
    result_json = result.model_dump()
    with open(os.path.join(temp_dir, f"{eval_tag}_{time_tag}_kbeval.json"), "w") as f:
        f.write(json.dumps(result_json, indent=4))

    return result

async def compile_and_eval_kernel(
    model_tag: str,
    task_tag: str,
    eval_tag: str,
    time_tag: str,
    reference_code: str,
    generated_code: str,
) -> KernelExecResult:
    global eval_queue, result_queue

    Model, get_init_inputs, get_inputs, ModelNew, metadata, context = compile_kernel(
        model_tag,
        task_tag,
        eval_tag,
        time_tag,
        reference_code,
        generated_code,
        build_dir=None,
        seed_num=42,
        verbose=True,
    )

    request = EvalKernelRequest(
        model_tag=model_tag,
        task_tag=task_tag,
        eval_tag=eval_tag,
        time_tag=time_tag,
        Model=Model,
        get_init_inputs=get_init_inputs,
        get_inputs=get_inputs,
        ModelNew=ModelNew,
        metadata=metadata,
        context=context,
    )

    result_queue_key = f"{model_tag}_{task_tag}_{eval_tag}_{time_tag}"
    try:
        result_queue[result_queue_key] = asyncio.Queue()

        # add request to eval queue
        await eval_queue.put(request)
   
        # get result from result queue
        result = await get_with_timeout(result_queue[result_queue_key], 120)

        return result

    except Exception as e:
        logger.error(f"Error adding request to eval queue: {e}")
        raise e

    finally:
        # delete result queue
        del result_queue[result_queue_key]



def compile_kernel(
    model_tag: str,
    task_tag: str,
    eval_tag: str,
    time_tag: str,
    original_model_src: str,
    custom_model_src: str,
    build_dir: str = None,
    seed_num: int = 42,
    verbose: bool = False,
) -> tuple[torch.nn.Module, callable, callable, torch.nn.Module, dict]: # Model, get_init_inputs, get_inputs, ModelNew, metadata, context
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    """
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    eval_key = f"{model_tag}_{task_tag}_{eval_tag}_{time_tag}"

    context = {}

    if verbose:
        logger.info(f"[Eval {eval_key}] Start Evalulation!")
        logger.info(f"[Eval {eval_key}] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )

    metadata = {}  # for storing result metadata

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model(custom_model_src, context, build_dir)
        # torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            logger.error(f"[Eval {eval_key}] Lock file error during compilation, Please retry. Error: {e}")
            graceful_eval_cleanup(context)
            return None
        else:
            metadata["compilation_error"] = e
            graceful_eval_cleanup(context)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    return Model, get_init_inputs, get_inputs, ModelNew, metadata, context

async def eval_kernel_against_ref_async(
    device: torch.device, # have to run on GPU
) -> KernelExecResult:

    global eval_queue, result_queue
    logger.info(f"[KB_Eval] Started on device {device}")

    seed_num: int = 42
    num_correct_trials: int = 2
    num_perf_trials: int = 50
    verbose: bool = False
    measure_performance: bool = True
    measure_performance_ref: bool = False

    context: dict = {}

    while True:
        try:
            request: EvalKernelRequest = await get_with_timeout(eval_queue, 10)
            if request is None:
                continue

            eval_key = f"{request.model_tag}_{request.task_tag}_{request.eval_tag}_{request.time_tag}"
            logger.info(f"[KB_Eval {eval_key}] Started...")

            context = request.context

            set_seed(seed_num)  # set seed for reproducible input
            init_inputs = request.get_init_inputs()
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
            ]

            with torch.no_grad():
                set_seed(seed_num)  # set seed for reproducible weights
                original_model = request.Model(*init_inputs)
                assert hasattr(original_model, "forward")
                if verbose:
                    logger.info(f"[KB_Eval {eval_key}] Original Model Loaded")
            if verbose:
                logger.info(f"[KB_Eval {eval_key}] Loading and Compiling New Model with Custom CUDA Kernel")

            request.metadata["hardware"] = torch.cuda.get_device_name(device=device)
            request.metadata["device"] = str(device)  # for debugging


            # at this point we passed compilation
            try:
                with torch.no_grad():
                    set_seed(seed_num)  # set seed for reproducible weights
                    custom_model = request.ModelNew(*init_inputs)
                    assert hasattr(custom_model, "forward")
                    torch.cuda.synchronize(device=device)
                if verbose:
                    logger.info(f"[KB_Eval {eval_key}] New Model with Custom CUDA Kernel Loaded")
            except RuntimeError as e:
                print(
                    f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
                )
                # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
                graceful_eval_cleanup(context, device)
                request.metadata["runtime_error"] = e
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=request.metadata
                )  # skip further steps

            kernel_exec_result = None

            # Check Correctness
            if verbose:
                logger.info(f"[KB_Eval {eval_key}] Checking Correctness")
            try:
                kernel_exec_result = run_and_check_correctness(
                    original_model,
                    custom_model,
                    request.get_inputs,
                    metadata=request.metadata,
                    num_correct_trials=num_correct_trials,
                    verbose=verbose,
                    seed=seed_num,
                    device=device,
                )
            except Exception as e:
                # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
                request.metadata["runtime_error"] = e
                kernel_exec_result = KernelExecResult(
                    compiled=True, correctness=False, metadata=request.metadata
                )

            # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
            if measure_performance:
                try:
                    if kernel_exec_result and kernel_exec_result.correctness:
                        if verbose:
                            logger.info(f"[KB_Eval {eval_key}] Measuring Performance as Sample is Correct")

                        torch.cuda.synchronize(device=device)
                        set_seed(seed_num)
                        inputs = request.get_inputs()
                        inputs = [
                            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                            for x in inputs
                        ]
                        model_new = custom_model.cuda(device=device)
                        torch.cuda.synchronize(device=device)

                        elapsed_times = time_execution_with_cuda_event(
                            model_new,
                            *inputs,
                            num_trials=num_perf_trials,
                            verbose=verbose,
                            device=device,
                        )
                        runtime_stats = get_timing_stats(elapsed_times, device=device)

                        if verbose:
                            logger.info(f"[KB_Eval {eval_key}] Performance Stats: {runtime_stats}")
                        kernel_exec_result.runtime = runtime_stats["mean"]
                        kernel_exec_result.runtime_stats = runtime_stats

                        if measure_performance_ref:
                            elapsed_times_ref = time_execution_with_cuda_event(
                                request.original_model,
                                *inputs,
                                num_trials=num_perf_trials,
                                verbose=verbose,
                                device=device,
                            )
                            runtime_stats_ref = get_timing_stats(elapsed_times_ref, device=device)
                            if verbose:
                                logger.info(f"[KB_Eval {eval_key}] Performance Stats (Reference): {runtime_stats_ref}")
                            kernel_exec_result.metadata["reference_runtime_stats"] = runtime_stats_ref

                except Exception as e:
                    if verbose:
                        logger.error(f"[KB_Eval] Error in Measuring Performance: {e}")
                    kernel_exec_result.metadata["error_during_performance"] = e

            await result_queue[eval_key].put(kernel_exec_result)
            logger.info(f"[KB_Eval {eval_key}] Result: {kernel_exec_result.model_dump()}")

        except Exception as e:
            # print exception and stack trace
            logger.error(f"[KB_Eval {eval_key}] Error in Evaluating Kernel: {e}")
            logger.error(traceback.format_exc())
            await result_queue[eval_key].put(KernelExecResult(
                compiled=False, correctness=False, metadata=request.metadata | {"error": str(e)}
            ))
        finally:
            # clean up
            graceful_eval_cleanup(context, device)


async def main(args):
    global eval_tasks
    devices = [int(d) for d in args.devices.split(",")]
    for device in devices:
        eval_tasks.append(asyncio.create_task(eval_kernel_against_ref_async(device)))

    import uvicorn
    logger.info(f"Starting server on port {args.port}")
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=args.port))
    server_task = asyncio.create_task(server.serve())
    logger.info(f"Server started on port {args.port}")

    await asyncio.gather(*eval_tasks, server_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5678)
    parser.add_argument("-d", "--devices", type=str, default="0")
    args = parser.parse_args()

    asyncio.run(main(args))
