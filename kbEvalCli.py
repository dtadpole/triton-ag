import argparse
import sys
import time
import traceback
from kbEvalTest.kbeval import KernelExecResult, eval_kernel_against_ref, graceful_eval_cleanup, run_and_check_correctness, time_execution_with_cuda_event, get_timing_stats, load_original_model_and_inputs, load_custom_model, set_seed
import torch
import asyncio
import os
import json
from datetime import datetime
from util import logger
import random
from filelock import FileLock, Timeout

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

KB_EVAL_DIR = os.path.expanduser("~/.kbeval")

def compile_and_eval_kernel(
    model_tag: str,
    task_tag: str,
    eval_tag: str,
    time_tag: str,
    reference_code: str,
    generated_code: str,
    device: torch.device,
    args: argparse.Namespace,
) -> KernelExecResult:

    try:
        Model, get_init_inputs, get_inputs, ModelNew, metadata, context = compile_kernel_new(
            model_tag,
            task_tag,
            eval_tag,
            time_tag,
            reference_code,
            generated_code,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"[KB_Eval] Error compiling kernel: {e}")
        result = KernelExecResult(
            compiled=False, correctness=False, metadata={"compilation_error": e}
        )
        # return result
        return result

    # get my own process id
    pid = os.getpid()

    eval_key = f"{model_tag}_{task_tag}_{eval_tag}_{time_tag}"

    # lock file is {HOME}/.kbeval/lock_{str(device)}
    lock_file = os.path.join(KB_EVAL_DIR, f".lock_{str(device)}")
    lock = FileLock(lock_file)
    while True:
        try:
            with lock.acquire(timeout=1):
                logger.warning(f"[KB_Eval] Acquired lock {lock_file} [{eval_key}]")

                # verify lock is working by sleeping randome between 10 and 20 seconds
                # time.sleep(random.randint(10, 20)) # verified lock is working

                # write my pid to lock file
                with open(lock_file, "w") as f:
                    f.write(str(pid))

                result = eval_kernel_against_ref_new(
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
                    device=device,
                    seed_num=42,
                    verbose=args.verbose,
                    measure_performance=True,
                    measure_performance_ref=args.measure_performance_ref,
                )

            # os.remove(lock_file)
            logger.warning(f"[KB_Eval] Released lock {lock_file} [{eval_key}]")
            return result
        
        except Timeout:
            logger.info(f"[KB_Eval] Waiting for lock to be released {lock_file} [{eval_key}]")
            continue
        except Exception as e:
            logger.error(f"[KB_Eval] Error acquiring lock: {e} [{eval_key}]")
            result = KernelExecResult(
                compiled=True, correctness=False, metadata={"runtime_error": e}
            )
            return result
        finally:
            lock.release()
            # check lockfile modified time
            if os.path.exists(lock_file):
                lock_modified_time = os.path.getmtime(lock_file)
                # if modified time is more than 3 minutes, delete lock file
                if lock_modified_time < os.path.getmtime(lock_file) - 180:
                    logger.error(f"[KB_Eval] Lock file {lock_file} is older than 3 minutes, deleting... [{eval_key}]")
                    os.remove(lock_file)

def compile_kernel_new(
    model_tag: str,
    task_tag: str,
    eval_tag: str,
    time_tag: str,
    original_model_src: str,
    custom_model_src: str,
    verbose: bool = False,
) -> tuple[torch.nn.Module, callable, callable, torch.nn.Module, dict, dict]: # Model, get_init_inputs, get_inputs, ModelNew, metadata, context
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

    metadata = {}  # for storing result metadata

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model(custom_model_src, context, build_directory=None)
        # torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        logger.error(
            f"[KB_Eval] Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e} [{eval_key}]"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            logger.error(f"[KB_Eval] Lock file error during compilation, Please retry. Error: {e} [{eval_key}]")
            graceful_eval_cleanup(context, device=None)
            raise e
        else:
            metadata["compilation_error"] = e
            graceful_eval_cleanup(context, device=None)
            raise e

    if verbose:
        logger.info(f"[KB_Eval] Start Evalulation! [{eval_key}]")
        logger.info(f"[KB_Eval] Loading Original Model [{eval_key}]")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )

    return Model, get_init_inputs, get_inputs, ModelNew, metadata, context

def eval_kernel_against_ref_new(
    model_tag: str,
    task_tag: str,
    eval_tag: str,
    time_tag: str,
    Model: torch.nn.Module,
    get_init_inputs: callable,
    get_inputs: callable,
    ModelNew: torch.nn.Module,
    metadata: dict,
    context: dict,
    device: torch.device, # have to run on GPU
    seed_num: int = 42,
    verbose: bool = False,
    measure_performance: bool = True,
    measure_performance_ref: bool = False,
) -> KernelExecResult:

    global eval_queue, result_queue

    eval_key = f"{model_tag}_{task_tag}_{eval_tag}_{time_tag}"
    logger.info(f"[KB_Eval] Started on device {device} [{eval_key}]")

    num_correct_trials: int = 2
    num_perf_trials: int = 50

    try:
        set_seed(seed_num)  # set seed for reproducible input
        init_inputs = get_init_inputs()
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            original_model = Model(*init_inputs)
            assert hasattr(original_model, "forward")
            if verbose:
                logger.info(f"[KB_Eval] Original Model Loaded [{eval_key}]")
        if verbose:
            logger.info(f"[KB_Eval] Loading and Compiling New Model with Custom CUDA Kernel [{eval_key}]")

        metadata["hardware"] = torch.cuda.get_device_name(device=device)
        metadata["device"] = str(device)  # for debugging


        # at this point we passed compilation
        try:
            with torch.no_grad():
                set_seed(seed_num)  # set seed for reproducible weights
                custom_model = ModelNew(*init_inputs)
                assert hasattr(custom_model, "forward")
                torch.cuda.synchronize(device=device)
            if verbose:
                logger.info(f"[KB_Eval] New Model with Custom CUDA Kernel Loaded [{eval_key}]")
        except RuntimeError as e:
            logger.error(
                f"[KB_Eval] Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e} [{eval_key}]"
            )
            # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
            graceful_eval_cleanup(context, device)
            metadata["runtime_error"] = e
            return KernelExecResult(
                compiled=True, correctness=False, metadata=metadata
            )  # skip further steps

        kernel_exec_result = None

        # Check Correctness
        if verbose:
            logger.info(f"[KB_Eval] Checking Correctness [{eval_key}]")
        try:
            kernel_exec_result = run_and_check_correctness(
                original_model,
                custom_model,
                get_inputs,
                metadata=metadata,
                num_correct_trials=num_correct_trials,
                verbose=verbose,
                seed=seed_num,
                device=device,
            )
        except Exception as e:
            # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
            metadata["runtime_error"] = e
            kernel_exec_result = KernelExecResult(
                compiled=True, correctness=False, metadata=metadata
            )

        # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
        if measure_performance:
            try:
                if kernel_exec_result and kernel_exec_result.correctness:
                    if verbose:
                        logger.info(f"[KB_Eval] Measuring Performance as Sample is Correct [{eval_key}]")

                    torch.cuda.synchronize(device=device)
                    set_seed(seed_num)
                    inputs = get_inputs()
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
                        logger.info(f"[KB_Eval] Performance Stats: {runtime_stats} [{eval_key}]")
                    kernel_exec_result.runtime = runtime_stats["mean"]
                    kernel_exec_result.runtime_stats = runtime_stats

                    if measure_performance_ref:
                        elapsed_times_ref = time_execution_with_cuda_event(
                            original_model,
                            *inputs,
                            num_trials=num_perf_trials,
                            verbose=verbose,
                            device=device,
                        )
                        runtime_stats_ref = get_timing_stats(elapsed_times_ref, device=device)
                        if verbose:
                            logger.info(f"[KB_Eval] Performance Stats (Reference): {runtime_stats_ref} [{eval_key}]")
                        kernel_exec_result.metadata["reference_runtime_stats"] = runtime_stats_ref

            except Exception as e:
                if verbose:
                    logger.error(f"[KB_Eval] Error in Measuring Performance: {e} [{eval_key}]")
                kernel_exec_result.metadata["error_during_performance"] = e

        logger.info(f"[KB_Eval] Result: {kernel_exec_result.model_dump()} [{eval_key}]")
        return kernel_exec_result

    except Exception as e:
        # print exception and stack trace
        logger.error(f"[KB_Eval] Error in Evaluating Kernel: {e} [{eval_key}]")
        logger.error(traceback.format_exc())
        return KernelExecResult(
            compiled=False, correctness=False, metadata=metadata | {"evaluation_error": e}
        )
    finally:
        # clean up
        graceful_eval_cleanup(context, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str, default="./kbEvalTest")
    parser.add_argument("--model_tag", type=str, default="model_tag")
    parser.add_argument("--task_tag", type=str, default="task_tag")
    parser.add_argument("--eval_tag", type=str, default="eval_tag")
    parser.add_argument("--time_tag", type=str, default="time_tag")
    parser.add_argument("--reference_code", type=str, default="elemAddRef.py")
    parser.add_argument("--generated_code", type=str, default="elemAddCuda.py")
    parser.add_argument("--measure_performance_ref", action="store_true")
    parser.add_argument("--device-list", type=str, default="4")
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.environ["MAX_JOBS"] = str(args.max_jobs)

    # temp_dir is {HOME}/.kbeval/{model_tag}/{task_tag}
    temp_dir = os.path.join(KB_EVAL_DIR, args.model_tag, args.task_tag)
    os.makedirs(temp_dir, exist_ok=True)

    # read from file
    reference_model_src = open(os.path.join(args.wd, args.reference_code), "r").read()
    generated_model_src = open(os.path.join(args.wd, args.generated_code), "r").read()

    devices = args.device_list.split(",")
    # select a random device from devices
    device = torch.device(int(devices[random.randint(0, len(devices) - 1)]))

    result = compile_and_eval_kernel(
        model_tag=args.model_tag,
        task_tag=args.task_tag,
        eval_tag=args.eval_tag,
        time_tag=args.time_tag,
        reference_code=reference_model_src,
        generated_code=generated_model_src,
        device=device,
        args=args,
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

    # check if there is runtime error
    if 'error_during_performance' in result.metadata:
        # print exception and stack trace
        exception = result.metadata['error_during_performance']
        exception_traceback_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        # print to stderr
        logger.error(exception_traceback_str)
        result.metadata['error_during_performance'] = exception_traceback_str

    # check if there is evaluation error
    if 'evaluation_error' in result.metadata:
        # print exception and stack trace
        exception = result.metadata['evaluation_error']
        exception_traceback_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        # print to stderr
        logger.error(exception_traceback_str)
        result.metadata['evaluation_error'] = exception_traceback_str

    # write to file
    with open(os.path.join(temp_dir, f"{args.eval_tag}_{args.time_tag}_kbeval.json"), "w") as f:
        f.write(json.dumps(result.model_dump(), indent=4))


    logger.info(f"Evaluation result: {json.dumps(result.model_dump(), indent=4)}")
