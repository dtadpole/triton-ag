import argparse
import socket
import yaml
import asyncio
import json
import os
import signal
import random
import sys
import time
import fcntl
import errno
import traceback
from threading import Timer
from pydantic import BaseModel
from torch import nn
from kbEvalUtil import KernelExecResult, from_kbEval_yaml, format_exception, CorrectnessResult, CorrectnessError, CorrectnessShapeMismatchError, CorrectnessValueMismatchError, CorrectnessProcessingError, CompileError, CompileInstantiationError, CompileRuntimeError, set_seed, get_timing_stats, time_execution_with_cuda_event, load_model_and_inputs, load_custom_model, graceful_eval_cleanup, on_critical_alarm, on_critical_timeout, on_process_timeout, resolve_triton_code
import torch
import asyncio
import os
import json
import psutil
from datetime import datetime
from logger import logger
from armableWatchdog import ArmableWatchdog
import random
from configEndpoints import FileLock, cleanup_lockfile
from util import KB_EVAL_DIR
# from filelock import FileLock, Timeout


def verify_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    total_trials: int = 3,
    seed: int = 42,
    device: torch.device = None,
) -> CorrectnessResult:
    """
    run the model and check correctness,
    total_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(total_trials)
    ]

    # set device to cuda if not provided
    device = device or torch.device("cuda")

    with torch.no_grad():

        passed_trials = 0
        max_diff = 0
        avg_diff = 0
        for trial in range(total_trials):

            trial_seed = correctness_trial_seeds[trial]

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
            except Exception as e:
                raise CompileRuntimeError(f"Error in running custom model: [{type(e)}] [{e}]") from e

            try:
                output = model(*inputs)
                torch.cuda.synchronize(device=device)
            except Exception as e:
                # ensure all GPU operations are completed before checking results
                raise CompileRuntimeError(f"Error in running original model: [{type(e)}] [{e}]") from e

            try:
                if output.shape != output_new.shape:
                    raise CorrectnessShapeMismatchError(f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}")

                # check output value difference
                if torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):
                    passed_trials += 1
                else:
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()

            except Exception as e:
                raise CorrectnessProcessingError(f"Correctness processing error: {e}") from e

    # we are here because all trials passed
    return CorrectnessResult(
        trials=f"{passed_trials}/{total_trials}",
        total_trials=total_trials,
        passed_trials=passed_trials,
        output_shape=f"{output.shape}",
        max_diff=max_diff,
        avg_diff=avg_diff,
    )


def eval_kernel_custom(
    run_tag: str,
    model_tag: str,
    task_tag: str,
    eval_tag: str,
    reference_code: str,
    reference_path: str,
    generated_code: str,
    generated_path: str,
    device: torch.device,
    work_dir: str,
    seed_num: int = 42,
    num_verify_trials: int = 2,
    num_perf_trials: int = 10,
    num_warmups: int = 3,
    measure_reference: bool = False,
    code_type: str = "triton",
    max_critical_time: int = 20,
) -> KernelExecResult:
    """
    Evaluate the reference code against the original model
    """
    # test code
    # signal.signal(signal.SIGALRM, on_timeout)
    # signal.alarm(max_run_time)  # exit after max_run_time seconds

    context = {}
    if measure_reference:
        metadata = {
            "is_reference": True,
            "hardware": torch.cuda.get_device_name(device=device),
            "device": str(device),  # for debugging
        }
    else:
        metadata = {
            "hardware": torch.cuda.get_device_name(device=device),
            "device": str(device),  # for debugging
        }

    try:
        # always load reference model
        Model, get_init_inputs, get_inputs = load_model_and_inputs(
            reference_code,
            context,
            filename=reference_path,
        )

        if not measure_reference:
            # load custom model only if not measuring reference
            # remember to load models before acquiring lock
            ModelNew = load_custom_model(
                generated_code,
                context,
                build_directory=work_dir,
                filename=generated_path,
            )

            if code_type == "triton":
                # check there is function call from ModelNew.forward to @triton.jit function(s)
                resolve_triton_code(generated_code)

    except CompileError as e:
        formatted_error = format_exception(e)
        logger.warning(f"[KB_Eval_Cli] [{task_tag}/{eval_tag}] {formatted_error}")
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata=metadata | {
                "compilation_error": format_exception(e),
            },
        )
    except Exception as e:
        logger.warning(f"[KB_Eval_Cli] [{task_tag}/{eval_tag}] Error in code compilation: {e}")
        traceback.print_exc()
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata=metadata | {
                "compilation_error": format_exception(e),
            },
        )

    # we are here because models compiled successfully
    # now we need to acquire lock and run the evaluation
    lock_file = os.path.join(KB_EVAL_DIR, f".lock_{str(device)}")
    while True:
        correctness = False
        try:
            with FileLock(lock_file):
                logger.warning(f"[KB_Eval_Cli] [{task_tag}/{eval_tag}] Acquired lock [{lock_file}]")

                with ArmableWatchdog(timeout_sec=max_critical_time, grace_sec=2) as arm:
                    logger.warning(f"[KB_Eval_Cli] [{task_tag}/{eval_tag}] Arming for [{max_critical_time}] seconds")
                    arm()

                    # verify lock is working by sleeping randomly between 10 and 20 seconds
                    # time.sleep(random.randint(10, 15)) # verified lock is working

                    # Install the handler and arm the timer (in seconds)
                    # signal.signal(signal.SIGALRM, on_critical_alarm)
                    # signal.alarm(max_critical_time)  # exit after max_critical_time seconds
                    # critical_timer = Timer(max_critical_time * 1.5, on_critical_timeout) # insurance policy for critical timeout
                    # critical_timer.daemon = True
                    # critical_timer.start()
                    logger.warning(f"[KB_Eval_Cli] [{task_tag}/{eval_tag}] Alarm set for Critical Section with [{max_critical_time}] seconds")

                    init_inputs = get_init_inputs()
                    init_inputs = [
                        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
                    ]

                    inputs = get_inputs()
                    inputs = [
                        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs
                    ]

                    with torch.no_grad():
                        try:
                            set_seed(seed_num)  # set seed for reproducible weights
                            original_model = Model(*init_inputs)
                            original_model = original_model.cuda(device=device)
                            assert hasattr(original_model, "forward")
                        except Exception as e:
                            raise CompileInstantiationError(f"Error in instantiating original model: [{type(e)}] [{e}]") from e

                        if measure_reference:
                            elapsed_times_ref = time_execution_with_cuda_event(
                                original_model,
                                *inputs,
                                num_warmups=num_warmups,
                                num_trials=num_perf_trials,
                                device=device,
                            )
                            runtime_stats = get_timing_stats(elapsed_times_ref, device=device)

                            return KernelExecResult(
                                compiled=True,
                                correctness=True,
                                metadata=metadata,
                                runtime=runtime_stats["mean"],
                                runtime_stats=runtime_stats,
                            )

                        else:
                            try:
                                custom_model = ModelNew(*init_inputs)
                                custom_model = custom_model.cuda(device=device)
                                assert hasattr(custom_model, "forward")
                            except Exception as e:
                                raise CompileInstantiationError(f"Error in instantiating custom model: [{type(e)}] [{e}]") from e

                            correctness_result = verify_correctness(
                                original_model,
                                custom_model,
                                get_inputs,
                                total_trials=num_verify_trials,
                                seed=seed_num,
                                device=device,
                            )

                            if correctness_result.passed_trials == num_verify_trials:
                                correctness = True
                            else:
                                # if correctness is not met, return without measuring performance
                                return KernelExecResult(
                                    compiled=True,
                                    correctness=False,
                                    metadata=metadata | {
                                        "correctness": correctness_result.model_dump(),
                                    },
                                )

                            elapsed_times_perf = time_execution_with_cuda_event(
                                custom_model,
                                *inputs,
                                num_warmups=num_warmups,
                                num_trials=num_perf_trials,
                                device=device,
                            )
                            runtime_stats = get_timing_stats(elapsed_times_perf, device=device)

                            return KernelExecResult(
                                compiled=True,
                                correctness=correctness,
                                metadata=metadata | {
                                    "correctness": correctness_result.model_dump(),
                                },
                                runtime=runtime_stats["mean"],
                                runtime_stats=runtime_stats,
                            )

        except TimeoutError:
            graceful_eval_cleanup(context, device)
            logger.info(f"⏳ [KB_Eval_Cli] [{task_tag}/{eval_tag}] Waiting for lock to be released {lock_file}")
            time.sleep(random.uniform(0.5, 1.5)) # sleep randomly between 0.5 and 1.5 seconds, using float to avoid blocking
            continue

        except CompileError as e:
            logger.warning(f"[KB_Eval_Cli] [{task_tag}/{eval_tag}] Error in code compilation: [{type(e)}] [{e}]")
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata=metadata | {
                    "compilation_error": format_exception(e),
                },
            )

        except CorrectnessError as e:
            logger.warning(f"[KB_Eval_Cli] [{task_tag}/{eval_tag}] Correctness error: [{type(e)}] [{e}]")
            return KernelExecResult(
                compiled=True,
                correctness=False,
                metadata=metadata | {
                    "correctness_error": format_exception(e),
                },
            )

        except Exception as e:
            logger.warning(f"[KB_Eval_Cli] [{task_tag}/{eval_tag}] Error acquiring lock: [{type(e)}] [{e}]")
            result = KernelExecResult(
                compiled=True,
                correctness=correctness,
                metadata=metadata | {
                    "runtime_error": format_exception(e),
                },
            )
            return result

        finally:
            # torch.cuda.synchronize(device=device)
            graceful_eval_cleanup(context, device)
            cleanup_lockfile(lock_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str, default="./kbEvalTest")
    parser.add_argument("--run_tag", type=str, default="auto")
    parser.add_argument("--model_tag", type=str, default="model_tag")
    parser.add_argument("--task_tag", type=str, default="task_tag")
    parser.add_argument("--eval_tag", type=str, default="eval_tag")
    parser.add_argument("--code_type", type=str, required=True, choices=["triton", "cuda", "pytorch"]) # add rocm support later
    parser.add_argument("--reference_code", type=str,
                        # default="/home/centos/.kbeval/Qwen/Qwen3-8B-FP8/86_conv_depthwise_separable_2D/20250629_050843/reference_code.py")
                        default="elemAddRef.py")
    parser.add_argument("--generated_code", type=str, default="elemAddTriton.py")
    parser.add_argument("--measure_reference", action="store_true")
    parser.add_argument("--device-list", type=str, default="0")
    parser.add_argument("--max_critical_time", type=int, default=15)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cli_config = from_kbEval_yaml()
    for key, value in cli_config.get("env_vars", {}).items():
        os.environ[key] = str(value)

    # temp_dir is {HOME}/.kbeval/{run_tag}/{model_tag}/{task_tag}/{eval_tag}
    run_tag = args.run_tag if args.run_tag != "auto" else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    work_dir = os.path.join(KB_EVAL_DIR, run_tag, args.model_tag, args.task_tag, args.eval_tag)
    os.makedirs(work_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = work_dir
    logger.info(f"🔍 Setting TORCH_EXTENSIONS_DIR to [{work_dir}]")

    devices = args.device_list.split(",")
    # select the device from devices randomly (The randomness is a bit questionable, as one device got overloaded)
    # better strategy is to select the onlocked device
    device = torch.device(int(devices[random.randint(0, len(devices) - 1)]))
    logger.info(f"🗳️ There are {len(devices)} devices available. Using device {device} for evaluation.")

    # read reference code from file
    reference_code_path = os.path.join(args.wd, args.reference_code)
    reference_code_src = open(reference_code_path, "r").read()
    reference_eval_path = os.path.join(args.wd, f"{args.eval_tag}_kbeval.json")

    # read generated code from file ???
    if args.measure_reference:
        generated_code_path = None
        generated_code_src = None
        generated_eval_path = None
    else:
        generated_code_path = os.path.join(args.wd, args.generated_code)
        generated_code_src = open(generated_code_path, "r").read()
        generated_eval_path = os.path.join(args.wd, f"{args.eval_tag}_kbeval.json")

    result = None
    exit_code = 0

    # if measure_reference or code_type is `triton`, evaluate the reference code only
    try:
        if args.measure_reference:
            result = eval_kernel_custom(
                run_tag=run_tag,
                model_tag=args.model_tag,
                task_tag=args.task_tag,
                eval_tag=args.eval_tag,
                reference_code=reference_code_src,
                reference_path=reference_code_path,
                generated_code=reference_code_src,
                generated_path=generated_code_path,
                device=device,
                work_dir=work_dir,
                measure_reference=True,
                max_critical_time=args.max_critical_time,
            )
        else:
            result = eval_kernel_custom(
                run_tag=run_tag,
                model_tag=args.model_tag,
                task_tag=args.task_tag,
                eval_tag=args.eval_tag,
                reference_code=reference_code_src,
                reference_path=reference_code_path,
                generated_code=generated_code_src,
                generated_path=generated_code_path,
                device=device,
                work_dir=work_dir,
                code_type=args.code_type,
                max_critical_time=args.max_critical_time,
            )
    except Exception as exception:
        exit_code = 1
        # exception_traceback_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        traceback.print_exc()
        result = KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"processing_error": format_exception(exception)},
        )
    finally:
        if result is not None:
            # write to file
            if args.measure_reference:
                output_path = reference_eval_path
            else:
                output_path = generated_eval_path
            with open(output_path, "w") as f:
                f.write(json.dumps(result.model_dump(), indent=4))
            if args.quiet:
                logger.info(f"🔍 Evaluation result stored in [{output_path}]")
            else:
                logger.info(f"🔍 Evaluation result stored in [{output_path}]\n{json.dumps(result.model_dump(), indent=4)}")
            # exit with exit_code
            sys.exit(exit_code)

if __name__ == "__main__":
    main()
