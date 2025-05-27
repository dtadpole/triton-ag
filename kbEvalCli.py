import argparse
import sys
import traceback
from kbEvalTest.kbeval import eval_kernel_against_ref
import os
import json
from datetime import datetime

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str, default="./kbEvalTest")
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--reference_code", type=str, default="elemAddRef.py")
    parser.add_argument("--generated_code", type=str, default="elemAddCuda.py")
    parser.add_argument("--measure_performance_ref", action="store_true")
    parser.add_argument("--num_correct_trials", type=int, default=2)
    parser.add_argument("--num_perf_trials", type=int, default=100)
    args = parser.parse_args()

    # read from file
    reference_model_src = open(os.path.join(args.wd, args.reference_code), "r").read()
    generated_model_src = open(os.path.join(args.wd, args.generated_code), "r").read()

    result = eval_kernel_against_ref(
        reference_model_src, 
        generated_model_src, 
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        verbose=True, 
        measure_performance=True,
        measure_performance_ref=args.measure_performance_ref,
    )

    # check if there is compilation error
    if 'compilation_error' in result.metadata:
        # print exception and stack trace
        exception = result.metadata['compilation_error']
        exception_traceback_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        # print to stderr
        print(exception_traceback_str, file=sys.stderr)
        result.metadata['compilation_error'] = exception_traceback_str

    # check if there is runtime error
    if 'runtime_error' in result.metadata:
        # print exception and stack trace
        exception = result.metadata['runtime_error']
        exception_traceback_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        # print to stderr
        print(exception_traceback_str, file=sys.stderr)
        result.metadata['runtime_error'] = exception_traceback_str

    with open(os.path.join(args.wd, f"kbeval_{args.tag}.json"), "w") as f:
        f.write(json.dumps(result.model_dump(), indent=4))
