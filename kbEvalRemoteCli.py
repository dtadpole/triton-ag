import argparse
import requests
import sys
import traceback
from fastapi import FastAPI
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
    parser.add_argument("--model_tag", type=str, default="model_tag")
    parser.add_argument("--task_tag", type=str, default="task_tag")
    parser.add_argument("--eval_tag", type=str, default="eval_tag")
    parser.add_argument("--reference_code", type=str, default="elemAddRef.py")
    parser.add_argument("--generated_code", type=str, default="elemAddCuda.py")
    parser.add_argument("--measure_reference", action="store_true")
    args = parser.parse_args()

    # read from file
    reference_model_src = open(os.path.join(args.wd, args.reference_code), "r").read()
    generated_model_src = open(os.path.join(args.wd, args.generated_code), "r").read()

    # connect to FastAPI server
    server_url = "http://localhost:5678"
    if args.measure_reference:
        response = requests.post(f"{server_url}/kb_eval_ref", data=json.dumps({
            "model_tag": args.model_tag,
            "task_tag": args.task_tag,
            "time_tag": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "reference_code": reference_model_src,
        }), headers={"Content-Type": "application/json"})
        print(json.dumps(response.json(), indent=4))
    else:
        response = requests.post(f"{server_url}/kb_eval", data=json.dumps({
            "model_tag": args.model_tag,
            "task_tag": args.task_tag,
            "eval_tag": args.eval_tag,
            "time_tag": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "reference_code": reference_model_src,
            "generated_code": generated_model_src,
        }), headers={"Content-Type": "application/json"})
        print(json.dumps(response.json(), indent=4))

    