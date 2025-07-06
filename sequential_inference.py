import os
import re
import json
import time
import asyncio
import traceback
import argparse
import yaml
import random
from datetime import datetime
from typing import List
import boto3
import requests
from pathlib import Path
from transformers import AutoTokenizer
from inferenceClient import InferenceClient
from kbEvalClient import KbEvalClient
from logger import logger
from kbEvalTest.kbeval import KernelExecResult
from inferenceCodeGenEval import CodeGenEvalClient


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="./kernel_bench/", help="Input directory containing Python files")
    parser.add_argument("--config", type=str, default="inferenceClient.yaml")
    parser.add_argument("--client_type", type=str, default="deepinfra-r1")
    parser.add_argument("--epoch_id", type=int, default=1)
    parser.add_argument("--batch_id", type=int, default=1)
    parser.add_argument("--run_tag", type=str, default="sequential_inference")
    parser.add_argument("--include_timestamp", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--parallel_tasks", type=int, default=16)
    args = parser.parse_args()
    
    # Set up directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Error: Input directory {input_dir} does not exist")
        return

    # recursively get all the python files under the input directory and store in a list
    reference_code_json = []
    for root, dirs, files in os.walk(input_dir, followlinks=True):
        for file in files:
            if file.endswith(".py"):
                # read the file content
                with open(os.path.join(root, file), 'r') as f:
                    reference_code = f.read()
                reference_code_json.append({
                    "reference_code": reference_code,
                    "task_tag": os.path.relpath(os.path.join(root, file), input_dir),
                })

    # randomly pick args.num_samples files from the list
    random.shuffle(reference_code_json)
    reference_code_json = reference_code_json[:args.num_samples]

    # create the codeGenEvalClient
    run_tag = args.run_tag if not args.include_timestamp else f"{args.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    codeGenEvalClient = CodeGenEvalClient(config_file=args.config, run_tag=run_tag, client_type=args.client_type)

    # run the codeGenEvalClient
    reference_code_contents = [item['reference_code'] for item in reference_code_json]
    task_tags = [item['task_tag'] for item in reference_code_json]
    # now run inference
    await codeGenEvalClient.run(args.epoch_id, args.batch_id, reference_code_contents, task_tags, args.num_generations, args.parallel_tasks)


if __name__ == "__main__":
    asyncio.run(main())
