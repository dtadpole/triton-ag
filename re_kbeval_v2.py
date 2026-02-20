import asyncio
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from kbEvalClient import KbEvalClient


CONFIG_FILE = "kbEval.yaml"
PROVIDER = ["a4_2"]
OUTPUT_DIR = "shared/re_kbeval"
MAX_CONCURRENT = 30
CODE_TYPE = "triton"

OVERWRITE = True


def read_code_file(filename):
    with open(filename, "r") as file:
        code = file.read()
    return code


def get_output_filename(generated_code_path):
    """
    Generate output path for the JSON result file by mirroring the generated code's
    folder structure under OUTPUT_DIR, replacing the file extension with .json,
    and ensuring the output directory exists.
    """
    generated_path = Path(generated_code_path)
    # Remove the root (if any) and reconstruct the relative path
    relative_path = generated_path.relative_to(*generated_path.parts[:1])
    # Change the suffix to .json
    output_relative = relative_path.with_suffix(".json")
    # Prepend OUTPUT_DIR
    output_path = Path(OUTPUT_DIR) / output_relative
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_reference_and_generated(folder_path):
    """
    Search for reference and generated code files in a given folder.

    Args:
        folder_path (str): Path to the folder to search

    Returns:
        tuple: (reference_code_path, list_of_generated_code_paths)

    Raises:
        ValueError: If no reference code or more than one reference code is found
    """
    # Convert to Path object for easier handling
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder_path}")

    # Search for reference code files
    reference_pattern = "*reference*.py"
    reference_files = list(folder.glob(reference_pattern))

    # Check reference code count
    if len(reference_files) == 0:
        raise ValueError(
            f"No reference code found with pattern '{reference_pattern}' in {folder_path}"
        )
    elif len(reference_files) > 1:
        reference_paths = [str(f) for f in reference_files]
        raise ValueError(
            f"Found multiple reference codes in {folder_path}: {reference_paths}"
        )

    reference_code_path = str(reference_files[0])

    # Search for generated code files
    generated_files = []

    # Pattern 1: *generated*.py
    generated_pattern1 = "*generated*.py"
    generated_files.extend(folder.glob(generated_pattern1))

    # Pattern 2: kernel*.py
    generated_pattern2 = "*kernel*.py"
    generated_files.extend(folder.glob(generated_pattern2))

    # Remove duplicates while preserving order
    seen = set()
    unique_generated_files = []
    for f in generated_files:
        if f not in seen:
            seen.add(f)
            unique_generated_files.append(f)

    # Convert to string paths
    generated_code_paths = [str(f) for f in unique_generated_files]

    return reference_code_path, generated_code_paths


async def process_codes(reference_code_path, generated_code_path, generated_only=False):
    """Process a pair of reference and generated code files"""
    kbeval_client = KbEvalClient(config_file=CONFIG_FILE)

    # Read code files
    reference_code = read_code_file(reference_code_path)
    generated_code = read_code_file(generated_code_path)

    # Run evaluation with retry logic
    max_retries = 3
    eval_results = {}
    eval_results_ref = {}
    # Create a hash of the generated_code to use as eval_tag
    eval_tag = hashlib.sha256(generated_code.encode("utf-8")).hexdigest()

    # generate output filename for reference
    reference_output_path = get_output_filename(reference_code_path)
    ensure_output_dir()
    # Generate output filename
    output_path = get_output_filename(generated_code_path)
    if os.path.exists(output_path) and OVERWRITE is False:
        print(f"Output file already exists: {output_path}")
        return None


    for attempt in range(1, max_retries + 1):
        try:
            eval_results = await kbeval_client.kb_eval(
                PROVIDER,
                reference_code=reference_code,
                generated_code=generated_code,
                run_tag="re_kbeval",
                eval_tag=eval_tag,
                code_type=CODE_TYPE,
            )
            break  # Success, exit loop
        except Exception as e:
            if attempt == max_retries:
                print(f"Evaluation failed after {max_retries} attempts: {e}")
                raise
            else:
                print(f"Attempt {attempt} failed with error: {e}. Retrying...")

    if len(eval_results) == 0:
        print(f"Evaluation failed for {generated_code_path}")
        return None
    # Ensure output directory exists
    ensure_output_dir()


    # Save results as JSON
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Results saved to: {output_path}")

    if generated_only is True:
        return eval_results

    # evaluation for reference
    for attempt in range(1, max_retries + 1):
        try:
            task_tag = reference_code_path.split("/")[-2]
            eval_results_ref = await kbeval_client.kb_eval_ref(
                PROVIDER,
                reference_code=reference_code,
                run_tag="re_kbeval",
                task_tag=task_tag,
            )
            break  # Success, exit loop
        except Exception as e:
            if attempt == max_retries:
                print(f"Evaluation failed after {max_retries} attempts: {e}")
                raise
            else:
                print(f"Attempt {attempt} failed with error: {e}. Retrying...")

    with open(reference_output_path, "w") as f:
        json.dump(eval_results_ref, f, indent=2)
    print(f"Results saved to: {reference_output_path}")

    return eval_results


async def process_all_pairs(file_pairs, max_concurrent=100):
    """
    Process all file pairs with a limit on concurrent tasks.

    Args:
        file_pairs: List of (reference_code_path, generated_code_path) tuples
        max_concurrent: Maximum number of concurrent tasks

    Returns:
        List of evaluation results
    """
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(ref_path, gen_path, generated_only):
        async with semaphore:
            return await process_codes(ref_path, gen_path, generated_only)

    # Create tasks for all file pairs
    tasks = [
        process_with_semaphore(ref_path, gen_path, generated_only) for ref_path, gen_path, generated_only in file_pairs
    ]

    # Run all tasks and gather results
    total_tasks = len(tasks)
    print(f"Starting processing of {total_tasks} tasks...")

    # Process tasks and track progress
    filtered_results = []
    for i, task in enumerate(asyncio.as_completed(tasks)):
        try:
            result = await task
            filtered_results.append(result)
        except Exception as e:
            # Find which file pair caused the exception
            # This is approximate since we can't directly map as_completed results back to inputs
            print(f"Error processing task: {e}")

        # Print progress update every 5 tasks or when all tasks are done
        if (i + 1) % 200 == 0 or (i + 1) == total_tasks:
            print(
                f"Progress: {i + 1}/{total_tasks} tasks completed ({(i + 1)/total_tasks*100:.1f}%)"
            )

    print(f"All {total_tasks} tasks processed. {len(filtered_results)} succeeded.")
    return filtered_results


# Example usage:
if __name__ == "__main__":
    # To run with default paths:
    file_pairs = []

    claude_run_folder = "shared/claude_run/0205_l1"
    level = 1
    all_tasks = [os.path.join(claude_run_folder, f_) for f_ in os.listdir(claude_run_folder) if "md" not in f_ and "json" not in f_]

    reference_folder = f"kernel_bench/level{level}"
    for task in all_tasks:
        reference_file = os.path.join(reference_folder, os.path.basename(task) + ".py")
        for i, generated_code in enumerate([_ for _ in os.listdir(task) if "cuda_kernel" in _]):
            generated_file = os.path.join(task, generated_code)
            if i == 0:
                file_pairs.append([reference_file, generated_file, True])
            else:
                file_pairs.append([reference_file, generated_file, False])


    claude_run_folder = "shared/claude_run/0205_l2"
    level = 2
    all_tasks = [os.path.join(claude_run_folder, f_) for f_ in os.listdir(claude_run_folder) if "md" not in f_ and "json" not in f_]

    reference_folder = f"kernel_bench/level{level}"
    for task in all_tasks:
        reference_file = os.path.join(reference_folder, os.path.basename(task) + ".py")
        for i, generated_code in enumerate([_ for _ in os.listdir(task) if "cuda_kernel" in _]):
            generated_file = os.path.join(task, generated_code)
            if i == 0:
                file_pairs.append([reference_file, generated_file, True])
            else:
                file_pairs.append([reference_file, generated_file, False])


    print(f"Total pairs: {len(file_pairs)}")
    # print(file_pairs[0:10])
    asyncio.run(process_all_pairs(file_pairs, max_concurrent=MAX_CONCURRENT))
    print("All evaluations completed!")
