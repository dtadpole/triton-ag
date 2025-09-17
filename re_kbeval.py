import asyncio
import glob
import json
import os
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from kbEvalClient import KbEvalClient


CONFIG_FILE = "kbEval.yaml"
PROVIDER = "h8_1"
OUTPUT_DIR = "shared/re_kbeval"


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
    output_relative = relative_path.with_suffix('.json')
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


async def process_codes(reference_code_path, generated_code_path):
    """Process a pair of reference and generated code files"""
    kbeval_client = KbEvalClient(config_file=CONFIG_FILE)

    # Read code files
    reference_code = read_code_file(reference_code_path)
    generated_code = read_code_file(generated_code_path)

    # Run evaluation with retry logic
    max_retries = 3
    eval_results = {}
    # Create a hash of the generated_code to use as eval_tag
    eval_tag = hashlib.sha256(generated_code.encode("utf-8")).hexdigest()

    for attempt in range(1, max_retries + 1):
        try:
            eval_results = await kbeval_client.kb_eval(
                [PROVIDER], reference_code=reference_code, generated_code=generated_code, run_tag="re_kbeval", eval_tag=eval_tag,
            )
            break  # Success, exit loop
        except Exception as e:
            if attempt == max_retries:
                print(f"Evaluation failed after {max_retries} attempts: {e}")
                raise
            else:
                print(f"Attempt {attempt} failed with error: {e}. Retrying...")

    # Ensure output directory exists
    ensure_output_dir()

    # Generate output filename
    output_path = get_output_filename(generated_code_path)

    # Save results as JSON
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Results saved to: {output_path}")
    return eval_results


# async def main(reference_code_path=None, generated_code_path=None):
#     """Main function that can be called with specific paths or use defaults"""
#     if reference_code_path is None:
#         reference_code_path = "shared/.inference/codeGenEval/cudacoder_eval_4_turn.qwen32b_repeat_000_00/qwen3-32b/level2_14_Gemm_Divide_Sum/reference_code.py"
#     if generated_code_path is None:
#         generated_code_path = "shared/.inference/codeGenEval/cudacoder_eval_4_turn.qwen32b_repeat_000_00/qwen3-32b/level2_14_Gemm_Divide_Sum/gen_03_t00_generated_code.py"

#     return await process_codes(reference_code_path, generated_code_path)


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

    async def process_with_semaphore(ref_path, gen_path):
        async with semaphore:
            return await process_codes(ref_path, gen_path)

    # Create tasks for all file pairs
    tasks = [
        process_with_semaphore(ref_path, gen_path)
        for ref_path, gen_path in file_pairs
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
            print(f"Progress: {i + 1}/{total_tasks} tasks completed ({(i + 1)/total_tasks*100:.1f}%)")

    print(f"All {total_tasks} tasks processed. {len(filtered_results)} succeeded.")
    return filtered_results


# Example usage:
if __name__ == "__main__":
    # To run with default paths:
    # results = asyncio.run(main())
    # print("Evaluation completed!")

    # reference_file, generated_files = get_reference_and_generated("shared/.inference/codeGenEval/cudacoder_eval_4_turn.qwen32b_repeat_000_00/qwen3-32b/level2_14_Gemm_Divide_Sum")
    # print(f"Reference file: {reference_file}")
    # print(f"Generated files: {generated_files}")
    # print("Done!")
    file_pairs = []
    for i in range(1, 13):
        test_folders = "shared/.inference/codeGenEval/cudacoder_eval_4_turn.qwen32b_000_%02d"%(i)
        foler_level1 = [os.path.join(test_folders,f_) for f_ in  os.listdir(test_folders)]
        for folder_ in foler_level1:
            folder_level2 = [os.path.join(folder_,f_) for f_ in  os.listdir(folder_)]
            for folder_2 in folder_level2:
                reference_file, generated_files = get_reference_and_generated(folder_2)
                file_pairs.extend([(reference_file, generated_file) for generated_file in generated_files])

    r1_folder1 = "shared/deepseek/deepseek-reasoner_2025_07_21_h03"
    r1_folder2 = "shared/deepseek/deepseek-reasoner_2025_07_26_h16"
    r1_folder1_folders = [os.path.join(r1_folder1, f_ + "/current/") for f_ in  os.listdir(r1_folder1)]
    for folder_ in r1_folder1_folders:
        reference_file, generated_files = get_reference_and_generated(folder_)
        file_pairs.extend([(reference_file, generated_file) for generated_file in generated_files])
    r1_folder2_folders = [os.path.join(r1_folder2, f_ + "/current/") for f_ in  os.listdir(r1_folder2)]
    for folder_ in r1_folder2_folders:
        reference_file, generated_files = get_reference_and_generated(folder_)
        file_pairs.extend([(reference_file, generated_file) for generated_file in generated_files])

    print(f"Total pairs: {len(file_pairs)}")

    asyncio.run(process_all_pairs(file_pairs, max_concurrent=150))
    print("All evaluations completed!")
