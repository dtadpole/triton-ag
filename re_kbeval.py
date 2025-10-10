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
PROVIDER = ["h8_4", "h8_2"]
OUTPUT_DIR = "shared/re_kbeval"
MAX_CONCURRENT = 50


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

    # for i in range(0, 13):
    #     test_folders = (
    #         "shared/.inference/codeGenEval/cudacoder_eval_4_turn.qwen32b_000_%02d"
    #         % (i)
    #     )
    #     foler_level1 = [
    #         os.path.join(test_folders, f_) for f_ in os.listdir(test_folders) if "deepseek-reasoner" in f_
    #     ]
    #     for folder_ in foler_level1:
    #         folder_level2 = [os.path.join(folder_, f_) for f_ in os.listdir(folder_)]
    #         for folder_2 in folder_level2:
    #             reference_file, generated_files = get_reference_and_generated(folder_2)
    #             current_paris = [
    #                     [reference_file, generated_file, True]
    #                     for generated_file in generated_files
    #                 ]
    #             current_paris[0][2] = False
    #             file_pairs.extend(
    #                 current_paris
    #             )

    # for i in range(0, 13):
    #     test_folders = (
    #         "shared/.inference/codeGenEval/cudacoder_eval_4_turn.qwen32b_repeat_000_%02d"
    #         % (i)
    #     )
    #     foler_level1 = [
    #         os.path.join(test_folders, f_) for f_ in os.listdir(test_folders) if "qwen3-32b" in f_
    #     ]
    #     for folder_ in foler_level1:
    #         folder_level2 = [os.path.join(folder_, f_) for f_ in os.listdir(folder_)]
    #         for folder_2 in folder_level2:
    #             reference_file, generated_files = get_reference_and_generated(folder_2)
    #             current_paris = [
    #                     [reference_file, generated_file, True]
    #                     for generated_file in generated_files
    #                 ]
    #             current_paris[0][2] = False
    #             file_pairs.extend(
    #                 current_paris
    #             )
    # for i in range(0, 13):
    #     test_folders = (
    #         "shared/.inference/codeGenEval/cudacoder_eval_4_turn.qwen32b_000_%02d"
    #         % (i)
    #     )
    #     foler_level1 = [
    #         os.path.join(test_folders, f_) for f_ in os.listdir(test_folders) if "gpt-oss-120b" in f_
    #     ]
    #     for folder_ in foler_level1:
    #         folder_level2 = [os.path.join(folder_, f_) for f_ in os.listdir(folder_)]
    #         for folder_2 in folder_level2:
    #             reference_file, generated_files = get_reference_and_generated(folder_2)
    #             current_paris = [
    #                     [reference_file, generated_file, True]
    #                     for generated_file in generated_files
    #                 ]
    #             current_paris[0][2] = False
    #             file_pairs.extend(
    #                 current_paris
    #             )
    # for i in range(0, 13):
    #     test_folders = (
    #         "shared/.inference/codeGenEval/cudacoder_eval_4_turn.qwen32b_repeat_000_%02d"
    #         % (i)
    #     )
    #     foler_level1 = [
    #         os.path.join(test_folders, f_) for f_ in os.listdir(test_folders) if "cudacoder_gspo_qwen32b_t03_ckpt2400" in f_
    #     ]
    #     for folder_ in foler_level1:
    #         folder_level2 = [os.path.join(folder_, f_) for f_ in os.listdir(folder_)]
    #         for folder_2 in folder_level2:
    #             reference_file, generated_files = get_reference_and_generated(folder_2)
    #             current_paris = [
    #                     [reference_file, generated_file, True]
    #                     for generated_file in generated_files
    #                 ]
    #             current_paris[0][2] = False
    #             file_pairs.extend(
    #                 current_paris
    #             )

    # r1_folder1 = "shared/deepseek/deepseek-reasoner_2025_07_21_h03"
    # r1_folder2 = "shared/deepseek/deepseek-reasoner_2025_07_26_h16"
    # r1_folder1_folders = [os.path.join(r1_folder1, f_ + "/current/") for f_ in  os.listdir(r1_folder1)]
    # for folder_ in r1_folder1_folders:
    #     reference_file, generated_files = get_reference_and_generated(folder_)
    #     file_pairs.extend([(reference_file, generated_file) for generated_file in generated_files])
    # r1_folder2_folders = [os.path.join(r1_folder2, f_ + "/current/") for f_ in  os.listdir(r1_folder2)]
    # for folder_ in r1_folder2_folders:
    #     reference_file, generated_files = get_reference_and_generated(folder_)
    #     file_pairs.extend([(reference_file, generated_file) for generated_file in generated_files])

    # folders_to_return = 'qwen3-32b/level1_17_Matmul_with_transposed,qwen3-32b/level1_18_Matmul_with_transposed,qwen3-32b/level1_1_Square_matrix_multiplication,qwen3-32b/level1_33_BatchNorm.py,qwen3-32b/level1_35_GroupNorm_.py,qwen3-32b/level1_36_RMSNorm_.py,qwen3-32b/level1_46_Average_Pooling_3D.py,qwen3-32b/level1_5_Matrix_scalar_multiplication.py,qwen3-32b/level1_7_Matmul_with_small,qwen3-32b/level1_80_conv_standard_2D,qwen3-32b/level1_9_Tall_skinny_matrix,qwen3-32b/level1_87_conv_pointwise_2D.py,qwen3-32b/level1_60_conv_standard_3D,qwen3-32b/level1_8_Matmul_with_irregular,qwen3-32b/level1_40_LayerNorm.py,qwen3-32b/level1_43_Max_Pooling_3D.py,qwen3-32b/level1_15_Matmul_for_lower,qwen3-32b/level1_13_Matmul_for_symmetric,qwen3-32b/level1_10_3D_tensor_matrix'.split(",")
    # folders_to_return = folders_to_return + 'deepseek-reasoner/level1_33_BatchNorm.py,deepseek-reasoner/level1_4_Matrix_vector_multiplication,deepseek-reasoner/level1_97_CosineSimilarityLoss.py,deepseek-reasoner/level1_91_cumsum_reverse.py,deepseek-reasoner/level1_40_LayerNorm.py,deepseek-reasoner/level1_12_Matmul_with_diagonal'.split(",")
    # folders_to_return = folders_to_return + 'deepseek-reasoner/level2_18_Matmul_Sum_Max,deepseek-reasoner/level2_24_Conv3d_Min_Softmax.py,deepseek-reasoner/level2_55_Matmul_MaxPool_Sum,deepseek-reasoner/level2_56_Matmul_Sigmoid_Sum.py,deepseek-reasoner/level2_9_Matmul_Subtract_Multiply,deepseek-reasoner/level2_69_Conv2d_HardSwish_ReLU.py,deepseek-reasoner/level2_64_Gemm_LogSumExp_LeakyReLU,deepseek-reasoner/level2_63_Gemm_ReLU_Divide.py,deepseek-reasoner/level2_45_Gemm_Sigmoid_Sum,deepseek-reasoner/level2_40_Matmul_Scaling_ResidualAdd.py,deepseek-reasoner/level2_12_Gemm_Multiply_LeakyReLU.py,deepseek-reasoner/level2_10_ConvTranspose2d_MaxPool_Hardtanh,deepseek-reasoner/level2_14_Gemm_Divide_Sum,deepseek-reasoner/level2_13_ConvTranspose3d_Mean_Add'.split(",")
    # folders_to_return = folders_to_return + 'qwen3-32b/level1_34_InstanceNorm.py,qwen3-32b/level1_53_Min_reduction_over,qwen3-32b/level1_6_Matmul_with_large,qwen3-32b/level1_3_Batched_matrix_multiplication.py,qwen3-32b/level1_42_Max_Pooling_2D.py,qwen3-32b/level1_45_Average_Pooling_2D.py'.split(",")
    # folders_to_return = folders_to_return + 'qwen3-32b/level2_59_Matmul_Swish_Scaling.py,qwen3-32b/level2_43_Conv3d_Max_LogSumExp,qwen3-32b/level2_13_ConvTranspose3d_Mean_Add'.split(",")
    # folders_to_return = folders_to_return + 'cudacoder_gspo_qwen32b_t03_ckpt2400/level1_12_Matmul_with_diagonal,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_11_4D_tensor_matrix,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_15_Matmul_for_lower,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_14_Matmul_for_upper,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_18_Matmul_with_transposed,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_59_conv_standard_3D,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_5_Matrix_scalar_multiplication.py,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_95_CrossEntropyLoss.py,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_77_conv_transposed_3D,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_33_BatchNorm.py,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_36_RMSNorm_.py,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_37_FrobeniusNorm_.py,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_66_conv_standard_3D,cudacoder_gspo_qwen32b_t03_ckpt2400/level1_40_LayerNorm.py'.split(",")
    # folders_to_return = folders_to_return + 'cudacoder_gspo_qwen32b_t03_ckpt2400/level2_18_Matmul_Sum_Max,cudacoder_gspo_qwen32b_t03_ckpt2400/level2_23_Conv3d_GroupNorm_Mean.py,cudacoder_gspo_qwen32b_t03_ckpt2400/level2_47_Conv3d_Mish_Tanh.py,cudacoder_gspo_qwen32b_t03_ckpt2400/level2_70_Gemm_Sigmoid_Scaling,cudacoder_gspo_qwen32b_t03_ckpt2400/level2_40_Matmul_Scaling_ResidualAdd.py'.split(",")

    # print(folders_to_return)

    # filtered_pairs = []
    # for pair in file_pairs:
    #     key =  "/".join(pair[0].split("/")[-3:-1])
    #     if key in folders_to_return:
    #         filtered_pairs.append(pair)
    # file_pairs = filtered_pairs

    df = pd.read_pickle("notebooks/error_255_t07.pkl")
    for r, g in df[["reference_code", "generated_code"]].values:
        file_pairs.append([r, g, True])

    print(f"Total pairs: {len(file_pairs)}")
    asyncio.run(process_all_pairs(file_pairs, max_concurrent=MAX_CONCURRENT))
    print("All evaluations completed!")
