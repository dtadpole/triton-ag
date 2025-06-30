#!/usr/bin/env python3
"""
Simple script to upload Python code files to HuggingFace Hub as a dataset.

Reads all .py files recursively and creates a dataset with:
- task_tag: relative filepath from the starting folder
- reference_code: actual content of the Python code
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from logger import logger


def read_python_files(folder: str, init_folder: str = ".") -> List[Dict[str, Any]]:
    """
    Read Python files recursively and create dataset entries.
    
    Args:
        folder_path: Path to folder containing Python files
    
    Returns:
        List of dataset entries with task_tag and reference_code columns
    """
    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Folder {folder} does not exist")
    
    data = []
    
    # Read all .py files recursively, must check subfolders or softlinks if they are present
    for file_path in folder.rglob("*"):
        if file_path.is_dir() or file_path.is_symlink():
            # recursively read the subfolder
            data.extend(read_python_files(file_path, init_folder=init_folder))
        elif file_path.is_file() and file_path.suffix == ".py":
            try:
                # Read the Python file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                task_tag = str(file_path.relative_to(init_folder))
                # print(task_tag)
                level_id = int(task_tag.split("/")[0][len("level"):])
                source_id = int(task_tag.split("/")[1].split("_")[0])
                # Create dataset entry with only the required columns
                data.append({
                    'task_tag': task_tag,
                    'level_id': level_id,
                    'source_id': source_id,
                    'reference_code': code_content
                })
                        
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                continue
    
    # add info icon to the beginning of the message
    logger.info(f"💡 Processed {len(data)} Python files from [{folder}]")
    return data


def upload_to_huggingface(
    data: List[Dict[str, Any]],
    repo_name: str,
    hf_token: Optional[str] = None,
    private: bool = False
) -> str:
    """
    Upload processed data to HuggingFace Hub.
    
    Args:
        data: Processed data entries
        repo_name: HuggingFace repository name (format: username/dataset-name)
        hf_token: HuggingFace token (uses HF_TOKEN env var if None)
        private: Whether to create private repository
    
    Returns:
        URL of uploaded dataset
    """
    if not data:
        raise ValueError("No data to upload")
    
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN')
        if hf_token is None:
            # read from ~/.keys/huggingface.api.key
            with open(os.path.expanduser("~/.keys/huggingface.api.key"), "r") as f:
                hf_token = f.read().strip()
    
    # Create dataset
    dataset = Dataset.from_list(data)
    logger.info(f"Created dataset with {len(dataset)} entries")
    logger.info(f"Dataset columns: {dataset.column_names}")
    
    # Upload to HuggingFace
    dataset.push_to_hub(
        repo_name,
        token=hf_token,
        private=private
    )
    
    dataset_url = f"https://huggingface.co/datasets/{repo_name}"
    logger.info(f"✅ Successfully uploaded dataset to [{dataset_url}]!")

    return dataset_url


def main():
    """Main function to upload Python code dataset."""
    
    # Configuration
    FOLDER_PATH = "kernel_bench"  # Folder containing Python files
    REPO_NAME = "dtadpole/kernel-bench"
    PRIVATE = False
    
    # Validate folder exists
    if not Path(FOLDER_PATH).exists():
        logger.error(f"Folder {FOLDER_PATH} does not exist")
        return
    
    try:
        # Read Python files recursively
        logger.info(f"Reading Python files recursively from: {FOLDER_PATH}")
        data = read_python_files(FOLDER_PATH, init_folder=FOLDER_PATH)
        # sort data by level_id and source_id
        data.sort(key=lambda x: (x['level_id'], x['source_id']))
        
        if not data:
            logger.error("No Python files found to upload")
            return
        else:
            logger.info(f"Loaded {len(data)} files.")
        
        # Upload to HuggingFace
        logger.info(f"Uploading to HuggingFace: {REPO_NAME}")
        dataset_url = upload_to_huggingface(data, REPO_NAME, private=PRIVATE)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
