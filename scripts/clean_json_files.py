#!/usr/bin/env python3
"""
Script to recursively find JSON files in a folder and delete empty or null ones in parallel.
"""

import argparse
import json
import logging
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def is_json_empty_or_null(file_path: Path) -> bool:
    """
    Check if a JSON file is empty, contains only null, or has CUDA processing errors.

    Args:
        file_path: Path to the JSON file

    Returns:
        True if file should be deleted (empty, null, or has CUDA errors), False otherwise
    """
    try:
        if file_path.stat().st_size == 0:
            return True

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return True

        try:
            parsed = json.loads(content)
            if parsed is None:
                return True

            # Check for CUDA unknown error in metadata
            if isinstance(parsed, dict) and "metadata" in parsed:
                metadata = parsed["metadata"]
                if isinstance(metadata, dict) and "processing_error" in metadata:
                    processing_error = metadata["processing_error"]
                    if (
                        isinstance(processing_error, str)
                        and "CUDA unknown error" in processing_error
                    ):
                        return True

            return False
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON format in {file_path}, keeping file")
            return False

    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return False


def process_json_file(file_path: Path) -> Tuple[Path, bool, str]:
    """
    Process a single JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Tuple of (file_path, was_deleted, status_message)
    """
    try:
        if is_json_empty_or_null(file_path):
            file_path.unlink()
            return file_path, True, "Deleted (empty or null)"
        else:
            return file_path, False, "Kept (contains data)"
    except Exception as e:
        return file_path, False, f"Error: {e}"


def find_json_files(folder_path: Path) -> List[Path]:
    """
    Recursively find all JSON files in the given folder.

    Args:
        folder_path: Path to the folder to search

    Returns:
        List of paths to JSON files
    """
    return list(folder_path.rglob("*.json"))


def clean_json_files_parallel(
    folder_path: Path, max_workers: int | None = None
) -> None:
    """
    Clean JSON files in parallel.

    Args:
        folder_path: Path to the folder to process
        max_workers: Maximum number of worker threads
    """
    if not folder_path.exists():
        logging.error(f"Folder {folder_path} does not exist")
        return

    if not folder_path.is_dir():
        logging.error(f"{folder_path} is not a directory")
        return

    logging.info(f"Searching for JSON files in {folder_path}")
    json_files = find_json_files(folder_path)

    if not json_files:
        logging.info("No JSON files found")
        return

    logging.info(f"Found {len(json_files)} JSON files")

    deleted_count = 0
    kept_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_json_file, file_path): file_path
            for file_path in json_files
        }

        for future in as_completed(future_to_file):
            file_path, was_deleted, status = future.result()

            if was_deleted:
                deleted_count += 1
                logging.info(f"Deleted: {file_path}")
            elif "Error:" in status:
                error_count += 1
                logging.error(f"Error processing {file_path}: {status}")
            else:
                kept_count += 1
                logging.debug(f"Kept: {file_path}")

    logging.info(f"Processing complete:")
    logging.info(f"  - Files deleted: {deleted_count}")
    logging.info(f"  - Files kept: {kept_count}")
    logging.info(f"  - Errors: {error_count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Recursively find JSON files and delete empty or null ones in parallel"
    )
    parser.add_argument("--folder", type=str, help="Path to the folder to process")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Maximum number of worker threads (default: None, uses system default)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_logging()

    folder_path = Path(args.folder).resolve()
    clean_json_files_parallel(folder_path, args.max_workers)


if __name__ == "__main__":
    main()
