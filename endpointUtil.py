import os
import json
import re
import asyncio
import fcntl
import time
import psutil
from pathlib import Path
from logger import logger
from typing import Any

VALID_RECORD_FORMATS = ["json", "text"]

MAX_LOCK_AGE = 20 # 20 seconds

class Recorder:
    def __init__(self):
        pass
    
    def save(self, path: str, data: Any, format: str = "json"):
        if format not in VALID_RECORD_FORMATS:
            raise ValueError(f"Invalid format: {format}, must be one of {VALID_RECORD_FORMATS}")
        # check that the folder exists
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        # save the data to the file
        if format == "json":
            with open(path, 'w') as f:
                f.write(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        elif format == "text":
            with open(path, 'w') as f:
                f.write(data)
        else:
            raise ValueError(f"Invalid format: {format}, must be one of {VALID_RECORD_FORMATS}")

class CodeExtractor:
    def __init__(self):
        pass
    
    def extract_code(self, completion: str) -> str:
        # first extract <think>...</think> block
        think_blocks = re.findall(r"(<think>.*?</think>)", completion, re.DOTALL)
        for think_block in think_blocks:
            completion = completion.replace(think_block, "")
        # then extract ```python block
        code_blocks = re.findall(r"```python\n(.*?)\n```", completion, re.DOTALL)
        # if not found, try ``` block
        if not code_blocks:
            code_blocks = re.findall(r"```\n(.*?)\n```", completion, re.DOTALL)
        # if still not found, use the whole completion
        if not code_blocks:
            code_blocks = [completion]
        return {
            "code": code_blocks[0],
            "reasoning": '\n'.join(think_blocks)
        }


class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock_fd = None
        self.pid = os.getpid()

    def __enter__(self):
        try:
            # if open file with 'w', it will change modified timestamp even without writing to the file
            self.lock_fd = open(self.lock_file, 'r+')
        except FileNotFoundError:
             # if file does not exist, open file with 'w'
            self.lock_fd = open(self.lock_file, 'w')
        try:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # write my pid to lock file
            self.lock_fd.truncate(0)
            self.lock_fd.write(str(self.pid) + "\n")
            self.lock_fd.flush()
            # os.fsync(self.lock_fd.fileno())
        except BlockingIOError as e:
            self.lock_fd.close()
            raise TimeoutError("Could not acquire lock")
        logger.debug(f"Lock [{self.lock_file}] acquired.")
        return self

    def __exit__(self, type, value, traceback):
        if self.lock_fd:
            self.lock_fd.write('\n[done]\n')
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            logger.debug(f"Lock [{self.lock_file}] released.")

def cleanup_lockfile(lock_file: str):
    if os.path.exists(lock_file):
        my_pid = os.getpid()
        lock_modified_time = os.path.getmtime(lock_file)
        with open(lock_file, 'r') as file:
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                first_line = file.readline().strip()
                digits_only = ""
                for c in first_line:
                    if c.isdigit():
                        digits_only += c
                if digits_only:
                    file_pid = int(digits_only)
                    if my_pid != file_pid:
                        if psutil.pid_exists(file_pid):
                            logger.warning(f"[{my_pid}] Lock file [{lock_file}] for [pid={digits_only}] is running...")
                        else:
                            # if process is not running
                            logger.error(f"[{my_pid}] Lock file [{lock_file}] [pid={digits_only}] is not running, deleting...")
                            os.remove(lock_file)
            except BlockingIOError:
                if lock_modified_time < time.time() - MAX_LOCK_AGE:
                    # safety net: if modified time is more than MAX_LOCK_AGE, delete lock file
                    logger.error(f"[{my_pid}] Lock file [{lock_file}] older than [{MAX_LOCK_AGE}s], deleting...")
                    try:
                        os.remove(lock_file)
                    except Exception as e:
                        pass

REFERENCE_CATEGORY = "reference"
class StatsClient:
    def __init__(self, stats_dir: str = "~/.trainer/stats"):
        self.stats_dir = Path(os.path.expanduser(stats_dir))
        os.makedirs(self.stats_dir, exist_ok=True)

    async def add_data_point(self, prefix_tag: str, model_tag: str, task_tag: str, category: str, data: dict):
        if category == REFERENCE_CATEGORY:
            # if category is reference, ignore model_tag
            file_path = os.path.join(
                self.stats_dir,
                prefix_tag,
                REFERENCE_CATEGORY,
                f"{task_tag}",
                f"{REFERENCE_CATEGORY}.jsonl"
            )
        else:
            # save the data to the file
            file_path = os.path.join(
                self.stats_dir,
                prefix_tag,
                f"{model_tag}",
                f"{task_tag}",
                f"{category}.jsonl"
            )
        dirname = os.path.dirname(file_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        lock_file = file_path + ".lock"
        # lock the file first
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                with FileLock(lock_file):
                    with open(file_path, 'a+') as f:
                        f.write(json.dumps(data) + "\n")
                    break
            except TimeoutError:
                if retry_count == max_retries - 1:
                    logger.error(f"[{os.getpid()}] Error adding data point to [{file_path}] after {max_retries} retries")
                    break
                else:
                    await asyncio.sleep(2 ** retry_count) # sleep for 2^retry_count seconds
                    continue
            finally:
                retry_count += 1
                cleanup_lockfile(lock_file)
    
    async def wait_for_stats(self, prefix_tag: str, model_tag: str, task_tag: str, category: str, timeout: int = 120, last_n_lines: int = 100) -> dict:
        if category == REFERENCE_CATEGORY:
            # if category is reference, ignore model_tag
            file_path = os.path.join(
                self.stats_dir,
                prefix_tag,
                REFERENCE_CATEGORY,
                f"{task_tag}",
                f"{REFERENCE_CATEGORY}.jsonl"
            )
        else:
            # save the data to the file
            file_path = os.path.join(
                self.stats_dir,
                prefix_tag,
                f"{model_tag}",
                f"{task_tag}",
                f"{category}.jsonl"
            )
        # wait until the file exists
        start_time = time.time()
        runtime_list = []
        while True:
            try:
                if not os.path.exists(file_path):
                    continue
                # read the last 50 lines
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    lines = lines[-last_n_lines:]
                # parse the lines and get 'runtime'
                last_row = None
                for line in lines:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"[{os.getpid()}] Ignore invalid JSON line in [{file_path}]: [{line}]")
                        continue
                    runtime_list.append(data['runtime'])
                    last_row = data
                # break if there is at least one valid runtime
                if len(runtime_list) > 0:
                    break
            except Exception as e:
                logger.error(f"[{os.getpid()}] Error waiting for [{file_path}]: [{e}]")
                await asyncio.sleep(1)
                continue
            finally:
                await asyncio.sleep(0.1)
                if time.time() - start_time > timeout:
                    logger.error(f"[{os.getpid()}] Timeout waiting for [{file_path}] after [{timeout}s]")
                    raise TimeoutError(f"Timeout waiting for [{file_path}] after [{timeout}s]")

        # calculate 25th, 50th, 75th percentile
        runtime_list.sort()
        percentile_stats = {
            'count': len(runtime_list),
            'min': runtime_list[0],
            'max': runtime_list[-1],
            'mean': sum(runtime_list) / len(runtime_list),
            '25th': runtime_list[int(len(runtime_list) * 0.25)],
            '50th': runtime_list[int(len(runtime_list) * 0.5)],
            '75th': runtime_list[int(len(runtime_list) * 0.75)]
        }
        last_row['runtime'] = percentile_stats['50th']
        last_row['metadata']['stats'] = percentile_stats
        return last_row

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="test")
    parser.add_argument("--model_tag", type=str, default="fireworks_deepseek-v3")
    parser.add_argument("--task_tag", type=str, default="level1_24_LogSoftmax.py")
    parser.add_argument("--category", type=str, default="reference")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--last_n_lines", type=int, default=100)
    args = parser.parse_args()

    stats_client = StatsClient()
    result = asyncio.run(stats_client.wait_for_reference_stats(
        prefix_tag=args.prefix_tag,
        model_tag=args.model_tag,
        task_tag=args.task_tag,
        category=args.category,
        timeout=args.timeout,
        last_n_lines=args.last_n_lines
    ))
    logger.info(json.dumps(result, indent=2))