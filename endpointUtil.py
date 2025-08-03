import os
import json
import asyncio
import fcntl
import time
import psutil
from pathlib import Path
from logger import logger
from typing import Any

VALID_RECORD_FORMATS = ["json", "txt"]

MAX_LOCK_AGE = 20 # 20 seconds

class Recorder:
    def __init__(self):
        pass
    
    async def save(self, path: str, data: Any, format: str = "json"):
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
        elif format == "txt":
            with open(path, 'w') as f:
                f.write(data)

class CodeExtractor:
    def __init__(self):
        pass
    
    async def extract_code(self, completion: str) -> str:
        return completion


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
        logger.info(f"Lock [{self.lock_file}] acquired.")
        return self

    def __exit__(self, type, value, traceback):
        if self.lock_fd:
            self.lock_fd.write('\n[done]\n')
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            logger.info(f"Lock [{self.lock_file}] released.")

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

class StatsClient:
    def __init__(self, stats_dir: str = "~/.trainer/stats"):
        self.stats_dir = Path(os.path.expanduser(stats_dir))
        os.makedirs(self.stats_dir, exist_ok=True)

    async def add_data_point(self, prefix_tag: str, model_tag: str, task_tag: str, category: str, data: dict):
        # save the data to the file
        file_path = os.path.join(
            self.stats_dir,
            prefix_tag,
            f"model_tag={model_tag}",
            f"task_tag={task_tag}",
            f"category={category}.jsonl"
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
    
    # TODO: add code to get stats
