import os
import json
import re
import asyncio
import fcntl
import time
import httpx
import yaml
import psutil
from pathlib import Path
from logger import logger
from typing import Any
import duckdb

VALID_RECORD_FORMATS = ["json", "text"]

MAX_LOCK_AGE = 20 # 20 seconds

class DuckDBClient:
    def __init__(self):
        pass
    
    def sql(self, query: str) -> Any:
        result = duckdb.sql(query)
        return result

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
            "code": code_blocks[-1],
            "reasoning": '\n'.join(think_blocks)
        }

class VLLMClient:
    def __init__(self, short_hostname: str = "two"):
        self.short_hostname = short_hostname
        self.vllm_config = self.load_config().get('vllm_clients', {}).get(short_hostname, {})
        api_key_path = os.path.expanduser(self.vllm_config.get('api_key_path', '~/.keys/local.api.key'))
        with open(api_key_path, 'r') as f:
            self.api_key = f.read().strip()
        self.timeout = self.vllm_config.get('timeout', 60)
        self.retries = self.vllm_config.get('retries', 3)

    def load_config(self, config_path: str = "configEndpoints.yaml"):
        """Load config from yaml file"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    async def load_lora_adapter(self, lora_name: str, lora_path: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"http://{self.host}:{self.port}/v1/load_lora_adapter",
                        json={
                            "lora_name": lora_name,
                            "lora_path": lora_path,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    logger.warning(f"🔍 [VLLMClient] Response: {response.text}") # use text instead of json
                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Loaded lora adapter from [{lora_path}]")
                    return
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error loading lora adapter from [{lora_path}]: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to load lora adapter from [{lora_path}] in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to load lora adapter from [{lora_path}] after {self.retries} retries")
                    raise e

    async def unload_lora_adapter(self, lora_name: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"http://{self.host}:{self.port}/v1/unload_lora_adapter",
                        json={"lora_name": lora_name},
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    logger.warning(f"🔍 [VLLMClient] Response: {response.text}") # use text instead of json
                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Unloaded lora adapter from [{lora_name}]")
                    return
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error unloading lora adapter from [{lora_name}]: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to unload lora adapter from [{lora_name}] in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to unload lora adapter from [{lora_name}] after {self.retries} retries")
                    raise e

    async def get_models(self):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.get(
                        f"http://{self.host}:{self.port}/v1/models",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Got models")
                    return response.json()
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error getting models: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to get models in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to get models after {self.retries} retries")
                    raise e

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
    
    async def wait_for_stats(self, prefix_tag: str, model_tag: str, task_tag: str, category: str, timeout: int = 90, last_n_lines: int = 50, return_percentile: str = "25th") -> dict:
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
        last_row = None
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
                for line in lines:
                    try:
                        data = json.loads(line)
                        if data is None:
                            logger.warning(f"[{os.getpid()}] Ignore null JSON line in [{file_path}]: [{line}]")
                            continue
                        runtime_list.append(data['runtime'])
                        last_row = data
                    except json.JSONDecodeError:
                        logger.warning(f"[{os.getpid()}] Ignore invalid JSON line in [{file_path}]: [{line}]")
                        continue
                    except Exception as e:
                        logger.warning(f"[{os.getpid()}] Ignore invalid JSON line in [{file_path}]: [{line}] [{type(e).__name__}: {e}]")
                        continue
                # break if there is at least one valid runtime
                if len(runtime_list) > 0 and last_row is not None:
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
        if return_percentile not in percentile_stats:
            logger.error(f"[{os.getpid()}] Invalid return percentile: [{return_percentile}], must be one of {list(percentile_stats.keys())}")
            return_percentile = "25th"
        last_row['runtime'] = percentile_stats[return_percentile]
        last_row['metadata']['stats'] = percentile_stats
        return last_row

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="TC_0.1.0_14B.m")
    parser.add_argument("--model_tag", type=str, default="fireworks_deepseek-v3")
    parser.add_argument("--task_tag", type=str, default="level1_24_LogSoftmax.py")
    parser.add_argument("--category", type=str, default="reference")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--last_n_lines", type=int, default=50)
    args = parser.parse_args()

    stats_client = StatsClient()
    result = asyncio.run(stats_client.wait_for_stats(
        prefix_tag=args.prefix_tag,
        model_tag=args.model_tag,
        task_tag=args.task_tag,
        category=args.category,
        timeout=args.timeout,
        last_n_lines=args.last_n_lines
    ))
    logger.info(json.dumps(result, indent=2))