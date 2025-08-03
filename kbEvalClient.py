import argparse
import asyncio
import requests
import sys
import traceback
from fastapi import FastAPI
from kbEvalTest.kbeval import eval_kernel_against_ref
import os
import json
from pathlib import Path
import time
from datetime import datetime
import yaml
import httpx
from kbEvalTest.kbeval import KernelExecResult
from logger import logger
from typing import Dict

# reusable client for calling kbEvalRemoteServer
class KbEvalClient:
    """Client for calling kbEvalRemoteServer to evaluate generated code."""

    def __init__(self, config_file: str = "kbEval.yaml"):
        """Initialize the client with configuration"""
        self.config = self._load_config(config_file)
        # Get kbEval config from kbEval.yaml
        kb_eval_config = self.config.get('kbEvalClient', {})
        self.base_url = kb_eval_config.get('servers', [])[0].get('url', 'http://localhost:44456')
        self.timeout = kb_eval_config.get('servers', [])[0].get('timeout', 450)
        self.num_retries = kb_eval_config.get('servers', [])[0].get('num_retries', 7)
        self.server_last_refresh_time = time.time()
        self.server_stats = {}
        self.kb_eval_config = kb_eval_config
        # expand the api_key_file
        api_key_file = kb_eval_config.get('servers', [])[0].get('api_key', '~/.keys/kbeval.api.key').replace("${HOME}", os.path.expanduser("~")).replace("~", os.path.expanduser("~"))
        # read the api_key from the file
        with open(api_key_file, 'r') as f:
            self.api_key = f.read().strip()
            logger.info(f"🔑 [kbEvalRemoteCli] API key loaded from [{api_key_file}]")

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            # Try relative to script directory
            config_path = Path(__file__).parent / config_file

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # use file emoji
                logger.info(f"📁 [kbEvalClient] Config file [{config_file}] loaded")
                return config or {}
        else:
            logger.warning(f"⚠️ [kbEvalClient] Warning: Config file [{config_file}] not found")
            return {}

    async def kb_eval_ref(
        self,
        run_tag: str="auto",
        model_tag: str="model_tag",
        task_tag: str="task_tag",
        reference_code: str="reference_code",
    ) -> KernelExecResult:
        """Call the kbEvalRemoteServer with evaluation parameters"""
        if len(self.kb_eval_config) > 0 and len(self.kb_eval_config["servers"]) > 1:
            self.pick_server()
        run_tag = run_tag if run_tag != "auto" else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"{self.base_url}/kb_eval_ref",
                        json={
                            "run_tag": run_tag,
                            "model_tag": model_tag,
                            "task_tag": task_tag,
                            "reference_code": reference_code,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout  # 5 minute timeout
                    )

                    response.raise_for_status()

                    result = KernelExecResult(**response.json())
                    if (not result.compiled or not result.correctness) and "retriable" in result.metadata and result.metadata["retriable"]:
                        sleep_seconds = 2 ** retry_count
                        if retry_count < self.num_retries:
                            # retry_count -= 0.5 # reduce retry count by 0.5 to avoid infinite loop
                            logger.warning(f"⚠️ [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] Retriable error, retrying... [{retry_count}/{self.num_retries}] in [{sleep_seconds}s]")
                            await asyncio.sleep(sleep_seconds)
                            continue
                        else:
                            logger.warning(f"⚠️ [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] Return the last result from retriable error... [{retry_count}/{self.num_retries}]")
                            return result

                    return result

            except Exception as e:
                logger.warning(f"🔍 [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] Error calling server: [{e}] [{retry_count}/{self.num_retries}]")
                if retry_count < self.num_retries:
                    sleep_seconds = 2 ** retry_count
                    logger.info(f"🔄 [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] Retrying in {sleep_seconds} seconds... ({retry_count}/{self.num_retries})")
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(f"❌ [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] Failed after {retry_count} retries")
                    return None

    async def kb_eval(
        self,
        run_tag: str="auto",
        model_tag: str="model_tag",
        task_tag: str="task_tag",
        eval_tag: str="eval_tag",
        reference_code: str="reference_code",
        generated_code: str="generated_code",
        code_type: str="cuda",
    ) -> KernelExecResult:
        """Call the kbEvalRemoteServer with evaluation parameters"""
        if len(self.kb_eval_config) > 0 and len(self.kb_eval_config["servers"]) > 1:
            self.pick_server()
        run_tag = run_tag if run_tag != "auto" else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"{self.base_url}/kb_eval",
                        json={
                            "run_tag": run_tag,
                            "model_tag": model_tag,
                            "task_tag": task_tag,
                            "eval_tag": eval_tag,
                            "reference_code": reference_code,
                            "generated_code": generated_code,
                            "code_type": code_type,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout  # 5 minute timeout
                    )

                    response.raise_for_status()

                    result = KernelExecResult(**response.json())
                    if (not result.compiled or not result.correctness) and "retriable" in result.metadata and result.metadata["retriable"]:
                        sleep_seconds = 2 ** retry_count
                        if retry_count < self.num_retries:
                            # retry_count -= 0.5 # reduce retry count by 0.5 to avoid infinite loop
                            logger.warning(f"⚠️ [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Retriable error, retrying... [{retry_count}/{self.num_retries}] in [{sleep_seconds}s]")
                            await asyncio.sleep(sleep_seconds)
                            continue
                        else:
                            logger.warning(f"⚠️ [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Return the last result from retriable error... [{retry_count}/{self.num_retries}]")
                            return result

                    return result

            except Exception as e:
                # add retry emoji to beginning and end of the string
                logger.warning(f"⚠️ [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Error calling server: [{type(e).__name__}: {str(e)}] [{retry_count}/{self.num_retries}]")
                if retry_count < self.num_retries:
                    sleep_seconds = 2 ** retry_count
                    logger.info(f"🔄 [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Retrying in {sleep_seconds} seconds... ({retry_count}/{self.num_retries})") # no emoji
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(f"❌ [kbEvalClient] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Failed after {retry_count} retries")
                    return None

    def get_server_stats(self):
        time_now = time.time()
        kb_eval_config = self.config.get("kbEvalClient", {})
        if "servers" not in kb_eval_config:
            return {}
        self.server_stats = {}
        if len(self.server_stats) == 0 or time_now - self.server_last_refresh_time > 10:
            for server in kb_eval_config["servers"]:
                try:
                    response = requests.get(f"{server['url']}/stats")
                    self.server_stats[server["url"]] = response.json()
                except:
                    pass

    def pick_server(self):
        self.get_server_stats()
        min_avg_load = float("inf")
        min_avg_load_server = None
        for server in self.server_stats:
            if self.server_stats[server]["pending_requests"] / self.server_stats[server]["num_devices"] < min_avg_load:
                min_avg_load = self.server_stats[server]["pending_requests"] / self.server_stats[server]["num_devices"]
                min_avg_load_server = server
        logger.info(f"[kbEvalClient] choose {min_avg_load_server}")
        self.base_url = min_avg_load_server


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str, default="./kbEvalTest")
    parser.add_argument("--run_tag", type=str, default="auto")
    parser.add_argument("--model_tag", type=str, default="model_tag")
    parser.add_argument("--task_tag", type=str, default="task_tag")
    parser.add_argument("--eval_tag", type=str, default="eval_tag")
    parser.add_argument("--reference_code", type=str, default="elemAddRef.py")
    parser.add_argument("--generated_code", type=str, default="elemAddCuda.py")
    parser.add_argument("--code_type", type=str, default="cuda")
    parser.add_argument("--measure_reference", action="store_true")
    args = parser.parse_args()

    # read from file
    reference_model_src = open(os.path.join(args.wd, args.reference_code), "r").read()
    generated_model_src = open(os.path.join(args.wd, args.generated_code), "r").read()

    client = KbEvalClient()

    if args.measure_reference:
        result = await client.kb_eval_ref(run_tag=args.run_tag, model_tag=args.model_tag, task_tag=args.task_tag, reference_code=reference_model_src)
        logger.info(f"🔍 [kbEvalClient] [{args.run_tag}] [{args.model_tag}] [{args.task_tag}] Reference code evaluation result: {json.dumps(result.model_dump() if result else None, indent=4)}")
    else:
        result = await client.kb_eval(run_tag=args.run_tag, model_tag=args.model_tag, task_tag=args.task_tag, eval_tag=args.eval_tag, reference_code=reference_model_src, generated_code=generated_model_src, code_type=args.code_type)
        logger.info(f"🔍 [kbEvalClient] [{args.run_tag}] [{args.model_tag}] [{args.task_tag}] [{args.eval_tag}] Generated code evaluation result: {json.dumps(result.model_dump() if result else None, indent=4)}")

if __name__ == "__main__":
    asyncio.run(main())
