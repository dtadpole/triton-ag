import argparse
import asyncio
import random
import sys
import traceback
import os
import json
from pathlib import Path
import time
from datetime import datetime
import yaml
import httpx
import gzip
from typing import Dict, Any, List
from logger import logger
from kbEvalUtil import KernelExecResult

# reusable client for calling kbEvalRemoteServer
class KbEvalClient:
    """Client for calling kbEvalRemoteServer to evaluate generated code."""

    def __init__(self, config_file: str = "kbEval.yaml"):
        """Initialize the client with configuration"""
        self.provider_config_cache = {}

        # Get kbEval config from kbEval.yaml
        self.server_last_refresh_time = time.time()
        self.server_stats = {}

    def _provider_config_from_yaml(
        self,
        provider_name: str,
        yaml_file: str="kbEval.yaml",
    ) -> Dict[str, Any]:
        """Load provider config from YAML file."""
        if provider_name in self.provider_config_cache:
            return self.provider_config_cache[provider_name]

        # if not found, load from yaml file
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        if provider_name not in config.get("providers", {}):
            raise ValueError(f"Provider [{provider_name}] not found in config file [{yaml_file}]")

        provider_config = config.get("providers", {}).get(provider_name, {})

        base_url = provider_config.get('base_url', 'http://localhost:8091')
        api_key_path = os.path.expanduser(provider_config.get('api_key_path', '~/.keys/local.api.key'))
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip() # read the api key from the file
                logger.info(f"🔑 [KbEvalClient] API key loaded from [{api_key_path}]")
        except FileNotFoundError:
            logger.info(f"🔑 [KbEvalClient] API key not found at [{api_key_path}], using [dummy] key")
            api_key = "dummy"  # vLLM often doesn't require real auth

        result = {
            "provider_name": provider_name,
            "base_url": base_url,
            "hostname": base_url.split("://")[1].split(":")[0],
            "port": base_url.split(":")[2].split("/")[0],
            "api_key": api_key,
            "retry_count": provider_config.get('retry_count', 3),
            "initial_retry_interval": provider_config.get('initial_retry_interval', 3),
            "timeout": provider_config.get('timeout', 300),
        }

        self.provider_config_cache[provider_name] = result
        logger.info(f"🔍 [kbEvalClient] Provider config [{provider_name}] created with base_url [{base_url}] and api_key_path [{api_key_path}]")

        return result

    async def kb_eval_ref(
        self,
        provider: str|List[str],
        reference_code: str,
        run_tag: str="auto",
        model_tag: str="model_tag",
        task_tag: str="task_tag",
    ) -> dict[str, Any]:
        """Call the kbEvalServer with evaluation parameters"""
        default_provider_name = provider[0] if isinstance(provider, list) else provider
        provider_config = self._provider_config_from_yaml(default_provider_name)
        num_retries = provider_config.get('retry_count', 3)

        run_tag = run_tag if run_tag != "auto" else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        retry_count = 0
        while retry_count < num_retries:
            try:
                retry_count += 1
                start_time = time.time()
                provider_name = random.choice(provider) if isinstance(provider, list) else provider
                provider_config = self._provider_config_from_yaml(provider_name)
                base_url = provider_config.get('base_url')
                api_key = provider_config.get('api_key')
                initial_retry_interval = provider_config.get('initial_retry_interval', 3)
                timeout = provider_config.get('timeout', 300)

                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                timeout = httpx.Timeout(connect=5.0, write=30.0, read=timeout, pool=5.0) # 5s connect, 30s write, 300s read, 5s pool
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close", "Accept-Encoding": "identity"}, # disable gzip
                    http2=False, # disable http2
                    trust_env=False, # disable trust env
                    timeout=timeout,
                ) as client:
                    response = await client.post(
                        url=f"{base_url}/kb_eval_ref",
                        json={
                            "run_tag": run_tag,
                            "model_tag": model_tag,
                            "task_tag": task_tag,
                            "reference_code": reference_code,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    )

                    response.raise_for_status()

                    result = KernelExecResult(**response.json())
                    if (not result.compiled or not result.correctness) and "retriable" in result.metadata and result.metadata["retriable"]:
                        elapsed_time = time.time() - start_time
                        sleep_seconds = initial_retry_interval ** retry_count
                        if retry_count < num_retries:
                            # retry_count -= 0.5 # reduce retry count by 0.5 to avoid infinite loop
                            logger.warning(f"⚠️ [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] Retriable error, retrying... [{retry_count}/{num_retries}] in [{sleep_seconds}s] [elapsed_time: {elapsed_time:.2f}s]")
                            await asyncio.sleep(sleep_seconds)
                            continue
                        else:
                            logger.warning(f"⚠️ [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] Return the last result from retriable error... [{retry_count}/{self.num_retries}] [elapsed_time: {elapsed_time:.2f}s]")
                            return result.model_dump()

                    return result.model_dump()

            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.warning(f"🔍 [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] Error calling server [{base_url}]: [{type(e).__name__}: {str(e)}] [{retry_count}/{num_retries}] [elapsed_time: {elapsed_time:.2f}s]")
                if retry_count < num_retries:
                    sleep_seconds = initial_retry_interval ** retry_count
                    logger.info(f"🔄 [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] Retrying in {sleep_seconds} seconds... ({retry_count}/{num_retries}) [elapsed_time: {elapsed_time:.2f}s]")
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(f"❌ [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] Failed after {retry_count} retries [elapsed_time: {elapsed_time:.2f}s]")
                    return None

    async def kb_eval(
        self,
        provider: str|List[str],
        reference_code: str,
        generated_code: str,
        run_tag: str="auto",
        model_tag: str="model_tag",
        task_tag: str="task_tag",
        eval_tag: str="eval_tag",
        code_type: str="cuda",
    ) -> dict[str, Any]:
        """Call the kbEvalServer with evaluation parameters"""
        default_provider_name = provider[0] if isinstance(provider, list) else provider
        provider_config = self._provider_config_from_yaml(default_provider_name)
        num_retries = provider_config.get('retry_count', 3)

        run_tag = run_tag if run_tag != "auto" else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        retry_count = 0
        while retry_count < num_retries:
            try:
                retry_count += 1
                start_time = time.time()
                provider_name = random.choice(provider) if isinstance(provider, list) else provider
                provider_config = self._provider_config_from_yaml(provider_name)
                base_url = provider_config.get('base_url')
                api_key = provider_config.get('api_key')
                initial_retry_interval = provider_config.get('initial_retry_interval', 3)
                timeout = provider_config.get('timeout', 300)

                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                timeout = httpx.Timeout(connect=5.0, write=30.0, read=timeout, pool=5.0) # 5s connect, 30s write, 300s read, 5s pool
                json_body = {
                    "run_tag": run_tag,
                    "model_tag": model_tag,
                    "task_tag": task_tag,
                    "eval_tag": eval_tag,
                    "reference_code": reference_code,
                    "generated_code": generated_code,
                    "code_type": code_type,
                }
                logger.info(f"🔍 [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Sending request to server [{base_url}] with json body: {json.dumps(json_body, indent=4)}")
                body = gzip.compress(json.dumps(json_body).encode("utf-8"))
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={
                        "Connection": "close",
                        "Accept-Encoding": "gzip",
                        "Authorization": f"Bearer {api_key}",
                    }, # disable gzip
                    http2=False, # disable http2
                    trust_env=False, # disable trust env
                    timeout=timeout,
                ) as client:
                    response = await client.post(
                        url=f"{base_url}/kb_eval",
                        content=body,
                    )

                    response.raise_for_status()

                    result = KernelExecResult(**response.json())
                    if (not result.compiled or not result.correctness) and "retriable" in result.metadata and result.metadata["retriable"]:
                        elapsed_time = time.time() - start_time
                        sleep_seconds = self.initial_retry_interval ** retry_count
                        if retry_count < self.num_retries:
                            # retry_count -= 0.5 # reduce retry count by 0.5 to avoid infinite loop
                            logger.warning(f"⚠️ [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Retriable error, retrying... [{retry_count}/{self.num_retries}] in [{sleep_seconds}s] [elapsed_time: {elapsed_time:.2f}s]")
                            await asyncio.sleep(sleep_seconds)
                            continue
                        else:
                            logger.warning(f"⚠️ [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Return the last result from retriable error... [{retry_count}/{self.num_retries}] [elapsed_time: {elapsed_time:.2f}s]")
                            return result.model_dump()

                    return result.model_dump()

            except Exception as e:
                elapsed_time = time.time() - start_time
                # add retry emoji to beginning and end of the string
                logger.warning(f"⚠️ [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Error calling server [{base_url}]: [{type(e).__name__}: {str(e)}] [{retry_count}/{num_retries}] [elapsed_time: {elapsed_time:.2f}s]")
                if retry_count < num_retries:
                    sleep_seconds = initial_retry_interval ** retry_count
                    logger.info(f"🔄 [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Retrying in {sleep_seconds} seconds... ({retry_count}/{num_retries}) [elapsed_time: {elapsed_time:.2f}s]") # no emoji
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(f"❌ [kbEvalClient] [{provider_name}] [{run_tag}] [{model_tag}] [{task_tag}] [{eval_tag}] Failed after {retry_count} retries")
                    return None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="local")
    parser.add_argument("--wd", type=str, default=".")
    parser.add_argument("--run_tag", type=str, default="auto")
    parser.add_argument("--model_tag", type=str, default="model_tag")
    parser.add_argument("--task_tag", type=str, default="task_tag")
    parser.add_argument("--eval_tag", type=str, default="eval_tag")
    parser.add_argument("--reference_code_path", type=str, default="kbEvalTest/elemAddRef.py")
    parser.add_argument("--generated_code_path", type=str, default="kbEvalTest/elemAddTriton.py")
    parser.add_argument("--code_type", type=str, default="triton")
    parser.add_argument("--measure_reference", action="store_true")
    args = parser.parse_args()

    # read from file
    reference_model_src = open(os.path.join(args.wd, args.reference_code_path), "r").read()
    generated_model_src = open(os.path.join(args.wd, args.generated_code_path), "r").read()

    client = KbEvalClient()

    if args.measure_reference:
        result = await client.kb_eval_ref(
            provider=args.provider.split(","),
            reference_code=reference_model_src,
            run_tag=args.run_tag,
            model_tag=args.model_tag,
            task_tag=args.task_tag,
        )
        logger.info(f"🔍 [kbEvalClient] [{args.provider_name}] [{args.run_tag}] [{args.model_tag}] [{args.task_tag}] Reference code evaluation result: {json.dumps(result if result else None, indent=4)}")
    else:
        result = await client.kb_eval(
            provider=args.provider.split(","),
            reference_code=reference_model_src,
            generated_code=generated_model_src,
            run_tag=args.run_tag,
            model_tag=args.model_tag,
            task_tag=args.task_tag,
            eval_tag=args.eval_tag,
            code_type=args.code_type,
        )
        logger.info(f"🔍 [kbEvalClient] [{args.run_tag}] [{args.model_tag}] [{args.task_tag}] [{args.eval_tag}] Generated code evaluation result: {json.dumps(result if result else None, indent=4)}")

if __name__ == "__main__":
    asyncio.run(main())
