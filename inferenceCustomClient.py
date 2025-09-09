import argparse
import asyncio
import os
import gzip
import json
import random
from typing import Dict, Any, List
import httpx
import yaml
from logger import logger
from transformers import AutoTokenizer


def from_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class InferenceCustomClient:
    def __init__(self, config_path="inferenceCustom.yaml"):
        self.provider_config_cache = {}

    def _provider_config_from_yaml(
        self,
        provider_name: str,
        yaml_file: str="inferenceCustom.yaml",
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
                logger.info(f"🔑 [InferenceCustomClient] API key loaded from [{api_key_path}]")
        except FileNotFoundError:
            logger.info(f"🔑 [InferenceCustomClient] API key not found at [{api_key_path}], using [dummy] key")
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
        logger.info(f"🔍 [InferenceCustomClient] Provider config [{provider_name}] created with base_url [{base_url}] and api_key_path [{api_key_path}]")

        return result

    async def get_models(self, provider: str|List[str]):
        default_provider_name = provider[0] if isinstance(provider, list) else provider
        provider_config = self._provider_config_from_yaml(default_provider_name)
        num_retries = provider_config.get('retry_count', 3)

        retry_count = 0
        while retry_count < num_retries:
            try:
                retry_count += 1
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
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(
                        f"{base_url}/models",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        timeout=timeout,
                    )

                    response.raise_for_status()

                    return response.json()

            except Exception as e:
                logger.warning(
                    f"🔍 [InferenceCustomClient] [{provider_name}] [get_models]: [{e}] [{retry_count}/{num_retries}]"
                )
                if retry_count < num_retries:
                    sleep_seconds = initial_retry_interval**retry_count
                    logger.info(
                        f"🔄 [InferenceCustomClient] [{provider_name}] [get_models] Retrying in {sleep_seconds} seconds... ({retry_count}/{num_retries})"
                    )
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(
                        f"❌ [InferenceCustomClient] [{provider_name}] [get_models] Failed after {retry_count} retries"
                    )
                    return None

    async def load_lora_adapter(self, provider: str|List[str], lora_name: str, lora_path: str = None):
        default_provider_name = provider[0] if isinstance(provider, list) else provider
        provider_config = self._provider_config_from_yaml(default_provider_name)
        num_retries = provider_config.get('retry_count', 3)

        retry_count = 0
        while retry_count < num_retries:
            try:
                retry_count += 1
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
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.post(
                        f"{base_url}/load_lora_adapter",
                        json={
                            "lora_name": lora_name,
                            "lora_path": lora_path if lora_path else lora_name,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        timeout=timeout,
                    )

                    response.raise_for_status()

                    return response.json()

            except Exception as e:
                logger.warning(
                    f"🔍 [InferenceCustomClient] [{provider_name}] [load_lora_adapter]: [{e}] [{retry_count}/{num_retries}]"
                )
                if retry_count < num_retries:
                    sleep_seconds = initial_retry_interval**retry_count
                    logger.info(
                        f"🔄 [InferenceCustomClient] [{provider_name}] [load_lora_adapter] Retrying in {sleep_seconds} seconds... ({retry_count}/{num_retries})"
                    )
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(
                        f"❌ [InferenceCustomClient] [{provider_name}] [load_lora_adapter] Failed after {retry_count} retries"
                    )
                    return None

    async def unload_lora_adapter(self, provider: str|List[str], lora_name: str):
        default_provider_name = provider[0] if isinstance(provider, list) else provider
        provider_config = self._provider_config_from_yaml(default_provider_name)
        num_retries = provider_config.get('retry_count', 3)

        retry_count = 0
        while retry_count < num_retries:
            try:
                retry_count += 1
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
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.post(
                        f"{base_url}/unload_lora_adapter",
                        json={
                            "lora_name": lora_name,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        timeout=timeout,
                    )

                    response.raise_for_status()

                    return response.json()

            except Exception as e:
                logger.warning(
                    f"🔍 [InferenceCustomClient] [{provider_name}] [unload_lora_adapter]: [{e}] [{retry_count}/{num_retries}]"
                )
                if retry_count < num_retries:
                    sleep_seconds = initial_retry_interval**retry_count
                    logger.info(
                        f"🔄 [InferenceCustomClient] [{provider_name}] [unload_lora_adapter] Retrying in {sleep_seconds} seconds... ({retry_count}/{num_retries})"
                    )
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(
                        f"❌ [InferenceCustomClient] [{provider_name}] [unload_lora_adapter] Failed after {retry_count} retries"
                    )
                    return None

    async def logps(self, provider: str|List[str], model_name: str, input_ids: list[int]):
        default_provider_name = provider[0] if isinstance(provider, list) else provider
        provider_config = self._provider_config_from_yaml(default_provider_name)
        num_retries = provider_config.get('retry_count', 3)

        retry_count = 0
        while retry_count < num_retries:
            try:
                retry_count += 1
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
                json_body = {
                    "model_name": model_name,
                    "input_ids": input_ids,
                }
                body = gzip.compress(json.dumps(json_body).encode("utf-8"))
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={
                        "Connection": "close",
                        "Content-Encoding": "gzip",
                        "Authorization": f"Bearer {api_key}",
                    },
                    http2=False, # disable http2
                    trust_env=False, # disable trust env
                    timeout=timeout,
                ) as client:
                    response = await client.post(
                        f"{base_url}/logps",
                        content=body,
                    )

                    response.raise_for_status()

                    result = response.json()
                    # logger.info(f"🔍 [InferenceCustomClient] [{self.provider_name}] [logps] Got response: {result}")
                    return result

            except Exception as e:
                logger.warning(
                    f"🔍 [InferenceCustomClient] [{provider_name}] [logps]: [{type(e)}] {e} [{retry_count}/{num_retries}]"
                )
                if retry_count < num_retries:
                    sleep_seconds = initial_retry_interval**retry_count
                    logger.info(
                        f"🔄 [InferenceCustomClient] [{provider_name}] [logps] Retrying in {sleep_seconds} seconds... ({retry_count}/{num_retries})"
                    )
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(
                        f"❌ [InferenceCustomClient] [{provider_name}] [logps] Failed after {retry_count} retries"
                    )
                    return None


async def test_client(
    model_or_adapter_name: str,
    client: InferenceCustomClient,
    provider_name: str,
    tokenizer: AutoTokenizer,
    iterations: int = 10,
):
    # now prepare the input_ids
    system_prompt = "You are a helpful assistant."
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "What is machine learning?",
        "How does the human brain work?",
        "What is the meaning of life?",
        "Can you tell me a joke?",
        "Show me a picture of a cat.",
        "Does the moon have a face?",
        "What is the best way to learn programming?",
        "How AI will change the world?",
    ]

    async def test_prob_task():
        for _ in range(iterations):
            try:
                # randomly select 1 prompt from prompts
                prompt = random.choice(prompts)
                formatted_prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                )
                input_ids = tokenizer.encode(formatted_prompt)
                logger.info(f"🔍 Input IDs: [{len(input_ids)}] {input_ids}")
                response = await client.logps(provider_name, model_or_adapter_name, input_ids)
                logger.info(f"✅ Got response: {response}")
            except Exception as e:
                logger.error(f"❌ Error: [{type(e).__name__}] {e}")
            finally:
                await asyncio.sleep(random.uniform(0.1, 1))

    # create 10 tasks
    tasks = [test_prob_task() for _ in range(8)]
    await asyncio.gather(*tasks)


async def main():
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, default="inferenceCustom.yaml")
    args.add_argument("--provider_name", type=str, default="local")
    args.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    args.add_argument("--adapter_name", type=str, default=None)
    args.add_argument("--iterations", type=int, default=5)
    args = args.parse_args()

    client = InferenceCustomClient(args.config_path)
    model_list = await client.get_models(args.provider_name)
    logger.info(f"✅ Got models: {model_list}")

    if args.adapter_name and args.adapter_name not in model_list.get("data", []):
        logger.warning(
            f"🔍 Adapter [{args.adapter_name}] not found on [{args.provider_name}], loading it..."
        )
        # load the model
        msg = await client.load_lora_adapter(args.provider_name, args.adapter_name, args.adapter_name)
        logger.info(
            f"✅ Loaded adapter [{args.adapter_name}] on [{args.provider_name}]: {msg}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    await test_client(
        args.adapter_name if args.adapter_name else args.model_name,
        client,
        args.provider_name,
        tokenizer,
        iterations=args.iterations,
    )

    if args.adapter_name:
        msg = await client.unload_lora_adapter(args.provider_name, args.adapter_name)
        logger.info(
            f"✅ Unloaded adapter [{args.adapter_name}] on [{args.provider_name}]: {msg}"
        )


if __name__ == "__main__":
    asyncio.run(main())
