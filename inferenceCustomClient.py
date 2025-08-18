import os
import httpx
import random
import yaml
import asyncio
import argparse
from logger import logger
from transformers import AutoTokenizer

def from_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class InferenceCustomClient:
    def __init__(self, provider_name: str, config_path="inferenceCustom.yaml"):
        self.raw_config = from_yaml(config_path).get("providers", {})
        if provider_name not in self.raw_config:
            raise ValueError(f"Provider {provider_name} not found in config file [{config_path}]")
        self.provider_name = provider_name
        self.provider_config = self.raw_config.get(provider_name, {})
        self.base_url = self.provider_config.get("base_url", "http://localhost:8092/v1")
        self.hostname = self.base_url.split('://')[1].split(':')[0]
        self.port = self.base_url.split(':')[2].split('/')[0]
        self.api_key_path = os.path.expanduser(self.provider_config.get('api_key_path', '~/.keys/local.api.key'))
        with open(self.api_key_path, 'r') as f:
            self.api_key = f.read().strip()
        self.num_retries = self.provider_config.get('num_retries', 3)
        self.initial_retry_interval = self.provider_config.get('initial_retry_interval', 3)
        self.timeout = self.provider_config.get('timeout', 300)

    async def get_models(self):
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.get(
                        f"{self.base_url}/models",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )

                    response.raise_for_status()

                    return response.json()

            except Exception as e:
                logger.warning(f"🔍 [InferenceCustomClient] [{self.provider_name}] [get_models]: [{e}] [{retry_count}/{self.num_retries}]")
                if retry_count < self.num_retries:
                    sleep_seconds = self.initial_retry_interval ** retry_count
                    logger.info(f"🔄 [InferenceCustomClient] [{self.provider_name}] [get_models] Retrying in {sleep_seconds} seconds... ({retry_count}/{self.num_retries})")
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(f"❌ [InferenceCustomClient] [{self.provider_name}] [get_models] Failed after {retry_count} retries")
                    return None

    async def load_lora_adapter(self, lora_name: str, lora_path: str = None):
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"{self.base_url}/load_lora_adapter",
                        json={
                            "lora_name": lora_name,
                            "lora_path": lora_path if lora_path else lora_name,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )

                    response.raise_for_status()

                    return response.json()

            except Exception as e:
                logger.warning(f"🔍 [InferenceCustomClient] [{self.provider_name}] [load_lora_adapter]: [{e}] [{retry_count}/{self.num_retries}]")
                if retry_count < self.num_retries:
                    sleep_seconds = self.initial_retry_interval ** retry_count
                    logger.info(f"🔄 [InferenceCustomClient] [{self.provider_name}] [load_lora_adapter] Retrying in {sleep_seconds} seconds... ({retry_count}/{self.num_retries})")
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(f"❌ [InferenceCustomClient] [{self.provider_name}] [load_lora_adapter] Failed after {retry_count} retries")
                    return None

    async def unload_lora_adapter(self, lora_name: str):
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"{self.base_url}/unload_lora_adapter", 
                        json={
                            "lora_name": lora_name,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )

                    response.raise_for_status()

                    return response.json()

            except Exception as e:
                logger.warning(f"🔍 [InferenceCustomClient] [{self.provider_name}] [unload_lora_adapter]: [{e}] [{retry_count}/{self.num_retries}]")
                if retry_count < self.num_retries:
                    sleep_seconds = self.initial_retry_interval ** retry_count
                    logger.info(f"🔄 [InferenceCustomClient] [{self.provider_name}] [unload_lora_adapter] Retrying in {sleep_seconds} seconds... ({retry_count}/{self.num_retries})")
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(f"❌ [InferenceCustomClient] [{self.provider_name}] [unload_lora_adapter] Failed after {retry_count} retries")
                    return None


    async def logps(self, model_name: str, input_ids: list[int]):
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"{self.base_url}/logps",
                        json={
                            "model_name": model_name,
                            "input_ids": input_ids,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    
                    response.raise_for_status()

                    result = response.json()
                    logger.info(f"🔍 [InferenceCustomClient] [{self.provider_name}] [logps] Got response: {result}")
                    return result

            except Exception as e:
                logger.warning(f"🔍 [InferenceCustomClient] [{self.provider_name}] [logps]: [{e}] [{retry_count}/{self.num_retries}]")
                if retry_count < self.num_retries:
                    sleep_seconds = self.initial_retry_interval ** retry_count
                    logger.info(f"🔄 [InferenceCustomClient] [{self.provider_name}] [logps] Retrying in {sleep_seconds} seconds... ({retry_count}/{self.num_retries})")
                    # exponential backoff
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    # add error emoji to beginning and end of the string
                    logger.error(f"❌ [InferenceCustomClient] [{self.provider_name}] [logps] Failed after {retry_count} retries")
                    return None

async def test_client(model_or_adapter_name: str, client: InferenceCustomClient, tokenizer: AutoTokenizer, iterations: int = 10):
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
                        {"role": "user", "content": prompt}
                    ],
                    tokenize=False,
                )
                input_ids = tokenizer.encode(formatted_prompt)
                logger.info(f"🔍 Input IDs: [{len(input_ids)}] {input_ids}")
                response = await client.logps(model_or_adapter_name, input_ids)
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
    args.add_argument('--config', type=str, default="inferenceCustom.yaml")
    args.add_argument('--provider', type=str, default="local")
    args.add_argument('--model_name', type=str, default="Qwen/Qwen3-32B")
    args.add_argument('--adapter_name', type=str, default=None)
    args.add_argument('--iterations', type=int, default=5)
    args = args.parse_args()

    client = InferenceCustomClient(args.provider, args.config)
    model_list = await client.get_models()
    logger.info(f"✅ Got models: {model_list}")

    if args.adapter_name and args.adapter_name not in model_list.get("data", []):
        logger.warning(f"🔍 Adapter [{args.adapter_name}] not found on [{args.provider}], loading it...")
        # load the model
        msg = await client.load_lora_adapter(args.adapter_name, args.adapter_name)
        logger.info(f"✅ Loaded adapter [{args.adapter_name}] on [{args.provider}]: {msg}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    await test_client(args.adapter_name if args.adapter_name else args.model_name, client, tokenizer, iterations=args.iterations)

    if args.adapter_name:
        msg = await client.unload_lora_adapter(args.adapter_name)
        logger.info(f"✅ Unloaded adapter [{args.adapter_name}] on [{args.provider}]: {msg}")

if __name__ == "__main__":
    asyncio.run(main())