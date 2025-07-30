import asyncio
import argparse
from globalRegClient import GlobalRegClient
from globalRegistry import ADAPTER_PREFIX
from logger import logger
from trainerUtil import VLLMClient

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_name", type=str, default=None)
    args = parser.parse_args()

    reg_client = GlobalRegClient()

    vllm_client = VLLMClient()

    if args.lora_name is None:
        keys = await reg_client.keys()
        for key in keys:
            try:
                if key.startswith(f"{ADAPTER_PREFIX}"):
                    adapter_path = await reg_client.get(key)
                    logger.info(f"🔍 [trainerReloadLora] Loading adapter [{key}] from [{adapter_path}]")
                    await vllm_client.load_lora_adapter(adapter_path, adapter_path)
                    logger.info(f"🔍 [trainerReloadLora] Loaded adapter [{key}] from [{adapter_path}]")
            except Exception as e:
                logger.error(f"🔍 [trainerReloadLora] Error loading adapter [{key}]: {e}")
    else:
        await vllm_client.load_lora_adapter(args.lora_name, args.lora_name)
        logger.info(f"🔍 Loaded adapter [{args.lora_name}] from [{args.lora_name}]")


if __name__ == "__main__":
    asyncio.run(main())