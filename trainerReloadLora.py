import asyncio
import argparse
from workflowClient import WorkflowClient
from workflowRegistry import ADAPTER_PREFIX
from logger import logger
from configEndpoints import VLLMClient

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, required=True)
    parser.add_argument("--lora_name", type=str, default=None)
    parser.add_argument("--short_hostname", type=str, default="two")
    args = parser.parse_args()

    reg_client = WorkflowClient(prefix_tag=args.prefix_tag)

    vllm_client = VLLMClient(short_hostname=args.short_hostname)

    uploaded_set = set()
    if args.lora_name is None:
        keys = await reg_client.keys()
        for key in keys:
            try:
                if key.startswith(f"{ADAPTER_PREFIX}"):
                    adapter_path = await reg_client.get(key)
                    if key not in uploaded_set:
                        logger.info(f"🔍 [trainerReloadLora] Loading adapter [{key}] from [{adapter_path}]")
                        await vllm_client.load_lora_adapter(adapter_path, adapter_path)
                        uploaded_set.add(key)
                        logger.info(f"🔍 [trainerReloadLora] Loaded adapter [{key}] from [{adapter_path}]")
                    else:
                        logger.info(f"🔍 [trainerReloadLora] Skipping adapter [{key}] because it has already been loaded")
            except Exception as e:
                logger.error(f"🔍 [trainerReloadLora] Error loading adapter [{key}]: {e}")
    else:
        await vllm_client.load_lora_adapter(args.lora_name, args.lora_name)
        logger.info(f"🔍 Loaded adapter [{args.lora_name}] from [{args.lora_name}]")


if __name__ == "__main__":
    asyncio.run(main())