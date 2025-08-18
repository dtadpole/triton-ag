
async def load_lora_adapters(vllm_provider: str, checkpoint_name: str):
    from configEndpoints import VLLMClient
    # get the vllm client
    vllm_client = VLLMClient(provider_name=vllm_provider)
    await vllm_client.load_lora_adapter(lora_name=checkpoint_name, lora_path=checkpoint_name)

async def unload_lora_adapters(vllm_provider: str, checkpoint_name: str):
    from configEndpoints import VLLMClient
    # get the vllm client
    vllm_client = VLLMClient(provider_name=vllm_provider)
    await vllm_client.unload_lora_adapter(lora_name=checkpoint_name)

async def get_models(vllm_provider: str):
    from configEndpoints import VLLMClient
    # get the vllm client
    vllm_client = VLLMClient(provider_name=vllm_provider)
    return await vllm_client.get_models()

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_provider", type=str, default="h8_1")
    parser.add_argument("--action", type=str, default="get", choices=["load", "unload", "get"]) # load, unload, get
    parser.add_argument("--checkpoint_name", type=str, default="TC_0.1.0_32B.b/checkpoint-1500")
    args = parser.parse_args()

    from logger import logger
    if args.action == "load":
        result = await load_lora_adapters(args.vllm_provider, args.checkpoint_name)
        logger.info(f"Loaded adapter: {args.checkpoint_name}")
    elif args.action == "unload":
        result = await unload_lora_adapters(args.vllm_provider, args.checkpoint_name)
        logger.info(f"Unloaded adapter: {args.checkpoint_name}")
    elif args.action == "get":
        models = await get_models(args.vllm_provider)
        import json
        logger.info(f"Models: {json.dumps(models, indent=4)}")

if __name__ == "__main__":
    # add parent folder to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import asyncio
    asyncio.run(main())
