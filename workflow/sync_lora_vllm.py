async def vllm_load_lora_adapters(vllm_provider: str, checkpoint_name: str):
    from configEndpoints import VLLMClient
    import os
    # get the vllm client
    vllm_client = VLLMClient(provider_name=vllm_provider)
    lora_path = os.path.join(vllm_client.lora_folder, checkpoint_name)
    await vllm_client.load_lora_adapter(lora_name=checkpoint_name, lora_path=lora_path)

async def vllm_unload_lora_adapters(vllm_provider: str, checkpoint_name: str):
    from configEndpoints import VLLMClient
    # get the vllm client
    vllm_client = VLLMClient(provider_name=vllm_provider)
    await vllm_client.unload_lora_adapter(lora_name=checkpoint_name)

async def vllm_get_models(vllm_provider: str):
    from configEndpoints import VLLMClient
    # get the vllm client
    vllm_client = VLLMClient(provider_name=vllm_provider)
    return await vllm_client.get_models()

async def vllm_get_unused_lora_adapters(
    prefix_tag: str,
    vllm_provider: str,
    recent_checkpoint_name: str = None,
    last_modified_within: int = 3600,
):
    from configEndpoints import VLLMClient
    from workflowClient import WorkflowClient
    ADAPTER_PREFIX = "adapter."
    # get the vllm client
    vllm_client = VLLMClient(provider_name=vllm_provider)
    # get the workflow client
    workflowClient = WorkflowClient(prefix_tag=prefix_tag)
    # find out all the loaded lora adapters
    models = await vllm_client.get_models()
    models = models.get('data', [])
    # print('models', models)
    available_lora_adapters = [model['id'] for model in models if model['parent'] is not None]
    # print('available_lora_adapters', available_lora_adapters)
    # fine out who is using the lora adapters
    in_use_lora_adapters = [recent_checkpoint_name] if recent_checkpoint_name else []
    keys = await workflowClient.keys()
    for key in keys:
        if key.startswith(ADAPTER_PREFIX):
            value = await workflowClient.get(key, last_modified_within=last_modified_within)
            in_use_lora_adapters.append(value)
    # unused lora adapters are the ones in available_lora_adapters but not in in_use_lora_adapters
    unused_lora_adapters = [lora_adapter for lora_adapter in available_lora_adapters if lora_adapter not in in_use_lora_adapters]
    return unused_lora_adapters

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_provider", type=str, default="h8_3")
    parser.add_argument("--action", type=str, default="get", choices=["get", "load", "unload", "unused"]) # load, unload, get
    parser.add_argument("--checkpoint_name", type=str, default="auto.trainer.grpo/checkpoint-66")
    args = parser.parse_args()

    import json
    from logger import logger
    if args.action == "get":
        models = await vllm_get_models(args.vllm_provider)
        logger.info(f"Models: {json.dumps(models, indent=4, default=str)}")
    elif args.action == "load":
        result = await vllm_load_lora_adapters(args.vllm_provider, args.checkpoint_name)
        logger.info(f"Loaded adapter: {args.checkpoint_name}")
    elif args.action == "unload":
        result = await vllm_unload_lora_adapters(args.vllm_provider, args.checkpoint_name)
        logger.info(f"Unloaded adapter: {args.checkpoint_name}")
    elif args.action == "unused":
        unused_lora_adapters = await vllm_get_unused_lora_adapters(
            prefix_tag=args.prefix_tag,
            vllm_provider=args.vllm_provider,
            recent_checkpoint_name=args.checkpoint_name,
        )
        logger.info(f"Unused lora adapters: {json.dumps(unused_lora_adapters, indent=4)}")
    else:
        raise ValueError(f"Invalid action: {args.action}")

if __name__ == "__main__":
    # add parent folder to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import asyncio
    asyncio.run(main())
