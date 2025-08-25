async def logp_load_lora_adapters(logp_provider: str, checkpoint_name: str):
    from inferenceCustomClient import InferenceCustomClient
    # get the custom client
    inference_custom_client = InferenceCustomClient()
    await inference_custom_client.load_lora_adapter(provider=logp_provider, lora_name=checkpoint_name, lora_path=checkpoint_name)

async def logp_unload_lora_adapters(logp_provider: str, checkpoint_name: str):
    from inferenceCustomClient import InferenceCustomClient
    # get the custom client
    inference_custom_client = InferenceCustomClient()
    await inference_custom_client.unload_lora_adapter(provider=logp_provider, lora_name=checkpoint_name)

async def logp_get_models(logp_provider: str):
    from inferenceCustomClient import InferenceCustomClient
    # get the custom client
    inference_custom_client = InferenceCustomClient()
    return await inference_custom_client.get_models(provider=logp_provider)

async def logp_get_unused_lora_adapters(
    prefix_tag: str,
    logp_provider: str,
    recent_checkpoint_name: str = None,
    last_modified_within: int = 3600,
):
    from inferenceCustomClient import InferenceCustomClient
    from workflowClient import WorkflowClient
    ADAPTER_PREFIX = "adapter."
    # get the custom client
    logp_client = InferenceCustomClient()
    # get the workflow client
    workflowClient = WorkflowClient(prefix_tag=prefix_tag)
    # find out all the loaded lora adapters
    models = await logp_client.get_models(provider=logp_provider)
    models = models.get('data', [])
    available_lora_adapters = [model['id'] for model in models if model['parent'] is not None]
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
    parser.add_argument("--logp_provider", type=str, default="h8_3_2")
    parser.add_argument("--action", type=str, default="get", choices=["get", "load", "unload", "unused"]) # load, unload, get
    parser.add_argument("--checkpoint_name", type=str, default="auto.trainer.grpo/checkpoint-66")
    args = parser.parse_args()

    import json
    from logger import logger
    if args.action == "get":
        models = await logp_get_models(args.logp_provider)
        logger.info(f"Models: {json.dumps(models, indent=4, default=str)}")
    elif args.action == "load":
        result = await logp_load_lora_adapters(args.logp_provider, args.checkpoint_name)
        logger.info(f"Loaded adapter: {args.checkpoint_name}")
    elif args.action == "unload":
        result = await logp_unload_lora_adapters(args.logp_provider, args.checkpoint_name)
        logger.info(f"Unloaded adapter: {args.checkpoint_name}")
    elif args.action == "unused":
        unused_lora_adapters = await logp_get_unused_lora_adapters(
            prefix_tag=args.prefix_tag,
            logp_provider=args.logp_provider,
            recent_checkpoint_name=args.checkpoint_name,
        )
        logger.info(f"Unused lora adapters: {json.dumps(unused_lora_adapters, indent=4)}")

if __name__ == "__main__":
    # add parent folder to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import asyncio
    asyncio.run(main())
