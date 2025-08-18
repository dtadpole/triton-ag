import asyncio

async def get_unused_lora_adapters(prefix_tag: str, vllm_provider: str, recent_checkpoint_name: str = None, last_modified_within=7200):
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
    parser.add_argument("--prefix_tag", type=str, default="auto.workflow.vllm")
    parser.add_argument("--vllm_provider", type=str, default="h8_1")
    parser.add_argument("--recent_checkpoint_name", type=str, default="TC_0.1.0_32B.b/checkpoint-1500")
    args = parser.parse_args()

    unused_lora_adapters = await get_unused_lora_adapters(
        prefix_tag=args.prefix_tag,
        vllm_provider=args.vllm_provider,
        recent_checkpoint_name=args.recent_checkpoint_name,
    )
    import json
    from logger import logger
    logger.info(f"Unused lora adapters: {json.dumps(unused_lora_adapters, indent=4)}")

if __name__ == "__main__":
    # add parent folder to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    asyncio.run(main())
