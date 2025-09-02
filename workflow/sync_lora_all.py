from workflow.sync_lora_vllm import vllm_load_lora_adapters, vllm_unload_lora_adapters, vllm_get_unused_lora_adapters
from workflow.sync_lora_logp import logp_load_lora_adapters, logp_unload_lora_adapters, logp_get_unused_lora_adapters

MODEL_OVERRIDE_KEY = "adapter.model_override"

async def sync_lora_all(
    prefix_tag: str,
    checkpoint_name: str,
    vllm_providers: list[str],
    logp_providers: list[str],
):
    from logger import logger
    # sync vllm servers
    for vllm_provider in vllm_providers:
        logger.info(f"🔍 Loading LoRA adapter: [{checkpoint_name}] to VLLM server [{vllm_provider}]")
        await vllm_load_lora_adapters(vllm_provider, checkpoint_name)
        logger.info(f"✅ Loaded LoRA adapter: [{checkpoint_name}] to VLLM server [{vllm_provider}]")
        # check unused vllm adapters
        unused_vllm_adapters = await vllm_get_unused_lora_adapters(
            prefix_tag=prefix_tag,
            vllm_provider=vllm_provider,
            recent_checkpoint_name=checkpoint_name,
        )
        # unload unused vllm adapters
        logger.info(f"🔍 Unused LoRA adapters: [{unused_vllm_adapters}] on vLLM server [{vllm_provider}]")
        for unused_vllm_adapter in unused_vllm_adapters:
            await vllm_unload_lora_adapters(vllm_provider, unused_vllm_adapter)
            logger.info(f"✅ Unloaded LoRA adapter: [{unused_vllm_adapter}] from VLLM server [{vllm_provider}]")
    # sync logp servers
    for logp_provider in logp_providers:
        await logp_load_lora_adapters(logp_provider, checkpoint_name)
        logger.info(f"✅ Loaded LoRA adapter: [{checkpoint_name}] to LogP server [{logp_provider}]")
        # check unused logp adapters
        unused_logp_adapters = await logp_get_unused_lora_adapters(
            prefix_tag=prefix_tag,
            logp_provider=logp_provider,
            recent_checkpoint_name=checkpoint_name,
        )
        logger.info(f"🔍 Unused LoRA adapters: [{unused_logp_adapters}] on LogP server [{logp_provider}]")
        for unused_logp_adapter in unused_logp_adapters:
            await logp_unload_lora_adapters(logp_provider, unused_logp_adapter)
            logger.info(f"✅ Unloaded LoRA adapter: [{unused_logp_adapter}] from LogP server [{logp_provider}]")
    # if everything is done, update the adapter.model_override
    from workflowClient import WorkflowClient
    workflow_client = WorkflowClient(prefix_tag=prefix_tag)
    await workflow_client.put(
        key=MODEL_OVERRIDE_KEY,
        value=checkpoint_name,
    )
    logger.info(f"✅ Updated [{MODEL_OVERRIDE_KEY}] to [{checkpoint_name}]")
    return True

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="auto.trainer.grpo")
    parser.add_argument("--checkpoint_name", type=str, default="auto.trainer.grpo/checkpoint-66")
    parser.add_argument("--vllm_providers", type=list, default=["h8_3"])
    parser.add_argument("--logp_providers", type=list, default=["h8_3_2", "h8_3_3"])
    args = parser.parse_args()

    from logger import logger
    logger.info("Starting [sync.lora.all]")
    await sync_lora_all(
        prefix_tag="test",
        checkpoint_name=args.checkpoint_name,
        vllm_providers=args.vllm_providers,
        logp_providers=args.logp_providers,
    )
    logger.info("Completed [sync.lora.all]")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import asyncio
    asyncio.run(main())