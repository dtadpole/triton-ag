ADAPTER_PREFIX = "adapter"
MODEL_OVERRIDE_KEY = "model_override"

async def update_model_override(prefix_tag: str, worker_name: str, proc_id: str, worker_id: str):
    from workflowClient import WorkflowClient
    workflow_client = WorkflowClient(prefix_tag=prefix_tag)
    model_override = await workflow_client.get(key=f"{ADAPTER_PREFIX}.{MODEL_OVERRIDE_KEY}", return_none_if_not_found=True)
    if model_override is None:
        return None

    # we have a model override, record with workflow client
    target_key = f"{ADAPTER_PREFIX}.{MODEL_OVERRIDE_KEY}.{worker_name}.p{proc_id}.w{worker_id}"
    await workflow_client.put(key=target_key, value=model_override)
    # return model override
    return model_override

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="test")
    parser.add_argument("--worker_name", type=str, default="codeGenEval.base")
    parser.add_argument("--proc_id", type=str, default="01")
    parser.add_argument("--worker_id", type=str, default="001")
    args = parser.parse_args()

    from logger import logger
    result = await update_model_override(args.prefix_tag, args.worker_name, args.proc_id, args.worker_id)
    logger.info(f"Model override: {result}")

if __name__ == "__main__":
    # add parent folder to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import asyncio
    asyncio.run(main())
