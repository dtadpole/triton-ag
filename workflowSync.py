import os
import asyncio
import traceback
import argparse
import json
import yaml
from logger import logger
from workflowClient import WorkflowClient
from workflowUtil import TrainerBlock, WorkflowSyncBlock
from configInterpreter import ConfigInterpreter
from trainerUtil import make_checkpoint_callback
from util import INFERENCE_DIR, TRAINER_DIR

LAST_MODIFIED_WITHIN = 7200

class WorkflowSync:
    def __init__(
        self,
        prefix_tag: str,
        queue_name: str,
        module_file: str = "workflow/sync.module.vllm.yaml",
    ):
        self.prefix_tag = prefix_tag
        self.queue_name = queue_name
        self.module_file = module_file
        self.workflowClient = WorkflowClient(prefix_tag=prefix_tag)
        self.configInterpreter = ConfigInterpreter()
        self.module_config = self.from_yaml(module_file)
        self.context_vars = {
            "os": os,
            "yaml": yaml,
            "json": json,
        }

    def from_yaml(self, module_file: str):
        with open(module_file, "r") as f:
            return yaml.safe_load(f)

    async def get(self, timeout: int = 10):
        return await asyncio.wait_for(self.rsync_queue.get(), timeout=timeout)

    def put(self, checkpoint_path: str):
        self.rsync_queue.put_nowait(checkpoint_path)

    def load_config(self, config_path: str = "workflowSync.yaml"):
        """Load config from yaml file"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    async def handle_sync_task(self, context_vars: dict = {}):
        sync_context_vars = self.context_vars | context_vars
        sync_task_config = self.module_config.get('sync_task', {})
        success = await self.configInterpreter.execute(
            runtime=self,
            config=sync_task_config,
            context_vars=sync_context_vars,
        )
        if not success:
            logger.error(f"❌ [SyncClient] Failed to execute sync task: [{context_vars}]")
            return
        logger.info(f"✅ [SyncClient] Sync task executed successfully: [{context_vars}]")

    async def run(self):
        while True:
            try:
                qsize = await self.workflowClient.qsize(self.queue_name)
                if qsize == 0:
                    logger.info(f"🔍 [SyncClient] No sync task found, skipping...")
                    await asyncio.sleep(10)
                    continue
                sync_block_json = await self.workflowClient.dequeue(queue_name=self.queue_name)
                sync_block = WorkflowSyncBlock(**sync_block_json)
                context_vars = {
                    "block": sync_block,
                }
                await self.handle_sync_task(context_vars)
            except Exception as e:
                logger.error(f"❌ [SyncClient] Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)
            finally:
                await asyncio.sleep(10)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_provider", type=str, default="default")
    parser.add_argument("--prefix_tag", type=str, default='auto.workflow.sync')
    parser.add_argument("--module_file", type=str, default="workflow/sync.module.vllm+logp.yaml")
    parser.add_argument("--queue_name", type=str, default="sync.sync.1")
    parser.add_argument("--test_callback", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=TRAINER_DIR + "/TC_0.1.0_32B.c/checkpoint-100")
    parser.add_argument("--test_manual_sync", action="store_true")
    parser.add_argument("--manual_sync_context", type=str, default="""{
        "vllm_providers": ["h8_3"],
        "logp_providers": ["h8_3_2", "h8_3_3"],
        "sync_hosts": [],
        "checkpoint_name": "TC_0.1.0_32B.c/checkpoint-100"
    }""")
    args = parser.parse_args()

    if args.prefix_tag.startswith('auto') and not args.test_callback and not args.test_manual_sync:
        logger.error(f"❌ [RsyncClient] --prefix_tag is required!")
        return

    # test callback
    if args.test_callback:
        callback_func = make_checkpoint_callback(
            prefix_tag=args.prefix_tag,
            trainer_block=TrainerBlock(
                queue_name="grpo.1",
                prefix_tag=args.prefix_tag,
                epoch_id=0,
                block_id=0,
                input_tag=f"{args.prefix_tag}_000_00",
            ),
            workflow_provider=args.workflow_provider,
            env_vars=json.loads(args.manual_sync_context),
        )
        checkpoint_path = os.path.expanduser(args.checkpoint_path)
        # callback_func is a sync function
        task = callback_func(checkpoint_path)
        await task
    # test manual sync
    elif args.test_manual_sync:
        manual_sync_context = json.loads(args.manual_sync_context)
        sync_block = WorkflowSyncBlock(
            queue_name=args.queue_name,
            prefix_tag=args.prefix_tag,
            epoch_id=0,
            block_id=0,
            input_tag=f"{args.prefix_tag}_000_00",
            module_file=args.module_file,
            context=manual_sync_context,
        )
        sync_client = WorkflowSync(args.prefix_tag, args.queue_name, module_file=args.module_file)
        await sync_client.handle_sync_task(context_vars={
            "block": sync_block,
        })
    # run the sync client
    else:
        sync_client = WorkflowSync(args.prefix_tag, args.queue_name, module_file=args.module_file)
        await sync_client.run()

if __name__ == "__main__":
    asyncio.run(main())
