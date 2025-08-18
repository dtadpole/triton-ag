import os
import asyncio
import traceback
import argparse
import json
import yaml
from logger import logger
from workflowClient import WorkflowClient
from configEndpoints import VLLMClient
from workflowUtil import MODEL_OVERRIDE_KEY
from workflowClient import WorkflowClient
from configInterpreter import ConfigInterpreter
from trainerUtil import read_stream

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

    def enqueue(self, checkpoint_path: str):
        """Enqueue checkpoint path to the rsync_queue"""
        logger.info(f"📞 [RsyncClient] Enqueue path: [{checkpoint_path}] ...")
        checkpoint_name = '/'.join(str(os.path.expanduser(checkpoint_path)).split('/')[-2:])
        logger.info(f"📞 [RsyncClient] Enqueue name: [{checkpoint_name}]")
        loop = asyncio.get_event_loop()
        # create task to enqueue
        task = asyncio.create_task(
            self.workflowClient.enqueue(self.queue_name, {"checkpoint_name": checkpoint_name})
        )
        # run task in background
        logger.info(f"📞 [RsyncClient] Enqueue name: [{checkpoint_name}] done.")

    async def handle_sync_task(self, checkpoint_name: str, context_vars: dict = {}):
        sync_context_vars = self.context_vars | context_vars | {
            "checkpoint_name": checkpoint_name,
        }
        sync_task_config = self.module_config.get('sync_task', {})
        success = await self.configInterpreter.execute(runtime=self, config=sync_task_config, context_vars=sync_context_vars)
        if not success:
            logger.error(f"❌ [SyncClient] Failed to execute sync task: [{checkpoint_name}] with context: [{sync_context_vars}]")
            return
        logger.info(f"✅ [SyncClient] Sync task executed successfully: [{checkpoint_name}] with context: [{sync_context_vars}]")

    async def run(self):
        while True:
            try:
                qsize = await self.workflowClient.qsize(self.queue_name)
                if qsize == 0:
                    logger.info(f"🔍 [SyncClient] No sync task found, skipping...")
                    await asyncio.sleep(10)
                    continue
                item_json = await self.workflowClient.dequeue(queue_name=self.queue_name)
                checkpoint_name = str(item_json.get('checkpoint_name', ''))
                context_vars = item_json.get('context_vars', {})
                if not checkpoint_name:
                    logger.info(f"🔍 [SyncClient] No checkpoint path found, skipping...")
                    continue
                await self.handle_sync_task(checkpoint_name, context_vars)
            except Exception as e:
                logger.error(f"❌ [SyncClient] Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)
            finally:
                await asyncio.sleep(10)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default='TC_0.1.0_32B.b') # "auto.workflow.sync")
    parser.add_argument("--module_file", type=str, default="workflow/sync.module.vllm+logps.yaml")
    parser.add_argument("--queue_name", type=str, default="sync.sync.1")
    parser.add_argument("--manual_sync_checkpoint", type=str, default='TC_0.1.0_32B.b/checkpoint-1500')
    parser.add_argument("--manual_sync_context", type=str, default="""{
        "vllm_provider": "h8_1",
        "custom_provider": "h8_1",
        "vllm_host": "devvm3317.eag0.facebook.com"
    }""")
    args = parser.parse_args()

    if args.prefix_tag == 'auto':
        logger.error(f"❌ [RsyncClient] --prefix_tag is required!")
        return

    sync_client = WorkflowSync(args.prefix_tag, args.queue_name, module_file=args.module_file)
    if args.manual_sync_checkpoint:
        manual_sync_context = json.loads(args.manual_sync_context) | {
            "prefix_tag": args.prefix_tag,
            "checkpoint_name": args.manual_sync_checkpoint,
        }
        await sync_client.handle_sync_task(args.manual_sync_checkpoint, manual_sync_context)
    else:
        await sync_client.run()

if __name__ == "__main__":
    asyncio.run(main())
