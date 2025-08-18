import os
import asyncio
import traceback
import argparse
import yaml
from logger import logger
from workflowClient import WorkflowClient
from configEndpoints import VLLMClient
from workflowUtil import MODEL_OVERRIDE_KEY
from workflowClient import WorkflowClient
from configInterpreter import ConfigInterpreter
from trainerUtil import read_stream

LAST_MODIFIED_WITHIN = 7200
RSYNC_QUEUE_NAME = 'rsync.rsync.1'

async def rsync_file(source_path: str, target_path: str, rsync_path: str = "rsync") -> int:
    """
    Rsync a file from source to target path
    """
    # run command: rsync -azP <source_path> <target_path>
    command = f"rsync -azP --rsync-path '{rsync_path}' '{source_path}' '{target_path}'"
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )

    logger.info(f"[rsync_file] START ====================")
    logger.info(f"[rsync_file] command: {command}")

    # Create tasks to read stdout and stderr concurrently
    stdout_task = asyncio.create_task(
        read_stream(process.stdout, "rsync", is_error=False)
    )
    stderr_task = asyncio.create_task(
        read_stream(
            process.stderr, "rsync", is_error=True
        )  # seems taking warning message as error message
    )

    # Wait for the process to complete
    return_code = await process.wait()

    # Wait for all output to be processed
    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
    if process.returncode != 0:
        logger.error(f"[rsync_file] return code: {process.returncode}")
    else:
        logger.info(f"[rsync_file] return code: {process.returncode}")

    logger.info(f"[rsync_file] END ====================")

    return return_code


class SyncClient:
    def __init__(self, prefix_tag: str):
        self.rsync_config = self.load_config().get('rsync', {})
        self.retries = self.rsync_config.get('retries', 3)
        self.timeout = self.rsync_config.get('timeout', 10)
        target_short_hostname = self.rsync_config.get('vllm_client', 'two')
        self.vllm_client = VLLMClient(target_short_hostname)
        self.workflowClient = WorkflowClient(prefix_tag=prefix_tag)
        self.configInterpreter = ConfigInterpreter()

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
            self.workflowClient.enqueue(RSYNC_QUEUE_NAME, {"checkpoint_name": checkpoint_name})
        )
        # run task in background
        logger.info(f"📞 [RsyncClient] Enqueue name: [{checkpoint_name}] done.")
        # future = asyncio.run_coroutine_threadsafe(
        #     self.workflowClient.enqueue(RSYNC_QUEUE_NAME, {"checkpoint_name": checkpoint_name}),
        #     loop
        # )
        # logger.info(f"📞 [RsyncClient] Enqueue: [{checkpoint_name}] done.")
        # return future

    async def upload_lora_adapter(self, checkpoint_name: str):
        if not self.rsync_config.get('run_upload', False):
            logger.info(f"🔍 [RsyncClient] Skipping upload because [run_upload] is [false]")
            return
        # get with timeout
        source_prefix = os.path.expanduser(self.rsync_config.get('rsync_source_prefix', '~/.trainer'))
        target_prefix = self.rsync_config.get('rsync_target_prefix', '192.168.1.205:.trainer')
        target_path = target_prefix + '/' + checkpoint_name
        # rsync_path is the path to the rsync command
        rsync_path = self.rsync_config.get('rsync_path', 'rsync').format(checkpoint_name=checkpoint_name)
        # get a list of files to rsync
        file_list = self.rsync_config.get('file_list', [])
        for file in file_list:
            source_file = os.path.join(source_prefix, checkpoint_name, file)
            # target_file = os.path.join(target_path, file)
            logger.info(f"📞 [RsyncClient] Rsyncing: [{source_file}] to [{target_path}]")
            await rsync_file(source_file, target_path + '/', rsync_path=rsync_path)
        # we are here if rsync is successful
        logger.info(f"🌐 [RsyncClient] Rsynced: [{checkpoint_name}] to [{target_path}]")

    async def clean_up_lora_adapters(self, lora_path: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                # find out all the loaded lora adapters
                models = await self.vllm_client.get_models()
                models = models.get('data', [])
                available_lora_adapters = [model['id'] for model in models if model['parent'] is not None]
                # fine out who is using the lora adapters
                used_lora_adapters = [lora_path]
                keys = await self.workflowClient.keys()
                for key in keys:
                    if key.startswith("adapter."):
                        value = await self.workflowClient.get(key, last_modified_within=LAST_MODIFIED_WITHIN)
                        used_lora_adapters.append(value)
                # unused lora adapters are the ones in available_lora_adapters but not in used_lora_adapters
                unused_lora_adapters = [lora_adapter for lora_adapter in available_lora_adapters if lora_adapter not in used_lora_adapters]
                # unload the unused lora adapters
                for lora_adapter in unused_lora_adapters:
                    await self.vllm_client.unload_lora_adapter(lora_adapter)
                    logger.info(f"🔍 [RsyncClient] Unloaded unused lora adapter: {lora_adapter}")
                # we are here if everything executed successfully
                return
            except Exception as e:
                if retry_count < self.retries:
                    logger.info(f"🔍 [RsyncClient] Retrying to load and unload lora adapter: {lora_path} in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                    continue
                else:
                    logger.error(f"❌ [RsyncClient] Failed to load and unload lora adapter: {lora_path} after {self.retries} retries")
                    raise e

    async def handle_rsync_task(self, checkpoint_name: str):
        # upload the lora adapter
        await self.upload_lora_adapter(checkpoint_name)
        logger.info(f"🔍 [RsyncClient] Uploaded lora adapter: [{checkpoint_name}]")
        # load the lora adapter
        await self.vllm_client.load_lora_adapter(checkpoint_name, checkpoint_name)
        logger.info(f"🔍 [RsyncClient] Loaded lora adapter: [{checkpoint_name}]")
        # update the model_override in the global registry
        await self.workflowClient.put(MODEL_OVERRIDE_KEY, checkpoint_name)
        logger.info(f"🔍 [RsyncClient] Model override [{MODEL_OVERRIDE_KEY}] updated to [{checkpoint_name}]")
        # clean up the unused lora adapters
        await self.clean_up_lora_adapters(checkpoint_name)
        logger.info(f"🔍 [RsyncClient] Cleaned up lora adapters: [{checkpoint_name}]")

    async def rsync_task(self):
        """Push rsync task to the rsync_queue"""
        while True:
            try:
                qsize = await self.workflowClient.qsize(RSYNC_QUEUE_NAME)
                if qsize == 0:
                    logger.info(f"🔍 [RsyncClient] No checkpoint path found, skipping...")
                    continue
                checkpoint_json = await self.workflowClient.dequeue(queue_name=RSYNC_QUEUE_NAME)
                checkpoint_name = str(checkpoint_json.get('checkpoint_name', ''))
                if not checkpoint_name:
                    logger.info(f"🔍 [RsyncClient] No checkpoint path found, skipping...")
                    continue
                await self.handle_rsync_task(checkpoint_name)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"❌ [RsyncClient] Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)
            finally:
                await asyncio.sleep(10)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="auto")
    parser.add_argument("--manual_upload", type=str, default=None)
    args = parser.parse_args()

    if args.prefix_tag == 'auto':
        logger.error(f"❌ [RsyncClient] --prefix_tag is required!")
        return

    rsync_client = SyncClient(args.prefix_tag)
    if args.manual_upload:
        await rsync_client.handle_rsync_task(args.manual_upload)
    else:
        await rsync_client.rsync_task()

if __name__ == "__main__":
    asyncio.run(main())
