import asyncio
import argparse
import sys
import random
import httpx
import yaml
import traceback
import os
from globalUtils import MODEL_OVERRIDE_KEY, TrainerGRPOBlock, TrainerRFTBlock, TrainerSFTBlock
from logger import logger
from globalRegClient import GlobalRegClient
from trainerSFT import sft_train_block, sft_get_trainer
from trainerRFT import rft_train_block, rft_get_trainer
from trainerGRPO import grpo_train_block, grpo_get_trainer
from trainerUtil import rsync_file


client = GlobalRegClient()

LAST_MODIFIED_WITHIN = 3600
ALPHA = 1.1

class VLLMClient:
    def __init__(self):
        self.vllm_config = self.load_config().get('vllm', {})
        self.host = self.vllm_config.get('host', '10.12.0.202')
        self.port = self.vllm_config.get('port', 8091)
        api_key_path = os.path.expanduser(self.vllm_config.get('api_key_path', '~/.keys/local.api.key'))
        with open(api_key_path, 'r') as f:
            self.api_key = f.read().strip()
        self.timeout = self.vllm_config.get('timeout', 60)
        self.retries = self.vllm_config.get('retries', 3)

    def load_config(self, config_path: str = "trainerMain.yaml"):
        """Load config from yaml file"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    async def load_lora_adapter(self, lora_name: str, lora_path: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"http://{self.host}:{self.port}/v1/load_lora_adapter",
                        json={
                            "lora_name": lora_name,
                            "lora_path": lora_path,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )

                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Loaded lora adapter from [{lora_path}]")
                    return
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error loading lora adapter from [{lora_path}]: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to load lora adapter from [{lora_path}] in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to load lora adapter from [{lora_path}] after {self.retries} retries")
                    raise e

    async def unload_lora_adapter(self, lora_name: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"http://{self.host}:{self.port}/v1/unload_lora_adapter",
                        json={"lora_name": lora_name},
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Unloaded lora adapter from [{lora_name}]")
                    return
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error unloading lora adapter from [{lora_name}]: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to unload lora adapter from [{lora_name}] in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to unload lora adapter from [{lora_name}] after {self.retries} retries")
                    raise e

    async def get_models(self):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.get(
                        f"http://{self.host}:{self.port}/v1/models",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Got models")
                    return response.json()
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error getting models: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to get models in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to get models after {self.retries} retries")
                    raise e

class RsyncQueue:
    def __init__(self):
        self.rsync_queue = asyncio.Queue()
        self.rsync_config = self.load_config().get('rsync', {})
        self.retries = self.rsync_config.get('retries', 3)
        self.timeout = self.rsync_config.get('timeout', 10)
        self.vllm_client = VLLMClient()
        self.reg_client = GlobalRegClient()

    async def get(self, timeout: int = 10):
        return await asyncio.wait_for(self.rsync_queue.get(), timeout=timeout)

    def put(self, checkpoint_path: str):
        self.rsync_queue.put_nowait(checkpoint_path)

    def load_config(self, config_path: str = "trainerMain.yaml"):
        """Load config from yaml file"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def enqueue(self, checkpoint_path: str):
        """Enqueue checkpoint path to the rsync_queue"""
        logger.info(f"📞 [RsyncQueue] Enqueue: [{checkpoint_path}]")
        self.rsync_queue.put_nowait(checkpoint_path)

    async def upload_lora_adapter(self, checkpoint_path: str):
        # get with timeout
        source_prefix = os.path.expanduser(self.rsync_config.get('rsync_source_prefix', '~/.trainer'))
        target_prefix = self.rsync_config.get('rsync_target_prefix', '192.168.1.205:.trainer')
        if checkpoint_path.startswith(source_prefix):
            target_path = checkpoint_path.replace(source_prefix, target_prefix)
        else:
            target_path = target_prefix + checkpoint_path
        # get a list of files to rsync
        file_list = self.rsync_config.get('file_list', [])
        for file in file_list:
            source_file = os.path.join(checkpoint_path, file)
            # target_file = os.path.join(target_path, file)
            logger.info(f"📞 [RsyncQueue] Rsyncing: [{source_file}] to [{target_path}]")
            await rsync_file(source_file, target_path + '/')
        # we are here if rsync is successful
        logger.info(f"🌐 [RsyncQueue] Rsynced: [{checkpoint_path}] to [{target_path}]")

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
                keys = await self.reg_client.keys()
                for key in keys:
                    if key.startswith("adapter."):
                        value = await self.reg_client.get(key, last_modified_within=LAST_MODIFIED_WITHIN)
                        used_lora_adapters.append(value)
                # unused lora adapters are the ones in available_lora_adapters but not in used_lora_adapters
                unused_lora_adapters = [lora_adapter for lora_adapter in available_lora_adapters if lora_adapter not in used_lora_adapters]
                # unload the unused lora adapters
                for lora_adapter in unused_lora_adapters:
                    await self.vllm_client.unload_lora_adapter(lora_adapter)
                    logger.info(f"🔍 [RsyncQueue] Unloaded unused lora adapter: {lora_adapter}")
                # we are here if everything executed successfully
                return
            except Exception as e:
                if retry_count < self.retries:
                    logger.info(f"🔍 [RsyncQueue] Retrying to load and unload lora adapter: {lora_path} in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                    continue
                else:
                    logger.error(f"❌ [RsyncQueue] Failed to load and unload lora adapter: {lora_path} after {self.retries} retries")
                    raise e

    async def rsync_task(self):
        """Push rsync task to the rsync_queue"""
        while True:
            try:
                checkpoint_path = await asyncio.wait_for(self.rsync_queue.get(), timeout=self.timeout)
                checkpoint_path = str(checkpoint_path)
                if self.rsync_queue.qsize() > 0:
                    logger.info(f"🔍 [RsyncQueue] Skipping [{checkpoint_path}] because queue size: [{self.rsync_queue.qsize()}]")
                    continue
                await self.upload_lora_adapter(checkpoint_path)
                # get the last 2 parts of the checkpoint_path
                lora_path = checkpoint_path.split('/')[-2] + '/' + checkpoint_path.split('/')[-1]
                await self.vllm_client.load_lora_adapter(lora_path, lora_path)
                # update the model_override in the global registry
                await self.reg_client.put(f"{MODEL_OVERRIDE_KEY}", lora_path)
                # clean up the unused lora adapters
                await self.clean_up_lora_adapters(lora_path)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"❌ [RsyncQueue] Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

async def main_loop_task(rsync_queue: RsyncQueue, prefix_tag: str, test_mode: bool):

    logger.info(f"🌀 [trainerMain] Main loop started for prefix: {prefix_tag}")

    # initialize trainers (for now, we only have sft and grpo)
    sft_trainer = sft_get_trainer(None, prefix_tag)
    rft_trainer = rft_get_trainer(sft_trainer, prefix_tag)
    grpo_trainer = grpo_get_trainer(rft_trainer, prefix_tag)

    loop = asyncio.get_event_loop()

    while True:
        try:
            sft_qsize = await client.qsize('trainer.sft')
            rft_qsize = await client.qsize('trainer.rft')
            grpo_qsize = await client.qsize('trainer.grpo')

            # randomly pick a queue to dequeue from based on the qsize as the probability
            # sft_qsize / (sft_qsize + rft_qsize + grpo_qsize)
            # rft_qsize / (sft_qsize + rft_qsize + grpo_qsize)
            # grpo_qsize / (sft_qsize + rft_qsize + grpo_qsize)
            total_qsize = sft_qsize + rft_qsize + grpo_qsize
            # randomly pick a queue to dequeue from based on the probability
            if total_qsize == 0:
                logger.info(f"🔍 [trainerMain] No items to process, sleeping for [10] seconds")
                await asyncio.sleep(10)
                continue
            else:
                logger.info(f"🔍 [trainerMain] Queue sizes: sft: [{sft_qsize}], rft: [{rft_qsize}], grpo: [{grpo_qsize}]")

            # calculate the probability for each queue, use ALPHA (>1.0) to enhance the probability for larger queues
            sft_prob = (sft_qsize / total_qsize) ** ALPHA
            rft_prob = (rft_qsize / total_qsize) ** ALPHA
            grpo_prob = (grpo_qsize / total_qsize) ** ALPHA

            if random.random() < sft_prob:
                sft_item = await client.dequeue(queue_name="trainer.sft")
                if sft_item['prefix_tag'] != prefix_tag:
                    logger.error(f"❌ [trainerMain] Skipping item with prefix: {sft_item['prefix_tag']}")
                    continue
                sft_block = TrainerSFTBlock(**sft_item)
                logger.info(f"🧊 [trainerMain] Training SFT block: {sft_block.input_tag}")
                # run in executor to avoid blocking the event loop
                await loop.run_in_executor(None, sft_train_block, sft_block, sft_trainer, rsync_queue.enqueue)

            if random.random() < rft_prob:
                rft_item = await client.dequeue(queue_name="trainer.rft")
                if rft_item['prefix_tag'] != prefix_tag:
                    logger.error(f"❌ [trainerMain] Skipping item with prefix: {rft_item['prefix_tag']}")
                    continue
                rft_block = TrainerRFTBlock(**rft_item)
                logger.info(f"🧊 [trainerMain] Training RFT block: {rft_block.input_tag}")
                # run in executor to avoid blocking the event loop
                await loop.run_in_executor(None, rft_train_block, rft_block, rft_trainer, rsync_queue.enqueue)

            if random.random() < grpo_prob:
                grpo_item = await client.dequeue(queue_name="trainer.grpo")
                if grpo_item['prefix_tag'] != prefix_tag:
                    logger.error(f"❌ [trainerMain] Skipping item with prefix: {grpo_item['prefix_tag']}")
                    continue
                grpo_block = TrainerGRPOBlock(**grpo_item)
                logger.info(f"🧊 [trainerMain] Training GRPO block: {grpo_block.input_tag}")
                # run in executor to avoid blocking the event loop
                await loop.run_in_executor(None, grpo_train_block, grpo_block, grpo_trainer, rsync_queue.enqueue)

        except Exception as e:
            logger.error(f"❌ [trainerMain] Error: [{type(e)}: {e}]")
            traceback.print_exc()
            await asyncio.sleep(5)

async def main():
    parser = argparse.ArgumentParser(description="Train a model using mixed SFT and GRPO trainers")
    parser.add_argument("--prefix_tag", type=str, default="TC_0.1.0_14B")
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    if args.test_mode:
        # read from trainerMain.yaml
        with open("trainerMain.yaml", "r") as f:
            test_config = yaml.safe_load(f).get('test', {})
        prefix_tag = test_config.get('prefix_tag', 'test_0.1.0')
        sft_trainer = sft_get_trainer(None, prefix_tag)
        rft_trainer = rft_get_trainer(sft_trainer, prefix_tag)
        grpo_trainer = grpo_get_trainer(rft_trainer, prefix_tag)
        for item in test_config.get('loop', []):
            try:
                if item['type'] == 'trainer.grpo':
                    grpo_block = TrainerGRPOBlock(prefix_tag=prefix_tag, **item)
                    grpo_train_block(grpo_block, grpo_trainer)
                elif item['type'] == 'trainer.rft':
                    rft_block = TrainerRFTBlock(prefix_tag=prefix_tag, **item)
                    rft_train_block(rft_block, rft_trainer)
                elif item['type'] == 'trainer.sft':
                    sft_block = TrainerSFTBlock(prefix_tag=prefix_tag, **item)
                    sft_train_block(sft_block, sft_trainer)
            except Exception as e:
                logger.error(f"❌ [trainerMain] Error: [{type(e)}: {e}]")
                traceback.print_exc()
                continue
    else:
        rsync_queue = RsyncQueue()

        # create tasks: 1/ main loop, 2/ rsync_queue
        rsync_task = asyncio.create_task(rsync_queue.rsync_task())
        main_task = asyncio.create_task(main_loop_task(rsync_queue, args.prefix_tag, args.test_mode))

        # wait for the tasks to complete
        await asyncio.gather(rsync_task, main_task)

if __name__ == "__main__":
    # test
    asyncio.run(main())
