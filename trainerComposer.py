import asyncio
from datetime import datetime
import argparse
import sys
import random
import httpx
import yaml
import traceback
import os
from workflowUtil import MODEL_OVERRIDE_KEY, TrainerGRPOBlock, TrainerRFTBlock, TrainerSFTBlock, get_prefix_tag
from logger import logger
from workflowServer import WorkflowServer
from workflowClient import WorkflowClient
from trainerBase import TrainerConfig
from trainerSFT import sft_train_block, sft_get_trainer, SFTConfig
from trainerRFT import rft_train_block, rft_get_trainer, RFTConfig
from trainerGRPO import grpo_train_block, grpo_get_trainer, GRPOConfig
from trainerUtil import rsync_file, VLLMClient

class TrainerComposer:
    def __init__(self, workflow_client: WorkflowClient, config_file: str = "trainerComposer.yaml"):
        self.workflow_client = workflow_client
        self.config_file = config_file
        self.config = yaml.safe_load(open(config_file, "r"))
        self.global_config = self.config.get("global", {})
        self.last_modified_within = self.global_config.get("last_modified_within", 3600)
        self.alpha = self.global_config.get("alpha", 1.1)
        self.queues = self.config.get("queues", [])
        self._build_trainers()

    def _build_trainers(self):
        base_config_file = "trainerBase.yaml"
        sft_config_file = "trainerSFT.yaml"
        rft_config_file = "trainerRFT.yaml"
        grpo_config_file = "trainerGRPO.yaml"
        # initialize trainers (for now, we only have sft and grpo)
        self.sft_trainer = sft_get_trainer(None, self.workflow_client.prefix_tag, base_config_file, sft_config_file)
        self.rft_trainer = rft_get_trainer(self.sft_trainer, self.workflow_client.prefix_tag, base_config_file, rft_config_file)
        self.grpo_trainer = grpo_get_trainer(self.rft_trainer, self.workflow_client.prefix_tag, base_config_file, grpo_config_file)

    async def main_loop_task(self):
        logger.info(f"🌀 [trainerComposer] Main loop started for prefix: {self.workflow_client.prefix_tag}")

        loop = asyncio.get_event_loop()

        while True:
            try:
                # reinitialize the reg_client to avoid stale connection
                self.reg_client = WorkflowClient(prefix_tag=self.trainer_prefix_tag)
                # queue name is {task_type}:{task_name}
                SFT_QUEUE_NAME = 'trainer.sft:sft.1'
                RFT_QUEUE_NAME = 'trainer.rft:rft.1'
                GRPO_QUEUE_NAME = 'trainer.grpo:grpo.1'

                sft_qsize = await self.reg_client.qsize(SFT_QUEUE_NAME)
                rft_qsize = await self.reg_client.qsize(RFT_QUEUE_NAME)
                grpo_qsize = await self.reg_client.qsize(GRPO_QUEUE_NAME)

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
                sft_prob = (sft_qsize / total_qsize) ** self.alpha
                rft_prob = (rft_qsize / total_qsize) ** self.alpha
                grpo_prob = (grpo_qsize / total_qsize) ** self.alpha

                if random.random() < sft_prob:
                    sft_item = await self.reg_client.dequeue(queue_name=SFT_QUEUE_NAME)
                    if sft_item['prefix_tag'] != self.trainer_prefix_tag:
                        logger.error(f"❌ [trainerMain] Skipping item with prefix: {sft_item['prefix_tag']}")
                        continue
                    sft_block = TrainerSFTBlock(**sft_item)
                    logger.info(f"🧊 [trainerMain] Training SFT block: {sft_block.input_tag}")
                    # update config before running
                    base_config = TrainerConfig.from_yaml(base_config_file, override_yaml_path=sft_config_file)
                    sft_config = SFTConfig.from_yaml(sft_config_file)
                    self.sft_trainer._update_config(base_config)
                    self.sft_trainer._update_sft_config(sft_config)
                    logger.info(f"🔍 [trainerMain] Base config: {self.sft_trainer.config.model_dump_json()}")
                    logger.info(f"🔍 [trainerMain] SFT config: {self.sft_trainer.sft_config.model_dump_json()}")
                    # run in executor to avoid blocking the event loop
                    await sft_train_block(sft_block, self.sft_trainer, self.rsync_queue.enqueue)

                if random.random() < rft_prob:
                    rft_item = await self.reg_client.dequeue(queue_name=RFT_QUEUE_NAME)
                    if rft_item['prefix_tag'] != self.trainer_prefix_tag:
                        logger.error(f"❌ [trainerMain] Skipping item with prefix: {rft_item['prefix_tag']}")
                        continue
                    rft_block = TrainerRFTBlock(**rft_item)
                    logger.info(f"🧊 [trainerMain] Training RFT block: {rft_block.input_tag}")
                    # update config before running
                    base_config = TrainerConfig.from_yaml(base_config_file, override_yaml_path=rft_config_file)
                    rft_config = RFTConfig.from_yaml(rft_config_file)
                    self.rft_trainer._update_config(base_config)
                    self.rft_trainer._update_rft_config(rft_config)
                    logger.info(f"🔍 [trainerMain] Base config: {self.rft_trainer.config.model_dump_json()}")
                    logger.info(f"🔍 [trainerMain] RFT config: {self.rft_trainer.rft_config.model_dump_json()}")
                    # run in executor to avoid blocking the event loop
                    await rft_train_block(rft_block, self.rft_trainer, self.rsync_queue.enqueue)

                if random.random() < grpo_prob:
                    grpo_item = await self.reg_client.dequeue(queue_name=GRPO_QUEUE_NAME)
                    if grpo_item['prefix_tag'] != self.trainer_prefix_tag:
                        logger.error(f"❌ [trainerMain] Skipping item with prefix: {grpo_item['prefix_tag']}")
                        continue
                    grpo_block = TrainerGRPOBlock(**grpo_item)
                    logger.info(f"🧊 [trainerMain] Training GRPO block: {grpo_block.input_tag}")
                    # update config before running
                    base_config = TrainerConfig.from_yaml(base_config_file, override_yaml_path=grpo_config_file)
                    grpo_config = GRPOConfig.from_yaml(grpo_config_file)
                    self.grpo_trainer._update_config(base_config)
                    self.grpo_trainer._update_grpo_config(grpo_config)
                    logger.info(f"🔍 [trainerMain] Base config: {self.grpo_trainer.config.model_dump_json()}")
                    logger.info(f"🔍 [trainerMain] GRPO config: {self.grpo_trainer.grpo_config.model_dump_json()}")
                    # run in executor to avoid blocking the event loop
                    await grpo_train_block(grpo_block, self.grpo_trainer, self.rsync_queue.enqueue)

            except Exception as e:
                logger.error(f"❌ [trainerMain] Error: [{type(e)}: {e}]")
                traceback.print_exc()
                await asyncio.sleep(5)

async def main():
    parser = argparse.ArgumentParser(description="Train a model using mixed SFT and GRPO trainers")
    parser.add_argument("--prefix_tag", type=str, default="auto")
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    if args.test_mode:
        # read from trainerMain.yaml
        with open("trainerComposer.yaml", "r") as f:
            test_config = yaml.safe_load(f).get('test', {})
        prefix_tag = test_config.get('prefix_tag', 'test_0.1.0') + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
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
        workflow_client = WorkflowClient(prefix_tag=args.prefix_tag)
        logger.info(f"🌀 [trainerMain] Starting with prefix: {workflow_client.prefix_tag}")
        main_trainer = TrainerComposer(workflow_client)

        # create tasks: 1/ main loop, 2/ rsync_queue
        rsync_task = asyncio.create_task(main_trainer.rsync_queue.rsync_task())
        main_task = asyncio.create_task(main_trainer.main_loop_task())

        # wait for the tasks to complete
        await asyncio.gather(rsync_task, main_task)

if __name__ == "__main__":
    # test
    asyncio.run(main())
