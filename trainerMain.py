import asyncio
from datetime import datetime
import argparse
import sys
import random
import httpx
import yaml
import traceback
import os
from workflowUtil import MODEL_OVERRIDE_KEY, TrainerGRPOBlock, TrainerRFTBlock, get_prefix_tag
from logger import logger
from workflowServer import WorkflowServer
from workflowClient import WorkflowClient
from trainerBase import TrainerConfig
from trainerRFT import RFTConfig, rft_get_trainer
from trainerGRPO import GRPOConfig, grpo_get_trainer
from workflowRsync import RsyncClient


ALPHA = 1.1

class TrainerMain:
    def __init__(self, prefix_tag: str):
        self.prefix_tag = prefix_tag
        self.rsync_client = RsyncClient(prefix_tag)
        self.trainer_prefix_tag = get_prefix_tag(prefix_tag)
        self.reg_client = WorkflowClient(prefix_tag=self.trainer_prefix_tag)

    async def main_loop_task(self):
        logger.info(f"🌀 [trainerMain] Main loop started for prefix: {self.trainer_prefix_tag}")

        base_config_file = "trainerBase.yaml"
        rft_config_file = "trainerRFT.yaml"
        grpo_config_file = "trainerGRPO.yaml"
        # initialize trainers (for now, we only have grpo)
        rft_trainer = rft_get_trainer(None, self.trainer_prefix_tag, base_config_file, rft_config_file)
        grpo_trainer = grpo_get_trainer(rft_trainer, self.trainer_prefix_tag, base_config_file, grpo_config_file)

        while True:
            try:
                # reinitialize the reg_client to avoid stale connection
                self.reg_client = WorkflowClient(prefix_tag=self.trainer_prefix_tag)
                # queue name is {task_type}:{task_name}
                RFT_QUEUE_NAME = 'trainer.rft.1'
                GRPO_QUEUE_NAME = 'trainer.grpo.1'

                rft_qsize = await self.reg_client.qsize(RFT_QUEUE_NAME)
                grpo_qsize = await self.reg_client.qsize(GRPO_QUEUE_NAME)

                # randomly pick a queue to dequeue from based on the qsize as the probability
                # sft_qsize / (sft_qsize + rft_qsize + grpo_qsize)
                # rft_qsize / (sft_qsize + rft_qsize + grpo_qsize)
                # grpo_qsize / (sft_qsize + rft_qsize + grpo_qsize)
                total_qsize = rft_qsize + grpo_qsize
                # randomly pick a queue to dequeue from based on the probability
                if total_qsize == 0:
                    logger.info(f"🔍 [trainerMain] No items to process, sleeping for [10] seconds")
                    await asyncio.sleep(10)
                    continue
                else:
                    logger.info(f"🔍 [trainerMain] Queue sizes: rft: [{rft_qsize}], grpo: [{grpo_qsize}]")

                # calculate the probability for each queue, use ALPHA (>1.0) to enhance the probability for larger queues
                rft_prob = (rft_qsize / total_qsize) ** ALPHA
                grpo_prob = (grpo_qsize / total_qsize) ** ALPHA

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
                    rft_trainer._update_config(base_config)
                    rft_trainer._update_rft_config(rft_config)
                    logger.info(f"🔍 [trainerMain] Base config: {rft_trainer.config.model_dump_json()}")
                    logger.info(f"🔍 [trainerMain] RFT config: {rft_trainer.rft_config.model_dump_json()}")
                    # run in executor to avoid blocking the event loop
                    await rft_trainer.train_rft_block(rft_block, self.rsync_client.enqueue)

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
                    grpo_trainer._update_config(base_config)
                    grpo_trainer._update_grpo_config(grpo_config)
                    logger.info(f"🔍 [trainerMain] Base config: {grpo_trainer.config.model_dump_json()}")
                    logger.info(f"🔍 [trainerMain] GRPO config: {grpo_trainer.grpo_config.model_dump_json()}")
                    # run in executor to avoid blocking the event loop
                    await grpo_trainer.train_grpo_block(grpo_block, self.rsync_client.enqueue)

            except Exception as e:
                logger.error(f"❌ [trainerMain] Error: [{type(e)}: {e}]")
                traceback.print_exc()
                await asyncio.sleep(5)

async def main():
    parser = argparse.ArgumentParser(description="Train a model using mixed RFT and GRPO trainers")
    parser.add_argument("--prefix_tag", type=str, default="auto")
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    if args.test_mode:
        # read from trainerMain.yaml
        with open("trainerMain.yaml", "r") as f:
            test_config = yaml.safe_load(f).get('test', {})
        prefix_tag = test_config.get('prefix_tag', 'test_0.1.0') + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
        rft_trainer = rft_get_trainer(None, prefix_tag)
        grpo_trainer = grpo_get_trainer(rft_trainer, prefix_tag)
        for item in test_config.get('loop', []):
            try:
                if item['type'] == 'trainer.grpo':
                    grpo_block = TrainerGRPOBlock(prefix_tag=prefix_tag, **item)
                    await grpo_trainer.train_grpo_block(grpo_block)
                elif item['type'] == 'trainer.rft':
                    rft_block = TrainerRFTBlock(prefix_tag=prefix_tag, **item)
                    await rft_trainer.train_rft_block(rft_block)
            except Exception as e:
                logger.error(f"❌ [trainerMain] Error: [{type(e)}: {e}]")
                traceback.print_exc()
                continue
    else:
        if args.prefix_tag == 'auto':
            logger.error(f"❌ [trainerMain] --prefix_tag is required")
            return

        logger.info(f"🌀 [trainerMain] Starting with prefix: {args.prefix_tag}")
        main_trainer = TrainerMain(args.prefix_tag)

        await main_trainer.main_loop_task()

if __name__ == "__main__":
    # test
    asyncio.run(main())
