import asyncio
from datetime import datetime
import argparse
import sys
import random
import httpx
import yaml
import traceback
import os
from workflowUtil import MODEL_OVERRIDE_KEY, TrainerBlock
from logger import logger
from workflowServer import WorkflowServer
from workflowClient import WorkflowClient
from trainerUtil import make_checkpoint_callback
from engineBase import EngineConfig, EngineBase, TrainerStatus
from trainerRFT import RFTConfig, rft_get_trainer
from trainerGRPO import GRPOConfig, grpo_get_trainer


ALPHA = 1.1

class TrainerMain:
    def __init__(self, engine: EngineBase, prefix_tag: str):
        self.engine = engine
        self.prefix_tag = prefix_tag
        self.reg_client = WorkflowClient(prefix_tag=prefix_tag)

    async def main_loop_task(self):
        logger.info(f"🌀 [trainerMain] Main loop started for prefix: {self.prefix_tag}")

        engine_config_file = "engineBase.yaml"
        rft_config_file = "trainerRFT.yaml"
        grpo_config_file = "trainerGRPO.yaml"
        # initialize trainers (for now, we only have grpo)
        rft_trainer = rft_get_trainer(self.engine, self.prefix_tag, engine_config_file, rft_config_file)
        grpo_trainer = grpo_get_trainer(self.engine, self.prefix_tag, engine_config_file, grpo_config_file)

        while True:
            try:
                # reinitialize the reg_client to avoid stale connection
                self.reg_client = WorkflowClient(prefix_tag=self.prefix_tag)
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
                    if rft_item['prefix_tag'] != self.prefix_tag:
                        logger.error(f"❌ [trainerMain] Skipping item with prefix: {rft_item['prefix_tag']}")
                        continue
                    rft_block = TrainerBlock(**rft_item)
                    logger.info(f"🧊 [trainerMain] Training RFT block: {rft_block.input_tag}")
                    # update config before running
                    engine_config = EngineConfig.from_yaml(engine_config_file, override_yaml_path=rft_config_file)
                    rft_config = RFTConfig.from_yaml(rft_config_file)
                    rft_trainer.engine._update_config(engine_config)
                    rft_trainer._update_rft_config(rft_config)
                    logger.info(f"🔍 [trainerMain] Base config: {rft_trainer.engine.config.model_dump_json()}")
                    logger.info(f"🔍 [trainerMain] RFT config: {rft_trainer.rft_config.model_dump_json()}")
                    # run in executor to avoid blocking the event loop
                    callback_func = make_checkpoint_callback(
                        prefix_tag=self.prefix_tag,
                        trainer_block=rft_block,
                        workflow_provider="default",
                    )
                    await rft_trainer.train_rft_block(rft_block, callback_func)

                if random.random() < grpo_prob:
                    grpo_item = await self.reg_client.dequeue(queue_name=GRPO_QUEUE_NAME)
                    if grpo_item['prefix_tag'] != self.prefix_tag:
                        logger.error(f"❌ [trainerMain] Skipping item with prefix: {grpo_item['prefix_tag']}")
                        continue
                    grpo_block = TrainerBlock(**grpo_item)
                    logger.info(f"🧊 [trainerMain] Training GRPO block: {grpo_block.input_tag}")
                    # update config before running
                    engine_config = EngineConfig.from_yaml(engine_config_file, override_yaml_path=grpo_config_file)
                    grpo_config = GRPOConfig.from_yaml(grpo_config_file)
                    grpo_trainer.engine._update_config(engine_config)
                    grpo_trainer._update_grpo_config(grpo_config)
                    logger.info(f"🔍 [trainerMain] Base config: {grpo_trainer.engine.config.model_dump_json()}")
                    logger.info(f"🔍 [trainerMain] GRPO config: {grpo_trainer.grpo_config.model_dump_json()}")
                    # run in executor to avoid blocking the event loop
                    callback_func = make_checkpoint_callback(
                        prefix_tag=self.prefix_tag,
                        trainer_block=grpo_block,
                        workflow_provider="default",
                    )
                    await grpo_trainer.train_grpo_block(grpo_block, callback_func)

            except Exception as e:
                logger.error(f"❌ [trainerMain] Error: [{type(e)}: {e}]")
                traceback.print_exc()
                await asyncio.sleep(5)

async def main():
    parser = argparse.ArgumentParser(description="Train a model using mixed RFT and GRPO trainers")
    parser.add_argument("--engine", type=str, default="unsloth")
    parser.add_argument("--engine_config", type=str, default="engineBase.yaml")
    parser.add_argument("--prefix_tag", type=str, default="auto.trainer.main")
    args = parser.parse_args()

    if args.prefix_tag.startswith('auto'):
        logger.error(f"❌ [trainerMain] --prefix_tag is required")
        return

    if args.engine == "unsloth":
        import unsloth

    engine_config = EngineConfig.from_yaml(args.engine_config)
    engine_config.model.engine = args.engine
    engine = EngineBase.create_engine(args.prefix_tag, engine_config) # no status for testing

    logger.info(f"🌀 [trainerMain] Starting with prefix: {args.prefix_tag}")
    main_trainer = TrainerMain(engine, args.prefix_tag)

    await main_trainer.main_loop_task()

if __name__ == "__main__":
    # test
    asyncio.run(main())
