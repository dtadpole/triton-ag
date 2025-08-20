import os
import sys
import json
import duckdb
import unsloth
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Any, Callable
import yaml
import asyncio
import argparse
from pydantic import BaseModel
from trainerBase import BaseTrainer, TrainerConfig, TrainerStatus, train_async
from trainerUtil import format_conversation
from logger import logger
import torch
from workflowUtil import TrainerRFTBlock
from configInterpreter import ConfigInterpreter
from configEndpoints import DuckDBClient
from workflowRsync import RsyncClient
from util import INFERENCE_DIR

class RFTConfig(BaseModel):
    """RFT configuration"""
    max_seq_length: int = 4096
    mask_non_assistant_tokens: bool = True
    mask_non_last_assistant_tokens: bool = True
    discard_long_conversations: bool = True
    return_top_percentile: float = 0.25

    @classmethod
    def from_yaml(cls, file_path: str) -> "RFTConfig":
        """Load RFT configuration from YAML file"""
        with open(os.path.expanduser(file_path), 'r') as f:
            config = yaml.safe_load(f)
        rft_config = config.get('rft', {})
        return cls(**rft_config)

class RFTTrainer(BaseTrainer):
    """Rejection Fine-Tuning trainer for conversational datasets"""

    def __init__(self,
        prefix_tag: str,
        rft_config: RFTConfig,
        base_config: TrainerConfig,
        module_file: str = "trainer/rft.module.yaml",
        status: Optional[TrainerStatus] = None,
        base_trainer: BaseTrainer = None,
    ):
        """Initialize RFT trainer"""
        super().__init__(prefix_tag, base_config, status, base_trainer)
        self.configInterpreter = ConfigInterpreter()
        self.duckdbClient = DuckDBClient()
        self.rft_config = rft_config
        self.module_file = module_file
        self.module_config = yaml.safe_load(open(module_file, 'r')).get('module', {})
        self.context_vars = {
            # built-in context vars
            "self": self,
            "os": os,
            "json": json,
            "yaml": yaml,
            "rft_config": self.rft_config,
            "format_conversation": format_conversation,
            "tokenizer": self.tokenizer,
        }
        context_var_config = self.module_config.get('context_vars', {})
        self.context_vars = self.configInterpreter.prepare_context_vars(self, context_var_config, self.context_vars)
        logger.info(f"📜 [RFTTrainer] Initialized for conversational fine-tuning with RFTConfig: {rft_config}")

    def short_name(self):
        return 'rft'

    def _update_rft_config(self, rft_config: RFTConfig):
        """Update RFT config"""
        self.rft_config = rft_config

    def log_raw_data(self, data: Any, context_vars: Dict[str, Any]):
        """Log raw data"""
        logger.info(f"🔍 [RFTTrainer] DuckDB search has found [{len(data)}] rows.\n{data}")

    async def train_rft_block(self, block: TrainerRFTBlock, callback: Optional[Callable] = None):
        """Train the model for one block"""
        logger.info(f"👉 [RFTTrainer] [{block.input_tag}] RFT Training started for block...")

        # Create data loader
        context_vars = self.configInterpreter.prepare_context_vars(
            runtime=self,
            context_config=self.module_config.get('context_vars', {}),
            context_vars=self.context_vars | {
                "block": block
            },
        )
        success = await self.configInterpreter.execute(
            runtime=self,
            config=self.module_config.get('input_processor', {}),
            context_vars=context_vars,
        )
        if not success:
            logger.error(f"❌ [RFTTrainer] [{block.input_tag}] Failed to execute module config: {self.module_config.get('processor', {})}")
            return

        rft_datasets = context_vars.get('__result__', {})

        logger.info(f"👉 [{self.__class__.__name__}] [{block.input_tag}] Block started with [{len(rft_datasets)}] tasks, Initial global step: [{self.trainer_status.global_step}]")

        await super().train_block(block.input_tag, rft_datasets, callback=callback)
        await asyncio.sleep(1)


def rft_get_trainer(
    base_trainer: BaseTrainer,
    prefix_tag: str,
    base_config_file: str = "trainerBase.yaml",
    rft_config_file: str = "trainerRFT.yaml",
    module_file: str = "trainer/rft.module.yaml",
):
    """Get a RFT trainer"""
    try:
        base_config = TrainerConfig.from_yaml(base_config_file, override_yaml_path=rft_config_file)
        logger.info(f"⚙️ [RFTTrainer] [{prefix_tag}] Base configuration loaded from [{base_config_file}]")
    except Exception as e:
        logger.error(f"❌ [RFTTrainer] [{prefix_tag}] Failed to load base configuration: {e}")
        raise e

    try:
        rft_config = RFTConfig.from_yaml(rft_config_file)
        logger.info(f"⚙️ [RFTTrainer] [{prefix_tag}] RFT configuration loaded from [{rft_config_file}]")
    except Exception as e:
        logger.error(f"❌ [RFTTrainer] [{prefix_tag}] Failed to load RFT configuration: {e}")
        raise e

    try:
        trainer = RFTTrainer(prefix_tag, rft_config, base_config, module_file=module_file, base_trainer=base_trainer)
        logger.info(f"⭐ [RFTTrainer] [{prefix_tag}] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [RFTTrainer] [{prefix_tag}] Initialization failed: {e}")
        raise e

    return trainer


async def main():
    """Main function for RFT training"""
    parser = argparse.ArgumentParser(description="Train a model using RFTTrainer")
    parser.add_argument("--prefix_tag", type=str, default="auto")
    parser.add_argument("--epoch_id", type=int, default=0)
    parser.add_argument("--block_id", type=int, default=0)
    parser.add_argument("--input_dir", type=str, default=INFERENCE_DIR + "/codeGenEval")
    parser.add_argument("--output_dir", type=str, default="~/.trainer/rft")
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_14B.n_000_00") # {prefix}_{timestamp} or {prefix}_{epoch_id}_{block_id}
    parser.add_argument("--base_config", type=str, default="trainerBase.yaml")
    parser.add_argument("--rft_config", type=str, default="trainerRFT.yaml")
    parser.add_argument("--module_file", type=str, default="trainer/rft.module.yaml")
    parser.add_argument("--target_short_hostname", type=str, default="two")
    args = parser.parse_args()

    trainer = rft_get_trainer(None, args.prefix_tag, args.base_config, args.rft_config, args.module_file)
    rft_block = TrainerRFTBlock(
        prefix_tag=args.prefix_tag,
        epoch_id=args.epoch_id,
        block_id=args.block_id,
        input_tag=args.input_tag,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    rft_block.input_dir = os.path.expanduser(rft_block.input_dir)
    rft_block.output_dir = os.path.expanduser(rft_block.output_dir)

    rsync_client = RsyncClient(prefix_tag=args.prefix_tag)

    loop = asyncio.get_event_loop()
    # await loop.run_in_executor(None, trainer.train_rft_block, rft_block, rsync_client.enqueue)
    # asyncio.run_coroutine_threadsafe(trainer.train_rft_block(rft_block, rsync_client.enqueue), loop)
    await trainer.train_rft_block(rft_block, rsync_client.enqueue)
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
