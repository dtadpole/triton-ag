import os
import json
from typing import Dict, Optional, Any, Callable
import yaml
import asyncio
import argparse
from pydantic import BaseModel
from engineBase import EngineBase, EngineConfig, TrainerStatus
from trainerUtil import format_conversation, make_checkpoint_callback
from logger import logger
from workflowUtil import TrainerBlock
from configInterpreter import ConfigInterpreter
from configEndpoints import DuckDBClient
from workflowSync import WorkflowSync

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

class RFTTrainer():
    """Rejection Fine-Tuning trainer for conversational datasets"""

    def __init__(self,
        engine: EngineBase,
        prefix_tag: str,
        rft_config: RFTConfig,
        module_file: str = "trainer/rft.module.yaml",
    ):
        """Initialize RFT trainer"""
        self.engine = engine
        self.tokenizer = self.engine.tokenizer
        self.status = self.engine.status
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

    async def train_rft_block(self, block: TrainerBlock, callback: Optional[Callable] = None):
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

        logger.info(f"👉 [RFTTrainer] [{block.input_tag}] Block started with [{len(rft_datasets)}] tasks, Initial global step: [{self.engine.status.global_step}]")

        await self.engine.train_block(block.input_tag, rft_datasets, callback=callback)
        await asyncio.sleep(1)


def rft_get_trainer(
    engine: EngineBase,
    prefix_tag: str,
    engine_config_file: str = "engineBase.yaml",
    rft_config_file: str = "trainerRFT.yaml",
    module_file: str = "trainer/rft.module.yaml",
):
    """Get a RFT trainer"""
    try:
        engine._update_config(EngineConfig.from_yaml(engine_config_file, override_yaml_path=rft_config_file))
        logger.info(f"⚙️ [RFTTrainer] [{prefix_tag}] Engine config [{engine.__class__.__name__}] loaded from [{engine_config_file}]")
    except Exception as e:
        logger.error(f"❌ [RFTTrainer] [{prefix_tag}] Engine config [{engine.__class__.__name__}] failed to load: [{type(e)}] {e}")
        raise e

    try:
        rft_config = RFTConfig.from_yaml(rft_config_file)
        logger.info(f"⚙️ [RFTTrainer] [{prefix_tag}] RFT configuration loaded from [{rft_config_file}]")
    except Exception as e:
        logger.error(f"❌ [RFTTrainer] [{prefix_tag}] Failed to load RFT configuration: {e}")
        raise e

    try:
        trainer = RFTTrainer(engine=engine, prefix_tag=prefix_tag, rft_config=rft_config, module_file=module_file)
        logger.info(f"⭐ [RFTTrainer] [{prefix_tag}] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [RFTTrainer] [{prefix_tag}] Initialization failed: {e}")
        raise e

    return trainer


async def main():
    """Main function for RFT training"""
    parser = argparse.ArgumentParser(description="Train a model using RFTTrainer")
    parser.add_argument("--queue_name", type=str, default="rft.1")
    parser.add_argument("--engine", type=str, default="unsloth")
    parser.add_argument("--engine_config", type=str, default="engineBase.yaml")
    parser.add_argument("--prefix_tag", type=str, default="auto.trainer.rft")
    parser.add_argument("--epoch_id", type=int, default=0)
    parser.add_argument("--block_id", type=int, default=0)
<<<<<<< HEAD
    parser.add_argument("--input_dir", type=str, default=INFERENCE_DIR + "/codeGenEval")
    parser.add_argument("--output_dir", type=str, default=TRAINER_DIR + "/rft")
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_14B.n_000_00") # {prefix}_{timestamp} or {prefix}_{epoch_id}_{block_id}
    parser.add_argument("--base_config", type=str, default="trainerBase.yaml")
=======
    parser.add_argument("--input_dir", type=str, default="~/.inference/codeGenEval")
    parser.add_argument("--output_dir", type=str, default="~/.trainer/rft")
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_32B.b_006_05") # {prefix}_{timestamp} or {prefix}_{epoch_id}_{block_id}
>>>>>>> deepspeed
    parser.add_argument("--rft_config", type=str, default="trainerRFT.yaml")
    parser.add_argument("--module_file", type=str, default="trainer/rft.module.yaml")
    parser.add_argument("--target_short_hostname", type=str, default="two")
    args = parser.parse_args()

    if args.engine == "unsloth":
        import unsloth

    engine_config = EngineConfig.from_yaml(args.engine_config)
    engine_config.model.engine = args.engine
    engine = EngineBase.create_engine(args.prefix_tag, engine_config) # no status for testing
    trainer = rft_get_trainer(engine, args.prefix_tag, args.engine_config, args.rft_config, args.module_file)
    rft_block = TrainerBlock(
        queue_name=args.queue_name,
        prefix_tag=args.prefix_tag,
        epoch_id=args.epoch_id,
        block_id=args.block_id,
        input_tag=args.input_tag,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    rft_block.input_dir = os.path.expanduser(rft_block.input_dir)
    rft_block.output_dir = os.path.expanduser(rft_block.output_dir)

    callback_func = make_checkpoint_callback(
        prefix_tag=args.prefix_tag,
        trainer_block=rft_block,
        workflow_provider="default",
    )

    await trainer.train_rft_block(rft_block, callback_func)
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
