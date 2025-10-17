import os
import duckdb
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Any, Callable
import yaml
import asyncio
import argparse
from pydantic import BaseModel
from engineBase import EngineBase, EngineConfig, TrainerStatus
from trainerUtil import format_conversation, make_checkpoint_callback
from logger import logger
import torch
from workflowUtil import TrainerBlock
from workflowClient import WorkflowClient
from workflowServer import WorkflowServer
from util import TRAINER_DIR, INFERENCE_DIR


class SFTConfig(BaseModel):
    """SFT configuration"""
    max_seq_length: int = 4096
    mask_non_assistant_tokens: bool = True
    mask_non_last_assistant_tokens: bool = True
    discard_long_conversations: bool = True

    @classmethod
    def from_yaml(cls, file_path: str) -> "SFTConfig":
        """Load SFT configuration from YAML file"""
        with open(os.path.expanduser(file_path), 'r') as f:
            config = yaml.safe_load(f)
        sft_config = config.get('sft', {})
        return cls(**sft_config)

class MessageDataset(Dataset):
    """Dataset class for handling conversational message data for SFT training"""

    def __init__(self,
                 messages_list: List[Dict[str, Any]],
                 tokenizer: AutoTokenizer,
                 sft_config: SFTConfig):
        """
        Initialize the MessageDataset

        Args:
            messages_list: List of conversation dictionaries, each containing 'messages' key
            tokenizer: Tokenizer to use for encoding
            sft_config: SFT configuration
        """
        self.tokenizer = tokenizer
        self.sft_config = sft_config

        # Ensure tokenizer has necessary tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"📊 [MessageDataset] Initializing with [{len(messages_list)}] conversations")

        self.messages_list = []
        for messages in messages_list:
            formatted_data = format_conversation(
                messages,
                tokenizer,
                mask_non_assistant_tokens=sft_config.mask_non_assistant_tokens,
                mask_non_last_assistant_tokens=sft_config.mask_non_last_assistant_tokens,
            )

            # Convert tensors to lists for the data collator
            assert isinstance(formatted_data['input_ids'], torch.Tensor)
            assert isinstance(formatted_data['attention_mask'], torch.Tensor)
            assert isinstance(formatted_data['labels'], torch.Tensor)
            input_ids = formatted_data['input_ids'].flatten().tolist()
            attention_mask = formatted_data['attention_mask'].flatten().tolist()
            labels = formatted_data['labels'].flatten().tolist()
            assert len(input_ids) == len(attention_mask) == len(labels)

            # Handle sequence length
            if len(input_ids) > self.sft_config.max_seq_length:
                if self.sft_config.discard_long_conversations:
                    logger.warning(f"⚠️ [MessageDataset] Discarding conversation [{formatted_data['text'][:100]}] because it is too long [{len(input_ids)} tokens]")
                    continue
                else:
                    # truncate the conversation to the max_seq_length
                    input_ids = input_ids[:self.sft_config.max_seq_length]
                    attention_mask = attention_mask[:self.sft_config.max_seq_length]
                    labels = labels[:self.sft_config.max_seq_length]

            condensed_data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            # add to the messages_list
            self.messages_list.append(condensed_data)

        logger.info(f"📊 [MessageDataset] Loaded [{len(self.messages_list)}] conversations, sft_config=[{sft_config}]")

    def __len__(self):
        return len(self.messages_list)

    def __getitem__(self, idx):
        return self.messages_list[idx]

class SFTTrainer():
    """Supervised Fine-Tuning trainer for conversational datasets"""

    def __init__(self, engine: EngineBase, prefix_tag: str, sft_config: SFTConfig):
        """Initialize SFT trainer"""
        self.engine = engine
        self.tokenizer = self.engine.tokenizer
        self.status = self.engine.status
        self.sft_config = sft_config
        logger.info(f"📜 [SFTTrainer] Initialized for conversational fine-tuning with SFTConfig: {sft_config}")

    def short_name(self):
        return 'sft'

    def _update_sft_config(self, sft_config: SFTConfig):
        """Update SFT config"""
        self.sft_config = sft_config

def sft_get_trainer(engine: EngineBase, prefix_tag: str, engine_config_file: str = "engineBase.yaml", sft_config_file: str = "trainerSFT.yaml"):
    """Get a SFT trainer"""
    try:
        engine._update_config(EngineConfig.from_yaml(engine_config_file, override_yaml_path=sft_config_file))
        logger.info(f"⚙️ [SFTTrainer] [{prefix_tag}] Engine config [{engine.__class__.__name__}] loaded from [{engine_config_file}]")
    except Exception as e:
        logger.error(f"❌ [SFTTrainer] [{prefix_tag}] Engine config [{engine.__class__.__name__}] failed to load: [{type(e)}] {e}")
        raise e

    try:
        sft_config = SFTConfig.from_yaml(sft_config_file)
        logger.info(f"⚙️ [SFTTrainer] [{prefix_tag}] SFT configuration loaded from [{sft_config_file}]")
    except Exception as e:
        logger.error(f"❌ [SFTTrainer] [{prefix_tag}] Failed to load SFT configuration: {e}")
        raise e

    try:
        trainer = SFTTrainer(engine=engine, prefix_tag=prefix_tag, sft_config=sft_config)
        logger.info(f"⭐ [SFTTrainer] [{prefix_tag}] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [SFTTrainer] [{prefix_tag}] Initialization failed: {e}")
        raise e

    return trainer

async def sft_train_block(block: TrainerBlock, trainer: SFTTrainer, callback: Optional[Callable] = None):
    """Train the model for one block"""
    logger.info(f"👉 [SFTTrainer] [{block.input_tag}] SFT Training started for block...")

    try:
        # Load or create dataset
        search_path = os.path.expanduser(f"{block.input_dir}/{block.input_tag}")
        if not os.path.exists(search_path):
            error_msg = f"❌ [SFTTrainer] [{block.input_tag}] Error: Input directory [{search_path}] does not exist"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # query from search_path folder, find all the conversation_*.json files, and load them into a dataframe
        result = duckdb.sql(f"""
        WITH convs AS (
            SELECT regexp_replace(filename, '_conversation\.json$', '') AS stem,
                messages,
                metadata
            FROM read_json_auto('{search_path}/**/*_conversation.json', sample_size=-1, ignore_errors=true)
            WHERE messages[3]['content'] IS NOT NULL
        ),
        evals AS(
            SELECT
                filename AS eval_file,
                regexp_replace(filename, '_generated_eval\.json$', '') AS stem,
                regexp_replace(filename, '/[^/]*$', '') AS task_id,
                compiled, correctness, runtime, reference_runtime as ref_runtime, runtime_stats
            FROM read_json_auto('{search_path}/**/*_generated_eval.json', filename = true)
        )
        SELECT convs.messages, convs.metadata, evals.*
        FROM convs
        LEFT JOIN evals  USING(stem)
        """)
        result_df = result.df()
        result_df = result_df[result_df["correctness"] == True] # only keep the correct ones
        # create a dataset from result_df['messages']
        message_dataset = MessageDataset(result_df['messages'].tolist(), trainer.tokenizer, trainer.sft_config)

        logger.info(f"📊 [SFTTrainer] [{block.input_tag}] Dataset created - Train: {len(message_dataset)}")

    except Exception as e:
        error_msg = f"❌ [SFTTrainer] [{block.input_tag}] Failed to load dataset: [{type(e)}: {e}]"
        logger.error(error_msg)
        raise e

    # train the block
    try:
        await trainer.engine.train_block(block.input_tag, message_dataset, callback=callback)
        await asyncio.sleep(1)
        logger.info(f"🎉 [SFTTrainer] [{block.input_tag}] Training completed successfully!")
    except Exception as e:
        error_msg = f"❌ [SFTTrainer] [{block.input_tag}] Training failed: {e}"
        logger.error(error_msg)
        raise e


async def main():
    """Main function for SFT training"""
    parser = argparse.ArgumentParser(description="Train a model using SFTTrainer")
    parser.add_argument("--queue_name", type=str, default="sft.1")
    parser.add_argument("--engine", type=str, default="unsloth")
    parser.add_argument("--engine_config", type=str, default="engineBase.yaml")
    parser.add_argument("--prefix_tag", type=str, default="TC_0.1.0_0.6B.a")
    parser.add_argument("--epoch_id", type=int, default=0)
    parser.add_argument("--block_id", type=int, default=0)
    parser.add_argument("--input_dir", type=str, default=INFERENCE_DIR + "/codeGenEval")
    parser.add_argument("--output_dir", type=str, default=TRAINER_DIR + "/sft")
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_14B.n_000_00") # {prefix}_{timestamp} or {prefix}_{epoch_id}_{block_id}
    parser.add_argument("--sft_config", type=str, default="trainerSFT.yaml")
    args = parser.parse_args()

    if args.engine == "unsloth":
        import unsloth

    engine_config = EngineConfig.from_yaml(args.engine_config)
    engine_config.model.engine = args.engine
    engine = EngineBase.create_engine(args.prefix_tag, engine_config) # no status for testing
    trainer = sft_get_trainer(engine, args.prefix_tag, args.engine_config, args.sft_config)
    sft_block = TrainerBlock(
        queue_name=args.queue_name,
        prefix_tag=args.prefix_tag,
        epoch_id=args.epoch_id,
        block_id=args.block_id,
        input_tag=args.input_tag,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    sft_block.input_dir = os.path.expanduser(sft_block.input_dir)
    sft_block.output_dir = os.path.expanduser(sft_block.output_dir)

    callback_func = make_checkpoint_callback(
        prefix_tag=args.prefix_tag,
        trainer_block=sft_block,
        workflow_provider="default",
    )

    await sft_train_block(sft_block, trainer, callback_func)
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
