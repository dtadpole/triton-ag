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
from globalUtils import TrainerRFTBlock
from globalRegClient import GlobalRegClient
from globalWorkflow import GlobalWorkflow


class RFTConfig(BaseModel):
    """RFT configuration"""
    max_seq_length: int = 4096
    mask_non_assistant_tokens: bool = True
    discard_long_conversations: bool = True

    @classmethod
    def from_yaml(cls, file_path: str) -> "RFTConfig":
        """Load RFT configuration from YAML file"""
        with open(os.path.expanduser(file_path), 'r') as f:
            config = yaml.safe_load(f)
        rft_config = config.get('rft', {})
        return cls(**rft_config)

class RFTDataset(Dataset):
    """Dataset class for handling conversational message data for RFT training"""
    
    def __init__(self, 
                 messages_list: List[Dict[str, Any]], 
                 tokenizer: AutoTokenizer, 
                 rft_config: RFTConfig):
        """
        Initialize the RFTDataset
        
        Args:
            messages_list: List of conversation dictionaries, each containing 'messages' key
            tokenizer: Tokenizer to use for encoding
            rft_config: RFT configuration
        """
        self.tokenizer = tokenizer
        self.rft_config = rft_config
        
        # Ensure tokenizer has necessary tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"📊 [RFTDataset] Initializing with [{len(messages_list)}] conversations")

        self.messages_list = []
        for messages in messages_list:
            formatted_data = format_conversation(messages, tokenizer, rft_config.mask_non_assistant_tokens)
            
            # Convert tensors to lists for the data collator
            assert isinstance(formatted_data['input_ids'], torch.Tensor)
            assert isinstance(formatted_data['attention_mask'], torch.Tensor)
            assert isinstance(formatted_data['labels'], torch.Tensor)
            input_ids = formatted_data['input_ids'].flatten().tolist()
            attention_mask = formatted_data['attention_mask'].flatten().tolist()
            labels = formatted_data['labels'].flatten().tolist()
            assert len(input_ids) == len(attention_mask) == len(labels)
            
            # Handle sequence length
            if len(input_ids) > self.rft_config.max_seq_length:
                if self.rft_config.discard_long_conversations:
                    logger.warning(f"⚠️ [RFTDataset] Discarding conversation [{formatted_data['text'][:100]}] because it is too long [{len(input_ids)} tokens]")
                    continue
                else:
                    # truncate the conversation to the max_seq_length
                    input_ids = input_ids[:self.rft_config.max_seq_length]
                    attention_mask = attention_mask[:self.rft_config.max_seq_length]
                    labels = labels[:self.rft_config.max_seq_length]
            
            condensed_data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            # add to the messages_list
            self.messages_list.append(condensed_data)

        logger.info(f"📊 [RFTDataset] Loaded [{len(self.messages_list)}] conversations, rft_config=[{rft_config}]")
    
    def __len__(self):
        return len(self.messages_list)
    
    def __getitem__(self, idx):
        return self.messages_list[idx]

class RFTTrainer(BaseTrainer):
    """Rejection Fine-Tuning trainer for conversational datasets"""
    
    def __init__(self, prefix_tag: str, rft_config: RFTConfig, base_config: TrainerConfig, status: Optional[TrainerStatus] = None, base_trainer: BaseTrainer = None):
        """Initialize RFT trainer"""
        super().__init__(prefix_tag, base_config, status, base_trainer)
        self.rft_config = rft_config
        logger.info(f"📜 [RFTTrainer] Initialized for conversational fine-tuning with RFTConfig: {rft_config}")
        

def rft_get_trainer(base_trainer: BaseTrainer, prefix_tag: str, base_config_file: str = "trainerBase.yaml", rft_config_file: str = "trainerRFT.yaml"):
    """Get a RFT trainer"""
    try:
        base_config = TrainerConfig.from_yaml(base_config_file)
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
        trainer = RFTTrainer(prefix_tag, rft_config, base_config, base_trainer=base_trainer)
        logger.info(f"⭐ [RFTTrainer] [{prefix_tag}] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [RFTTrainer] [{prefix_tag}] Initialization failed: {e}")
        raise e
    
    return trainer

def rft_train_block(block: TrainerRFTBlock, trainer: RFTTrainer, callback: Optional[Callable] = None):
    """Train the model for one block"""
    logger.info(f"👉 [RFTTrainer] [{block.input_tag}] RFT Training started for block...")

    try:
        # Load or create dataset
        search_path = os.path.expanduser(f"{block.input_dir}/{block.input_tag}")
        if not os.path.exists(search_path):
            error_msg = f"❌ [RFTTrainer] [{block.input_tag}] Error: Input directory [{search_path}] does not exist"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # query from search_path folder, find all the conversation_*.json files, and load them into a dataframe
        result = duckdb.sql(f"""SELECT filename, compiled, correctness, metadata, runtime, runtime_stats
                            FROM read_json_auto('{search_path}/**/gen_*_eval.json', sample_size=-1, ignore_errors=true) 
                        """)
        
        result_df = result.df()

        # filter out the rows where compiled or correctness not True
        filtered_df = result_df[(result_df['compiled'] == True) & (result_df['correctness'] == True)]
        filtered_length = len(filtered_df)

        # for each filtered row, load the messages from corresponding *_conversation.json file
        filtered_df['messages'] = None
        filtered_df['metadata'] = None
        for index, row in filtered_df.iterrows():
            conversation_filename = row['filename'].replace("_eval.json", "_conversation.json")
            with open(conversation_filename, 'r') as f:
                messages = json.load(f)
            # add a new column 'messages' to the filtered_df
            filtered_df.at[index, 'messages'] = messages['messages']
            filtered_df.at[index, 'metadata'] = messages['metadata']

        # create a dataset from result_df['messages']
        rft_dataset = RFTDataset(filtered_df['messages'].tolist(), trainer.tokenizer, trainer.rft_config)
        if len(rft_dataset) == 0:
            logger.warning(f"🗑️ [RFTTrainer] [{block.input_tag}] No tasks for RFT in [{search_path}]")
            return

        logger.info(f"🔍 [RFTTrainer] [{block.input_tag}] Loaded [{len(rft_dataset)}/{filtered_length}] tasks for RFT in [{search_path}]\n[{filtered_df}]")
    
    except Exception as e:
        error_msg = f"❌ [RFTTrainer] [{block.input_tag}] Failed to load dataset: [{type(e)}: {e}]"
        logger.error(error_msg)
        raise e
    
    # train the block
    try:
        trainer.train_block(block.input_tag, rft_dataset, callback=callback)
        logger.info(f"🎉 [RFTTrainer] [{block.input_tag}] Training completed successfully!")
    except Exception as e:
        error_msg = f"❌ [RFTTrainer] [{block.input_tag}] Training failed: {e}"
        logger.error(error_msg)
        raise e


async def main():
    """Main function for RFT training"""
    parser = argparse.ArgumentParser(description="Train a model using RFTTrainer")
    parser.add_argument("--prefix_tag", type=str, default="KC_0.1.0_14B")
    parser.add_argument("--epoch_id", type=int, default=0)
    parser.add_argument("--block_id", type=int, default=0)
    parser.add_argument("--input_dir", type=str, default="~/.exemplar")
    parser.add_argument("--output_dir", type=str, default="~/.trainer")
    parser.add_argument("--input_tag", type=str, default="KC_0.1.0_14B_000_00") # {prefix}_{timestamp} or {prefix}_{epoch_id}_{block_id}
    parser.add_argument("--base_config", type=str, default="trainerBase.yaml")
    parser.add_argument("--rft_config", type=str, default="trainerRFT.yaml")
    args = parser.parse_args()

    trainer = rft_get_trainer(None, args.prefix_tag, args.base_config, args.rft_config)
    rft_block = TrainerRFTBlock(
        prefix_tag=args.prefix_tag,
        epoch_id=args.epoch_id,
        block_id=args.block_id,
        input_tag=args.input_tag,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    rft_train_block(rft_block, trainer)
    
if __name__ == "__main__":
    asyncio.run(main())
