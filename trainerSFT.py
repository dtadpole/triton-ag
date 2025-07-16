import os
import sys
import duckdb
import unsloth
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Any
import yaml
import asyncio
import argparse
from pydantic import BaseModel
from trainerBase import BaseTrainer, TrainerConfig, TrainerStatus, train_async
from trainerUtil import format_conversation
from logger import logger
import torch


class SFTConfig(BaseModel):
    """SFT configuration"""
    max_seq_length: int = 4096
    mask_non_assistant_tokens: bool = True
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
            formatted_data = format_conversation(messages, tokenizer, sft_config.mask_non_assistant_tokens)
            
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

class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning trainer for conversational datasets"""
    
    def __init__(self, prefix_tag: str, sft_config: SFTConfig, base_config: TrainerConfig, status: Optional[TrainerStatus] = None):
        """Initialize SFT trainer"""
        super().__init__(prefix_tag, base_config, status)
        self.sft_config = sft_config
        logger.info(f"🎯 [SFTTrainer] Initialized for conversational fine-tuning with SFTConfig: {sft_config}")
        

async def main():
    """Main function for SFT training"""
    parser = argparse.ArgumentParser(description="Train a model using SFTTrainer")
    parser.add_argument("--input_dir", type=str, default="~/.critique")
    parser.add_argument("--run_tag", type=str, default="v0.1_20250714_050308") # {prefix}_{timestamp} or {prefix}_{epoch_id}_{block_id}
    parser.add_argument("--base-config", type=str, default="trainerBase.yaml")
    parser.add_argument("--config", type=str, default="trainerSFT.yaml")
    args = parser.parse_args()
    
    # Load configuration
    try:
        base_config = TrainerConfig.from_yaml(args.base_config)
        logger.info(f"✅ [SFTTrainer] Configuration loaded from {args.base_config}")
    except Exception as e:
        logger.error(f"❌ [SFTTrainer] Failed to load Base configuration: {e}")
        sys.exit(1)

    try:
        sft_config = SFTConfig.from_yaml(args.config)
        logger.info(f"✅ [SFTTrainer] Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"❌ [SFTTrainer] Failed to load SFT configuration: {e}")
        sys.exit(1)
    
    try:
        trainer = SFTTrainer(args.run_tag, sft_config, base_config)
        logger.info("✅ [SFTTrainer] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [SFTTrainer] Initialization failed: {e}")
        sys.exit(1)
    
    # Load or create dataset
    search_path = os.path.expanduser(f"{args.input_dir}/{args.run_tag}")
    if not os.path.exists(search_path):
        logger.error(f"❌ [SFTTrainer] [{args.run_tag}] Error: Input directory [{search_path}] does not exist")
        return

    # query from search_path folder, find all the conversation_*.json files, and load them into a dataframe
    result = duckdb.sql(f"""SELECT filename, messages, metadata
                        FROM read_json_auto('{search_path}/**/*_conversation.json', sample_size=-1, ignore_errors=true) 
                        WHERE messages[3]['content'] IS NOT NULL
                    """)
    
    result_df = result.df()
    # create a dataset from result_df['messages']
    message_dataset = MessageDataset(result_df['messages'].tolist(), trainer.tokenizer, sft_config)

    logger.info(f"📊 [SFTTrainer] Dataset created - Train: {len(message_dataset)}")
    
    # train the block
    try:
        trainer.train_block(args.run_tag, message_dataset)
        logger.info("🎉 [SFTTrainer] Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ [SFTTrainer] Training failed: {e}")
        sys.exit(0)
    
if __name__ == "__main__":
    asyncio.run(main())
