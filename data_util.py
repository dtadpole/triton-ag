import os
import json
import glob
import torch
import argparse
from torch.utils.data import Dataset
from util import logger


class ExperienceDataset(Dataset):
    """Dataset for loading processed experience files."""
    
    def __init__(self, data_dir, max_length=16384):
        self.max_length = max_length
        self.data = []
        
        # Load all JSONL files from the data directory
        jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        
        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            
                            # Ensure all required fields exist
                            if 'input_ids' not in item:
                                raise ValueError(f"Missing 'input_ids' field in {file_path}")
                            if 'attention_mask' not in item:
                                raise ValueError(f"Missing 'attention_mask' field in {file_path}")
                            if 'labels' not in item:
                                raise ValueError(f"Missing 'labels' field in {file_path}")
                            
                            input_ids = item['input_ids']
                            attention_mask = item['attention_mask']
                            labels = item['labels']
                            
                            # Filter out sequences longer than max_length
                            if len(input_ids) <= max_length:
                                self.data.append({
                                    'input_ids': input_ids,
                                    'attention_mask': attention_mask,
                                    'labels': labels
                                })
                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"Loaded {len(self.data)} examples from {len(jsonl_files)} files")
        logger.info(f"Filtered out sequences longer than {max_length} tokens")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SimpleDataCollator:
    """Simple data collator with padding."""
    
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def __call__(self, batch):
        # Find max length in batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        # Pad to multiple of pad_to_multiple_of
        if self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Prepare batch tensors
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            # Pad sequences
            padding_length = max_len - len(input_ids)
            
            if padding_length > 0:
                input_ids = input_ids + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length  # -100 is ignored in loss computation
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long)
        } 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--max_length", type=int, default=16384)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    collator = SimpleDataCollator(tokenizer)
    dataset = ExperienceDataset(data_dir="finetune_processed_experiences", max_length=args.max_length)
    batch = collator([dataset[0], dataset[1]])
    logger.info(f"Sample batch: {batch}")
