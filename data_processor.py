#!/usr/bin/env python3
"""
Data processing utilities for Qwen fine-tuning.
Handles loading, processing, and formatting of conversation data.
"""

import os
import json
import torch
import argparse
import traceback
from datasets import Dataset
from typing import List, Dict, Any, Union
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from util import logger
from huggingface_hub import HfApi


class CustomDataCollatorWithMasking(DataCollatorForLanguageModeling):
    """Custom data collator that masks system, user, and function tokens with dynamic length support."""
    
    def __init__(self, tokenizer, mlm=False, ignore_index=-100, return_tensors="pt", pad_to_multiple_of=8, max_length=None, use_dynamic_padding=True):
        super().__init__(tokenizer=tokenizer, mlm=mlm, return_tensors=return_tensors, pad_to_multiple_of=pad_to_multiple_of)
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.use_dynamic_padding = use_dynamic_padding
        
    def torch_call(self, examples):
        if self.use_dynamic_padding:
            # Use dynamic padding - find max length in this batch
            max_len_in_batch = max(len(ex['input_ids']) for ex in examples)
            
            # Optionally limit to global max_length
            if self.max_length is not None:
                max_len_in_batch = min(max_len_in_batch, self.max_length)
            
            # Pad to multiple for efficiency
            if self.pad_to_multiple_of is not None:
                max_len_in_batch = ((max_len_in_batch + self.pad_to_multiple_of - 1) 
                                   // self.pad_to_multiple_of * self.pad_to_multiple_of)
            
            # Manually pad each example to the batch max length
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            for example in examples:
                input_ids = example['input_ids'][:max_len_in_batch]
                attention_mask = example.get('attention_mask', [1] * len(input_ids))[:max_len_in_batch]
                labels = example.get('labels', input_ids.copy())[:max_len_in_batch]
                
                # Pad sequences to batch max length
                padding_length = max_len_in_batch - len(input_ids)
                if padding_length > 0:
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                    
                    input_ids = input_ids + [pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                    labels = labels + [self.ignore_index] * padding_length
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
            
            # Create batch dict
            batch = {
                'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
                'labels': torch.tensor(batch_labels, dtype=torch.long)
            }
            
            # Log occasionally to show dynamic padding is working
            if torch.rand(1).item() < 0.01:  # Log ~1% of batches
                logger.info(f"Dynamic batch - Max length in batch: {max_len_in_batch}, Batch size: {len(examples)}")
        else:
            # Use original super() method for fixed-length padding
            batch = super().torch_call(examples).data

        # Apply masking to labels
        input_ids = batch["input_ids"]
        labels = batch["labels"].clone()
        
        total_masked = 0
        total_unmasked = 0
        
        # Create masks for different roles
        for i, input_seq in enumerate(input_ids):
            labels[i] = self._mask_non_assistant_tokens(input_seq, labels[i])
            
            # Count masked vs unmasked tokens for this sequence
            masked_count = (labels[i] == self.ignore_index).sum().item()
            unmasked_count = (labels[i] != self.ignore_index).sum().item()
            
            total_masked += masked_count
            total_unmasked += unmasked_count
        
        # Print masking statistics
        total_tokens = total_masked + total_unmasked
        if total_tokens > 0:
            masked_pct = (total_masked / total_tokens) * 100
            unmasked_pct = (total_unmasked / total_tokens) * 100
            # log only if unmasked percentage is less than 5%
            if unmasked_pct < 5.0:
                logger.info(f"Masking stats - Total: {total_tokens}, Masked: {total_masked} ({masked_pct:.1f}%), Unmasked: {total_unmasked} ({unmasked_pct:.1f}%)")
        
        batch["labels"] = labels
        
        return batch
    
    def _tokenize_and_prepare(self, texts):
        """Tokenize texts and prepare batch."""
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=getattr(self.tokenizer, 'model_max_length', 2048),
            return_tensors="pt"
        )
        # Create labels from input_ids
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    def _mask_non_assistant_tokens(self, input_ids, labels):
        """Mask tokens that are not assistant responses."""
        # Convert to list for easier processing and ensure 1D
        input_ids_list = input_ids.squeeze().tolist() if input_ids.dim() > 1 else input_ids.tolist()
        labels_list = labels.squeeze().tolist() if labels.dim() > 1 else labels.tolist()
        original_loss_tokens = sum(1 for l in labels_list if l != self.ignore_index)
        
        # Special tokens for Qwen format - ensure we catch all variations
        system_start = self.tokenizer.encode("<|im_start|>system", add_special_tokens=False)
        user_start = self.tokenizer.encode("<|im_start|>user", add_special_tokens=False)
        assistant_start = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        function_start = self.tokenizer.encode("<|im_start|>function", add_special_tokens=False)
        im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        function_call_start = self.tokenizer.encode("<function_call>", add_special_tokens=False)
        function_call_end = self.tokenizer.encode("</function_call>", add_special_tokens=False)
        
        # Track current state and what we're masking
        current_role = "unknown"  # Track current role for debugging
        in_assistant_content = False  # Only true when in actual assistant response content
        in_function_call = False
        i = 0
        
        # Initially mask everything until we know the role
        while i < len(input_ids_list):
            # Check for role transitions
            if self._token_sequence_match(input_ids_list, i, system_start):
                current_role = "system"
                in_assistant_content = False
                in_function_call = False
                # Mask the entire system role marker and advance
                for j in range(len(system_start)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                i += len(system_start)
                
            elif self._token_sequence_match(input_ids_list, i, user_start):
                current_role = "user"
                in_assistant_content = False
                in_function_call = False
                # Mask the entire user role marker and advance
                for j in range(len(user_start)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                i += len(user_start)
                
            elif self._token_sequence_match(input_ids_list, i, assistant_start):
                current_role = "assistant"
                in_assistant_content = True
                in_function_call = False
                # Mask the assistant start tokens themselves (role marker should not be trained on)
                for j in range(len(assistant_start)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                i += len(assistant_start)
                
            elif self._token_sequence_match(input_ids_list, i, function_start):
                current_role = "function"
                in_assistant_content = False
                in_function_call = False
                # Mask the entire function role marker and advance
                for j in range(len(function_start)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                i += len(function_start)
                
            elif self._token_sequence_match(input_ids_list, i, function_call_start):
                # Function calls within assistant responses should be TRAINED ON (not masked)
                # Only set the flag to track we're in a function call, but don't mask
                in_function_call = True
                i += len(function_call_start)
                
            elif self._token_sequence_match(input_ids_list, i, function_call_end):
                in_function_call = False
                i += len(function_call_end)
                
            elif self._token_sequence_match(input_ids_list, i, im_end):
                # End of any role - mask the end marker and reset state
                for j in range(len(im_end)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                current_role = "unknown"
                in_assistant_content = False
                in_function_call = False
                i += len(im_end)
                
            else:
                # Apply masking based on current state
                should_mask = True
                
                if current_role == "assistant" and in_assistant_content:
                    # Train on ALL assistant content including function calls
                    should_mask = False
                    
                # Always mask: system instructions, user prompts, function outputs
                # Note: function calls within assistant responses are now trained on
                if current_role in ["system", "user", "function"]:
                    should_mask = True
                
                if should_mask:
                    labels_list[i] = self.ignore_index
                
                i += 1
        
        # Convert back to tensor - ensure proper shape
        result = torch.tensor(labels_list, dtype=labels.dtype)
        if labels.dim() > 1:
            result = result.view(labels.shape)
        
        # CRITICAL: Prevent completely masked samples (which cause division by zero)
        unmasked_count = (result != self.ignore_index).sum().item()
        if unmasked_count == 0:
            # If everything is masked, unmask the last non-padding token
            # Find the last meaningful token (not padding)
            for idx in range(len(result) - 1, -1, -1):
                if result[idx] != self.ignore_index and input_ids_list[idx] != self.tokenizer.pad_token_id:
                    result[idx] = input_ids_list[idx]
                    break
            else:
                # Fallback: unmask the last token regardless
                if len(result) > 0:
                    result[-1] = input_ids_list[-1] if len(input_ids_list) > 0 else 0
        
        return result
    
    def _token_sequence_match(self, input_ids, start_idx, target_sequence):
        """Check if token sequence matches at given position."""
        if start_idx + len(target_sequence) > len(input_ids):
            return False
        return input_ids[start_idx:start_idx + len(target_sequence)] == target_sequence


def load_experiences(data_dir):
    """Load and process conversation experiences."""
    experiences = []
    logger.info(f"Loading experiences from: {data_dir}")
    
    idx = 0
    for root, _, files in os.walk(data_dir):
        for filename in files:
            idx += 1
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                if "old-agent-0.1" in file_path:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            processed = process_old_0_1_conversation(data)
                            if processed:
                                experiences.append({
                                    "functions": [],
                                    "messages": processed
                                })
                    except Exception as e:
                        logger.warning(f"Error processing [{idx}] {file_path}: {e}")
                else:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            experiences.append(data)
                    except Exception as e:
                        logger.warning(f"Error processing [{idx}] {file_path}: {e}")
    
    logger.info(f"Loaded {len(experiences)} conversations")
    return experiences

def process_old_0_1_conversation(data):
    """Process and clean conversation data."""
    processed = []
    function_name = None
    
    for msg in data:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            processed.append({"role": "system", "content": content})
        elif role == "user":
            if isinstance(content, dict) and content.get("type") == "function_call_output":
                processed.append({
                    "role": "function",
                    "name": function_name or "unknown",
                    "content": content["output"]
                })
            elif isinstance(content, str):
                processed.append({"role": "user", "content": content})
        elif role == "assistant":
            if isinstance(content, dict) and content.get("type") == "function_call":
                function_name = content["name"]
                processed.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": content["name"],
                        "arguments": content["arguments"]
                    }
                })
            elif isinstance(content, list) and content:
                if content[0].get("type") == "output_text":
                    processed.append({"role": "assistant", "content": content[0]["text"]})
            elif isinstance(content, str):
                processed.append({"role": "assistant", "content": content})
    
    return processed


def format_conversation(example) -> str:
    """Format conversations for training with token length validation."""
    messages = example.get("messages", [])
    functions = example.get("functions", [])

    conversation = ""
    system_message_found = False
    prev_role = None
    
    for message in messages:
        role = message["role"]
        content = message.get("content", "")
        
        if role == "system":
            system_message_found = True
            # If we have functions, incorporate them into the system message
            if functions:
                functions_text = "You have access to the following functions:\n\n"
                for func in functions:
                    if not func.get("name"):
                        raise ValueError(f"Function name is required: {func}")
                    functions_text += f"Function: {func.get('name', 'unknown')}\n"
                    if 'description' in func:
                        functions_text += f"Description: {func['description']}\n"
                    if 'parameters' in func:
                        functions_text += f"Parameters: {json.dumps(func['parameters'], indent=2)}\n"
                    functions_text += "\n"
                
                # Combine original system content with functions
                enhanced_content = content
                if content and not content.endswith('\n'):
                    enhanced_content += "\n\n"
                elif not content:
                    enhanced_content = ""
                enhanced_content += functions_text.rstrip()
                
                conversation += f"<|im_start|>system\n{enhanced_content}<|im_end|>\n"
            else:
                conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
            prev_role = "system"
        elif role == "user":
            # If no system message was found but we have functions, add them at the beginning
            if not system_message_found and functions:
                functions_text = "You have access to the following functions:\n\n"
                for func in functions:
                    functions_text += f"Function: {func.get('name', 'unknown')}\n"
                    if 'description' in func:
                        functions_text += f"Description: {func['description']}\n"
                    if 'parameters' in func:
                        functions_text += f"Parameters: {json.dumps(func['parameters'], indent=2)}\n"
                    functions_text += "\n"
                
                conversation = f"<|im_start|>system\n{functions_text.rstrip()}<|im_end|>\n" + conversation
                system_message_found = True
            
            conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
            prev_role = "user"
        elif role == "assistant":
            if prev_role == "assistant":
                # Merge with previous assistant message - remove the last <|im_end|>\n and append content
                if conversation.endswith("<|im_end|>\n"):
                    conversation = conversation[:-11]  # Remove "<|im_end|>\n"
                    # Add the new content
                    if "function_call" in message:
                        func_call = message["function_call"]
                        conversation += f"\n\n<function_call>\n{json.dumps(func_call)}\n</function_call>"
                    if content:
                        # Handle complex content structure (list of objects with text)
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    conversation += "\n\n" + item["text"]
                                elif isinstance(item, str):
                                    conversation += "\n\n" + item
                        else:
                            conversation += "\n\n" + content
                    conversation += "<|im_end|>\n"
            else:
                # Start new assistant message
                conversation += f"<|im_start|>assistant\n"
                if "function_call" in message:
                    func_call = message["function_call"]
                    conversation += f"<function_call>\n{json.dumps(func_call)}\n</function_call>"
                if content:
                    # Handle complex content structure (list of objects with text)
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                conversation += item["text"]
                            elif isinstance(item, str):
                                conversation += item
                    else:
                        conversation += content
                conversation += "<|im_end|>\n"
            prev_role = "assistant"
        elif role == "function":
            if not "name" in message:
                raise ValueError(f"Function name is required: {message}")
            name = message.get("name")
            conversation += f"<|im_start|>function\nname={name}\n{content}<|im_end|>\n"
            prev_role = "function"

    return conversation


# this will filter out examples that are too long
def create_dataset(experiences, tokenizer, max_length, local_rank=0):
    """Create and prepare training dataset with token length validation."""
    """Create and prepare training dataset with token length validation."""
    if not tokenizer:
        raise ValueError("Tokenizer is required")

    formatted_records = []
    for example_idx, example in enumerate(experiences):
        try:
            result = format_conversation(example)
            # use tokenizer to tokenize the result
            tokens = tokenizer.encode(result, add_special_tokens=True)
            if max_length > 0 and len(tokens) > max_length:
                logger.warning(f"Example [{example_idx}]: Token length {len(tokens)} exceeds max length {max_length} - skipping")
                continue
            if tokens is not None:
                formatted_records.append({
                    "input_ids": tokens,
                    # "text": result,
                })
        except Exception as e:
            if local_rank == 0:
                # print stack trace
                logger.warning(f"Error formatting example [{example_idx}]: {e}")
                logger.warning(traceback.format_exc())
            continue

    formatted_dataset = Dataset.from_list(formatted_records)
    if local_rank == 0:
        logger.info(f"Created dataset with {len(formatted_dataset)} examples")

    return formatted_dataset 

def print_masking_analysis(batch, tokenizer):
    """Print detailed analysis of masked vs unmasked tokens in a batch."""
    buffer = ""
    for example_idx in range(len(batch["input_ids"])):
        prev_masked = None
        logger.info("=" * 50)
        logger.info(f"EXAMPLE_IDX: [{example_idx}]")
        logger.info("-" * 50)
        for i in range(len(batch["input_ids"][example_idx])):
            is_masked = batch["labels"][example_idx][i] == -100
            if is_masked != prev_masked:
                masked_str = f"\nEXAMPLE [{example_idx}]: {'MASKED' if prev_masked else 'UNMASKED'}\n"
                if prev_masked is not None:
                    if prev_masked and buffer != "":
                        logger.info(masked_str + buffer)
                    elif buffer != "":
                        logger.warning(masked_str + buffer)
                buffer = ""
            # print token if not padding or eos
            token_id = batch["input_ids"][example_idx][i]
            if token_id != tokenizer.pad_token_id and token_id != tokenizer.eos_token_id:
                # print tokenizer.decode(token_id) without new line
                buffer += tokenizer.decode(token_id)
            prev_masked = is_masked

        if buffer != "":
            masked_str = f"\nEXAMPLE [{example_idx}]: {'MASKED' if prev_masked else 'UNMASKED'}\n"
            if prev_masked:
                logger.info(masked_str + buffer)
            else:
                logger.warning(masked_str + buffer)

        logger.info("=" * 50)
        logger.info("DONE")

def test_data_util():
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    experiences = [{
        "functions": [],
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is today's date?"
            },
            {
                "role": "assistant",
                "content": "I will use the get_current_time function to get the current time."
            },
            {
                "role": "assistant",
                "function_call": {
                    "name": "get_current_time",
                    "arguments": {}
                }
            },
            {
                "role": "function",
                "name": "get_current_time",
                "content": "2025-06-14 10:00:00"
            },
            {
                "role": "assistant",
                "content": "Today's date is 2025-06-14."
            }
        ]
    },
    {
        "functions": [],
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the weather in Tokyo?"
            },
            {
                "role": "assistant",
                "content": "I will use the get_weather function to get the weather in Tokyo."
            },
            {
                "role": "assistant",
                "function_call": {
                    "name": "get_weather",
                    "arguments": {
                        "city": "Tokyo"
                    }
                }
            },
            {
                "role": "function",
                "name": "get_weather",
                "content": "Tokyo's weather is sunny."
            },
            {
                "role": "assistant",
                "content": "The weather in Tokyo is sunny."
            }
        ]
    }]
    dataset = create_dataset(experiences, tokenizer, 4096, 0)

    collate_fn = CustomDataCollatorWithMasking(tokenizer, mlm=False, return_tensors="pt", max_length=4096)

    # Pass it to dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=2)

    # this will end in error
    for batch in dataloader:
        # recursively convert batch data from Tensor to list
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.tolist()

        print(json.dumps(batch, indent=4))

        # print masked vs unmasked tokens
        print_masking_analysis(batch, tokenizer)

def process_data(args):
    """Process and clean conversation data."""
    experiences = load_experiences(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    dataset = create_dataset(experiences, tokenizer, args.max_length, 0)

    collate_fn = CustomDataCollatorWithMasking(tokenizer, mlm=False, return_tensors="pt", max_length=None if args.max_length <=0 else args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    idx = 0
    for batch in dataloader:
        # recursively convert batch data from Tensor to list
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.tolist()

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            for example_idx in range(len(batch["input_ids"])):
                idx += 1
                filename = f"experience_{idx:04d}.jsonl"
                # delete file if it exists
                if os.path.exists(os.path.join(args.output_dir, filename)):
                    os.remove(os.path.join(args.output_dir, filename))
                with open(os.path.join(args.output_dir, filename), "a") as f:
                    example = {
                        "input_ids": batch["input_ids"][example_idx],
                        "attention_mask": batch["attention_mask"][example_idx],
                        "labels": batch["labels"][example_idx],
                    }
                    f.write(json.dumps(example) + "\n")

    logger.info(f"Processed {idx} experiences")

    if args.upload_to_hf:
        # create repo if it doesn't exist
        api = HfApi()
        if not api.repo_exists(args.upload_to_hf):
            logger.info(f"Creating repo {args.upload_to_hf} ...")
            api.create_repo(args.upload_to_hf, repo_type="dataset", private=False)
            logger.info(f"Repo {args.upload_to_hf} created .")
        else:
            logger.info(f"Repo {args.upload_to_hf} already exists .")

        logger.info(f"Uploading {idx} experiences to HF ...")
        api = HfApi()
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.upload_to_hf,
            repo_type="dataset",
        )
        logger.info(f"Uploaded {idx} experiences to HF .")

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--process", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default="./finetune_raw_experiences")
    parser.add_argument("--output_dir", type=str, default="./finetune_processed_experiences")
    parser.add_argument("--upload_to_hf", type=str, default="dtadpole/finetune_processed_experiences")
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if args.test:
        test_data_util()

    if args.process:
        process_data(args)