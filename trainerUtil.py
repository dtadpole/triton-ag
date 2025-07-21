import torch
import json
from transformers import AutoTokenizer
from logger import logger
from typing import List, Dict, Any


def format_conversation(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer, mask_non_assistant_tokens: bool = True, ignore_index: int = -100) -> Dict[str, Any]:
    # Format conversation using chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        try:
            # Use the tokenizer's chat template
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"⚠️ [MessageDataset] Chat template failed, falling back to manual formatting: {e}")
            formatted_text = _manual_format_conversation(messages)
    else:
        # Manual formatting for models without chat template
        formatted_text = _manual_format_conversation(messages)
    
    # Tokenize the formatted conversation
    encoding = tokenizer(
        formatted_text,
        truncation=False,
        padding=False,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].squeeze()
    attention_mask = encoding['attention_mask'].squeeze()
    
    # Create labels for loss computation
    labels = input_ids.clone()
    
    if mask_non_assistant_tokens:
        # Mask user tokens if requested (only train on assistant responses)
        labels = _mask_non_assistant_tokens(input_ids, labels, tokenizer, ignore_index)
        # convert labels to tensor
        labels = torch.tensor(labels)
    
    return {
        'text': formatted_text,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
def _manual_format_conversation(messages: List[Dict[str, str]]) -> str:
    """Manually format conversation when chat template is not available"""
    formatted_parts = []
    
    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if role == 'system':
            formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == 'user':
            formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == 'assistant':
            formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == 'tool':
            formatted_parts.append(f"<|im_start|>tool\n{content}<|im_end|>")
        elif role == 'function':
            formatted_parts.append(f"<|im_start|>function\n{content}<|im_end|>")
        else:
            raise ValueError(f"Unknown role: {role}")
    
    return '\n'.join(formatted_parts)

def _token_sequence_match(input_ids, start_idx, target_sequence):
    """Check if token sequence matches at given position."""
    if start_idx + len(target_sequence) > len(input_ids):
        return False
    return input_ids[start_idx:start_idx + len(target_sequence)] == target_sequence

def _mask_non_assistant_tokens(input_ids, labels, tokenizer, ignore_index) -> torch.Tensor:
    """Mask tokens that are not assistant responses."""
    # Convert to list for easier processing and ensure 1D
    input_ids_list = input_ids.squeeze().tolist() if input_ids.dim() > 1 else input_ids.tolist()
    labels_list = labels.squeeze().tolist() if labels.dim() > 1 else labels.tolist()

    # Special tokens for Qwen format - ensure we catch all variations
    system_start = tokenizer.encode("<|im_start|>system", add_special_tokens=False)
    user_start = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
    assistant_start = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    tool_start = tokenizer.encode("<|im_start|>tool", add_special_tokens=False)
    function_start = tokenizer.encode("<|im_start|>function", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    tool_call_start = tokenizer.encode("<tool_call>", add_special_tokens=False)
    tool_call_end = tokenizer.encode("</tool_call>", add_special_tokens=False)

    # Track current state and what we're masking
    current_role = "unknown"  # Track current role for debugging
    in_assistant_content = False  # Only true when in actual assistant response content
    i = 0

    # Initially mask everything until we know the role
    while i < len(input_ids_list):
        # Check for role transitions
        if _token_sequence_match(input_ids_list, i, system_start):
            current_role = "system"
            in_assistant_content = False
            # Mask the entire system role marker and advance
            for j in range(len(system_start)):
                if i + j < len(labels_list):
                    labels_list[i + j] = ignore_index
            i += len(system_start)

        elif _token_sequence_match(input_ids_list, i, user_start):
            current_role = "user"
            in_assistant_content = False
            # Mask the entire user role marker and advance
            for j in range(len(user_start)):
                if i + j < len(labels_list):
                    labels_list[i + j] = ignore_index
            i += len(user_start)

        elif _token_sequence_match(input_ids_list, i, assistant_start):
            current_role = "assistant"
            in_assistant_content = True
            # Mask the assistant start tokens themselves (role marker should not be trained on)
            for j in range(len(assistant_start)):
                if i + j < len(labels_list):
                    labels_list[i + j] = ignore_index
            i += len(assistant_start)

        elif _token_sequence_match(input_ids_list, i, tool_start):
            current_role = "tool"
            in_assistant_content = False
            # Mask the entire function role marker and advance
            for j in range(len(tool_start)):
                if i + j < len(labels_list):
                    labels_list[i + j] = ignore_index
            i += len(tool_start)

        elif _token_sequence_match(input_ids_list, i, function_start):
            current_role = "function"
            in_assistant_content = False
            # Mask the entire function role marker and advance
            for j in range(len(function_start)):
                if i + j < len(labels_list):
                    labels_list[i + j] = ignore_index
            i += len(function_start)

        elif _token_sequence_match(input_ids_list, i, tool_call_start):
            # Function calls within assistant responses should be TRAINED ON (not masked)
            # Only set the flag to track we're in a function call, but don't mask
            i += len(tool_call_start)

        elif _token_sequence_match(input_ids_list, i, tool_call_end):
            i += len(tool_call_end)

        elif _token_sequence_match(input_ids_list, i, im_end):
            # End of any role - mask the end marker and reset state
            for j in range(len(im_end)):
                if i + j < len(labels_list):
                    labels_list[i + j] = ignore_index
            current_role = "unknown"
            in_assistant_content = False
            i += len(im_end)

        else:
            # Apply masking based on current state
            should_mask = True

            if current_role == "assistant" and in_assistant_content:
                # Train on ALL assistant content including function calls
                should_mask = False

            # Always mask: system instructions, user prompts, function outputs
            # Note: function calls within assistant responses are now trained on
            if current_role in ["system", "user", "tool", "function"]:
                should_mask = True

            if should_mask:
                labels_list[i] = ignore_index

            i += 1

    # Convert back to tensor - ensure proper shape
    result = torch.tensor(labels_list, dtype=labels.dtype)
    if labels.dim() > 1:
        result = result.view(labels.shape)

    return result

class SimpleCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8, ignore_index=-100):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.ignore_index = ignore_index
        
    def __call__(self, batch):
        # Extract sequences
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        
        # Find max length
        max_len = max(len(seq) for seq in input_ids)
        
        # Round up to multiple if specified
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad input_ids
        input_ids_padded = []
        for seq in input_ids:
            pad_len = max_len - len(seq)
            padded = torch.cat([seq, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            input_ids_padded.append(padded)
        
        result = {'input_ids': torch.stack(input_ids_padded)}
        
        # Handle attention_mask if present
        if 'attention_mask' in batch[0]:
            attention_masks = []
            for item in batch:
                # if item['attention_mask'] is already a tensor, use it directly
                if isinstance(item['attention_mask'], torch.Tensor):
                    mask = item['attention_mask']
                else:
                    mask = torch.tensor(item['attention_mask'])
                pad_len = max_len - len(mask)
                padded_mask = torch.cat([mask, torch.zeros(pad_len)])
                attention_masks.append(padded_mask)
            result['attention_mask'] = torch.stack(attention_masks)
        
        # Handle labels if present
        if 'labels' in batch[0]:
            labels = []
            for item in batch:
                # if item['labels'] is already a tensor, use it directly
                if isinstance(item['labels'], torch.Tensor):
                    label = item['labels']
                else:
                    label = torch.tensor(item['labels'])
                pad_len = max_len - len(label)
                padded_label = torch.cat([label, torch.full((pad_len,), self.ignore_index)])
                labels.append(padded_label)
            result['labels'] = torch.stack(labels)

        # for all other keys, do not pad, do not convert to tensor
        for key in batch[0]:
            if key not in ['input_ids', 'attention_mask', 'labels']:
                values = []
                for item in batch:
                    values.append(item[key])
                result[key] = values
        
        return result

def print_masking_analysis(batch, tokenizer):
    """Print detailed analysis of masked vs unmasked tokens in a batch."""
    prev_masked = None
    logger.info("=" * 50)
    buffer = ""
    for i in range(len(batch["input_ids"])):
        is_masked = batch["labels"][i] == -100
        if is_masked != prev_masked:
            masked_str = f"\n{'MASKED' if prev_masked else 'UNMASKED'}\n"
            if prev_masked is not None:
                if prev_masked and buffer != "":
                    logger.info(masked_str + buffer)
                elif buffer != "":
                    logger.warning(masked_str + buffer)
            buffer = ""
        # print token if not padding or eos
        token_id = batch["input_ids"][i]
        if token_id != tokenizer.pad_token_id:
            # print tokenizer.decode(token_id) without new line
            buffer += tokenizer.decode(token_id)
        prev_masked = is_masked

    if buffer != "":
        masked_str = f"\n{'MASKED' if prev_masked else 'UNMASKED'}\n"
        if prev_masked:
            logger.info(masked_str + buffer)
        else:
            logger.warning(masked_str + buffer)

    logger.info("=" * 50)
    logger.info("DONE")


def test_data_util():
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    messages = [{
        "tools": [],
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
                "content": "I will use the get_current_time function to get the current time.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_time",
                            "arguments": {}
                        }
                    }
                ]
            },
            {
                "role": "tool",
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
        "tools": [],
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
                "content": "I will use the get_weather function to get the weather in Tokyo.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "city": "Tokyo"
                            }
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": json.dumps({
                    "city": "Tokyo",
                    "weather": "sunny"
                })
            },
            {
                "role": "assistant",
                "content": "The weather in Tokyo is sunny."
            }
        ]
    }]

    for message in messages:
        formatted_data = format_conversation(message["messages"], tokenizer, mask_non_assistant_tokens=True)
        logger.info(formatted_data['text'])
        print_masking_analysis(formatted_data, tokenizer)

if __name__ == "__main__":
    test_data_util()
