import json
import yaml
import numpy as np
import pandas as pd
import argparse
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer
from logger import logger
from workflowUtil import TrainerBlock
from typing import List, Dict, Any, Union, Callable
import warnings
import asyncio
import os
import httpx
import aiohttp

# Suppress specific warnings (modified by DevMate)
# warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

async def read_stream(stream, prefix: str, is_error: bool = False):
    """Read from a stream and print each line with a prefix."""
    while True:
        line = await stream.readline()
        if not line:
            break
        # Decode bytes to string and strip newline
        output = line.decode("utf-8").rstrip()
        if is_error:
            logger.error(f"[{prefix}] {output}")
        else:
            logger.info(f"[{prefix}] {output}")

def make_checkpoint_callback(
    prefix_tag: str,
    trainer_block: TrainerBlock,
    workflow_provider: str = "default",
    env: dict = {},
):
    """Make a callback function for the trainer"""
    def callback_func(checkpoint_path: str):
        """Callback function for the trainer"""
        from workflowClient import WorkflowClient
        workflow_client = WorkflowClient(prefix_tag=prefix_tag, provider_name=workflow_provider)
        logger.info(f"📞 [SyncClient] Enqueue path: [{checkpoint_path}] ...")
        checkpoint_name = '/'.join(str(os.path.expanduser(checkpoint_path)).split('/')[-2:])
        logger.info(f"📞 [SyncClient] Enqueue name: [{checkpoint_name}] ...")
        # create task to enqueue
        loop = asyncio.get_event_loop()
        task = loop.create_task(
            workflow_client.callback(
                callback_kind="checkpoint",
                queue_type=trainer_block.queue_type,
                queue_name=trainer_block.queue_name,
                block=trainer_block,
                env=env | {"checkpoint_name": checkpoint_name },
            )
        )
        # run task in background
        logger.info(f"📞 [SyncClient] Enqueue name: [{checkpoint_name}] done.")
        return task
    # return the callback function
    return callback_func

def format_conversation(messages: List[Dict[str, Any]],
                        tokenizer: AutoTokenizer,
                        mask_non_assistant_tokens: bool = True, # mask all token except assistant tokens
                        mask_non_last_assistant_tokens: bool = False, # mask all token except last assistant tokens
                        ignore_index: int = -100,
                        tools: List[Union[Dict, Callable]] = [],
                        messages_from_openai_agent: bool = False,
                        use_custom_chat_template_for_masking: bool = True) -> Dict[str, Any]:
    # Always format conversation using chat template to keep the formatted text consistent with pre-trained model
    if messages_from_openai_agent is True:
        raise ValueError("OpenAI Agent format is not supported yet")

    if use_custom_chat_template_for_masking is False:
        try:
            # Use the tokenizer's chat template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                toosl = tools,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"⚠️ [MessageDataset] Chat template failed, falling back to manual formatting: {e}")
            # Follow the model's chat template is very important, otherwise the model will not transfer the learned patterns from pre-training
            raise ValueError(f"Chat template failed, please fix the chat template for your model before continue: {e}")

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
            labels = _mask_non_assistant_tokens(input_ids, labels, tokenizer, ignore_index=ignore_index, mask_non_last_assistant_tokens=mask_non_last_assistant_tokens)
            # convert labels to tensor
            labels = torch.tensor(labels)
    else:
        try:
            if mask_non_last_assistant_tokens:
                custom_chat_template = "\n".join(qwen3_custom_chat_template_list_last_assistant)
            else:
                custom_chat_template = "\n".join(qwen3_custom_chat_template_list)
            # Use the tokenizer's chat template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                toosl = tools,
                chat_template=custom_chat_template,
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=False
            )
            encoding = tokenizer.apply_chat_template(
                messages,
                toosl = tools,
                chat_template=custom_chat_template,
                return_dict=True,
                tokenize=True,
                enable_thinking=True,
                return_tensors='pt',
                return_assistant_tokens_mask=True,
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"⚠️ [MessageDataset] Chat template failed, falling back to manual formatting: {e}")
            # Follow the model's chat template is very important, otherwise the model will not transfer the learned patterns from pre-training
            raise ValueError(f"Chat template failed, please fix the chat template for your model before continue: {e}")

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        assistant_mask = encoding['assistant_masks'].squeeze().bool()
        if mask_non_last_assistant_tokens:
            assistant_mask = torch.tensor(extract_last_assistant_mask(assistant_mask))
        labels = labels.masked_fill(~assistant_mask, ignore_index)
    return {
            'text': formatted_text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def extract_last_assistant_mask(mask: List[bool]) -> List[bool]:
    """Extract the last continuous assistant mask from the mask"""
    in_block = False
    start = end = None
    for i in reversed(range(len(mask))):
        if mask[i]:
            if not in_block:
                end = i
                in_block = True
            start = i
        elif in_block:
            break
    last_mask = [False] * len(mask)
    if start is not None and end is not None:
        for i in range(start, end + 1):
            last_mask[i] = True
    return last_mask

def _manual_format_conversation(messages: List[Dict[str, str]], messages_from_openai_agent: bool = True) -> str:
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

def _mask_non_assistant_tokens(input_ids, labels, tokenizer, ignore_index=-100, mask_non_last_assistant_tokens: bool = False) -> torch.Tensor:
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
    # Missing <tool_response> and </tool_response> for now

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
        input_ids = [
            item['input_ids'].detach().cpu().clone()
                if isinstance(item['input_ids'], torch.Tensor)
                else torch.tensor(item['input_ids'])
            for item in batch
        ]

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


def softmax_temperature_sampling(
    candidates_df,
    rewards_df,
    n_samples,
    temperature=1.0,
    min_exploration=0.05,
    task_id_col='task_id',
    reward_col='reward',
    return_probabilities=False,
    random_state=None
):
    """
    Sample candidates based on inverse normalized task rewards.

    The function:
    1. Normalizes rewards to [0, 1] range (0=worst, 1=best)
    2. Inverts them: sampling_weight = (1 - normalized_reward)^(1/temperature)
    3. Low-performing tasks (reward→0, normalized→0) get high sampling weight (→1)
    4. High-performing tasks (reward→1, normalized→1) get low sampling weight (→0)

    Parameters:
    -----------
    candidates_df : pd.DataFrame
        DataFrame containing candidate information
        Must have column specified by task_id_col

    rewards_df : pd.DataFrame
        DataFrame containing task rewards/performance
        Must have columns specified by task_id_col and reward_col

    n_samples : int
        Number of candidates to sample

    temperature : float, default=1.0
        Controls the sampling distribution:
        - Lower (e.g., 0.1-0.5): Strong focus on hard/low-reward tasks
        - 1.0: Linear inverse relationship
        - Higher (e.g., 2.0-5.0): More uniform sampling

    min_exploration : float, default=0.05
        Minimum probability for each task to prevent curriculum collapse
        Should be in range [0, 1]

    task_id_col : str, default='task_id'
        Name of the column containing task IDs (used in both DataFrames)

    reward_col : str, default='reward'
        Name of the column containing rewards in rewards_df

    return_probabilities : bool, default=False
        If True, also return the sampling probabilities for each candidate

    random_state : int or None, default=None
        Random seed for reproducibility

    Returns:
    --------
    sampled_df : pd.DataFrame
        Sampled candidates (low-reward tasks are sampled more frequently)

    probabilities : pd.Series (optional)
        Sampling probability for each candidate (if return_probabilities=True)
    """

    # Validate column names exist
    if task_id_col not in candidates_df.columns:
        raise ValueError(f"Column '{task_id_col}' not found in candidates_df. "
                        f"Available columns: {list(candidates_df.columns)}")

    if task_id_col not in rewards_df.columns:
        raise ValueError(f"Column '{task_id_col}' not found in rewards_df. "
                        f"Available columns: {list(rewards_df.columns)}")

    if reward_col not in rewards_df.columns:
        raise ValueError(f"Column '{reward_col}' not found in rewards_df. "
                        f"Available columns: {list(rewards_df.columns)}")

    # Validate inputs
    if n_samples > len(candidates_df):
        raise ValueError(f"n_samples ({n_samples}) cannot exceed number of candidates ({len(candidates_df)})")

    if not 0 <= min_exploration <= 1:
        raise ValueError(f"min_exploration must be in [0, 1], got {min_exploration}")

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Step 1: Normalize rewards to [0, 1]
    rewards_work = rewards_df.copy()

    min_reward = rewards_work[reward_col].min()
    max_reward = rewards_work[reward_col].max()

    if max_reward == min_reward:
        # All rewards are the same, use uniform sampling
        rewards_work['normalized_reward'] = 0.5
    else:
        # Min-max normalization
        rewards_work['normalized_reward'] = (rewards_work[reward_col] - min_reward) / (max_reward - min_reward)

    # Step 2: Compute inverse with temperature
    # Task with normalized_reward = 1.0 → inverse = 0.0 → almost no samples
    # Task with normalized_reward = 0.0 → inverse = 1.0 → most samples
    inverse_scores = 1.0 - rewards_work['normalized_reward']

    # Apply temperature: raise to power (1/temperature)
    # Lower temperature → more extreme differences
    # Higher temperature → more uniform
    inverse_scores_temp = np.power(inverse_scores, 1.0 / temperature)

    # Step 3: Convert to probabilities (normalize to sum to 1)
    if inverse_scores_temp.sum() == 0:
        # Edge case: all scores are 0
        task_probs = np.ones(len(rewards_work)) / len(rewards_work)
    else:
        task_probs = inverse_scores_temp / inverse_scores_temp.sum()

    # Step 4: Add minimum exploration probability
    n_tasks = len(rewards_work)
    uniform_prob = 1.0 / n_tasks
    task_probs = (1 - min_exploration) * task_probs + min_exploration * uniform_prob
    task_probs = task_probs / task_probs.sum()

    # Create task probability mapping
    rewards_work['task_prob'] = task_probs
    task_prob_map = rewards_work[[task_id_col, 'task_prob']].set_index(task_id_col)['task_prob'].to_dict()

    # Step 5: Assign probabilities to candidates based on their task
    candidates_work = candidates_df.copy()
    candidates_work['candidate_prob'] = candidates_work[task_id_col].map(task_prob_map)

    # Check for missing task mappings
    if candidates_work['candidate_prob'].isna().any():
        missing_tasks = candidates_work[candidates_work['candidate_prob'].isna()][task_id_col].unique()
        raise ValueError(f"Missing rewards for {task_id_col}(s): {missing_tasks}")

    # Normalize candidate probabilities to sum to 1
    candidate_probs = candidates_work['candidate_prob'].values
    candidate_probs = candidate_probs / candidate_probs.sum()

    # Step 6: Sample candidates based on probabilities
    sampled_indices = np.random.choice(
        len(candidates_work),
        size=n_samples,
        replace=False,
        p=candidate_probs
    )

    # Get sampled candidates
    sampled_df = candidates_df.iloc[sampled_indices].copy()
    sampled_df = sampled_df.reset_index(drop=True)

    if return_probabilities:
        sampled_probs = pd.Series(candidate_probs[sampled_indices])
        return sampled_df, sampled_probs.reset_index(drop=True)

    return sampled_df



def test_data_util():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    args.add_argument("--mask_non_assistant_tokens", type=bool, default=True)
    args.add_argument("--mask_non_last_assistant_tokens", type=bool, default=True)
    args = args.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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
    }
    ]

    for message in messages:
        ## Use chat_completion to get the formatted text
        formatted_data = format_conversation(message["messages"], tokenizer, tools=message["tools"], mask_non_assistant_tokens=args.mask_non_assistant_tokens, mask_non_last_assistant_tokens=args.mask_non_last_assistant_tokens, use_custom_chat_template_for_masking=False)
        logger.info(f"🔍 [test_data_util] Formatted text: {formatted_data['text']}")
        print_masking_analysis(formatted_data, tokenizer)
        ## Use the tokenizer to get the formatted text and masks
        formatted_data = format_conversation(message["messages"], tokenizer, tools=message["tools"], mask_non_assistant_tokens=args.mask_non_assistant_tokens, mask_non_last_assistant_tokens=False, use_custom_chat_template_for_masking=True)
        logger.info(f"🔍 [test_data_util] Formatted text: {formatted_data['text']}")
        print_masking_analysis(formatted_data, tokenizer)

        ## Use the tokenizer to get the formatted text and masks
        formatted_data = format_conversation(message["messages"], tokenizer, tools=message["tools"], mask_non_assistant_tokens=args.mask_non_assistant_tokens, mask_non_last_assistant_tokens=True, use_custom_chat_template_for_masking=True)
        logger.info(f"🔍 [test_data_util] Formatted text: {formatted_data['text']}")
        print_masking_analysis(formatted_data, tokenizer)

# added the support for return_assistant_tokens_mask in apply_chat_template
# {% generation %} and {% endgeneration %} are used to mark the assistant tokens
qwen3_custom_chat_template_list_last_assistant = ['{%- if tools %}',
 "    {{- '<|im_start|>system\\n' }}",
 "    {%- if messages[0].role == 'system' %}",
 "        {{- messages[0].content + '\\n\\n' }}",
 '    {%- endif %}',
 '    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}',
 '    {%- for tool in tools %}',
 '        {{- "\\n" }}',
 '        {{- tool | tojson }}',
 '    {%- endfor %}',
 '    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}',
 '{%- else %}',
 "    {%- if messages[0].role == 'system' %}",
 "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}",
 '    {%- endif %}',
 '{%- endif %}',
 '{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1, last_assistant_index=-1) %}',
 '{%- for message in messages[::-1] %}',
 '    {%- set index = (messages|length - 1) - loop.index0 %}',
 '    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith(\'<tool_response>\') and message.content.endswith(\'</tool_response>\')) %}',
 '        {%- set ns.multi_step_tool = false %}',
 '        {%- set ns.last_query_index = index %}',
 '    {%- endif %}',
 '    {%- if ns.last_assistant_index == -1 and message.role == "assistant" %}',
 '        {%- set ns.last_assistant_index = index %}',
 '    {%- endif %}',
 '{%- endfor %}',
 '{%- for message in messages %}',
 '    {%- if message.content is string %}',
 '        {%- set content = message.content %}',
 '    {%- else %}',
 "        {%- set content = '' %}",
 '    {%- endif %}',
 '    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}',
 "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}",
 '    {%- elif message.role == "assistant" %}',
 "        {%- set reasoning_content = '' %}",
 '        {%- if message.reasoning_content is string %}',
 '            {%- set reasoning_content = message.reasoning_content %}',
 '        {%- else %}',
 "            {%- if '</think>' in content %}",
 "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}",
 "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}",
 '            {%- endif %}',
 '        {%- endif %}',
 '        {%- if loop.index0 == ns.last_assistant_index %}',
 '            {%- if loop.last or (not loop.last and reasoning_content) %}',
 "                {% generation %}{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }} {% endgeneration %}",
 '            {%- else %}',
 "                {% generation %}{{- '<|im_start|>' + message.role + '\\n' + content }}{% endgeneration %}",
 '            {%- endif %}',
 '        {%- else %}',
 "            {{- '<|im_start|>' + message.role + '\\n' + content }}",
 '        {%- endif %}',
 '        {%- if message.tool_calls %}',
 '            {%- for tool_call in message.tool_calls %}',
 '                {%- if (loop.first and content) or (not loop.first) %}',
 "                    {{- '\\n' }}",
 '                {%- endif %}',
 '                {%- if tool_call.function %}',
 '                    {%- set tool_call = tool_call.function %}',
 '                {%- endif %}',
 '                {%- if loop.index0 == ns.last_assistant_index %}',
 '                    {% generation %}{{- \'<tool_call>\\n{"name": "\' }}',
 '                    {{- tool_call.name }}',
 '                    {{- \'", "arguments": \' }}',
 '                    {%- if tool_call.arguments is string %}',
 '                        {{- tool_call.arguments }}',
 '                    {%- else %}',
 '                        {{- tool_call.arguments | tojson }}',
 '                    {%- endif %}',
 "                    {{- '}\\n</tool_call>' }}{% endgeneration %}",
 '                {%- else %}',
 '                    {{- \'<tool_call>\\n{"name": "\' }}',
 '                    {{- tool_call.name }}',
 '                    {{- \'", "arguments": \' }}',
 '                    {%- if tool_call.arguments is string %}',
 '                        {{- tool_call.arguments }}',
 '                    {%- else %}',
 '                        {{- tool_call.arguments | tojson }}',
 '                    {%- endif %}',
 "                    {{- '}\\n</tool_call>' }}",
 '                {%- endif %}',
 '            {%- endfor %}',
 '        {%- endif %}',
 '        {%- if loop.index0 == ns.last_assistant_index %}',
 "            {% generation %}{{- '<|im_end|>\\n' }}{% endgeneration %}",
 '        {%- else %}',
 "            {{- '<|im_end|>\\n' }}",
 '        {%- endif %}',
 '    {%- elif message.role == "tool" %}',
 '        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}',
 "            {{- '<|im_start|>user' }}",
 '        {%- endif %}',
 "        {{- '\\n<tool_response>\\n' }}",
 '        {{- content }}',
 "        {{- '\\n</tool_response>' }}",
 '        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}',
 "            {{- '<|im_end|>\\n' }}",
 '        {%- endif %}',
 '    {%- endif %}',
 '{%- endfor %}',
 '{%- if add_generation_prompt %}',
 "    {{- '<|im_start|>assistant\\n' }}",
 '    {%- if enable_thinking is defined and enable_thinking is false %}',
 "        {{- '<think>\\n\\n</think>\\n\\n' }}",
 '    {%- endif %}',
 '{%- endif %}']


qwen3_custom_chat_template_list = ['{%- if tools %}',
 "    {{- '<|im_start|>system\\n' }}",
 "    {%- if messages[0].role == 'system' %}",
 "        {{- messages[0].content + '\\n\\n' }}",
 '    {%- endif %}',
 '    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}',
 '    {%- for tool in tools %}',
 '        {{- "\\n" }}',
 '        {{- tool | tojson }}',
 '    {%- endfor %}',
 '    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}',
 '{%- else %}',
 "    {%- if messages[0].role == 'system' %}",
 "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}",
 '    {%- endif %}',
 '{%- endif %}',
 '{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}',
 '{%- for message in messages[::-1] %}',
 '    {%- set index = (messages|length - 1) - loop.index0 %}',
 '    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith(\'<tool_response>\') and message.content.endswith(\'</tool_response>\')) %}',
 '        {%- set ns.multi_step_tool = false %}',
 '        {%- set ns.last_query_index = index %}',
 '    {%- endif %}',
 '{%- endfor %}',
 '{%- for message in messages %}',
 '    {%- if message.content is string %}',
 '        {%- set content = message.content %}',
 '    {%- else %}',
 "        {%- set content = '' %}",
 '    {%- endif %}',
 '    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}',
 "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}",
 '    {%- elif message.role == "assistant" %}',
 "        {%- set reasoning_content = '' %}",
 '        {%- if message.reasoning_content is string %}',
 '            {%- set reasoning_content = message.reasoning_content %}',
 '        {%- else %}',
 "            {%- if '</think>' in content %}",
 "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}",
 "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}",
 '            {%- endif %}',
 '        {%- endif %}',
 '        {%- if loop.index0 > ns.last_query_index %}',
 '            {%- if loop.last or (not loop.last and reasoning_content) %}',
 "                {% generation %}{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }} {% endgeneration %}",
 '            {%- else %}',
 "                {% generation %}{{- '<|im_start|>' + message.role + '\\n' + content }}{% endgeneration %}",
 '            {%- endif %}',
 '        {%- else %}',
 "            {% generation %}{{- '<|im_start|>' + message.role + '\\n' + content }}{% endgeneration %}",
 '        {%- endif %}',
 '        {%- if message.tool_calls %}',
 '            {%- for tool_call in message.tool_calls %}',
 '                {%- if (loop.first and content) or (not loop.first) %}',
 "                    {{- '\\n' }}",
 '                {%- endif %}',
 '                {%- if tool_call.function %}',
 '                    {%- set tool_call = tool_call.function %}',
 '                {%- endif %}',
 '                {% generation %}{{- \'<tool_call>\\n{"name": "\' }}',
 '                {{- tool_call.name }}',
 '                {{- \'", "arguments": \' }}',
 '                {%- if tool_call.arguments is string %}',
 '                    {{- tool_call.arguments }}',
 '                {%- else %}',
 '                    {{- tool_call.arguments | tojson }}',
 '                {%- endif %}',
 "                {{- '}\\n</tool_call>' }}{% endgeneration %}",
 '            {%- endfor %}',
 '        {%- endif %}',
 "        {% generation %}{{- '<|im_end|>\\n' }}{% endgeneration %}",
 '    {%- elif message.role == "tool" %}',
 '        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}',
 "            {{- '<|im_start|>user' }}",
 '        {%- endif %}',
 "        {{- '\\n<tool_response>\\n' }}",
 '        {{- content }}',
 "        {{- '\\n</tool_response>' }}",
 '        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}',
 "            {{- '<|im_end|>\\n' }}",
 '        {%- endif %}',
 '    {%- endif %}',
 '{%- endfor %}',
 '{%- if add_generation_prompt %}',
 "    {{- '<|im_start|>assistant\\n' }}",
 '    {%- if enable_thinking is defined and enable_thinking is false %}',
 "        {{- '<think>\\n\\n</think>\\n\\n' }}",
 '    {%- endif %}',
 '{%- endif %}']


if __name__ == "__main__":
    test_data_util()
