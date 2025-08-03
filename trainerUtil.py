import json
import yaml
import argparse
import torch
from transformers import AutoTokenizer
from logger import logger
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

def merge_dicts(a: dict, b: dict) -> dict:
    """
    Return a new dict that is a recursive merge of a and b.
    Keys in b override keys in a. If both values are dicts, merge them recursively.
    """
    result = a.copy()
    for key, b_val in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(b_val, dict):
            result[key] = merge_dicts(result[key], b_val)
        else:
            result[key] = b_val
    return result

async def rsync_file(source_path: str, target_path: str, rsync_path: str = "rsync") -> int:
    """
    Rsync a file from source to target path
    """
    # run command: rsync -azP <source_path> <target_path>
    command = f"rsync -azP --rsync-path '{rsync_path}' '{source_path}' '{target_path}'"
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )

    logger.info(f"[trainerUtil] [rsync_file] START ====================")
    logger.info(f"[trainerUtil] [rsync_file] command: {command}")

    # Create tasks to read stdout and stderr concurrently
    stdout_task = asyncio.create_task(
        read_stream(process.stdout, "reference", is_error=False)
    )
    stderr_task = asyncio.create_task(
        read_stream(
            process.stderr, "reference", is_error=True
        )  # seems taking warning message as error message
    )

    # Wait for the process to complete
    return_code = await process.wait()

    # Wait for all output to be processed
    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
    if process.returncode != 0:
        logger.error(f"[KB Eval] [reference] return code: {process.returncode}")
    else:
        logger.info(f"[KB Eval] [reference] return code: {process.returncode}")

    logger.info(f"[trainerUtil] [rsync_file] END ====================")

    return return_code

def format_conversation(messages: List[Dict[str, Any]],
                        tokenizer: AutoTokenizer,
                        mask_non_assistant_tokens: bool = True, # mask all token except assistant tokens
                        mask_non_last_assistant_tokens: bool = False, # mask all token except last assistant tokens
                        ignore_index: int = -100,
                        tools: List[Union[Dict, Callable]] = [],
                        messages_from_openai_agent: bool = False,
                        user_chat_template_for_masking: bool = True) -> Dict[str, Any]:
    # Always format conversation using chat template to keep the formatted text consistent with pre-trained model
    if messages_from_openai_agent is True:
        raise ValueError("OpenAI Agent format is not supported yet")

    if user_chat_template_for_masking is False:
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
            # Use the tokenizer's chat template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                toosl = tools,
                chat_template="\n".join(qwen3_custom_chat_template_list),
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=False
            )
            encoding = tokenizer.apply_chat_template(
                messages,
                toosl = tools,
                chat_template="\n".join(qwen3_custom_chat_template_list),
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
    def __init__(self, tokenizer_pad_token_id, pad_to_multiple_of=8, ignore_index=-100):
        self.tokenizer_pad_token_id = tokenizer_pad_token_id  # unsloth tokernizer can't be pickled
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
            padded = torch.cat([seq, torch.full((pad_len,), self.tokenizer_pad_token_id)])
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

class VLLMClient:
    def __init__(self):
        self.vllm_config = self.load_config().get('vllm', {})
        self.host = self.vllm_config.get('host', '10.12.0.202')
        self.port = self.vllm_config.get('port', 8091)
        api_key_path = os.path.expanduser(self.vllm_config.get('api_key_path', '~/.keys/local.api.key'))
        with open(api_key_path, 'r') as f:
            self.api_key = f.read().strip()
        self.timeout = self.vllm_config.get('timeout', 60)
        self.retries = self.vllm_config.get('retries', 3)

    def load_config(self, config_path: str = "trainerMain.yaml"):
        """Load config from yaml file"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    async def load_lora_adapter(self, lora_name: str, lora_path: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"http://{self.host}:{self.port}/v1/load_lora_adapter",
                        json={
                            "lora_name": lora_name,
                            "lora_path": lora_path,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    logger.warning(f"🔍 [VLLMClient] Response: {response.text}") # use text instead of json
                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Loaded lora adapter from [{lora_path}]")
                    return
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error loading lora adapter from [{lora_path}]: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to load lora adapter from [{lora_path}] in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to load lora adapter from [{lora_path}] after {self.retries} retries")
                    raise e

    async def unload_lora_adapter(self, lora_name: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.post(
                        f"http://{self.host}:{self.port}/v1/unload_lora_adapter",
                        json={"lora_name": lora_name},
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    logger.warning(f"🔍 [VLLMClient] Response: {response.text}") # use text instead of json
                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Unloaded lora adapter from [{lora_name}]")
                    return
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error unloading lora adapter from [{lora_name}]: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to unload lora adapter from [{lora_name}] in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to unload lora adapter from [{lora_name}] after {self.retries} retries")
                    raise e

    async def get_models(self):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                limits = httpx.Limits(max_keepalive_connections=0, keepalive_expiry=0)
                async with httpx.AsyncClient(limits=limits, headers={"Connection": "close"}, http2=False) as client:
                    response = await client.get(
                        f"http://{self.host}:{self.port}/v1/models",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    logger.info(f"🔍 [VLLMClient] Got models")
                    return response.json()
            except Exception as e:
                logger.warning(f"🔍 [VLLMClient] Error getting models: {e}")
                if retry_count < self.retries:
                    logger.info(f"🔍 [VLLMClient] Retrying to get models in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"❌ [VLLMClient] Failed to get models after {self.retries} retries")
                    raise e

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
        ## Use chat_completion to get the formatted text
        formatted_data = format_conversation(message["messages"], tokenizer, tools=message["tools"], mask_non_assistant_tokens=args.mask_non_assistant_tokens, mask_non_last_assistant_tokens=args.mask_non_last_assistant_tokens, user_chat_template_for_masking=False)
        logger.info(f"🔍 [test_data_util] Formatted text: {formatted_data['text']}")
        print_masking_analysis(formatted_data, tokenizer)
        ## Use the tokenizer to get the formatted text and masks
        formatted_data = format_conversation(message["messages"], tokenizer, tools=message["tools"], mask_non_assistant_tokens=args.mask_non_assistant_tokens, mask_non_last_assistant_tokens=args.mask_non_last_assistant_tokens, user_chat_template_for_masking=True)
        logger.info(f"🔍 [test_data_util] Formatted text: {formatted_data['text']}")
        print_masking_analysis(formatted_data, tokenizer)

# added the support for return_assistant_tokens_mask in apply_chat_template
# {% generation %} and {% endgeneration %} are used to mark the assistant tokens
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
