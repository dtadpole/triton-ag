#!/usr/bin/env python3
"""
SGLang and vLLM Generate API Client
===================================

Client code that uses SGLang's or vLLM's generate API to get results from given prompts.
Supports synchronous text generation with log probability capture.
"""

import os
import yaml
import time
import json
import argparse
import boto3
import requests
import asyncio
import aiofiles
import hashlib
import glob
import httpx
import traceback
from datetime import datetime
from typing import Dict, List, Union, Optional, Generator
from openai import OpenAI
from pathlib import Path


class VLLMClient:
    """Client for vLLM OpenAI-compatible API using synchronous generation with logprobs support."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config_file: str = "sequential_inference.yaml",
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server base URL (loaded from config if None)
            api_key: API key for authentication
            config_file: Path to YAML config file with server settings
        """
        # Load configuration from file
        self.config = self._load_config(config_file)
        common_config = self.config.get('vllm', {}).get('common', {})
        
        # Set defaults from config
        base_url = base_url or common_config.get('base_url', 'http://localhost:8000/v1')
        
        # Set up API key
        if api_key is None:
            api_key = os.environ.get("VLLM_API_KEY")
            if api_key is None:
                # Try to load from config or default location
                api_key_path = common_config.get('api_key', "${HOME}/.keys/local.api.key")
                if api_key_path.startswith('${HOME}/'):
                    api_key_path = os.path.expanduser(api_key_path.replace('${HOME}', '~'))
                try:
                    with open(api_key_path, 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    api_key = "dummy_key"  # vLLM often doesn't require real auth
        
        self.base_url = base_url
        self.api_key = api_key
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            # Try relative to script directory
            config_path = Path(__file__).parent / config_file
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}
        else:
            print(f"Warning: Config file {config_file} not found")
            return {}
    
    async def generate(
        self,
        source_code: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        logprobs: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text with asynchronous response using vLLM's OpenAI-compatible API.
        
        Args:
            source_code: Input source code
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            logprobs: Number of log probabilities to return for each token
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'text' and logprobs data
        """
        # Get defaults from config
        generation_config = self.config.get('vllm', {}).get('generation', {})
        model = model or generation_config.get('model', 'default')
        temperature = temperature if temperature is not None else generation_config.get('temperature', 0.7)
        max_tokens = max_tokens or generation_config.get('max_tokens', 1024)
        logprobs_enabled = generation_config.get('logprobs', True)
        
        # Use vLLM's OpenAI-compatible completions API
        completions_url = f"{self.base_url}/completions"
        
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(source_code)

        # format prompt with chatml format
        prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get('top_p', 1.0),
            "stream": False,
            "echo": False,  # We don't need to echo the prompt in output
        }
        
        # Add logprobs if requested - vLLM supports both prompt_logprobs and logprobs
        if logprobs_enabled:
            payload["logprobs"] = 1  # Need at least 1 to get logprobs
            payload["prompt_logprobs"] = 1  # Need at least 1 to get prompt logprobs
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    completions_url,
                    json=payload,
                    headers=headers,
                    timeout=600
                )
                response.raise_for_status()
                
                # Parse response
                data = response.json()
            
            # Extract text and logprobs from OpenAI-style response
            choices = data.get('choices', [])
            if not choices:
                return {'text': '', 'input_logprobs': [], 'output_logprobs': []}
            
            choice = choices[0]
            generated_text = choice.get('text', '')
            
            result = {
                'text': generated_text,
                'input_logprobs': [],
                'output_logprobs': []
            }
            
            # Extract logprobs if available
            logprobs_data = choice.get('logprobs')
            if logprobs_data:
                # Output token logprobs
                if 'tokens' in logprobs_data and 'token_logprobs' in logprobs_data:
                    tokens = logprobs_data['tokens']
                    token_logprobs = logprobs_data['token_logprobs']
                    top_logprobs = logprobs_data.get('top_logprobs', [])
                    
                    output_logprobs = []
                    for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
                        if logprob is not None:  # Skip None values
                            entry = {
                                'token': token,
                                'logprob': logprob
                            }
                            output_logprobs.append(entry)
                    result['output_logprobs'] = output_logprobs
            
            # Extract prompt logprobs (vLLM returns this at choice level, not in logprobs object)
            prompt_logprobs = choice.get('prompt_logprobs')
            if prompt_logprobs and isinstance(prompt_logprobs, list):
                input_logprobs = []
                for entry in prompt_logprobs:
                    if entry is not None and isinstance(entry, dict):
                        # vLLM format: dict with token_id as keys
                        if entry:
                            # Find the token with the highest rank (lowest rank number = most likely)
                            # or just take the first one as the chosen token
                            chosen_token_id = None
                            chosen_info = None
                            best_rank = float('inf')
                            
                            for token_id, token_info in entry.items():
                                if isinstance(token_info, dict):
                                    rank = token_info.get('rank', float('inf'))
                                    if rank < best_rank:
                                        best_rank = rank
                                        chosen_token_id = token_id
                                        chosen_info = token_info
                            
                            if chosen_info:
                                input_logprobs.append({
                                    'token': chosen_info.get('decoded_token', chosen_token_id),
                                    'logprob': chosen_info.get('logprob', 0.0)
                                })
                result['input_logprobs'] = input_logprobs
                        
            return result
                            
        except Exception as e:
            print(f"Error during vLLM generation: {e}")
            traceback.print_exc()
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            raise
    
    def get_system_prompt(self) -> str:
        """Get system prompt from configuration."""
        prompts_config = self.config.get('prompts', {})
        reference_code = self.get_example_reference_code()
        generated_code = self.get_example_generated_code()
        return prompts_config.get('system_prompt', 'You are a helpful assistant.').format(reference_code=reference_code, generated_code=generated_code)
    
    def get_user_prompt(self, source_code: str) -> str:
        """Get user prompt from configuration with source code substituted."""
        prompts_config = self.config.get('prompts', {})
        user_prompt_template = prompts_config.get('user_prompt', 'Analyze this code: {source_code}')
        return user_prompt_template.format(source_code=source_code)
    
    def get_example_reference_code(self) -> str:
        """Get example reference code from configuration."""
        prompts_config = self.config.get('prompts', {}).get('examples', {})
        return prompts_config.get('reference_code', '')

    def get_example_generated_code(self) -> str:
        """Get example generated code from configuration."""
        prompts_config = self.config.get('prompts', {}).get('examples', {})
        return prompts_config.get('generated_code', '')

    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            # Try a simple generation request
            result = await self.generate("Hello", max_tokens=1)
            return bool(result.get('text', '').strip())  # If we get any text response, server is healthy
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def get_models(self) -> List[str]:
        """Get available models from the server."""
        try:
            # Use vLLM's OpenAI-compatible models endpoint
            models_url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.get(models_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            if 'data' in data:
                for model_info in data['data']:
                    if 'id' in model_info:
                        models.append(model_info['id'])
            return models if models else ['default']
        except Exception as e:
            print(f"❌ Failed to get models: {e}")
            return ['default']


class SGLangClient:
    """Client for SGLang generate API using synchronous generation."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config_file: str = "sequential_inference.yaml",
    ):
        """
        Initialize SGLang client.
        
        Args:
            base_url: SGLang server base URL (loaded from config if None)
            api_key: API key for authentication
            config_file: Path to YAML config file with server settings
        """
        # Load configuration from file
        self.config = self._load_config(config_file)
        common_config = self.config.get('sglang', {}).get('common', {})
        
        # Set defaults from config
        base_url = base_url or common_config.get('base_url', 'http://localhost:8081/v1')
        
        # Set up API key
        if api_key is None:
            api_key = os.environ.get("SGLANG_API_KEY")
            if api_key is None:
                # Try to load from config or default location
                api_key_path = common_config.get('api_key', "${HOME}/.keys/local.api.key")
                if api_key_path.startswith('${HOME}/'):
                    api_key_path = os.path.expanduser(api_key_path.replace('${HOME}', '~'))
                try:
                    with open(api_key_path, 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    api_key = "dummy_key"  # SGLang often doesn't require real auth
        
        self.base_url = base_url
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            # Try relative to script directory
            config_path = Path(__file__).parent / config_file
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}
        else:
            print(f"Warning: Config file {config_file} not found")
            return {}
    
    async def generate(
        self,
        source_code: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        logprobs: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text with asynchronous response.
        
        Args:
            source_code: Input source code
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            logprobs: Number of log probabilities to return for each token
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'text' and logprobs data
        """
        # Get defaults from config
        generation_config = self.config.get('sglang', {}).get('generation', {})
        model = model or generation_config.get('model', 'default')
        temperature = temperature if temperature is not None else generation_config.get('temperature', 0.7)
        max_tokens = max_tokens or generation_config.get('max_tokens', 1024)
        logprobs = logprobs if logprobs is not None else generation_config.get('logprobs', 0)
        
        # Use SGLang's native generate API
        generate_url = self.base_url.replace('/v1', '') + '/generate'

        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(source_code)

        # format prompt with chatml format
        prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_k": kwargs.get('top_k', 40),
                "top_p": kwargs.get('top_p', 1.0),
            },
            "stream": False
        }
        
        # Add logprobs if requested
        if logprobs and logprobs > 0:
            payload["return_logprob"] = True
            payload["logprob_start_len"] = 0
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    generate_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=600,
                )
                response.raise_for_status()
                
                # Parse response
                data = response.json()
            
            result = {
                'text': data.get('text', ''),
                'meta_info': data.get('meta_info', {})
            }
            
            # Extract logprobs if available
            if 'meta_info' in data:
                meta_info = data['meta_info']
                
                if 'input_token_logprobs' in meta_info:
                    result['input_logprobs'] = meta_info['input_token_logprobs']
                
                if 'output_token_logprobs' in meta_info:
                    result['output_logprobs'] = meta_info['output_token_logprobs']
            
            return result
                            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise
    
    def get_system_prompt(self) -> str:
        """Get system prompt from configuration."""
        prompts_config = self.config.get('prompts', {})
        reference_code = self.get_example_reference_code()
        generated_code = self.get_example_generated_code()
        return prompts_config.get('system_prompt', 'You are a helpful assistant.').format(reference_code=reference_code, generated_code=generated_code)
    
    def get_user_prompt(self, source_code: str) -> str:
        """Get user prompt from configuration with source code substituted."""
        prompts_config = self.config.get('prompts', {})
        user_prompt_template = prompts_config.get('user_prompt', 'Analyze this code: {source_code}')
        return user_prompt_template.format(source_code=source_code)
    
    def get_example_reference_code(self) -> str:
        """Get example reference code from configuration."""
        prompts_config = self.config.get('prompts', {}).get('examples', {})
        return prompts_config.get('reference_code', '')

    def get_example_generated_code(self) -> str:
        """Get example generated code from configuration."""
        prompts_config = self.config.get('prompts', {}).get('examples', {})
        return prompts_config.get('generated_code', '')

    async def health_check(self) -> bool:
        """Check if SGLang server is healthy."""
        try:
            # Try a simple generation request using SGLang's native API
            result = await self.generate("Hello", max_tokens=1)
            return bool(result.get('text'))  # If we get any text response, server is healthy
        except Exception:
            return False
    
    def get_models(self) -> List[str]:
        """Get available models from the server."""
        try:
            # Use SGLang's native API to get models
            models_url = self.base_url.replace('/v1', '') + '/get_model_info'
            response = requests.get(models_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [data.get('model_path', 'default')]
        except Exception as e:
            print(f"Error getting models: {e}")
            return ["default"]


async def process_file_task(queue: asyncio.Queue, client: Union[SGLangClient, VLLMClient], input_base_dir: Path, output_base_dir: Path, time_tag: str, task_id: int):
    """
    Process files from the queue using the given client.
    
    Args:
        queue: Queue containing file paths to process
        client: Either SGLangClient or VLLMClient instance
        input_base_dir: Base directory for input files
        output_base_dir: Base directory for output files
        time_tag: Timestamp tag for output files
        task_id: Unique task identifier
    """
    client_type = "vLLM" if isinstance(client, VLLMClient) else "SGLang"

    while True:
        try:
            # Get file path from queue (blocking with timeout)
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            if item is None:
                print(f"[Task {task_id}] Received termination signal")
                break
            # break item into file_path and gen_id
            file_path = item["file_path"]
            gen_id = item["gen_id"]
            
            # add info emoji to beginning of the line
            print(f"🔍 [Task {task_id}] Processing [{f'{gen_id:02d}'}]: {file_path}")

            retry_count = 0   
            while retry_count < 3:
                retry_count += 1
                try:
                    # Read the Python file
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        source_code = await f.read()
                    
                    # Get relative path for output structure
                    relative_path = file_path.relative_to(input_base_dir)
                    output_sub_dir = output_base_dir / relative_path / f"{gen_id:02d}"
                    output_sub_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create conversation
                    system_prompt = client.get_system_prompt()
                    user_prompt = client.get_user_prompt(source_code)
                    
                    # Generate response
                    start_time = time.time()
                    result = await client.generate(source_code)
                    generation_time = time.time() - start_time
                    
                    generated_text = result.get('text', '')
                    input_logprobs = result.get('input_logprobs', [])
                    output_logprobs = result.get('output_logprobs', [])
                    
                    token_count = len(input_logprobs) + len(output_logprobs)

                    print(f"🔍 [Task {task_id}] Generated [{f'{token_count:04d}'} tokens] [{len(generated_text)} characters] in {generation_time:.2f}s")

                    # save response to a file
                    response_file = output_sub_dir / f"response.txt"
                    with open(response_file, 'w') as f:
                        f.write(generated_text)
                    
                    # Create conversation data
                    conversation = {
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": generated_text
                            }
                        ],
                        "metadata": {
                            "client_type": client_type,
                            "input_file": str(relative_path),
                            "generation_time": generation_time,
                            "time_tag": time_tag,
                            "task_id": task_id,
                            "input_tokens_count": len(input_logprobs),
                            "output_tokens_count": len(output_logprobs)
                        }
                    }
                    
                    # Save conversation file
                    conversation_file = output_sub_dir / f"conversation.json"
                    async with aiofiles.open(conversation_file, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(conversation, indent=2, ensure_ascii=False))
                    
                    # Create and save logprobs data
                    logprobs_data = {
                        "input_logprobs": input_logprobs,
                        "output_logprobs": output_logprobs,
                        "metadata": {
                            "client_type": client_type,
                            "input_file": str(relative_path),
                            "generation_time": generation_time,
                            "time_tag": time_tag,
                            "task_id": task_id,
                            "input_tokens_count": len(input_logprobs),
                            "output_tokens_count": len(output_logprobs)
                        }
                    }
                    
                    # Save logprobs file
                    logprobs_file = output_sub_dir / f"logprobs.json"
                    async with aiofiles.open(logprobs_file, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(logprobs_data, indent=2, ensure_ascii=False))

                    # remove <think> and </think> from the generated text
                    generated_text_no_think = generated_text.replace("<think>", "").replace("</think>", "")

                    # extract the generated code from the generated text
                    generated_code = generated_text_no_think.split("```python")[1].split("```")[0]
                    # save the generated code to a file
                    generated_code_file = output_sub_dir / f"generated_code.py"
                    with open(generated_code_file, 'w') as f:
                        f.write(generated_code)

                    # extract brief explanation from the generated text
                    brief_explaination = generated_text_no_think.split("```text")[1].split("```")[0]
                    # save the brief explaination to a file
                    brief_explaination_file = output_sub_dir / f"brief_explaination.txt"
                    with open(brief_explaination_file, 'w') as f:
                        f.write(brief_explaination)

                    print(f"✅ [Task {task_id}] Saved [{f'{gen_id:02d}'}]: {generated_code_file}")
                    
                except Exception as e:
                    print(f"[Task {task_id}] Error generating [{f'{gen_id:02d}'}] for {file_path}: {e}")
                    print(traceback.format_exc())
                
        except asyncio.TimeoutError:
            # Timeout waiting for queue item, check if queue is empty
            if queue.empty():
                print(f"[Task {task_id}] Queue is empty, terminating for {file_path}")
                break
        except Exception as e:
            print(f"[Task {task_id}] Unexpected error: {e}")
            print(traceback.format_exc())
            break
    
    print(f"[Task {task_id}] Task completed")


async def main():
    """Main function for batch processing Python files."""
    parser = argparse.ArgumentParser(description="Process Python files with SGLang or vLLM generate API")
    parser.add_argument("--input-dir", type=str, default="./kernel_bench/level1", help="Input directory containing Python files")
    parser.add_argument("--output-dir", type=str, default="./_output", help="Output directory for results")
    parser.add_argument("--num-tasks", type=int, default=8, help="Number of concurrent processing tasks")
    parser.add_argument("--num-generations", type=int, default=8, help="Number of generations to perform for each file")
    parser.add_argument("--client", type=str, default="vllm", choices=["sglang", "vllm"], 
                       help="Client type to use (sglang or vllm)")
    parser.add_argument("--epoch-id", type=int, default=1, help="Epoch ID to process")
    parser.add_argument("--bucket-size", type=int, default=10, help="Number of files to process in each bucket")
    parser.add_argument("--bucket-id", type=int, default=1, help="Bucket ID to process")
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir + "_" + f"{args.epoch_id:03d}" + "_" + f"{args.bucket_id:02d}" + "_" + time_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Find all Python files recursively
    python_files = list(input_dir.rglob("*.py"))
    # sort the python files by md5 of filepath
    python_files.sort(key=lambda x: hashlib.md5(str(str(x) + str(args.epoch_id)).encode('utf-8')).hexdigest())
    # split the python files into buckets
    bucket_files = [python_files[i:i + args.bucket_size] for i in range(0, len(python_files), args.bucket_size)]
    # get the bucket
    bucket_files = bucket_files[args.bucket_id - 1]
    
    if not bucket_files:
        print(f"Error: No Python files found in {input_dir}")
        return
    
    # add info emoji to beginning of the line
    print(f"🔍 Found {len(bucket_files)} Python files to process")
    print(f"🔍 Output directory: [{output_dir}]")
    print(f"🔍 Using {args.client} client with {args.num_tasks} concurrent tasks")

    # for each file in bucket_files, write file content to relevant path
    for file_path in bucket_files:
        # get the relative path
        relative_path = file_path.relative_to(input_dir)
        # get the output path
        output_path = output_dir / relative_path
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"reference_code.py"
        # write the file content to the output path
        with open(output_file, 'w') as f:
            f.write(file_path.read_text())

    # Create client based on selection
    if args.client == "vllm":
        client = VLLMClient(config_file="sequential_inference.yaml")
        print("Created vLLM client")
    else:
        client = SGLangClient(config_file="sequential_inference.yaml")
        print("Created SGLang client")
    
    # Test client connection
    print("Testing client connection...")
    if await client.health_check():
        print("✅ Client connection successful")
        models = client.get_models()
        print(f"Available models: {models}")
    else:
        print("❌ Client connection failed")
        return
    
    # Create queue and add all files
    queue = asyncio.Queue()
    for file_path in bucket_files:
        for gen_id in range(args.num_generations):
            await queue.put({
                "file_path": file_path,
                "gen_id": gen_id + 1,
            })
    
    # Add sentinel values to signal task completion
    for _ in range(args.num_tasks):
        await queue.put(None)
    
    # Create and start processing tasks
    tasks = []
    for task_id in range(args.num_tasks):
        task = asyncio.create_task(
            process_file_task(queue, client, input_dir, output_dir, time_tag, task_id+1)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    print(f"Starting {args.num_tasks} processing tasks...")
    await asyncio.gather(*tasks)
    
    print("All tasks completed!")
    print(f"Results saved in: {output_dir}")

    # upload the output directory to s3
    s3_client = boto3.client('s3')
    s3_client.upload_file(output_dir, 'agent-xyz', f'{output_dir.name}')


if __name__ == "__main__":
    asyncio.run(main())
