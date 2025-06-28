#!/usr/bin/env python3
"""
SGLang Generate API Client
==========================

Client code that uses SGLang's generate API to get results from given prompts.
Supports synchronous text generation with log probability capture.
"""

import os
import yaml
import time
import json
import argparse
import requests
import asyncio
import aiofiles
import glob
import httpx
from datetime import datetime
from typing import Dict, List, Union, Optional, Generator
from openai import OpenAI
from pathlib import Path


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
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        logprobs: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text with asynchronous response.
        
        Args:
            prompt: Input text prompt
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
        
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": kwargs.get('top_p', 1.0),
                "top_k": kwargs.get('top_k', -1),
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
                    timeout=600
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


def create_sglang_client_from_config(config_file: str = "sequential_inference.yaml") -> SGLangClient:
    """
    Create SGLang client from configuration file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        Configured SGLangClient instance
    """
    return SGLangClient(config_file=config_file)


async def process_file_task(queue: asyncio.Queue, client: SGLangClient, input_base_dir: Path, output_base_dir: Path, time_tag: str, task_id: int):
    """
    Async task function that processes Python files from the queue.
    
    Args:
        queue: Async queue containing relative file paths
        client: SGLang client instance
        input_base_dir: Base input directory for reading files
        output_base_dir: Base output directory
        time_tag: Time tag for this run
        task_id: Task identifier for logging
    """
    processed_count = 0
    
    while True:
        # Get next file from queue (blocks until item available)
        input_relative_filepath = await queue.get()
        
        # Check for sentinel value indicating no more work
        if input_relative_filepath is None:
            print(f"Task {task_id}: Processed {processed_count} files, exiting")
            # Put sentinel back for other tasks
            await queue.put(None)
            break
        
        try:
            print(f"Task {task_id}: Processing {input_relative_filepath}")
            
            # Construct full input file path
            full_input_path = os.path.join(input_base_dir, input_relative_filepath)
            full_output_path = os.path.join(output_base_dir, input_relative_filepath)
            os.makedirs(full_output_path, exist_ok=True)
            
            # Read the Python file content
            async with aiofiles.open(full_input_path, 'r', encoding='utf-8') as f:
                source_code = await f.read()
            
            # Get prompts for conversation
            system_prompt = client.get_system_prompt()
            user_prompt = client.get_user_prompt(source_code)

            # convert system_prompt and user_prompt to chatml syntax with <|im_start|> and <|im_end|>
            system_prompt_chatml = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            user_prompt_chatml = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            assistant_start_chatml = "<|im_start|>assistant\n"

            # combine the prompts into a single string
            prompt = f"{system_prompt_chatml}\n{user_prompt_chatml}\n{assistant_start_chatml}"
            
            # Generate response using SGLang
            result = await client.generate(prompt=prompt)
            
            assistant_response = result.get('text', '')
            all_input_logprobs = result.get('input_logprobs', [])
            all_output_logprobs = result.get('output_logprobs', [])

            # Create output directory for this file
            relative_dir = Path(input_relative_filepath).parent
            output_file_dir = output_base_dir / relative_dir
            output_file_dir.mkdir(parents=True, exist_ok=True)
            
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
                        "content": assistant_response
                    }
                ],
                "metadata": {
                    "time_tag": time_tag,
                    "source_file": str(input_relative_filepath),
                    "server_url": client.base_url,
                    "model": client.config.get('sglang', {}).get('generation', {}).get('model', 'default'),
                    "total_input_tokens": len(all_input_logprobs) if all_input_logprobs else 0,
                    "total_output_tokens": len(all_output_logprobs) if all_output_logprobs else 0,
                    "total_tokens": (len(all_input_logprobs) if all_input_logprobs else 0) + (len(all_output_logprobs) if all_output_logprobs else 0)
                }
            }
            
            # Save conversation file
            conversation_file = os.path.join(full_output_path, "conversation.json")
            async with aiofiles.open(conversation_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(conversation, indent=2, ensure_ascii=False))
            
            # Save logprobs file if available
            if all_input_logprobs or all_output_logprobs:
                logprobs_data = {
                    "input_logprobs": all_input_logprobs if all_input_logprobs else None,
                    "output_logprobs": all_output_logprobs if all_output_logprobs else None,
                    "metadata": {
                        "time_tag": time_tag,
                        "source_file": str(input_relative_filepath),
                        "server_url": client.base_url,
                        "model": client.config.get('sglang', {}).get('generation', {}).get('model', 'default'),
                        "logprobs_requested": client.config.get('sglang', {}).get('generation', {}).get('logprobs', 0),
                        "total_input_tokens": len(all_input_logprobs) if all_input_logprobs else 0,
                        "total_output_tokens": len(all_output_logprobs) if all_output_logprobs else 0,
                        "total_tokens": (len(all_input_logprobs) if all_input_logprobs else 0) + (len(all_output_logprobs) if all_output_logprobs else 0)
                    }
                }
                
                logprobs_file = os.path.join(full_output_path, "logprobs.json")
                async with aiofiles.open(logprobs_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(logprobs_data, indent=2, ensure_ascii=False))
            
            # extract the main section without the <think> and </think> tags
            assistant_response_no_think = assistant_response.split("<think>")[1].split("</think>")[0]
            # extract generated code from the assistant response
            generated_code = assistant_response_no_think.split("```python")[1].split("```")[0]
            # extract brief explanation from the assistant response
            brief_explanation = assistant_response_no_think.split("```text")[1].split("```")[0]

            # save the generated code and brief explanation to a file
            with open(os.path.join(full_output_path, "generated_code.py"), "w", encoding="utf-8") as f:
                f.write(generated_code)
            with open(os.path.join(full_output_path, "brief_explanation.txt"), "w", encoding="utf-8") as f:
                f.write(brief_explanation)
            

            processed_count += 1
            print(f"Task {task_id}: ✅ Saved {input_relative_filepath} -> {output_file_dir}")
            
        except Exception as e:
            print(f"Task {task_id}: ❌ Error processing {input_relative_filepath}: {e}")
            # print exception with traceback
            import traceback
            print(traceback.format_exc())


async def main():
    """Process multiple Python files using SGLang client with async tasks."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SGLang Generate API Client")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        # required=True,
        default="./kernel_bench/level1", 
        help="Directory to recursively search for .py files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./_output", 
        help="Directory to save conversation JSON files"
    )
    parser.add_argument(
        "--num-tasks", 
        type=int, 
        default=8,
        help="Number of concurrent async tasks to run"
    )

    args = parser.parse_args()
    
    print("SGLang Generate API Client - Batch Processing")
    print("=============================================")
    
    # Create output directory with time tag
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output_dir}_{time_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize client
    try:
        client = create_sglang_client_from_config()
        print(f"✅ Connected to SGLang server at {client.base_url}")    
    except Exception as e:
        print(f"❌ Failed to connect to SGLang server: {e}")
        exit(1)
    
    # Health check
    print("\n--- Performing health check ---")
    if await client.health_check():
        print("✅ SGLang server is healthy")
    else:
        print("❌ SGLang server health check failed")
        return
    
    # Find all Python files recursively
    print(f"\n--- Finding Python files in {args.input_dir} ---")
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory {args.input_dir} does not exist")
        return
    
    # Find all .py files recursively and get their relative paths
    py_files = []
    for py_file in input_dir.rglob("*.py"):
        relative_path = py_file.relative_to(input_dir)
        py_files.append(str(relative_path))
    
    if not py_files:
        print(f"❌ No Python files found in {args.input_dir}")
        return
    
    print(f"📄 Found {len(py_files)} Python files to process")
    
    # Create async queue and populate with file paths
    queue = asyncio.Queue()
    for py_file in py_files:
        await queue.put(py_file)  # Put relative paths in queue
    
    # Add sentinel values to signal task termination (one per task)
    for _ in range(args.num_tasks):
        await queue.put(None)
    
    print(f"🔄 Starting {args.num_tasks} async tasks...")
    
    # Create and start async tasks
    tasks = []
    for task_id in range(args.num_tasks):
        task = asyncio.create_task(
            process_file_task(queue, client, input_dir, output_dir, time_tag, task_id)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    print(f"\n✅ All tasks completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📄 Processed {len(py_files)} Python files with {args.num_tasks} concurrent tasks")


if __name__ == "__main__":
    asyncio.run(main())
