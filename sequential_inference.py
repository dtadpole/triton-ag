#!/usr/bin/env python3
"""
vLLM Generate API Client
========================

Client code that uses vLLM's generate API to get results from given prompts.
Supports synchronous text generation.
"""

import os
import re
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
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from datetime import datetime
from typing import Dict, List, Optional
from kbEvalRemoteServer import KernelExecResult
from pathlib import Path
import requests
import yaml

# Global configuration flags
STREAMING = True   # Set to True for streaming API, False for non-streaming single response

class VLLMClient:
    """Client for vLLM OpenAI-compatible API using OpenAI client for streaming or non-streaming generation."""
    
    def __init__(
        self,
        client_type: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
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
        self.client_type = client_type
        common_config = self.config.get(self.client_type, {}).get('common', {})
        
        # Set defaults from config
        self.model = model or self.config.get(self.client_type, {}).get('generation', {}).get('model', 'default')
        self.tokenizer = self.config.get(self.client_type, {}).get('generation', {}).get('tokenizer', self.model)
        self.base_url = base_url or common_config.get('base_url', 'http://localhost:8000/v1')

        # Set up API key
        if api_key is None:
            # Try to load from config or default location
            api_key_path = common_config.get('api_key', "${HOME}/.keys/local.api.key")
            if api_key_path.startswith('${HOME}/'):
                api_key_path = os.path.expanduser(api_key_path.replace('${HOME}', '~'))
            try:
                with open(api_key_path, 'r') as f:
                    self.api_key = f.read().strip()
            except FileNotFoundError:
                self.api_key = "dummy_key"  # vLLM often doesn't require real auth
        else:
            self.api_key = api_key
        
        # Create OpenAI client for vLLM
        self.openai_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # Print generation mode info
        if STREAMING:
            print("🚀 Using STREAMING mode with OpenAI client")
        else:
            print("📄 Using NON-STREAMING mode with OpenAI client")
        
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
            print(f"⚠️ Warning: Config file {config_file} not found")
            return {}
    
    async def generate(
        self,
        source_code: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text using vLLM's OpenAI-compatible API with streaming or non-streaming mode.
        
        Args:
            source_code: Input source code
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'text' and 'tokens' data
        """
        # Get defaults from config
        generation_config = self.config.get(self.client_type, {}).get('generation', {})
        temperature = temperature if temperature is not None else generation_config.get('temperature', 0.7)
        max_tokens = max_tokens or generation_config.get('max_tokens', 1024)

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(source_code)

        # format prompt with chatml format
        prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"
        
        try:
            generated_text = ""
            
            if STREAMING:
                # STREAMING MODE: Real-time token streaming using OpenAI client
                stream = await self.openai_client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=kwargs.get('top_p', 1.0),
                    stream=True,
                    timeout=600
                )
                
                # Process streaming response
                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if choice.text:
                            generated_text += choice.text
                            
            else:
                # NON-STREAMING MODE: Single response using OpenAI client
                response = await self.openai_client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=kwargs.get('top_p', 1.0),
                    stream=False,
                    timeout=600
                )
                
                # Extract text from response
                if response.choices:
                    choice = response.choices[0]
                    generated_text = choice.text or ""

            # tokenize the generated text
            tokens = tokenizer.encode(generated_text) if generated_text else []
            
            return {'text': generated_text, 'tokens': tokens}
                            
        except Exception as e:
            mode = "streaming" if STREAMING else "non-streaming"
            print(f"Error during vLLM {mode} generation with OpenAI client: {e}")
            traceback.print_exc()
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
            return bool(result.get('text', ''))  # If we get any text response, server is healthy
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def get_models(self) -> List[str]:
        """Get available models from the server."""
        try:
            # Use vLLM's OpenAI-compatible models endpoint
            models_url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.get(models_url, headers=headers, timeout=60)
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


async def _process_inference_task(
    client: VLLMClient,
    file_path: Path,
    gen_id: int,
    input_base_dir: Path,
    output_base_dir: Path,
    time_tag: str,
    task_id: int
) -> bool:
    """
    Process a single inference task for a file.
    
    Args:
        client: VLLMClient instance
        file_path: Path to the input Python file
        gen_id: Generation ID for this task
        input_base_dir: Base directory for input files
        output_base_dir: Base directory for output files
        time_tag: Timestamp tag for output files
        task_id: Task identifier for logging
        
    Returns:
        bool: True if successful, False if failed
    """
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

    print(f"🔍 [Task {task_id}] Responded [{f'{len(result.get('tokens', [])):04d}'} tokens] [{len(generated_text)} characters] in [{generation_time:.2f}s]")

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
            "client_type": client.client_type,
            "model": client.model,
            "input_file": str(relative_path),
            "generation_time": generation_time,
            "num_tokens": len(result.get('tokens', [])),
            "time_tag": time_tag,
            "task_id": task_id
        }
    }
    
    # Save conversation file
    conversation_file = output_sub_dir / f"conversation.json"
    async with aiofiles.open(conversation_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(conversation, indent=2, ensure_ascii=False))

    # remove the content between first <think> and the last </think> from the generated text
    # first_think_idx = generated_text.find("<think>")
    last_think_idx = generated_text.rfind("</think>")
    generated_text_no_think = generated_text[last_think_idx+len("</think>"):] if last_think_idx != -1 else generated_text

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

    # use generated emoji to beginning of the line
    print(f"👏 [Task {task_id}] Generated [{f'{gen_id:02d}'}]: {generated_code_file}")
    return True


class KbEvalClient:
    """Client for calling kbEvalRemoteServer to evaluate generated code."""
    
    def __init__(self, config_file: str = "sequential_inference.yaml"):
        """Initialize the client with configuration"""
        self.config = self._load_config(config_file)
        # Get kbEval config from sequential_inference.yaml
        kb_eval_config = self.config.get('kbEval', {})
        self.base_url = kb_eval_config.get('base_url', 'http://localhost:5678')
        
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
            print(f"⚠️ Warning: Config file {config_file} not found")
            return {}

    async def call_kb_eval_ref(self, eval_params: Dict[str, str]) -> KernelExecResult:
        """Call the kbEvalRemoteServer with evaluation parameters"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/kb_eval_ref",
                    json=eval_params,
                    headers={"Content-Type": "application/json"},
                    timeout=300  # 5 minute timeout
                )
                
                if response.status_code != 200:
                    raise Exception(f"Server returned status {response.status_code}: {response.text}")
                    
                result = KernelExecResult(**response.json())

                return result
            
        except httpx.TimeoutException:
            raise Exception("Server request timed out after 5 minutes")
        except httpx.ConnectError:
            raise Exception(f"Could not connect to server at {self.base_url}")
        except Exception as e:
            raise Exception(f"Error calling server: {e}")

    async def call_kb_eval_server(self, eval_params: Dict[str, str]) -> KernelExecResult:
        """Call the kbEvalRemoteServer with evaluation parameters"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/kb_eval",
                    json={
                        "model_tag": eval_params["model_tag"],
                        "task_tag": eval_params["task_tag"],
                        "eval_tag": eval_params["eval_tag"],
                        "time_tag": eval_params["time_tag"],
                        "reference_code": eval_params["reference_code"],
                        "generated_code": eval_params["generated_code"],
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=300  # 5 minute timeout
                )
                
                if response.status_code != 200:
                    raise Exception(f"Server returned status {response.status_code}: {response.text}")
                    
                result = KernelExecResult(**response.json())
                
                return result
            
        except httpx.TimeoutException:
            raise Exception("Server request timed out after 5 minutes")
        except httpx.ConnectError:
            raise Exception(f"Could not connect to server at {self.base_url}")
        except Exception as e:
            raise Exception(f"Error calling server: {e}")


async def _process_evaluation_task(
    kb_eval_client: KbEvalClient,
    file_path: Path,
    gen_id: int,
    input_base_dir: Path,
    output_base_dir: Path,
    time_tag: str,
    task_id: int,
    model_tag: str = "vllm"
) -> bool:
    """
    Process a single evaluation task for a generated file.
    
    Args:
        kb_eval_client: KbEvalClient instance
        file_path: Path to the original input Python file
        gen_id: Generation ID for this task
        input_base_dir: Base directory for input files
        output_base_dir: Base directory for output files
        time_tag: Timestamp tag for output files
        task_id: Task identifier for logging
        model_tag: Model tag for evaluation
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Get relative path for file structure
        relative_path = file_path.relative_to(input_base_dir)
        output_sub_dir = output_base_dir / relative_path / f"{gen_id:02d}"
        
        # Check if the required files exist
        reference_code_file = output_base_dir / relative_path / "reference_code.py"
        generated_code_file = output_sub_dir / "generated_code.py"
        
        if not reference_code_file.exists():
            print(f"⚠️ [Task {task_id}] Reference code file not found: {reference_code_file}")
            return False
            
        if not generated_code_file.exists():
            print(f"⚠️ [Task {task_id}] Generated code file not found: {generated_code_file}")
            return False
        
        # Read the reference and generated code
        with open(reference_code_file, 'r', encoding='utf-8') as f:
            reference_code = f.read()
        
        with open(generated_code_file, 'r', encoding='utf-8') as f:
            generated_code = f.read()
        
        # Prepare evaluation parameters
        task_name = str(relative_path)  # Get filename without extension
        eval_params = {
            "model_tag": model_tag,
            "task_tag": task_name,
            "eval_tag": f"gen_{gen_id:02d}",
            "time_tag": time_tag,
            "reference_code": reference_code,
            "generated_code": generated_code
        }
        
        # Call the evaluation server
        start_time = time.time()
        result = await kb_eval_client.call_kb_eval_server(eval_params)
        result = result.model_dump()
        evaluation_time = time.time() - start_time
        
        # Extract key metrics for logging
        compiled = result.get('compiled', False)
        correctness = result.get('correctness', False)
        runtime = result.get('runtime', -1.0)
        
        print(f"🔍 [Task {task_id}] Evaluated [{f'{gen_id:02d}'}]: [compiled={compiled}], [correct={correctness}], [runtime={runtime:.4f}ms] in [{evaluation_time:.2f}s]")
        
        # Save evaluation result
        evaluation_file = output_sub_dir / "generated_code_eval.json"
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        if compiled and correctness:
            print(f"✅ [Task {task_id}] Evaluated Correctly [{f'{gen_id:02d}'}]: {evaluation_file}")
        else:
            print(f"⚠️  [Task {task_id}] Evaluated Incorrectly [{f'{gen_id:02d}'}]: {evaluation_file}")
        return True
        
    except Exception as e:
        print(f"❌ [Task {task_id}] Error in _process_evaluation_task for {file_path}: {e}")
        print(traceback.format_exc())
        return False


async def inference_and_eval_task(queue: asyncio.Queue, inference_client: VLLMClient, kb_eval_client: KbEvalClient, input_base_dir: Path, output_base_dir: Path, time_tag: str, task_id: int):
    """
    Run inference task for a file.
    
    Args:
        queue: Queue containing file paths to process
        client: VLLMClient instance
        input_base_dir: Base directory for input files
        output_base_dir: Base directory for output files
        time_tag: Timestamp tag for output files
        task_id: Unique task identifier
    """

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
            max_retries = 3
            while retry_count < max_retries:
                retry_count += 1
                try:
                    # Process the inference task
                    success = await _process_inference_task(
                        client=inference_client,
                        file_path=file_path,
                        gen_id=gen_id,
                        input_base_dir=input_base_dir,
                        output_base_dir=output_base_dir,
                        time_tag=time_tag,
                        task_id=task_id
                    )
                    
                    if success:
                        # we are successful, break the retry loop
                        break
                    
                except Exception as e:
                    # add warning emoji to beginning of the line
                    if retry_count >= max_retries:
                        print(f"❌ [Task {task_id}] Error generating [{f'{gen_id:02d}'}] for {file_path}: {e}", f"[{retry_count}/{max_retries}]")
                    else:
                        print(f"⚠️  [Task {task_id}] Error generating [{f'{gen_id:02d}'}] for {file_path}: {e}", f"[{retry_count}/{max_retries}]")
                    print(traceback.format_exc())

            # add info emoji to beginning of the line
            print(f"🔍 [Task {task_id}] Evaluating [{f'{gen_id:02d}'}]: {file_path}")

            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                retry_count += 1
                try:
                    # Process the evaluation task
                    success = await _process_evaluation_task(
                        kb_eval_client=kb_eval_client,
                        file_path=file_path,
                        gen_id=gen_id,
                        input_base_dir=input_base_dir,
                        output_base_dir=output_base_dir,
                        time_tag=time_tag,
                        task_id=task_id,
                        model_tag=inference_client.model
                    )
                    
                    if success:
                        # we are successful, break the retry loop
                        break
                    
                except Exception as e:
                    # add warning emoji to beginning of the line
                    if retry_count >= max_retries:
                        print(f"❌ [Task {task_id}] Error evaluating [{f'{gen_id:02d}'}] for {file_path}: {e}", f"[{retry_count}/{max_retries}]")
                    else:
                        print(f"⚠️  [Task {task_id}] Error evaluating [{f'{gen_id:02d}'}] for {file_path}: {e}", f"[{retry_count}/{max_retries}]")
                    print(traceback.format_exc())

        except asyncio.TimeoutError:
            # Timeout waiting for queue item, check if queue is empty
            if queue.empty():
                print(f"❌ [Task {task_id}] Queue is empty, terminating for {file_path}")
                break
        except Exception as e:
            print(f"❌ [Task {task_id}] Unexpected error: {e}")
            print(traceback.format_exc())
            break
    
    # circle emoji to beginning of the line
    print(f"🔄 [Task {task_id}] completed")





async def async_eval_reference_code(kb_eval_client: KbEvalClient, model_tag: str, task_tag: str, time_tag: str, reference_code: str, output_path: Path) -> KernelExecResult:
    """
    Evaluate the reference code for a file.
    """
    # for each file in bucket_files, run kb_eval_ref
    result = await kb_eval_client.call_kb_eval_ref(eval_params={
        "model_tag": model_tag,
        "task_tag": task_tag,
        "time_tag": time_tag,
        "reference_code": reference_code,
    })
    # write the result to the output path
    output_eval_file = output_path / f"reference_code_eval.json"
    with open(output_eval_file, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"✅ Reference code [{task_tag}] evaluation result: [{result.runtime}ms]")

    return result

async def main():
    """Main function for batch processing Python files."""
    parser = argparse.ArgumentParser(description="Process Python files with SGLang or vLLM generate API")
    parser.add_argument("--input-dir", type=str, default="./kernel_bench/level1", help="Input directory containing Python files")
    parser.add_argument("--output-dir", type=str, default="./_output", help="Output directory for results")
    parser.add_argument("--num-tasks", type=int, default=8, help="Number of concurrent processing tasks")
    parser.add_argument("--num-generations", type=int, default=8, help="Number of generations to perform for each file")
    parser.add_argument("--client", type=str, default="vllm", help="Client type to use (vllm, runpod, sglang, deepseek, fireworks)")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming mode")
    parser.add_argument("--epoch-id", type=int, default=1, help="Epoch ID to process")
    parser.add_argument("--bucket-size", type=int, default=10, help="Number of files to process in each bucket")
    parser.add_argument("--bucket-id", type=int, default=1, help="Bucket ID to process")
    parser.add_argument("--bucket-seed", type=int, default=42, help="Seed for random number generator")
    
    args = parser.parse_args()

    global STREAMING
    STREAMING = args.streaming
    
    # Create timestamp for this run
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir + "_" + f"{args.epoch_id:03d}" + "_" + f"{args.bucket_id:02d}" + "_" + f"{args.bucket_seed:02d}" + "_" + time_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Find all Python files recursively
    python_files = list(input_dir.rglob("*.py"))
    # sort the python files by md5 of filepath
    python_files.sort(key=lambda x: hashlib.md5(str(str(x) + str(args.epoch_id) + str(args.bucket_seed)).encode('utf-8')).hexdigest())
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

    # Create vLLM client
    inference_client = VLLMClient(client_type=args.client, config_file="sequential_inference.yaml")
    print("Created vLLM client")
    
    # Test client connection
    print("Testing client connection...")
    if await inference_client.health_check():
        print("✅ Client connection successful")
        models = inference_client.get_models()
        print(f"Available models: {models}")
    else:
        print("❌ Client connection failed")
        return

    # Create KbEval client
    kb_eval_client = KbEvalClient(config_file="sequential_inference.yaml")
    print("Created KbEval client")
    
    # Test kbEvalRemoteServer connection
    print(f"Testing kbEvalRemoteServer connection [{kb_eval_client.base_url}]...")
    try:
        # test with /stats endpoint
        result = requests.get(f"{kb_eval_client.base_url}/stats", timeout=5)
        result.raise_for_status()
        print(f"✅ kbEvalRemoteServer stats: {result.json()}")
    except Exception as e:
        print(f"❌ kbEvalRemoteServer connection failed: {e}")
        return

    # for each file in bucket_files, write file content to relevant path
    ref_eval_tasks = []
    for file_path in bucket_files:
        # get the relative path
        relative_path = file_path.relative_to(input_dir)
        # get the output path
        output_path = output_dir / relative_path
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"reference_code.py"
        # write the file content to the output path
        with open(output_file, 'w') as f:
            reference_code = file_path.read_text()
            f.write(reference_code)

        ref_eval_task = asyncio.create_task(
            async_eval_reference_code(kb_eval_client, inference_client.model, str(relative_path), time_tag, reference_code, output_path)
        )
        ref_eval_tasks.append(ref_eval_task)

    # wait for all reference code evaluation tasks to complete
    await asyncio.gather(*ref_eval_tasks)

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
            inference_and_eval_task(queue, inference_client, kb_eval_client, input_dir, output_dir, time_tag, task_id+1)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    print(f"Starting {args.num_tasks} tasks...")
    await asyncio.gather(*tasks)

    metadata = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "time_tag": time_tag,
        "num_tasks": args.num_tasks,
        "num_generations": args.num_generations,
        "model_tag": inference_client.model,
        "epoch_id": args.epoch_id,
        "bucket_id": args.bucket_id,
        "bucket_seed": args.bucket_seed,
        "bucket_size": args.bucket_size,
        "bucket_files": [{
            "file_path": str(file_path),
            "md5": hashlib.md5(str(str(file_path) + str(args.epoch_id) + str(args.bucket_seed)).encode('utf-8')).hexdigest()
        } for file_path in bucket_files],
        "bucket_files_count": len(bucket_files),
    }

    # write the metadata to the output directory
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"All tasks completed!")
    print(f"Results saved in: {output_dir}")

    # upload the output directory to s3
    s3_client = boto3.client('s3')
    s3_client.upload_file(output_dir, 'agent-xyz', f'{args.epoch_id:03d}_{args.bucket_id:02d}/{output_dir.name}')


if __name__ == "__main__":
    asyncio.run(main())
