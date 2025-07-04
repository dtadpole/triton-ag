#!/usr/bin/env python3
"""
CodeGen and Evaluation Client
============================

Client code that uses OpenAI Chat Completion API to get results from given prompts.
Supports roll out on a batch of code generations and evaluations.
"""

import random
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
import traceback
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from datetime import datetime
from typing import Dict, List, Optional
from kbEvalClient import KbEvalClient
from kbEvalTest.kbeval import KernelExecResult
from pathlib import Path
import requests
import yaml
from inferenceClient import InferenceClient
from loguru import logger

STREAMING = True

class CodeGenEvalClient:
    def __init__(self, config_file: str, run_tag: str, client_type: str):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.run_tag = run_tag
        self.client_type = client_type
        self.model_tag = self.config.get(client_type, {}).get('generation', {}).get('model', 'deepseek-v3')
        self.tokenizer_name = self.config.get(client_type, {}).get('generation', {}).get('tokenizer', 'Qwen/Qwen3-8B')
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.inference_client = InferenceClient(client_type=client_type, streaming=STREAMING, config_file=config_file)
        self.kb_eval_client = KbEvalClient()
        self.output_dir = self._get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_dir(self) -> Path:
        """Get output sub directory for a task."""
        return Path(os.path.expanduser(f"~/.inferenceCodeGenEval/{self.run_tag}/{self.client_type}/{self.model_tag}"))

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

    async def _process_code_gen_task(
        self,
        task_tag: str,
        reference_code: str,
        gen_id: str,
    ) -> str:
        """
        Process a single inference task for a file.
        
        Args:
            task_tag: Task tag for this task
            reference_code: Reference code for this task
            gen_id: Generation ID for this task
            
        Returns:
            str: Generated code for this task
        """
        # Create conversation
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(reference_code)
        
        # Generate response
        start_time = time.time()
        result = await self.inference_client.chat_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        generation_time = time.time() - start_time
        
        content = result.get('content', '')
        reasoning_content = result.get('reasoning_content', '')

        tokens = self.tokenizer.encode(content) if content else []
        reasoning_tokens = self.tokenizer.encode(reasoning_content) if reasoning_content else []

        num_tokens = len(tokens)
        num_reasoning_tokens = len(reasoning_tokens)
        logger.info(f"🔍 [Task {task_tag}] Responded [{f'{num_tokens}'} tokens] [{f'{num_reasoning_tokens}'} reasoning tokens] in [{generation_time:.2f}s]")

        # save response to a file
        conversation_file = self.output_dir / task_tag / f"{gen_id}_conversation.json"

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
                    "reasoning_content": reasoning_content,
                    "content": content
                }
            ],
            "metadata": {
                "run_tag": self.run_tag,
                "client_type": self.client_type,
                "model_tag": self.model_tag,
                "task_tag": task_tag,
                "gen_id": gen_id,
                "reference_code": reference_code,
                "num_tokens": num_tokens,
                "num_reasoning_tokens": num_reasoning_tokens,
                "generation_time_seconds": generation_time,
            }
        }

        with open(conversation_file, 'w') as f:
            f.write(json.dumps(conversation, indent=2, ensure_ascii=False, default=str))

        # extract the generated code from the generated text
        try:
            # find the last Python code block using regex
            code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
            generated_code = code_blocks[-1].strip() if code_blocks else content.strip()
        except (IndexError, AttributeError) as e:
            logger.error(f"❌ [Task {task_tag}] Error extracting code: {e}, code generation failed")
            return None
        
        # save the generated code to a file
        generated_code_file = self.output_dir / task_tag / f"{gen_id}_generated_code.py"
        with open(generated_code_file, 'w') as f:
            f.write(generated_code)

        # use generated emoji to beginning of the line
        logger.info(f"👏 [Task {task_tag}] Generated [{f'{gen_id}'}]: {generated_code_file}")
        return generated_code


    async def _process_evaluation_task(
        self,
        task_tag: str,
        gen_id: str,
        generated_code: str,
    ) -> bool:
        """
        Process a single evaluation task for a generated file.
        
        Args:
            kb_eval_client: KbEvalClient instance
            file_path: Path to the original input Python file
            gen_id: Generation ID for this task
            input_base_dir: Base directory for input files
            output_base_dir: Base directory for output files
            run_tag: Run tag for evaluation
            task_id: Task identifier for logging
            model_tag: Model tag for evaluation
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Check if the required files exist
            reference_code_file = self.output_dir / task_tag / "reference_code.py"
            generated_code_file = self.output_dir / task_tag / f"{gen_id}_generated_code.py"
            
            if not reference_code_file.exists():
                logger.error(f"❌ [Task {task_tag}] Reference code file not found: {reference_code_file}")
                return False
                
            if not generated_code_file.exists():
                logger.error(f"❌ [Task {task_tag}] Generated code file not found: {generated_code_file}")
                return False
            
            # Read the reference and generated code
            with open(reference_code_file, 'r', encoding='utf-8') as f:
                reference_code = f.read()
            
            with open(generated_code_file, 'r', encoding='utf-8') as f:
                generated_code = f.read()
            
            # Call the evaluation server
            start_time = time.time()
            result = await self.kb_eval_client.kb_eval(run_tag=self.run_tag, model_tag=self.model_tag, task_tag=task_tag, eval_tag=f"gen_{gen_id:02d}", reference_code=reference_code, generated_code=generated_code)
            evaluation_time = time.time() - start_time
            
            # Save evaluation result
            evaluation_file = self.output_dir / task_tag / f"{gen_id}_eval.json"
            with open(evaluation_file, 'w') as f:
                f.write(json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str))
            
            if result.compiled and result.correctness:
                logger.info(f"✅ [Task {task_tag}] Evaluated [{f'{gen_id}'}] [{generated_code_file}] [{result.runtime:.3f}ms] in [{evaluation_time:.1f}s]")
            else:
                logger.warning(f"⚠️ [Task {task_tag}] Evaluated [{f'{gen_id}'}] [{generated_code_file}] [{'✅' if result.compiled else '❌'}compiled], [{'✅' if result.correctness else '❌'}correctness] in [{evaluation_time:.2f}s]")
            return result
            
        except Exception as e:
            logger.error(f"❌ [Task {task_tag}] Error in _process_evaluation_task for [gen_{gen_id}]: {e}")
            logger.error(traceback.format_exc())
            return None


    async def code_gen_and_eval_task(self, queue: asyncio.Queue, task_id: int):
        """
        Run inference task for a file.
        
        Args:
            queue: Queue containing file paths to process
            run_tag: Run tag for evaluation
            task_tag: Unique task identifier
        """

        while True:
            try:
                # Get file path from queue (blocking with timeout)
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                if item is None:
                    logger.info(f"[Task {task_id:02d}] Received termination signal")
                    break

                reference_code = item['reference_code']
                task_tag = item['task_tag']
                gen_id = item['gen_id']

                # add info emoji to beginning of the line
                logger.info(f"🔍 [Task {task_id:02d}] Processing [{task_tag}] [{f'{gen_id}'}]: {reference_code[:20]}...")

                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    retry_count += 1
                    try:
                        # Process the inference task
                        generated_code = await self._process_code_gen_task(task_tag=task_tag, reference_code=reference_code, gen_id=gen_id)
                        
                        if generated_code:
                            # we are successful, break the retry loop
                            break
                        
                    except Exception as e:
                        # add warning emoji to beginning of the line
                        if retry_count >= max_retries:
                            logger.error(f"❌ [Task {task_id:02d}] Error generating [{task_tag}] [{f'{gen_id}'}]: {e}", f"[{retry_count}/{max_retries}]")
                        else:
                            logger.warning(f"⚠️  [Task {task_id:02d}] Error generating [{task_tag}] [{f'{gen_id}'}]: {e}", f"[{retry_count}/{max_retries}]")
                        logger.error(traceback.format_exc())

                # add info emoji to beginning of the line
                logger.info(f"🔍 [Task {task_id:02d}] Evaluating [{task_tag}] [{f'{gen_id}'}]: {generated_code[:20]}...")

                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    retry_count += 1
                    try:
                        # Process the evaluation task
                        result = await self._process_evaluation_task(task_tag=task_tag, gen_id=gen_id, generated_code=generated_code)
                        
                        if result:
                            # we are successful, break the retry loop
                            break
                        
                    except Exception as e:
                        # add warning emoji to beginning of the line
                        if retry_count >= max_retries:
                            logger.error(f"❌ [Task {task_id:02d}] Error evaluating [{task_tag}] [{f'{gen_id}'}]: {e}", f"[{retry_count}/{max_retries}]")
                        else:
                            logger.warning(f"⚠️  [Task {task_id:02d}] Error evaluating [{task_tag}] [{f'{gen_id}'}]: {e}", f"[{retry_count}/{max_retries}]")
                        logger.error(traceback.format_exc())

            except asyncio.TimeoutError:
                # Timeout waiting for queue item, check if queue is empty
                if queue.empty():
                    # add info magnifying glass emoji to beginning of the line
                    logger.info(f"🔍 [Task {task_id:02d}] Queue is empty, terminating")
                    break
            except Exception as e:
                logger.error(f"❌ [Task {task_id:02d}] Unexpected error: {e}")
                logger.error(traceback.format_exc())
                break
        
        # circle emoji to beginning of the line
        logger.info(f"🔄 [Task {task_tag}] completed")


    async def reference_eval_task(self, task_tag: str, reference_code: str) -> KernelExecResult:
        """
        Evaluate the reference code for a file.
        """
        # for each file in bucket_files, run kb_eval_ref
        result = await self.kb_eval_client.kb_eval_ref(run_tag=self.run_tag, model_tag=self.model_tag, task_tag=task_tag, reference_code=reference_code)
        # write the result to the output path
        output_eval_file = self.output_dir / task_tag / f"reference_eval.json"
        with open(output_eval_file, 'w') as f:
            f.write(json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str))
        logger.info(f"✅ Reference code [{task_tag}] evaluation result: [{result.runtime:.3f}ms]")

        return result

async def main():
    """Main function for batch processing Python files."""
    parser = argparse.ArgumentParser(description="Process reference Python code with Inference and Evaluation API")
    parser.add_argument("--input-dir", type=str, default="./kernel_bench/", help="Input directory containing Python files")
    parser.add_argument("--output-dir", type=str, default="./_output", help="Output directory for results")
    parser.add_argument("--num-tasks", type=int, default=15, help="Number of concurrent processing tasks")
    parser.add_argument("--parallel-tasks", type=int, default=8, help="Number of parallel tasks to run in parallel")
    parser.add_argument("--num-generations", type=int, default=8, help="Number of generations to perform for each file")
    parser.add_argument("--client", type=str, default="fireworks-r1", help="Client type to use (vllm, runpod, sglang, deepseek, fireworks)")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming mode")
    parser.add_argument("--run-prefix", type=str, default="code_gen_v0.1", help="Run prefix")
    parser.add_argument("--epoch-id", type=int, default=1, help="Epoch ID to process")
    
    args = parser.parse_args()

    global STREAMING
    STREAMING = args.streaming

    # Set up directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Error: Input directory {input_dir} does not exist")
        return

    # recursively get all the python files under the input directory and store in a list
    python_files = []
    for root, dirs, files in os.walk(input_dir, followlinks=True):
        for file in files:
            if file.endswith(".py"):
                python_files.append({
                    "task_tag": os.path.relpath(os.path.join(root, file), input_dir),
                    "file_path": os.path.join(root, file),
                })

    # randomly pick args.num_tasks files from the list
    random.shuffle(python_files)
    bucket_files = python_files[:args.num_tasks]

    # add info emoji to beginning of the line
    logger.info(f"🔍 Using [{args.client}] client [{args.parallel_tasks}] parallel tasks")

    # Create vLLM client
    run_tag = f"{args.run_prefix}_epoch_{args.epoch_id:02d}"
    codeGenEvalClient = CodeGenEvalClient(config_file="inferenceClient.yaml", run_tag=run_tag, client_type=args.client)
    logger.info(f"Created [{args.client}] client for [{run_tag}]")
    
    # Test client connection
    logger.info("Testing client connection...")
    if await codeGenEvalClient.inference_client.health_check():
        logger.info("✅ Client connection successful")
        models = codeGenEvalClient.inference_client.get_models()
        logger.info(f"Available models: {models}")
    else:
        print("❌ Client connection failed")
        return

    # Create KbEval client
    kb_eval_client = KbEvalClient()
    print("Created KbEval client")
    
    # Test kbEvalRemoteServer connection
    logger.info(f"Testing kbEvalRemoteServer connection [{kb_eval_client.base_url}]...")
    try:
        # test with /stats endpoint
        result = requests.get(f"{kb_eval_client.base_url}/stats", timeout=5)
        result.raise_for_status()
        logger.info(f"✅ kbEvalRemoteServer stats: {result.json()}")
    except Exception as e:
        logger.error(f"❌ kbEvalRemoteServer connection failed: {e}")
        return

    # for each file in bucket_files, write file content to relevant path
    ref_eval_tasks = []
    for file in bucket_files:
        file_path = file['file_path']
        task_tag = file['task_tag']
        # get the output path
        output_path = codeGenEvalClient.output_dir / task_tag
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"reference_code.py"
        # write the file content to the output path
        with open(output_file, 'w') as f_out:
            with open(file_path, 'r') as f_in:
                reference_code = f_in.read()
            f_out.write(reference_code)

        ref_eval_task = asyncio.create_task(
            codeGenEvalClient.reference_eval_task(task_tag, reference_code)
        )
        ref_eval_tasks.append(ref_eval_task)

    # wait for all reference code evaluation tasks to complete
    await asyncio.gather(*ref_eval_tasks)

    # Create queue and add all files
    queue = asyncio.Queue()
    for file in bucket_files:
        file_path = file['file_path']
        task_tag = file['task_tag']
        with open(file_path, 'r') as f_in:
            reference_code = f_in.read()
        for gen_id in range(args.num_generations):
            await queue.put({
                "reference_code": reference_code,
                "task_tag": task_tag,
                "gen_id": f"gen_{gen_id:02d}",
            })
    
    # Add sentinel values to signal task completion
    for _ in range(args.num_tasks):
        await queue.put(None)
    
    # Create and start processing tasks
    tasks = []
    for task_id in range(args.parallel_tasks):
        task = asyncio.create_task(
            codeGenEvalClient.code_gen_and_eval_task(queue, task_id)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    print(f"Starting {args.num_tasks} tasks...")
    await asyncio.gather(*tasks)

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(codeGenEvalClient.output_dir),
        "run_tag": run_tag,
        "num_tasks": args.num_tasks,
        "num_generations": args.num_generations,
        "model_tag": codeGenEvalClient.model_tag,
        "epoch_id": args.epoch_id,
        "bucket_files": [{
            "file_path": str(file_path["file_path"]),
            "task_tag": file_path["task_tag"],
        } for file_path in bucket_files],
        "bucket_files_count": len(bucket_files),
    }

    # write the metadata to the output directory
    metadata_file = codeGenEvalClient.output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"All tasks completed!")
    logger.info(f"Results saved in: {codeGenEvalClient.output_dir}")

    # upload the output directory to s3
    s3_client = boto3.client('s3')
    # recursively upload folder to s3 (not a file)
    for root, dirs, files in os.walk(codeGenEvalClient.output_dir):
        for file in files:
            s3_client.upload_file(os.path.join(root, file), 'agent-xyz', f'{run_tag}/{file}')

if __name__ == "__main__":
    asyncio.run(main())
