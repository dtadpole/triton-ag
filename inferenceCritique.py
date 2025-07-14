
import os
import re
import json
import time
import asyncio
import traceback
import argparse
import yaml
import random
from datetime import datetime
from typing import List, Dict
import boto3
import requests
from pathlib import Path
from transformers import AutoTokenizer
from inferenceClient import InferenceClient, InferenceClientConfig, load_inference_client_config
from inferenceCodeGenEval import CODEGEN_EVAL_FOLDER
from kbEvalClient import KbEvalClient
from logger import logger
from kbEvalTest.kbeval import KernelExecResult
from globalRegistry import GlobalRegistry
import duckdb

CRITIQUE_FOLDER = Path(os.path.expanduser("~/.critique"))

class CritiqueClient:
    def __init__(
            self,
            run_tag: str,
            inference_client_config: InferenceClientConfig,
            config_file: str = "inferenceCritique.yaml",
    ):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.run_tag = run_tag
        self.inference_client_config = inference_client_config
        self.inference_client = InferenceClient(config=self.inference_client_config)
        self.tokenizer = self.inference_client.tokenizer
        self.model_tag = self.inference_client.model_tag
        self.output_dir = self._get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_dir(self) -> Path:
        """Get output sub directory for a task."""
        return CRITIQUE_FOLDER / self.run_tag / self.model_tag

    def get_system_prompt(self) -> str:
        """Get system prompt from configuration."""
        prompts_config = self.config.get('prompts', {})
        return prompts_config.get('system_prompt', 'You are a helpful assistant.')

    def get_user_prompt(self, reference_code: str, reference_code_eval: dict, generated_code: str, generated_code_eval: dict) -> str:
        """Get user prompt from configuration with source code substituted."""
        prompts_config = self.config.get('prompts', {})
        user_prompt_template = prompts_config.get('user_prompt', 'Reference code: {reference_code}\n\nReference code evaluation: {reference_code_eval}\n\nGenerated code: {generated_code}\n\nGenerated code evaluation: {generated_code_eval}')
        return user_prompt_template.format(reference_code=reference_code, reference_code_eval=json.dumps(reference_code_eval, indent=2), generated_code=generated_code, generated_code_eval=json.dumps(generated_code_eval, indent=2))

    async def _process_critique(
        self,
        task_tag: str,
        gen_tag: str,
        reference_code: str,
        reference_code_eval: dict,
        generated_code: str,
        generated_code_eval: dict,
    ) -> str:
        """
        Process a single inference task for a file.
        
        Args:
            task_tag: Task tag for this task
            reference_code: Reference code for this task
            gen_tag: Generation ID for this task
            
        Returns:
            str: Generated code for this task
        """
        # Create conversation
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(reference_code, reference_code_eval, generated_code, generated_code_eval)
        
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
        token_per_second = (num_tokens + num_reasoning_tokens) / generation_time

        # prepare the output directory
        output_dir = self.output_dir / task_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        # save response to a file
        conversation_file = output_dir / f"{gen_tag}_conversation.json"

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
                "model_tag": self.model_tag,
                "task_tag": task_tag,
                "gen_tag": gen_tag,
                "reference_code": reference_code,
                "reference_code_eval": reference_code_eval,
                "generated_code": generated_code,
                "evaluation_result": generated_code_eval,
                "num_tokens": num_tokens,
                "num_reasoning_tokens": num_reasoning_tokens,
                "generation_time_seconds": generation_time,
                "token_per_second": token_per_second,
            }
        }

        with open(conversation_file, 'w') as f:
            f.write(json.dumps(conversation, indent=2, ensure_ascii=False, default=str))

        # save the generated code to a file
        critique_file = self.output_dir / task_tag / f"{gen_tag}_critique.txt"
        with open(critique_file, 'w') as f:
            if reasoning_content:
                critique_content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"   
            else:
                critique_content = content
            f.write(critique_content)

        # use generated emoji to beginning of the line
        logger.info(f"👏 [Critique Client] [{self.run_tag}] Critiqued [{f'{task_tag}'}] [{f'{gen_tag}'}]: [{critique_file}] [{f'{num_tokens}'} tokens] [{f'{num_reasoning_tokens}'} reasoning tokens] in [{generation_time:.2f}s] [{token_per_second:.2f} tokens/s]")
        return critique_content


    async def critique_task(self, queue: asyncio.Queue, task_id: int):
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

                filename = item['filename']
                messages = item['messages']
                metadata = item['metadata']

                task_tag = metadata['task_tag']
                gen_tag = metadata['gen_tag']
                reference_code = metadata['reference_code']

                base_dir = os.path.dirname(filename)
                # read reference code evaluation result
                reference_code_eval_file = os.path.join(base_dir, f"reference_eval.json")
                if not os.path.exists(reference_code_eval_file):
                    logger.warning(f"⚠️ [Critique Client] [{self.run_tag}] Reference code evaluation file not found: {reference_code_eval_file}")
                    continue
                # read the reference code evaluation result from the file
                with open(reference_code_eval_file, 'r') as f:
                    reference_code_eval = json.load(f)

                # read generated code
                generated_code_file = os.path.join(base_dir, f"{gen_tag}_generated_code.py")
                if not os.path.exists(generated_code_file):
                    logger.warning(f"⚠️ [Critique Client] [{self.run_tag}] Generated code file not found: {generated_code_file}")
                    continue
                # read the generated code from the file
                with open(generated_code_file, 'r') as f:
                    generated_code = f.read()

                # read generated code evaluation result
                generated_code_eval_file = os.path.join(base_dir, f"{gen_tag}_eval.json")
                if not os.path.exists(generated_code_eval_file):
                    logger.warning(f"⚠️ [Critique Client] [{self.run_tag}] Evaluation result file not found: {generated_code_eval_file}")
                    continue
                # read the evaluation result from the file
                with open(generated_code_eval_file, 'r') as f:
                    generated_code_eval = json.load(f)

                output_base_dir = self.output_dir / task_tag
                critique_conversation_file = output_base_dir / f"{gen_tag}_conversation.json"
                critique_content_file = output_base_dir / f"{gen_tag}_critique.txt"
                if os.path.exists(critique_content_file) and os.path.exists(critique_conversation_file):
                    # add a skip emoji to beginning of the line
                    logger.info(f"⚡️ [Critique Client] [{self.run_tag}] Critique already exists: [{critique_content_file}], skipping...")
                    continue

                # add info emoji to beginning of the line
                logger.info(f"🔍 [Critique Client] [{self.run_tag}] Processing [{task_tag}] [{f'{gen_tag}'}]...")

                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    retry_count += 1
                    try:
                        # Process the inference task
                        critique_content = await self._process_critique(task_tag=task_tag, gen_tag=gen_tag, reference_code=reference_code, reference_code_eval=reference_code_eval, generated_code=generated_code, generated_code_eval=generated_code_eval)
                        
                        if critique_content:
                            # we are successful, break the retry loop
                            break
                        
                    except Exception as e:
                        # add warning emoji to beginning of the line
                        if retry_count >= max_retries:
                            logger.error(f"❌ [Task {task_id:02d}] Error critiquing [{task_tag}] [{f'{gen_tag}'}]: {e}", f"[{retry_count}/{max_retries}]")
                        else:
                            logger.warning(f"⚠️ [Task {task_id:02d}] Error critiquing [{task_tag}] [{f'{gen_tag}'}]: {e}", f"[{retry_count}/{max_retries}]")
                        logger.error(traceback.format_exc())

            except asyncio.TimeoutError:
                # Timeout waiting for queue item, check if queue is empty
                if queue.empty():
                    # add info magnifying glass emoji to beginning of the line
                    logger.info(f"🔍 [Critique Client] [{self.run_tag}] Queue is empty, terminating")
                    break
            except Exception as e:
                logger.error(f"❌ [Critique Client] [{self.run_tag}] Unexpected error: {e}")
                logger.error(traceback.format_exc())
                break
        
        # circle emoji to beginning of the line
        logger.info(f"🎯 [Critique Client] [{self.run_tag}] completed")


async def critique_mini_batch(run_tag: str, config: InferenceClientConfig, parallel_tasks: int=10):
    """
    Run one batch of code generation and evaluation.
    """
    try:
        # start running the batch
        logger.info(f"🔍 [Critique Client] [{run_tag}] Running batch...")

        # Set up directories
        search_path = os.path.expanduser(f"{CODEGEN_EVAL_FOLDER}/{run_tag}")
        if not os.path.exists(search_path):
            logger.error(f"❌ [Critique Client] [{run_tag}] Error: Input directory [{search_path}] does not exist")
            return

        # query from search_path folder, find all the conversation_*.json files, and load them into a dataframe
        result = duckdb.sql(f"""SELECT filename, messages, metadata
                            FROM read_json_auto('{search_path}/**/*_conversation.json') 
                            WHERE messages[3]['content'] IS NOT NULL
                        """)

        logger.info(f"✅ [Critique Client] [{run_tag}] Found [{len(result)}] tasks to critique in [{search_path}]\n[{result}]")

        df = result.df()

        queue = asyncio.Queue()
        # iterate the dataframe and put the items into the queue
        for index, row in df.iterrows(): 
            queue.put_nowait({
                "filename": row['filename'],
                "messages": row['messages'],
                "metadata": row['metadata']
            })


        critiqueClient = CritiqueClient(run_tag=run_tag, inference_client_config=config)

        critique_tasks = []
        for i in range(parallel_tasks):
            critique_tasks.append(asyncio.create_task(critiqueClient.critique_task(queue, i)))

        await asyncio.gather(*critique_tasks)

        logger.info(f"✅ [Critique Client] [{run_tag}] Batch completed")

    except Exception as e:
        logger.error(f"❌ [Critique Client] [{run_tag}] Error running batch: [{e}] in [{traceback.format_exc()}]")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="fireworks")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--model", type=str, default="deepseek-v3")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--run_tag", type=str, default="v0.1_20250714_050308") # {prefix}_{timestamp} or {prefix}_{epoch_id}_{block_id}
    parser.add_argument("--parallel_tasks", type=int, default=16)
    args = parser.parse_args()

    config = load_inference_client_config(
        provider_name=args.provider,
        model_short_name=args.model,
    )

    asyncio.run(critique_mini_batch(args.run_tag, config, args.parallel_tasks))
