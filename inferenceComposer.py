
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
from typing import List, Dict, Any
import boto3
import requests
import duckdb
from pathlib import Path
from transformers import AutoTokenizer
from inferenceClient import InferenceClient, InferenceClientConfig, load_inference_client_config
from logger import logger
from kbEvalClient import KbEvalClient
from globalUtils import CritiqueBlock
from globalRegClient import GlobalRegClient
from globalWorkflow import GlobalWorkflow
from endpointUtil import Recorder, CodeExtractor, StatsClient


class ComposeClient:
    def __init__(
            self,
            input_tag: str,
            inference_client_config: InferenceClientConfig,
            prompt_file: str = "inferenceCompose/prompt.exemplar.yaml",
            workflow_file: str = "inferenceCompose/workflow.exemplar.yaml",
            output_dir: str = "~/.inference/compose",
            stats_dir: str = "~/.trainer/stats",
    ):
        with open(prompt_file, 'r') as f:
            self.prompt_config = yaml.safe_load(f)
        self.input_tag = input_tag
        self.queue = asyncio.Queue()
        self.recorder = Recorder()
        self.kbEvalClient = KbEvalClient()
        self.statsClient = StatsClient()
        self.inference_client_config = inference_client_config
        self.inferenceClient = InferenceClient(config=self.inference_client_config)
        self.tokenizer = self.inference_client.tokenizer
        self.model_tag = self.inference_client.model_tag
        self.output_dir = self._get_output_dir(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_dir(self, output_dir: str) -> Path:
        """Get output sub directory for a task."""
        return Path(os.path.expanduser(output_dir)) / self.input_tag / self.model_tag

    def get_system_prompt(self) -> str:
        """Get system prompt from configuration."""
        prompts_config = self.prompt_config.get('prompts', {})
        return prompts_config.get('system_prompt', 'You are a helpful assistant.')

    def get_user_prompt(self, reference_code: str, reference_code_eval: dict, generated_code: str, generated_code_eval: dict) -> str:
        """Get user prompt from configuration with source code substituted."""
        prompts_config = self.prompt_config.get('prompts', {})
        user_prompt_template = prompts_config.get('user_prompt', 'Reference code: {reference_code}\n\nReference code evaluation: {reference_code_eval}\n\nGenerated code: {generated_code}\n\nGenerated code evaluation: {generated_code_eval}')
        return user_prompt_template.format(reference_code=reference_code, reference_code_eval=json.dumps(reference_code_eval, indent=2), generated_code=generated_code, generated_code_eval=json.dumps(generated_code_eval, indent=2))

    async def _process_tasks(
        self,
        context_vars: dict,
        task_id: int,
        task_tag: str,
        gen_tag: str,
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
                "run_tag": self.input_tag,
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
        logger.info(f"👏 [Critique] [{self.input_tag}] Critiqued [{f'{task_tag}'}] [{f'{gen_tag}'}]: [{critique_file}] [{f'{num_tokens}'} tokens] [{f'{num_reasoning_tokens}'} reasoning tokens] in [{generation_time:.2f}s] [{token_per_second:.2f} tokens/s]")
        return critique_content


    async def critique_task(self, worker_id: int):
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
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                if item is None:
                    logger.info(f"[Worker {worker_id:02d}] Received termination signal")
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
                    logger.warning(f"⚠️ [Critique] [{self.input_tag}] Reference code evaluation file not found: {reference_code_eval_file}")
                    continue
                # read the reference code evaluation result from the file
                with open(reference_code_eval_file, 'r') as f:
                    reference_code_eval = json.load(f)

                # read generated code
                generated_code_file = os.path.join(base_dir, f"{gen_tag}_generated_code.py")
                if not os.path.exists(generated_code_file):
                    logger.warning(f"⚠️ [Critique] [{self.input_tag}] Generated code file not found: {generated_code_file}")
                    continue
                # read the generated code from the file
                with open(generated_code_file, 'r') as f:
                    generated_code = f.read()

                # read generated code evaluation result
                generated_code_eval_file = os.path.join(base_dir, f"{gen_tag}_eval.json")
                if not os.path.exists(generated_code_eval_file):
                    logger.warning(f"⚠️ [Critique] [{self.input_tag}] Evaluation result file not found: {generated_code_eval_file}")
                    continue
                # read the evaluation result from the file
                with open(generated_code_eval_file, 'r') as f:
                    generated_code_eval = json.load(f)

                output_base_dir = self.output_dir / task_tag
                critique_conversation_file = output_base_dir / f"{gen_tag}_conversation.json"
                critique_content_file = output_base_dir / f"{gen_tag}_critique.txt"
                if os.path.exists(critique_content_file) and os.path.exists(critique_conversation_file):
                    # add a skip emoji to beginning of the line
                    logger.info(f"⚡️ [Critique] [{self.input_tag}] Critique already exists: [{critique_content_file}], skipping...")
                    continue

                # add info emoji to beginning of the line
                logger.info(f"🔍 [Critique] [{self.input_tag}] Processing [{task_tag}] [{f'{gen_tag}'}]...")

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
                            logger.error(f"❌ [Critique {worker_id:02d}] Error critiquing [{task_tag}] [{f'{gen_tag}'}]: {e}", f"[{retry_count}/{max_retries}]")
                        else:
                            logger.warning(f"⚠️ [Critique {worker_id:02d}] Error critiquing [{task_tag}] [{f'{gen_tag}'}]: {e}", f"[{retry_count}/{max_retries}]")
                        logger.error(traceback.format_exc())

            except asyncio.TimeoutError:
                # Timeout waiting for queue item, check if queue is empty
                if queue.empty():
                    # add info magnifying glass emoji to beginning of the line
                    logger.info(f"🔍 [Critique] [{self.input_tag}] Queue is empty, terminating")
                    break
            except Exception as e:
                logger.error(f"❌ [Critique] [{self.input_tag}] Unexpected error: {e}")
                logger.error(traceback.format_exc())
                break
        
        # circle emoji to beginning of the line
        logger.info(f"🎯 [Critique] [{self.input_tag}] completed. Remaining tasks: [{len(asyncio.all_tasks())}]")


async def compose_block(block: ComposeBlock):
    """
    Run one batch of code generation and evaluation.
    """
    try:
        config = load_inference_client_config(
            provider_name=block.provider_name,
            model_short_name=block.model_name,
        )

        # override the model name
        if block.model_override:
            config.model.model_name = block.model_override

        # start running the block
        logger.info(f"🔍 [Critique] [{block.input_tag}] Starting block...")

        # Set up directories
        search_path = os.path.expanduser(f"{block.input_dir}/{block.input_tag}")
        if not os.path.exists(search_path):
            logger.error(f"❌ [Critique] [{block.prefix_tag}] Error: Input directory [{search_path}] does not exist")
            return

        # query from search_path folder, find all the conversation_*.json files, and load them into a dataframe
        result = duckdb.sql(f"""SELECT filename, messages, metadata
                            FROM read_json_auto('{search_path}/**/*_conversation.json') 
                            WHERE messages[3]['content'] IS NOT NULL
                        """)

        logger.info(f"✅ [Critique] [{block.prefix_tag}] Found [{len(result)}] tasks to critique in [{search_path}]\n[{result}]")

        df = result.df()

        queue = asyncio.Queue()
        # iterate the dataframe and put the items into the queue
        for index, row in df.iterrows(): 
            queue.put_nowait({
                "filename": row['filename'],
                "messages": row['messages'],
                "metadata": row['metadata']
            })

        critiqueClient = CritiqueClient(input_tag=block.input_tag, inference_client_config=config)

        critique_tasks = []
        for i in range(block.parallel_tasks):
            critique_tasks.append(asyncio.create_task(critiqueClient.critique_task(queue, i)))

        await asyncio.gather(*critique_tasks)

        logger.info(f"🎉 [Critique] [{block.input_tag}] Block completed")

    except Exception as e:
        logger.error(f"❌ [Critique] [{block.input_tag}] Error running block: [{e}] in [{traceback.format_exc()}]")
        logger.error(traceback.format_exc())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="KC_0.1.0_14B")
    parser.add_argument("--epoch_id", type=int, default=-1)
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--input_tag", type=str, default="KC_0.1.0_14B_000_00")
    parser.add_argument("--input_dir", type=str, default="~/.codeGenEval", help="Input directory containing Python files")
    parser.add_argument("--output_dir", type=str, default="~/.critique", help="Output directory for the critique results")
    parser.add_argument("--provider", type=str, default="fireworks")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--model", type=str, default="deepseek-v3")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--parallel_tasks", type=int, default=16)
    parser.add_argument("--use_global_registry", action="store_true")
    args = parser.parse_args()

    try:
        if args.use_global_registry:
            # get the global registry
            global_reg_client = GlobalRegClient()
            # get the critiqueBlock from the global registry
            block_json = await global_reg_client.dequeue(f"inference.critique")
            # convert the block_json to a CritiqueBlock object
            block = CritiqueBlock(**block_json)
        else:
            block = CritiqueBlock(
                prefix_tag=args.prefix_tag,
                epoch_id=args.epoch_id,
                block_id=args.block_id,
                input_tag=args.input_tag,
                provider_name=args.provider,
                model_name=args.model,
                parallel_tasks=args.parallel_tasks,
                model_override=args.model_override,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
            )
        # run the block
        await critique_block(block)

        if args.use_global_registry:
            globalWorkflow = GlobalWorkflow(prefix_tag=block.prefix_tag)
            await globalWorkflow.post_critique(block)

    except Exception as e:
        logger.error(f"❌ [Critique] [{block.input_tag}] Error running block: [{e}]")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())