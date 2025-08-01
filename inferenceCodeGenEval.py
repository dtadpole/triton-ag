import os
import re
import sys
import shutil
import duckdb
import json
import time
import asyncio
import traceback
import argparse
import yaml
import random
from datetime import datetime
from typing import List, Dict, Optional
import boto3
import requests
from pathlib import Path
from transformers import AutoTokenizer
from inferenceClient import InferenceClient, InferenceClientConfig, load_inference_client_config
from kbEvalClient import KbEvalClient
from logger import logger
from kbEvalTest.kbeval import KernelExecResult
from globalUtils import MODEL_OVERRIDE_KEY, CodeGenEvalBlock, ExemplarBlock, CritiqueBlock
from globalRegClient import GlobalRegClient
from globalWorkflow import GlobalWorkflow

class CodeGenEvalClient:
    def __init__(
            self,
            run_tag: str,
            inference_client_config: InferenceClientConfig,
            config_file: str = "inferenceCodeGenEval.yaml",
            output_dir: str = "~/.codeGenEval",
            logprobs: bool = True,
            template: str="triton.1",
    ):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.run_tag = run_tag
        self.inference_client_config = inference_client_config
        self.inference_client_config.model.logprobs = logprobs # always use logprobs
        self.inference_client = InferenceClient(config=self.inference_client_config)
        self.tokenizer = self.inference_client.tokenizer
        self.model_tag = self.inference_client.model_tag
        self.kb_eval_client = KbEvalClient()
        self.output_dir = self._get_output_dir(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logprobs = logprobs
        self.template = template
        self.reference_eval_cache = {}

        logger.info(f"🔍 [CodeGenEvalClient] [{self.run_tag}] [{self.model_tag}] Using logprobs: [{self.inference_client_config.model.logprobs}] [{self.template}]")

    def _get_output_dir(self, output_dir: str) -> Path:
        """Get output sub directory for a task."""
        return Path(os.path.expanduser(output_dir)) / self.run_tag / self.model_tag

    def get_prompt_template(self) -> str:
        """Get prompt template from configuration."""
        prompts_config = self.config.get('prompts', {})
        if self.template not in prompts_config:
            raise ValueError(f"Template [{self.template}] not found in prompts config")
        return prompts_config[self.template]

    def get_code_type(self) -> str:
        """Get code type from configuration."""
        prompts_config = self.get_prompt_template()
        return prompts_config.get('code_type', 'triton')

    def get_system_prompt(self) -> str:
        """Get system prompt from configuration."""
        prompts_config = self.get_prompt_template()
        reference_code = self.get_example_reference_code()
        generated_code = self.get_example_generated_code()
        return prompts_config.get('system_prompt', 'You are a helpful assistant.').format(
            reference_code=reference_code,
            generated_code=generated_code,
        )

    def get_user_prompt(self, reference_code: str, reference_eval: Dict, prev_generated_code: Optional[str] = None, prev_generated_eval: Optional[Dict] = None) -> str:
        """Get user prompt from configuration with source code substituted."""
        prompts_config = self.get_prompt_template()
        if prev_generated_code is None or prev_generated_eval is None:
            user_prompt_template = prompts_config.get('user_prompt.init', 'Implement Triton code for the following reference code.\n\nReference code:\n```python\n{reference_code}\n```\n\nReference code evaluation:\n```json\n{reference_eval}\n```')
        else:
            if prev_generated_eval['compiled'] and prev_generated_eval['correctness']:
                user_prompt_template = prompts_config.get('user_prompt.perf', 'Compare to the reference code performance evaluation and improve your code performance by optimizing the code.\n\nGenerated code evaluation:\n```json\n{prev_generated_eval}\n```')
            else:
                user_prompt_template = prompts_config.get('user_prompt.fix', 'Your code did not compile or run correctly.  Please fix the code and return the correct code.\n\nGenerated code evaluation:\n```json\n{prev_generated_eval}\n```')
        return user_prompt_template.format(
            reference_code=reference_code,
            reference_eval=reference_eval,
            prev_generated_code=prev_generated_code,
            prev_generated_eval=prev_generated_eval,
        )

    def get_example_reference_code(self) -> str:
        """Get example reference code from configuration."""
        prompts_config = self.get_prompt_template()
        example_prompt = prompts_config.get('examples', {})
        return example_prompt.get('reference_code', '')

    def get_example_generated_code(self) -> str:
        """Get example generated code from configuration."""
        prompts_config = self.get_prompt_template()
        example_prompt = prompts_config.get('examples', {})
        return example_prompt.get('generated_code', '')

    async def _process_code_gen(
        self,
        task_tag: str,
        turn_tag: str,
        reference_code: str,
        reference_eval: Dict,
        prev_generated_code: Optional[str] = None,
        prev_generated_eval: Optional[Dict] = None,
        message_history: Optional[List[Dict]] = None,
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
        if message_history is None or len(message_history) == 0:
            system_prompt = self.get_system_prompt()
            user_prompt = self.get_user_prompt(reference_code, reference_eval, prev_generated_code, prev_generated_eval)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            user_prompt = self.get_user_prompt(reference_code, reference_eval, prev_generated_code, prev_generated_eval)
            messages = message_history + [
                {"role": "user", "content": user_prompt}
            ]

        if self.logprobs:
            # configure prompt tokenization parameters
            add_generation_prompt = True
            enable_thinking = self.inference_client_config.model.enable_thinking
            # use completion api if logprobs is True
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )

            # Generate response
            start_time = time.time()
            result = await self.inference_client.completion(prompt)
            generation_time = time.time() - start_time
            
            content = result.get('content', '')
            logprobs = result.get('logprobs', [])

            # prompt token ids and completion token ids can only be an estimate, not accurate
            # prompt_token_ids = self.tokenizer.encode(prompt)
            # num_prompt_tokens = len(prompt_token_ids)
            # completion_token_ids = self.tokenizer.encode(content) if content else []
            # num_completion_tokens = len(completion_token_ids)
            guessed_completion_tokens = self.tokenizer.encode(content) if content else []
            num_completion_tokens = len(guessed_completion_tokens)
            token_per_second = num_completion_tokens / generation_time

            metadata = {
                "run_tag": self.run_tag,
                "model_tag": self.model_tag,
                "task_tag": task_tag,
                "turn_tag": turn_tag,
                "code_type": self.get_code_type(),
                "reference_code": reference_code,
                "num_completion_tokens": num_completion_tokens,
                "generation_time_seconds": generation_time,
                "token_per_second": token_per_second,
            }

            # save response to a
            conversation_file = self.output_dir / task_tag / f"{turn_tag}_conversation.json"
            conversation = {
                "messages": messages + [
                    {"role": "assistant", "content": content}
                ],
                "metadata": metadata,
            }

            with open(conversation_file, 'w') as f:
                f.write(json.dumps(conversation, indent=2, ensure_ascii=False, default=str))

            completion_file = self.output_dir / task_tag / f"{turn_tag}_completion.json"
            completion = {
                "prompt": prompt,
                "generation": content,
                "logprobs": logprobs,
                "metadata": metadata,
            }

            with open(completion_file, 'w') as f:
                f.write(json.dumps(completion, indent=2, ensure_ascii=False, default=str))

        else:
            # use chat completion api if logprobs is False
            start_time = time.time()
            result = await self.inference_client.chat_completion(messages)
            generation_time = time.time() - start_time

            content = result.get('content', '')
            # we do not have access to the model's tokenizer, so we cannot get the number of prompt tokens
            # we can only rely on the usage attribute to get the number of completion tokens
            num_completion_tokens = result['usage']['completion_tokens'] if 'usage' in result else 0
            token_per_second = num_completion_tokens / generation_time

            metadata = {
                "run_tag": self.run_tag,
                "model_tag": self.model_tag,
                "task_tag": task_tag,
                "turn_tag": turn_tag,
                "code_type": self.get_code_type(),
                "reference_code": reference_code,
                "num_completion_tokens": num_completion_tokens,
                "generation_time_seconds": generation_time,
                "token_per_second": token_per_second,
            }

            conversation_file = self.output_dir / task_tag / f"{turn_tag}_conversation.json"
            conversation = {
                "messages": messages + [
                    {"role": "assistant", "content": content}
                ],
                "metadata": metadata,
            }

            with open(conversation_file, 'w') as f:
                f.write(json.dumps(conversation, indent=2, ensure_ascii=False, default=str))

        try:
            # find the last Python code block using regex
            code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
            generated_code = code_blocks[-1].strip() if code_blocks else content.strip()
        except (IndexError, AttributeError) as e:
            logger.error(f"❌ [CodeGenEval {task_tag}] Error extracting code [{e}] [{f'{num_completion_tokens}'} tokens] in [{generation_time:.2f}s] [{token_per_second:.2f} tokens/s]")
            return conversation, None
        
        # save the generated code to a file
        generated_code_file = self.output_dir / task_tag / f"{turn_tag}_generated_code.py"
        with open(generated_code_file, 'w') as f:
            f.write(generated_code)

        # use generated emoji to beginning of the line
        logger.info(f"👏 [CodeGenEval] [{self.run_tag}] Generated [{f'{task_tag}'}] [{f'{turn_tag}'}]: [{generated_code_file}] [{f'{num_completion_tokens}'} tokens] in [{generation_time:.2f}s] [{token_per_second:.2f} tokens/s]")
        return conversation, generated_code

    async def _process_code_eval(
        self,
        task_tag: str,
        turn_tag: str,
        generated_code: str,
        reference_eval: Optional[Dict] = None,
    ) -> Dict:
        """
        Process a single evaluation task for a generated file.
        
        Args:
            task_tag: Task tag for this task
            gen_tag: Generation ID for this task
            generated_code: Generated code for this task
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Check if the required files exist
            reference_code_file = self.output_dir / task_tag / "reference_code.py"
            generated_code_file = self.output_dir / task_tag / f"{turn_tag}_generated_code.py"
            
            if not reference_code_file.exists():
                logger.error(f"❌ [CodeGenEval] [{self.run_tag}] Reference code file not found: {reference_code_file}")
                return False
                
            if not generated_code_file.exists():
                logger.error(f"❌ [CodeGenEval] [{self.run_tag}] Generated code file not found: {generated_code_file}")
                return False
            
            # Read the reference and generated code
            with open(reference_code_file, 'r', encoding='utf-8') as f:
                reference_code = f.read()
            
            with open(generated_code_file, 'r', encoding='utf-8') as f:
                generated_code = f.read()
            
            # Call the evaluation server
            start_time = time.time()
            result = await self.kb_eval_client.kb_eval(run_tag=self.run_tag, model_tag=self.model_tag, task_tag=task_tag, eval_tag=turn_tag, reference_code=reference_code, generated_code=generated_code, code_type=self.get_code_type())
            eval_result_json = result.model_dump()
            # compare generated performance to reference performance
            if reference_eval and 'runtime' in reference_eval:
                reference_runtime = reference_eval['runtime']
                if 'runtime' in eval_result_json and eval_result_json['runtime'] > 0.0:
                    generated_runtime = eval_result_json['runtime']
                    speedup = reference_runtime / generated_runtime
                    speed_emoji = '🚀' if speedup > 1.0 else ('🦙' if speedup > 0.5 else '🐢')
                else:
                    generated_runtime = eval_result_json['runtime']
                    speedup = 0.0
                    speed_emoji = '🐢'
            else:
                reference_runtime = 0.0
                generated_runtime = eval_result_json['runtime']
                speedup = 0.0
                speed_emoji = '🐢'

            eval_result_json['metadata'] = {
                "run_tag": self.run_tag,
                "model_tag": self.model_tag,
                "task_tag": task_tag,
                "turn_tag": turn_tag,
                "code_type": self.get_code_type(),
                "performance": {
                    "reference_runtime": reference_runtime,
                    "generated_runtime": generated_runtime,
                    "speedup": speedup,
                },
            } | eval_result_json['metadata']
            evaluation_time = time.time() - start_time
            
            # Save evaluation result
            evaluation_file = self.output_dir / task_tag / f"{turn_tag}_eval.json"
            with open(evaluation_file, 'w') as f:
                f.write(json.dumps(eval_result_json, indent=2, ensure_ascii=False, default=str))
            if result.compiled and result.correctness:
                logger.info(f"✅ [CodeGenEval] [{self.run_tag}] Evaluated [{f'{task_tag}'}] [{f'{turn_tag}'}] [{generated_code_file}] [{result.runtime:.3f}ms] in [{evaluation_time:.1f}s] [{f'{speed_emoji}'} {f'{speedup:.2f}x'}]")
            else:
                logger.warning(f"⚠️ [CodeGenEval] [{self.run_tag}] Evaluated [{f'{task_tag}'}] [{f'{turn_tag}'}] [{generated_code_file}] [{'🟢' if result.compiled else '🔴'} compiled], [{'🟢' if result.correctness else '🔴'} correctness] in [{evaluation_time:.2f}s]")

            # return json
            return eval_result_json
            
        except Exception as e:
            logger.error(f"❌ [CodeGenEval] [{self.run_tag}] Error in _process_evaluation_task for [{task_tag}]: {e}")
            logger.error(traceback.format_exc())
            return None


    async def _process_ref_eval(self, task_tag: str, reference_code: str) -> Dict:
        """
        Evaluate the reference code for a file.
        """
        logger.info(f"🔍 [CodeGenEval Client] [{self.run_tag}] Evaluating reference code for [{task_tag}]...")
        try:
            # for each file in bucket_files, run kb_eval_ref
            start_time = time.time()
            result = await self.kb_eval_client.kb_eval_ref(run_tag=self.run_tag, model_tag=self.model_tag, task_tag=task_tag, reference_code=reference_code)
            result_json = result.model_dump()
            evaluation_time = time.time() - start_time
            result_json['metadata'] = {
                "run_tag": self.run_tag,
                "model_tag": self.model_tag,
                "task_tag": task_tag,
            } | result_json['metadata']

            # write the result to the output path
            reference_eval_file = self.output_dir / task_tag / f"reference_eval.json"
            with open(reference_eval_file, 'w') as f:
                f.write(json.dumps(result_json, indent=2, ensure_ascii=False, default=str))
            logger.info(f"📚 [CodeGenEval] [{self.run_tag}] Task [{task_tag}] Reference code evaluation result: [{result.runtime:.3f}ms] in [{evaluation_time:.2f}s]")

            self.reference_eval_cache[task_tag] = result_json

            # return json
            return result_json
        except Exception as e:
            logger.error(f"❌ [CodeGenEval] [{self.run_tag}] Error in _process_ref_eval: [{e}] in [{evaluation_time:.1f}s]")
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
                    logger.info(f"[CodeGenEval] [{self.run_tag}] Received termination signal")
                    break

                is_reference = item['is_reference']
                reference_code = item['reference_code']
                task_tag = item['task_tag']
                gen_tag = item['gen_tag']
                num_turns_per_generation = item['num_turns_per_generation']

                if is_reference:
                    # evaluate the reference code
                    ref_eval_result = await self._process_ref_eval(task_tag=task_tag, reference_code=reference_code)
                    continue

                # add info emoji to beginning of the line
                logger.info(f"🔍 [CodeGenEval] [{self.run_tag}] Processing [{task_tag}] [{f'{gen_tag}'}]...")

                while task_tag not in self.reference_eval_cache:
                    sleep_time = random.uniform(1, 10) # sleep randomly between 1 and 5 seconds, using float to avoid blocking
                    logger.info(f"⏳ [CodeGenEval] [{self.run_tag}] Waiting for reference eval for [{task_tag}] [{f'{gen_tag}'}], sleeping for [{f'{sleep_time:.2f}s'}]")
                    await asyncio.sleep(sleep_time)
                    
                if not self.reference_eval_cache[task_tag]['compiled'] or not self.reference_eval_cache[task_tag]['correctness']:
                    logger.warning(f"⚠️ [CodeGenEval] [{self.run_tag}] Skipping generation [{f'{gen_tag}'}] as reference code is not correct for [{task_tag}]")
                    continue

                reference_eval = self.reference_eval_cache[task_tag]
                reference_runtime = reference_eval['runtime']
                prev_generated_code = None
                prev_generated_eval = None
                message_history = None
                for turn_count in range(num_turns_per_generation):
                    turn_tag = f"{gen_tag}_t{turn_count:02d}"

                    retry_count = 0
                    max_retries = 3
                    while retry_count < max_retries:
                        retry_count += 1
                        try:
                            # add info emoji to beginning of the line
                            logger.info(f"🔍 [CodeGenEval] [{self.run_tag}] Generating code for [{task_tag}] [{f'{turn_tag}'}] [{f'{retry_count}/{max_retries}'}]...")
                            # Process the inference task
                            gen_conversation, generated_code = await self._process_code_gen(task_tag=task_tag, turn_tag=turn_tag, reference_code=reference_code, reference_eval=reference_eval, prev_generated_code=prev_generated_code, prev_generated_eval=prev_generated_eval, message_history=message_history)
                            # if failed try again until max retries
                            if not generated_code:
                                logger.warning(f"⚠️ [CodeGenEval] [{self.run_tag}] No generated code for [{task_tag}] [{f'{turn_tag}'}] [{f'{retry_count}/{max_retries}'}]")
                                continue

                            # add info emoji to beginning of the line
                            logger.info(f"🔍 [CodeGenEval] [{self.run_tag}] Evaluating code for [{task_tag}] [{f'{turn_tag}'}] [{f'{retry_count}/{max_retries}'}]...")
                            # Process the evaluation task
                            generated_eval = await self._process_code_eval(task_tag=task_tag, turn_tag=turn_tag, generated_code=generated_code, reference_eval=reference_eval)
                            # if failed try again until max retries
                            if not generated_eval:
                                logger.warning(f"⚠️ [CodeGenEval] [{self.run_tag}] No evaluation result for [{task_tag}] [{f'{turn_tag}'}] [{f'{retry_count}/{max_retries}'}]")
                                continue
                            
                            # update the message history only if both generated code and evaluation steps have completed successfully
                            modified_history = gen_conversation['messages']
                            if modified_history[-1]['role'] == 'assistant':
                                # remove all other generated content but only keep the generated code
                                modified_history[-1]['content'] = f"```python\n{generated_code}\n```"
                            else:
                                # log an error
                                logger.error(f"⚠️ [CodeGenEval] [{self.run_tag}] Error updating message history for [{task_tag}] [{f'{turn_tag}'}] [{f'{retry_count}/{max_retries}'}]")
                                logger.error(traceback.format_exc())
                                continue
                            # we are successful, update the message history and break the retry loop
                            message_history = modified_history
                            prev_generated_code = generated_code
                            prev_generated_eval = generated_eval
                            break

                        except Exception as e:
                            # add warning emoji to beginning of the line
                            if retry_count >= max_retries:
                                logger.error(f"❌ [CodeGenEval] [{self.run_tag}] Error generating or evaluating [{task_tag}] [{f'{gen_tag}'}]: {e}. Max retries reached [{retry_count}/{max_retries}]")
                            else:
                                logger.warning(f"⚠️ [CodeGenEval] [{self.run_tag}] Error generating or evaluating [{task_tag}] [{f'{gen_tag}'}]: {e}. Retrying... [{retry_count}/{max_retries}]")
                            logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"❌ [CodeGenEval] [{self.run_tag}] Error processing [{task_tag}] [{f'{gen_tag}'}]: {e}")
                logger.error(traceback.format_exc())

        # task completed
        logger.info(f"🎯 [CodeGenEval] [{self.run_tag}] [Task {task_id:02d}] completed. Remaining tasks: [{len(asyncio.all_tasks())}]")


    async def run_block(
        self,
        task_tags: List[str],
        reference_code_contents: List[str],
        reference_eval_contents: List[str] = None,
        num_generations: int=8,
        num_turns_per_generation: int=4,
        parallel_tasks: int=8,
        run_reference: bool=True,
    ):
        """
        Run the code generation and evaluation tasks.
        """
        # add info emoji to beginning of the line
        logger.info(f"🔍 [CodeGenEval] [{self.run_tag}] [{self.model_tag}] Using [{num_generations}] generations, [{parallel_tasks}] parallel tasks")

        # Create queue and add all files
        queue = asyncio.Queue()
        # for each reference code in reference_code_contents, write file content to relevant path
        # ref_eval_tasks = []
        if not reference_eval_contents:
            reference_eval_contents = [None] * len(reference_code_contents)
        for reference_code, reference_eval_content, task_tag in zip(reference_code_contents, reference_eval_contents, task_tags):
            # get the output path
            output_path = self.output_dir / task_tag
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"reference_code.py"
            # write the file content to the output path
            with open(output_file, 'w') as f_out:
                f_out.write(reference_code)

            if run_reference:
                # add reference tasks to the queue
                await queue.put({
                    "is_reference": True,
                    "task_tag": task_tag,
                    "gen_tag": f"reference",
                    "num_turns_per_generation": num_turns_per_generation,
                    "reference_code": reference_code,
                })
            else:
                # write the reference eval to the output directory
                reference_eval_file = self.output_dir / task_tag / f"reference_eval.json"
                with open(reference_eval_file, 'w') as f:
                    f.write(reference_eval_content)

        # add generation tasks to the queue
        for reference_code, task_tag in zip(reference_code_contents, task_tags):
            for gen_id in range(num_generations):
                await queue.put({
                    "is_reference": False,
                    "task_tag": task_tag,
                    "gen_tag": f"gen_{gen_id:02d}",
                    "num_turns_per_generation": num_turns_per_generation,
                    "reference_code": reference_code,
                })
        
        # Add sentinel values to signal task completion
        for _ in range(parallel_tasks):
            await queue.put(None)
        
        # Create and start processing tasks
        tasks = []
        for task_id in range(parallel_tasks):
            task = asyncio.create_task(
                self.code_gen_and_eval_task(queue, task_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        logger.info(f"🔍 [CodeGenEval] [{self.run_tag}] [{self.model_tag}] Starting [{parallel_tasks}] tasks...")
        await asyncio.gather(*tasks)

        metadata = {
            "output_dir": str(self.output_dir),
            "run_tag": self.run_tag,
            "model_tag": self.model_tag,
            "num_samples": len(reference_code_contents),
            "num_generations": num_generations,
            "num_turns_per_generation": num_turns_per_generation,
            "parallel_tasks": parallel_tasks,
            "run_reference": run_reference,
            "template": self.template,
        }

        # write the metadata to the output directory
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"All tasks completed!")
        logger.info(f"Results saved in: {self.output_dir}")

        # upload the output directory to s3
        # try:
        #     s3_client = boto3.client('s3')
        #     # recursively upload folder to s3 (not a file)
        #     for root, dirs, files in os.walk(self.output_dir):
        #         for file in files:
        #             relative_path = os.path.relpath(os.path.join(root, file), self.output_dir)
        #             s3_client.upload_file(os.path.join(root, file), 'agent-xyz', f'{self.run_tag}/{relative_path}')
        #             # add success emoji to beginning of the line
        #     logger.info(f"✅ [CodeGenEvalClient] [{self.run_tag}] Uploaded [{self.output_dir}] to [s3://agent-xyz/{self.run_tag}]")
        # except Exception as e:
        #     logger.error(f"❌ [CodeGenEvalClient] [{self.run_tag}] Error uploading to s3: {e}")
        #     logger.error(traceback.format_exc())

async def code_gen_eval_block(block: CodeGenEvalBlock):
    """
    Run one batch of code generation and evaluation.
    """
    try:
        config = load_inference_client_config(
            provider_name=block.provider_name,
            model_short_name=block.model_name,
        )

        if block.epoch_id < 0 or block.block_id < 0:
            run_tag = f"{block.prefix_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            run_tag = f"{block.prefix_tag}_{block.epoch_id:03d}_{block.block_id:02d}"

        # override the model name
        if block.model_override:
            config.model.model_name = block.model_override

        # start running the batch
        logger.info(f"🔍 [CodeGenEval] [{run_tag}] Starting block...")

        # Set up directories
        input_dir = Path(os.path.expanduser(block.input_dir))
        if not input_dir.exists():
            logger.error(f"❌ [CodeGenEval] [{run_tag}] Error: Input directory {input_dir} does not exist")
            return

        # recursively get all the python files under the input directory and store in a list
        reference_code_json = []
        for root, dirs, files in os.walk(input_dir, followlinks=True):
            for file in files:
                if file.endswith(".py"):
                    # read the file content
                    with open(os.path.join(root, file), 'r') as f:
                        reference_code = f.read()
                    relpath = os.path.relpath(os.path.join(root, file), input_dir).replace("/", "_")
                    task_tag = "_".join(relpath.split("_")[:5])
                    reference_code_json.append({
                        "reference_code": reference_code,
                        "task_tag": task_tag,
                    })

        # randomly pick args.num_samples files from the list
        random.shuffle(reference_code_json)
        reference_code_json = reference_code_json[:block.num_samples]

        reference_code_contents = [item['reference_code'] for item in reference_code_json]
        task_tags = [item['task_tag'] for item in reference_code_json]

        # now run inference
        codeGenEvalClient = CodeGenEvalClient(run_tag=run_tag, inference_client_config=config, output_dir=block.output_dir, logprobs=block.logprobs, template=block.template)
        await codeGenEvalClient.run_block(task_tags, reference_code_contents, num_generations=block.num_generations, num_turns_per_generation=block.num_turns_per_generation, parallel_tasks=block.parallel_tasks)
        logger.info(f"🎉 [CodeGenEval] [{run_tag}] Block completed. Remaining tasks: [{len(asyncio.all_tasks())}]")

    except Exception as e:
        logger.error(f"❌ [CodeGenEval] [{run_tag}] Error running block: [{e}] in [{traceback.format_exc()}]")
        logger.error(traceback.format_exc())

async def exemplar_block(block: ExemplarBlock):
    """
    Run one batch of code generation and evaluation.
    """
    try:
        config = load_inference_client_config(
            provider_name=block.provider_name,
            model_short_name=block.model_name,
        )

        # start running the batch
        logger.info(f"🔍 [Exemplar] [{block.input_tag}] Starting block...")

        # Set up directories
        search_path = Path(os.path.expanduser(block.input_dir)) / block.input_tag
        if not search_path.exists():
            logger.error(f"❌ [Exemplar] [{block.input_tag}] Error: Input directory {search_path} does not exist")
            return

        # use dockdb to get a list of reference code and generated code
        result = duckdb.sql(f"""SELECT filename, compiled, correctness, metadata, runtime, runtime_stats
                            FROM read_json_auto('{search_path}/**/reference_eval.json', sample_size=-1, ignore_errors=true) 
                        """)
        
        result_df = result.df()

        logger.info(f"🔍 [Exemplar] [{block.input_tag}] Found [{len(result_df)}] tasks for exemplar in [{search_path}]\n[{result_df}]")

        # for each reference code, copy over the reference code to the output directory
        reference_code_contents = []
        reference_eval_contents = []
        task_tags = []
        for index, row in result_df.iterrows():
            # copy the reference code to the output directory
            reference_eval_filename = row['filename']
            reference_code_filename = reference_eval_filename.replace("reference_eval.json", "reference_code.py")
            with open(reference_code_filename, 'r') as f:
                reference_code = f.read()
            reference_code_contents.append(reference_code)
            with open(reference_eval_filename, 'r') as f:
                reference_eval_content = f.read()
            reference_eval_contents.append(reference_eval_content)
            task_tags.append(row['metadata']['task_tag'])

        # now run inference
        codeGenEvalClient = CodeGenEvalClient(run_tag=block.input_tag, inference_client_config=config, output_dir=block.output_dir, logprobs=False, template=block.template)
        await codeGenEvalClient.run_block(task_tags, reference_code_contents, reference_eval_contents=reference_eval_contents, num_generations=block.num_generations, parallel_tasks=block.parallel_tasks, run_reference=False)
        logger.info(f"🎉 [Exemplar] [{block.input_tag}] Block completed. Remaining tasks: [{len(asyncio.all_tasks())}]")

    except Exception as e:
        logger.error(f"❌ [Exemplar] [{block.input_tag}] Error running block: [{e}] in [{traceback.format_exc()}]")
        logger.error(traceback.format_exc())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="TC_0.1.0_14B")
    parser.add_argument("--epoch_id", type=int, default=-1)
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_14B_000_00")
    parser.add_argument("--input_dir", type=str, default="~/triton-ag/kernel_bench", help="Input directory containing Python files")
    parser.add_argument("--output_dir", type=str, default="~/.codeGenEval", help="Output directory for the code generation and evaluation results")
    parser.add_argument("--provider", type=str, default="fireworks")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--model", type=str, default="deepseek-v3")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--model_override", type=str, default=None) # override the model name, e.g. "KC_0.1.0_14B/checkpoint-200"
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--num_turns_per_generation", type=int, default=4)
    parser.add_argument("--parallel_tasks", type=int, default=4)
    parser.add_argument("--template", type=str, default="triton.1")
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--run_exemplar", action="store_true")
    parser.add_argument("--use_global_registry", action="store_true")
    parser.add_argument("--proc_id", type=str, default=None)
    args = parser.parse_args()

    if args.proc_id is not None:
        PROC_ID = args.proc_id
    else:
        PROC_ID = os.environ.get("PROC_ID", None)

    if args.use_global_registry and not args.run_exemplar:
        if not PROC_ID:
            error_msg = f"❌ [CodeGenEvalClient] [{args.prefix_tag}] Unable to get PROC_ID to update model_override"
            logger.error(error_msg)
            sys.exit(1)
        else:
            logger.info(f"🔍 [CodeGenEvalClient] [{args.prefix_tag}] Using PROC_ID: [{PROC_ID}]")

    if not args.run_exemplar:
        try:
            run_tag = 'unknown'
            if args.use_global_registry:
                # get the global registry
                global_reg_client = GlobalRegClient()
                # get the codeGenEvalBlock from the global registry
                block_json = await global_reg_client.dequeue(f"inference.codeGenEval")
                # convert the block_json to a CodeGenEvalBlock object
                block = CodeGenEvalBlock(**block_json)
                # process the model override
                model_override = await global_reg_client.get(f"{MODEL_OVERRIDE_KEY}")
                if model_override:
                    logger.info(f"🔍 [CodeGenEvalClient] [{block.prefix_tag}] Using model override: [{model_override}]")
                    block.model_override = model_override
                    # update model_override in the global registry
                    if PROC_ID is None:
                        error_msg = f"❌ [CodeGenEvalClient] [{block.prefix_tag}] Unable to get PROC_ID to update model_override [{model_override}]"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    else:
                        await global_reg_client.put(f"adapter.codeGenEval.model_override.{PROC_ID}", model_override)
            else:
                block = CodeGenEvalBlock(
                    prefix_tag=args.prefix_tag,
                    epoch_id=args.epoch_id,
                    block_id=args.block_id,
                    provider_name=args.provider,
                    model_name=args.model,
                    num_samples=args.num_samples,
                    num_generations=args.num_generations,
                    num_turns_per_generation=args.num_turns_per_generation,
                    parallel_tasks=args.parallel_tasks,
                    model_override=args.model_override,
                    input_dir=args.input_dir,
                    output_dir=args.output_dir,
                    template=args.template,
                    logprobs=args.logprobs,
                )
            # run the block
            await code_gen_eval_block(block)

            if args.use_global_registry:
                globalWorkflow = GlobalWorkflow(prefix_tag=block.prefix_tag)
                await globalWorkflow.post_codeGenEval(block)

        except Exception as e:
            logger.error(f"❌ [CodeGenEval] [{run_tag}] Error running block: [{type(e).__name__}: {e}]")
            logger.error(traceback.format_exc())

    else:
        # exemplar
        try:
            if args.use_global_registry:
                # get the global registry
                global_reg_client = GlobalRegClient()
                # get the ExemplarBlock from the global registry
                block_json = await global_reg_client.dequeue(f"inference.exemplar")
                # convert the block_json to a ExemplarBlock object
                block = ExemplarBlock(**block_json)
            else:
                block = ExemplarBlock(
                    prefix_tag=args.prefix_tag,
                    epoch_id=args.epoch_id,
                    block_id=args.block_id,
                    input_tag=args.input_tag,
                    provider_name=args.provider,
                    model_name=args.model,
                    num_generations=args.num_generations,
                    num_turns_per_generation=args.num_turns_per_generation,
                    parallel_tasks=args.parallel_tasks,
                    model_override=args.model_override,
                    input_dir=args.input_dir,
                    output_dir=args.output_dir,
                    template=args.template,
                )
            # run the block
            await exemplar_block(block)

            if args.use_global_registry:
                globalWorkflow = GlobalWorkflow(prefix_tag=block.prefix_tag)
                await globalWorkflow.post_exemplar(block)

        except Exception as e:
            logger.error(f"❌ [Exemplar] Error running block: [{type(e).__name__}: {e}]")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
