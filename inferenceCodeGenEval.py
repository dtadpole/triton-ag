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
from kbEvalClient import KbEvalClient
from logger import logger
from kbEvalTest.kbeval import KernelExecResult
from globalUtils import CodeGenEvalBlock, TrainerGRPOBlock, CritiqueBlock
from globalRegClient import GlobalRegClient

class CodeGenEvalClient:
    def __init__(
            self,
            run_tag: str,
            inference_client_config: InferenceClientConfig,
            config_file: str = "inferenceCodeGenEval.yaml",
            output_dir: str = "~/.codeGenEval",
            logprobs: bool = True,         
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

        logger.info(f"🔍 [CodeGenEvalClient] [{self.run_tag}] [{self.model_tag}] Using logprobs: [{self.inference_client_config.model.logprobs}]")

    def _get_output_dir(self, output_dir: str) -> Path:
        """Get output sub directory for a task."""
        return Path(os.path.expanduser(output_dir)) / self.run_tag / self.model_tag

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

    async def _process_code_gen(
        self,
        task_tag: str,
        gen_tag: str,
        reference_code: str,
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
        user_prompt = self.get_user_prompt(reference_code)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.inference_client_config.model.enable_thinking)

        # Generate response
        start_time = time.time()
        result = await self.inference_client.completion(prompt)
        generation_time = time.time() - start_time
        
        content = result.get('content', '')
        reasoning_content = result.get('reasoning_content', '')
        logprobs = result.get('logprobs', [])

        tokens = self.tokenizer.encode(content) if content else []
        reasoning_tokens = self.tokenizer.encode(reasoning_content) if reasoning_content else []

        num_tokens = len(tokens)
        num_reasoning_tokens = len(reasoning_tokens)
        token_per_second = (num_tokens + num_reasoning_tokens) / generation_time

        metadata = {
            "run_tag": self.run_tag,
            "model_tag": self.model_tag,
            "task_tag": task_tag,
            "gen_tag": gen_tag,
            "reference_code": reference_code,
            "num_tokens": num_tokens,
            "num_reasoning_tokens": num_reasoning_tokens,
            "generation_time_seconds": generation_time,
            "token_per_second": token_per_second,
        }

        # save response to a
        conversation_file = self.output_dir / task_tag / f"{gen_tag}_conversation.json"
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
            "metadata": metadata,
        }

        with open(conversation_file, 'w') as f:
            f.write(json.dumps(conversation, indent=2, ensure_ascii=False, default=str))

        completion_file = self.output_dir / task_tag / f"{gen_tag}_completion.json"
        completion = {
            "prompt": prompt,
            "generation": content,
            "logprobs": logprobs,
            "metadata": metadata,
        }

        with open(completion_file, 'w') as f:
            f.write(json.dumps(completion, indent=2, ensure_ascii=False, default=str))

        # extract the generated code from the generated text
        try:
            # find the last Python code block using regex
            code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
            generated_code = code_blocks[-1].strip() if code_blocks else content.strip()
        except (IndexError, AttributeError) as e:
            logger.error(f"❌ [CodeGenEval Task {task_tag}] Error extracting code [{e}] [{f'{num_tokens}'} tokens] [{f'{num_reasoning_tokens}'} reasoning tokens] in [{generation_time:.2f}s] [{token_per_second:.2f} tokens/s]")
            return conversation, None
        
        # save the generated code to a file
        generated_code_file = self.output_dir / task_tag / f"{gen_tag}_generated_code.py"
        with open(generated_code_file, 'w') as f:
            f.write(generated_code)

        # use generated emoji to beginning of the line
        logger.info(f"👏 [CodeGenEval Client] [{self.run_tag}] Generated [{f'{task_tag}'}] [{f'{gen_tag}'}]: [{generated_code_file}] [{f'{num_tokens}'} tokens] [{f'{num_reasoning_tokens}'} reasoning tokens] in [{generation_time:.2f}s] [{token_per_second:.2f} tokens/s]")
        return conversation, generated_code

    async def _process_code_eval(
        self,
        task_tag: str,
        gen_tag: str,
        generated_code: str,
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
            generated_code_file = self.output_dir / task_tag / f"{gen_tag}_generated_code.py"
            
            if not reference_code_file.exists():
                logger.error(f"❌ [CodeGenEval Client] [{self.run_tag}] Reference code file not found: {reference_code_file}")
                return False
                
            if not generated_code_file.exists():
                logger.error(f"❌ [CodeGenEval Client] [{self.run_tag}] Generated code file not found: {generated_code_file}")
                return False
            
            # Read the reference and generated code
            with open(reference_code_file, 'r', encoding='utf-8') as f:
                reference_code = f.read()
            
            with open(generated_code_file, 'r', encoding='utf-8') as f:
                generated_code = f.read()
            
            # Call the evaluation server
            start_time = time.time()
            result = await self.kb_eval_client.kb_eval(run_tag=self.run_tag, model_tag=self.model_tag, task_tag=task_tag, eval_tag=gen_tag, reference_code=reference_code, generated_code=generated_code)
            eval_result_json = result.model_dump()
            eval_result_json['metadata'] = {
                "run_tag": self.run_tag,
                "model_tag": self.model_tag,
                "task_tag": task_tag
            } | eval_result_json['metadata']
            evaluation_time = time.time() - start_time
            
            # Save evaluation result
            evaluation_file = self.output_dir / task_tag / f"{gen_tag}_eval.json"
            with open(evaluation_file, 'w') as f:
                f.write(json.dumps(eval_result_json, indent=2, ensure_ascii=False, default=str))
            
            if result.compiled and result.correctness:
                logger.info(f"✅ [CodeGenEval Client] [{self.run_tag}] Evaluated [{f'{task_tag}'}] [{f'{gen_tag}'}] [{generated_code_file}] [{result.runtime:.3f}ms] in [{evaluation_time:.1f}s]")
            else:
                logger.warning(f"⚠️ [CodeGenEval Client] [{self.run_tag}] Evaluated [{f'{task_tag}'}] [{f'{gen_tag}'}] [{generated_code_file}] [{'✅' if result.compiled else '❌'}compiled], [{'✅' if result.correctness else '❌'}correctness] in [{evaluation_time:.2f}s]")

            # return json
            return eval_result_json
            
        except Exception as e:
            logger.error(f"❌ [CodeGenEval Client] [{self.run_tag}] Error in _process_evaluation_task for [{task_tag}]: {e}")
            logger.error(traceback.format_exc())
            return None


    async def _process_ref_eval(self, task_tag: str, reference_code: str) -> Dict:
        """
        Evaluate the reference code for a file.
        """
        logger.info(f"🔍 [CodeGenEval Client] [{self.run_tag}] Evaluating reference code...")
        try:
            # for each file in bucket_files, run kb_eval_ref
            start_time = time.time()
            result = await self.kb_eval_client.kb_eval_ref(run_tag=self.run_tag, model_tag=self.model_tag, task_tag=task_tag, reference_code=reference_code)
            result_json = result.model_dump()
            evaluation_time = time.time() - start_time
            result_json['metadata'] = {
                "run_tag": self.run_tag,
                "model_tag": self.model_tag,
                "task_tag": task_tag
            } | result_json['metadata']

            # write the result to the output path
            reference_eval_file = self.output_dir / task_tag / f"reference_eval.json"
            with open(reference_eval_file, 'w') as f:
                f.write(json.dumps(result_json, indent=2, ensure_ascii=False, default=str))
            logger.info(f"✅ [CodeGenEval Client] [{self.run_tag}] Task [{task_tag}] Reference code evaluation result: [{result.runtime:.3f}ms] in [{evaluation_time:.2f}s]")

            # return json
            return result_json
        except Exception as e:
            logger.error(f"❌ [CodeGenEval Client] [{self.run_tag}] Error in _process_ref_eval: [{e}] in [{evaluation_time:.1f}s]")
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
                    logger.info(f"[CodeGenEval Client] [{self.run_tag}] Received termination signal")
                    break

                is_reference = item['is_reference']
                reference_code = item['reference_code']
                task_tag = item['task_tag']
                gen_tag = item['gen_tag']

                if is_reference:
                    # evaluate the reference code
                    ref_eval_result = await self._process_ref_eval(task_tag=task_tag, reference_code=reference_code)
                    continue

                # add info emoji to beginning of the line
                logger.info(f"🔍 [CodeGenEval Client] [{self.run_tag}] Processing [{task_tag}] [{f'{gen_tag}'}]...")

                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    retry_count += 1
                    try:
                        # Process the inference task
                        gen_conversation, generated_code = await self._process_code_gen(task_tag=task_tag, gen_tag=gen_tag, reference_code=reference_code)
                        
                        if not generated_code:
                            logger.warning(f"⚠️ [CodeGenEval Client] [{self.run_tag}] No generated code for [{task_tag}] [{f'{gen_tag}'}]")
                            continue

                        # we are successful, break the retry loop
                        # add info emoji to beginning of the line
                        logger.info(f"🔍 [CodeGenEval Client] [{self.run_tag}] Evaluating [{task_tag}] [{f'{gen_tag}'}]...")

                        # Process the evaluation task
                        gen_eval_result = await self._process_code_eval(task_tag=task_tag, gen_tag=gen_tag, generated_code=generated_code)
                        
                        if not gen_eval_result:
                            logger.warning(f"⚠️ [CodeGenEval Client] [{self.run_tag}] No evaluation result for [{task_tag}] [{f'{gen_tag}'}]")
                            continue
                        
                        # we are successful, break the retry loop
                        break

                    except Exception as e:
                        # add warning emoji to beginning of the line
                        if retry_count >= max_retries:
                            logger.error(f"❌ [CodeGenEval Client] [{self.run_tag}] Error generating or evaluating [{task_tag}] [{f'{gen_tag}'}]: {e}. Max retries reached [{retry_count}/{max_retries}]")
                        else:
                            logger.warning(f"⚠️ [CodeGenEval Client] [{self.run_tag}] Error generating or evaluating [{task_tag}] [{f'{gen_tag}'}]: {e}. Retrying... [{retry_count}/{max_retries}]")
                        logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"❌ [CodeGenEval Client] [{self.run_tag}] Error processing [{task_tag}] [{f'{gen_tag}'}]: {e}")
                logger.error(traceback.format_exc())

        # task completed
        logger.info(f"🎯 [CodeGenEval Client] [{self.run_tag}] [Task {task_id:02d}] completed. Remaining tasks: [{len(asyncio.all_tasks())}]")


    async def run_block(
        self,
        reference_code_contents: List[str],
        task_tags: List[str],
        num_generations: int=8,
        parallel_tasks: int=8,
    ):
        """
        Run the code generation and evaluation tasks.
        """
        # add info emoji to beginning of the line
        logger.info(f"🔍 [CodeGenEvalClient] [{self.run_tag}] [{self.model_tag}] Using [{num_generations}] generations, [{parallel_tasks}] parallel tasks")

        # Create queue and add all files
        queue = asyncio.Queue()
        # for each reference code in reference_code_contents, write file content to relevant path
        # ref_eval_tasks = []
        for reference_code, task_tag in zip(reference_code_contents, task_tags):
            # get the output path
            output_path = self.output_dir / task_tag
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"reference_code.py"
            # write the file content to the output path
            with open(output_file, 'w') as f_out:
                f_out.write(reference_code)

            await queue.put({
                "is_reference": True,
                "task_tag": task_tag,
                "gen_tag": f"reference",
                "reference_code": reference_code,
            })

        for reference_code, task_tag in zip(reference_code_contents, task_tags):
            for gen_id in range(num_generations):
                await queue.put({
                    "is_reference": False,
                    "task_tag": task_tag,
                    "gen_tag": f"gen_{gen_id:02d}",
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
        logger.info(f"🔍 [CodeGenEvalClient] [{self.run_tag}] [{self.model_tag}] Starting [{parallel_tasks}] tasks...")
        await asyncio.gather(*tasks)

        metadata = {
            "output_dir": str(self.output_dir),
            "run_tag": self.run_tag,
            "model_tag": self.model_tag,
            "num_samples": len(reference_code_contents),
            "num_generations": num_generations,
            "parallel_tasks": parallel_tasks,
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
        logger.info(f"🔍 [CodeGenEvalClient] [{run_tag}] Running batch...")

        # Set up directories
        input_dir = Path(os.path.expanduser(block.input_dir))
        if not input_dir.exists():
            logger.error(f"❌ [CodeGenEvalClient] [{run_tag}] Error: Input directory {input_dir} does not exist")
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
        codeGenEvalClient = CodeGenEvalClient(run_tag=run_tag, inference_client_config=config, output_dir=block.output_dir, logprobs=block.logprobs)
        await codeGenEvalClient.run_block(reference_code_contents, task_tags, block.num_generations, block.parallel_tasks)
        logger.info(f"✅ [CodeGenEvalClient] [{run_tag}] Batch completed. Remaining tasks: [{len(asyncio.all_tasks())}]")

    except Exception as e:
        logger.error(f"❌ [CodeGenEvalClient] [{run_tag}] Error running batch: [{e}] in [{traceback.format_exc()}]")
        logger.error(traceback.format_exc())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="KC_0.1.0_14B")
    parser.add_argument("--epoch_id", type=int, default=-1)
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--input_dir", type=str, default="~/triton-ag/kernel_bench", help="Input directory containing Python files")
    parser.add_argument("--output_dir", type=str, default="~/.codeGenEval", help="Output directory for the code generation and evaluation results")
    parser.add_argument("--provider", type=str, default="deepinfra")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--model", type=str, default="qwen3-14b")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--model_override", type=str, default=None) # override the model name, e.g. "KC_0.1.0_14B/checkpoint-200"
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--parallel_tasks", type=int, default=24)
    parser.add_argument("--logprobs", type=bool, default=True)
    parser.add_argument("--use_global_registry", action="store_true", default=True)
    args = parser.parse_args()

    # class CodeGenEvalBlock(BaseModel):
    #     prefix_tag: str
    #     epoch_id: int
    #     block_id: int
    #     input_dir: str
    #     provider_name: str
    #     model_name: str
    #     num_samples: int
    #     num_generations: int
    #     parallel_tasks: int = Field(default=24)
    #     model_override: str = Field(default=None)
    #     logprobs: bool = Field(default=False)
    #     test_mode: bool = Field(default=False)

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
            model_override = await global_reg_client.get(f"inference.codeGenEval.model_override")
            model_override_value = model_override['value'] if 'value' in model_override else None
            if model_override_value:
                logger.info(f"🔍 [CodeGenEvalClient] [{block.prefix_tag}] Using model override: [{model_override_value}]")
                block.model_override = model_override_value
        else:
            block = CodeGenEvalBlock(
                prefix_tag=args.prefix_tag,
                epoch_id=args.epoch_id,
                block_id=args.block_id,
                provider_name=args.provider,
                model_name=args.model,
                num_samples=args.num_samples,
                num_generations=args.num_generations,
                parallel_tasks=args.parallel_tasks,
                model_override=args.model_override,
                logprobs=args.logprobs,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
            )
        # run the block
        await code_gen_eval_block(block)

        """
        if args.use_global_registry:
            # put next events into global registry
            grpo_block = TrainerGRPOBlock(
                prefix_tag=block.prefix_tag,
                epoch_id=block.epoch_id,
                block_id=block.block_id,
                input_tag=run_tag,
                input_dir='~/.codeGenEval',
                output_dir='~/.trainer',
            )
            await global_reg_client.enqueue(f"inference.trainerGRPO", grpo_block.model_dump())

            # put the critique block into the global registry
            critique_block = CritiqueBlock(
                prefix_tag=block.prefix_tag,
                epoch_id=block.epoch_id,
                block_id=block.block_id,
                input_tag=run_tag,
                input_dir='~/.codeGenEval',
                output_dir='~/.critique',
                provider_name=block.provider_name,
                model_name=block.model_name,
            )
            await global_reg_client.enqueue(f"inference.critique", critique_block.model_dump())
        """

    except Exception as e:
        logger.error(f"❌ [CodeGenEvalClient] [{run_tag}] Error running batch: [{e}]")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
