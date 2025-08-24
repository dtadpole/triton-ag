
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
from typing import List, Dict, Any, Optional
import boto3
import requests
import duckdb
from pathlib import Path
from transformers import AutoTokenizer
from inferenceClient import InferenceClient, InferenceClientConfig, load_inference_client_config
from inferenceCustomClient import InferenceCustomClient
from logger import logger
from kbEvalClient import KbEvalClient
from workflowUtil import InferenceBlock, MODEL_OVERRIDE_KEY
from workflowClient import WorkflowClient
from workflowServer import WorkflowServer
from configEndpoints import DuckDBClient, Recorder, CodeExtractor, StatsClient
from configInterpreter import ConfigInterpreter

VALID_INPUT_PROCESSORS = [
    "duckdb",
]

class ComposerClient:
    def __init__(
        self,
        input_tag: str,
        inference_client_config: InferenceClientConfig,
        module_file: str = "inference/codeGenEval.module.yaml",
        prompt_file: str = "inference/triton.prompt.yaml",
        example_file: str = "inference/triton.example.yaml",
        output_dir: str = "~/.inference/output",
        stats_dir: str = "~/.trainer/stats",
    ):
        with open(prompt_file, 'r') as f:
            self.prompt_config = yaml.safe_load(f)
        self.input_tag = input_tag
        self.logger = logger
        self.queue = asyncio.Queue()
        self.duckdbClient = DuckDBClient()
        self.recorder = Recorder()
        self.kbEvalClient = KbEvalClient()
        self.statsClient = StatsClient()
        self.codeExtractor = CodeExtractor()
        self.configInterpreter = ConfigInterpreter()
        self.inference_client_config = inference_client_config
        self.inferenceClient = InferenceClient(config=self.inference_client_config)
        self.provider_name = self.inference_client_config.provider.provider_name
        self.model_name = self.inference_client_config.model.model_name
        self.tokenizer = self.inferenceClient.tokenizer
        self.model_tag = self.inferenceClient.model_tag
        self.inferenceCustomClient = InferenceCustomClient(provider_name=self.custom_provider)
        self.output_dir = self._get_output_dir(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.module_file = module_file
        self.prompt_file = prompt_file
        self.module_config = yaml.safe_load(open(module_file, 'r')).get('module', {})
        self.prompt_config = yaml.safe_load(open(prompt_file, 'r')).get('prompts', {})
        self.example_file = example_file
        self.example_config = yaml.safe_load(open(example_file, 'r')).get('examples', {})
        self.context_vars = {
            # built-in context vars
            "self": self,
            "os": os,
            "json": json,
            "yaml": yaml,
            "stats_dir": os.path.expanduser(stats_dir),
            "__runtime_start_time__": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        context_var_config = self.module_config.get('context_vars', {})
        self.context_vars = self.configInterpreter.prepare_context_vars(self, context_var_config, self.context_vars)

    def _get_output_dir(self, output_dir: str) -> Path:
        """Get output sub directory for a task."""
        return Path(os.path.expanduser(output_dir)) / self.input_tag / self.model_tag

    def get_prompt_code_type(self, context_vars: dict) -> str:
        """Get prompt code type from configuration."""
        return self.prompt_config.get('code_type', 'triton')

    def get_system_prompt(self, context_vars: dict) -> str:
        """Get system prompt from configuration."""
        reference_code = self.get_example_reference_code()
        generated_code = self.get_example_generated_code()
        return self.prompt_config.get('system_prompt', 'You are a helpful assistant.').format(
            reference_code=reference_code,
            generated_code=generated_code,
        )

    def get_user_prompt(self, context_vars: dict) -> str:
        """Get user prompt from configuration with source code substituted."""
        generated_code = context_vars.get('generated_code', None)
        generated_eval = context_vars.get('generated_eval', None)
        reference_code = context_vars.get('reference_code', None)
        reference_eval = context_vars.get('reference_eval', None)
        if generated_code is None or generated_eval is None:
            user_prompt_template = self.prompt_config.get('user_prompt.init', 'Implement Triton code for the following reference code.\n\nReference code:\n```python\n{reference_code}\n```\n\nReference code evaluation:\n```json\n{reference_eval}\n```')
        else:
            if generated_eval['compiled'] and generated_eval['correctness']:
                user_prompt_template = self.prompt_config.get('user_prompt.perf', 'Compare to the reference code performance evaluation and improve your code performance by optimizing the code.\n\nGenerated code evaluation:\n```json\n{generated_eval}\n```')
            else:
                user_prompt_template = self.prompt_config.get('user_prompt.fix', 'Your code did not compile or run correctly.  Please fix the code and return the correct code.\n\nGenerated code evaluation:\n```json\n{generated_eval}\n```')
        return user_prompt_template.format(
            reference_code=reference_code,
            reference_eval=json.dumps(reference_eval),
            generated_code=generated_code,
            generated_eval=json.dumps(generated_eval),
        )

    def get_example_reference_code(self) -> str:
        """Get example reference code from configuration."""
        return self.example_config.get('reference_code', '')

    def get_example_generated_code(self) -> str:
        """Get example generated code from configuration."""
        return self.example_config.get('generated_code', '')

    def log_input_processor(self, result: Any, context_vars: dict) -> dict:
        """Process log input processor."""
        input_tag = context_vars.get('input_tag', None)
        logger.info(f"👏 [Composer] [{input_tag}] [{len(result)} samples]\n{result}")

    def log_completion(self, result: dict, context_vars: dict) -> dict:
        run_tag = context_vars.get('run_tag', None)
        model_tag = context_vars.get('model_tag', None)
        task_tag = context_vars.get('task_tag', None)
        turn_tag = context_vars.get('turn_tag', None)
        conversation_path = context_vars.get('conversation_path', None)
        num_completion_tokens = len(result['logprobs']) if 'logprobs' in result else 0
        completion_time_seconds = context_vars.get('__endpoint_time__', None)
        token_per_second = num_completion_tokens / completion_time_seconds if completion_time_seconds > 0 else 0.0
        logger.info(f"👏 [Composer] [{run_tag}] [{model_tag}] [{task_tag}] [{turn_tag}] [{conversation_path}] [{f'{num_completion_tokens}'} tokens] in [{completion_time_seconds:.2f}s] [{token_per_second:.2f} tokens/s]")

    def log_chat_completion(self, result: dict, context_vars: dict) -> dict:
        run_tag = context_vars.get('run_tag', None)
        model_tag = context_vars.get('model_tag', None)
        task_tag = context_vars.get('task_tag', None)
        turn_tag = context_vars.get('turn_tag', None)
        conversation_path = context_vars.get('conversation_path', None)
        num_completion_tokens = result['usage']['completion_tokens'] if 'usage' in result and 'completion_tokens' in result['usage'] else 0
        completion_time_seconds = context_vars.get('__endpoint_time__', None)
        token_per_second = num_completion_tokens / completion_time_seconds if completion_time_seconds > 0 else 0.0
        logger.info(f"👏 [Composer] [{run_tag}] [{model_tag}] [{task_tag}] [{turn_tag}] [{conversation_path}] [{f'{num_completion_tokens}'} tokens] in [{completion_time_seconds:.2f}s] [{token_per_second:.2f} tokens/s]")

    def log_logps(self, result: dict, context_vars: dict) -> dict:
        """Process log logps."""
        run_tag = context_vars.get('run_tag', None)
        model_tag = context_vars.get('model_tag', None)
        task_tag = context_vars.get('task_tag', None)
        turn_tag = context_vars.get('turn_tag', None)
        completion_time_seconds = context_vars.get('__endpoint_time__', None)
        len_input_ids = len(result['input_ids'])
        len_logps = len(result['logps'])
        token_per_second = len_logps / completion_time_seconds if completion_time_seconds > 0 else 0.0
        logger.info(f"📈 [Composer] [{run_tag}] [{model_tag}] [{task_tag}] [{turn_tag}] [{len_input_ids} input_ids] [{len_logps} logps] in [{completion_time_seconds:.2f}s] [{token_per_second:.2f} tokens/s]")

    def log_code_extraction(self, result: dict, context_vars: dict) -> dict:
        """Process log code extraction."""
        run_tag = context_vars.get('run_tag', None)
        model_tag = context_vars.get('model_tag', None)
        task_tag = context_vars.get('task_tag', None)
        turn_tag = context_vars.get('turn_tag', None)
        generated_code = context_vars['generated_code']
        generated_reasoning = context_vars['generated_reasoning']
        generated_code_path = context_vars.get('generated_code_path', None)
        logger.info(f"👏 [Composer] [{run_tag}] [{model_tag}] [{task_tag}] [{turn_tag}] [{generated_code_path}] [{generated_code[:100]}]... [{generated_reasoning[:100]}]...")

    def log_kb_eval_ref(self, result: dict, context_vars: dict) -> dict:
        """Process log kb eval ref."""
        run_tag = context_vars.get('run_tag', None)
        model_tag = context_vars.get('model_tag', None)
        task_tag = context_vars.get('task_tag', None)
        reference_eval_runtime = result['runtime'] if 'runtime' in result else -1.0
        reference_eval_path = context_vars.get('reference_eval_path', None)
        evaluation_time = context_vars.get('__endpoint_time__', None)
        logger.info(f"📚 [Composer] [{run_tag}] [{model_tag}] [{task_tag}] [{reference_eval_path}] [{reference_eval_runtime:.3f}ms] in [{evaluation_time:.2f}s]")

    async def log_kb_eval(self, result: dict, context_vars: dict) -> dict:
        """Process log eval."""
        prefix_tag = context_vars.get('prefix_tag', None)
        run_tag = context_vars.get('run_tag', None)
        model_tag = context_vars.get('model_tag', None)
        task_tag = context_vars.get('task_tag', None)
        turn_tag = context_vars.get('turn_tag', None)
        generated_eval_runtime = result['runtime'] if 'runtime' in result else -1.0
        generated_eval_path = context_vars.get('generated_eval_path', None)
        reference_eval = await self.statsClient.wait_for_stats(
            prefix_tag=prefix_tag,
            model_tag=model_tag,
            task_tag=task_tag,
            category="reference",
        )
        reference_eval_runtime = reference_eval['runtime'] if 'runtime' in reference_eval else -1.0
        if generated_eval_runtime > 0 and reference_eval_runtime > 0:
            speedup = reference_eval_runtime / generated_eval_runtime
        else:
            speedup = 0.0
        # fun emoji for speedup
        speedup_emoji = '🚀' if speedup > 1.0 else ('🦙' if speedup > 0.5 else '🐢')
        evaluation_time = context_vars.get('__endpoint_time__', None)
        if result['compiled'] and result['correctness']:
            logger.info(f"✅ [Composer] [{run_tag}] [{model_tag}] [{task_tag}] [{turn_tag}] [{generated_eval_path}] [{generated_eval_runtime:.3f}ms] [{speedup_emoji} {speedup:.2f}x] in [{evaluation_time:.1f}s]")
        else:
            logger.warning(f"⚠️ [Composer] [{run_tag}] [{model_tag}] [{task_tag}] [{turn_tag}] [{generated_eval_path}] [{'🟢' if result['compiled'] else '🔴'} compiled], [{'🟢' if result['correctness'] else '🔴'} correctness] in [{evaluation_time:.2f}s]")

    async def log_enqueue(self, result: dict, context_vars: dict) -> dict:
        """Process log enqueue."""
        run_tag = context_vars.get('run_tag', None)
        model_tag = context_vars.get('model_tag', None)
        task_tag = context_vars.get('task_tag', None)
        gen_tag = context_vars.get('gen_tag', None)
        logger.info(f"🔍 [Composer] [{run_tag}] Enqueued [{model_tag}] [{task_tag}] [{gen_tag}]...")

    async def input_processor(self, block: InferenceBlock):
        """Process input variables."""
        for processor_name, processor_config in self.module_config.get('input_processor', {}).items():
            try:
                context_vars = self.context_vars | {
                    "block": block,
                } | block.model_dump()

                # run the input processor
                success = await self.configInterpreter.execute(
                    runtime=self,
                    config=processor_config,
                    context_vars=context_vars,
                )
                if not success:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Input processor [{processor_name}] failed.")
                    continue

                logger.info(f"👌 [Composer] [{self.input_tag}] Input processor [{processor_name}] completed.")

            except Exception as e:
                logger.error(f"🔴 [Composer] [{self.input_tag}] Input processor [{processor_name}] error: [{type(e)}: {e}]")
                logger.error(traceback.format_exc())
                continue

    async def queue_worker(self, worker_id: int):
        """Run queue worker."""
        while True:
            try:
                # Get file path from queue (blocking with timeout)
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                if item is None:
                    logger.info(f"🔍 [Composer] [{self.input_tag}] [Worker {worker_id:02d}] Received termination signal")
                    break

                if 'worker' not in item:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Worker not configured in item: {item}")
                    continue

                worker_name = item['worker']
                worker_config = self.module_config.get('queue_worker', {}).get(worker_name, {})
                if not worker_config:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Worker config not found in module [{self.module_file}] for worker [{worker_name}].")
                    continue

                # start processing the worker config, create a new context_vars dictionary
                context_vars = self.context_vars | {
                    "__input__": item,
                } | item # add the input variables to the context variables

                success = await self.configInterpreter.execute(
                    runtime=self,
                    config=worker_config,
                    context_vars=context_vars,
                )
                if not success:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Worker [{worker_name}] failed.")
                    continue

                logger.info(f"👌 [Composer] [{self.input_tag}] [worker {worker_id:02d}] [{worker_name}] completed.")

            except asyncio.TimeoutError:
                continue

            except Exception as e:
                logger.error(f"🔴 [Composer] [{self.input_tag}] Worker [{worker_id:02d}] error: [{type(e)}: {e}]")
                logger.error(traceback.format_exc())
                break

            finally:
                await asyncio.sleep(1)

        # circle emoji to beginning of the line
        logger.info(f"🎯 [Composer] [{self.input_tag}] Worker [{worker_id:02d}] completed. Remaining tasks: [{len(asyncio.all_tasks())}]")


async def run_inference_block(block: InferenceBlock, use_global_registry: bool = False):
    """
    Run one batch of code generation and evaluation.
    """
    try:
        config = load_inference_client_config(
            provider_name=block.vllm_providers[0],
            model_short_name=block.model_name,
        )

        # override the model name
        # if block.model_override:
        #     config.model.model_name = block.model_override

        # start running the block
        logger.info(f"🔍 [Composer] [{block.input_tag}] Starting block...")

        composerClient = ComposerClient(
            input_tag=block.input_tag,
            inference_client_config=config,
            custom_provider=block.custom_provider,
            model_override=block.model_override,
            module_file=block.module_file,
            prompt_file=block.prompt_file,
            example_file=block.example_file,
            output_dir=block.output_dir,
        )

        # start the input processor
        await composerClient.input_processor(block)

        # start the queue workers
        queue_workers = []
        for i in range(block.parallel_workers):
            queue_workers.append(asyncio.create_task(composerClient.queue_worker(i)))
            await composerClient.queue.put(None) # add exit signals

        # wait for the queue workers to complete
        await asyncio.gather(*queue_workers)

        if use_global_registry:
            globalWorkflow = WorkflowServer(prefix_tag=block.prefix_tag)
            await globalWorkflow.post_composer(block)

        logger.info(f"🎉 [Composer] [{block.input_tag}] Block completed")

    except Exception as e:
        logger.error(f"❌ [Composer] [{block.input_tag}] Error running block: [{e}] in [{traceback.format_exc()}]")
        logger.error(traceback.format_exc())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="codeGenEval.base") # "TC_0.1.0_14B.m"
    parser.add_argument("--prefix_tag", type=str, default="auto.inference.composer") # "TC_0.1.0_14B.m"
    parser.add_argument("--epoch_id", type=int, default=-1)
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_32B.b_006_05")
    parser.add_argument("--input_dir", type=str, default="~/KernelBench/KernelBench/level1", help="Input directory containing Python files")
    parser.add_argument("--output_dir", type=str, default="~/.inference/output", help="Output directory for the composer results")
    parser.add_argument("--parallel_workers", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--num_turns_per_generation", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="qwen3-32b")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--vllm_providers", type=str, default="local")
    parser.add_argument("--logp_providers", type=str, default="local")
    parser.add_argument("--kbeval_providers", type=str, default="local")
    parser.add_argument("--use_global_queue", type=str, default=None) # this is the task_name of the global queue
    parser.add_argument("--proc_id", type=str, default=None)
    parser.add_argument("--module_file", type=str, default="inference/codeGenEval.module.vllm+logp.yaml")
    parser.add_argument("--prompt_file", type=str, default="inference/triton.prompt.yaml")
    parser.add_argument("--example_file", type=str, default="inference/triton.example.yaml")
    parser.add_argument("--context", type=str, default="{}")
    args = parser.parse_args()

    if args.proc_id is not None:
        PROC_ID = args.proc_id
    else:
        PROC_ID = os.environ.get("PROC_ID", None)

    QUEUE_TYPE = "inference"

    try:
        if args.use_global_queue:
            if args.prefix_tag == 'auto':
                logger.error(f"❌ [Composer] --prefix_tag is required")
                return

            prefix_tag = args.prefix_tag
            QUEUE_NAME = args.use_global_queue # user configurable name
            queue_name = f"{QUEUE_TYPE}.{QUEUE_NAME}"
            # get the global registry
            global_reg_client = WorkflowClient(prefix_tag=prefix_tag)
            # get the critiqueBlock from the global registry
            block_json = await global_reg_client.dequeue(queue_name)
            # convert the block_json to a ComposerBlock object
            block = InferenceBlock(**block_json)
            # process the model override
            if "codeGen" in QUEUE_NAME: # a hack for now. TODO: fix this
                if PROC_ID is None:
                    error_msg = f"❌ [Composer] [{block.prefix_tag}] Unable to get PROC_ID to update model_override"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                model_override = await global_reg_client.get(f"{MODEL_OVERRIDE_KEY}")
                if model_override:
                    logger.info(f"🔍 [Composer] [{block.prefix_tag}] Using model override: [{model_override}]")
                    block.model_override = model_override
                    # update model_override in the global registry
                    await global_reg_client.put(f"adapter.{queue_name}.model_override.{PROC_ID}", model_override)
        else:
            block = InferenceBlock(
                name=args.name,
                prefix_tag=args.prefix_tag,
                epoch_id=args.epoch_id,
                block_id=args.block_id,
                input_tag=args.input_tag,
                num_samples=args.num_samples,
                num_generations=args.num_generations,
                num_turns_per_generation=args.num_turns_per_generation,
                parallel_workers=args.parallel_workers,
                model_name=args.model_name,
                vllm_providers=args.vllm_providers.split(","),
                logp_providers=args.logp_providers.split(","),
                kbeval_providers=args.kbeval_providers.split(","),
                module_file=args.module_file,
                prompt_file=args.prompt_file,
                example_file=args.example_file,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                context=json.loads(args.context),
            )
        # run the block
        await run_inference_block(block)

        if args.use_global_queue:
            global_reg_client = WorkflowClient(prefix_tag=block.prefix_tag)
            await global_reg_client.post_block(QUEUE_TYPE, QUEUE_NAME, block)

    except Exception as e:
        logger.error(f"❌ [Composer] Error running block: [{type(e).__name__}: {e}]")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
