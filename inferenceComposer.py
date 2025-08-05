
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
from logger import logger
from kbEvalClient import KbEvalClient
from globalUtils import ComposerBlock, MODEL_OVERRIDE_KEY
from globalRegClient import GlobalRegClient
from globalWorkflow import GlobalWorkflow
from endpointUtil import Recorder, CodeExtractor, StatsClient

VALID_INPUT_PROCESSORS = [
    "duckdb",
]

class ComposerClient:
    def __init__(
            self,
            input_tag: str,
            inference_client_config: InferenceClientConfig,
            module_file: str = "inferenceComposer/codeGen.module.yaml",
            prompt_file: str = "inferenceComposer/codeGen.prompt.triton.yaml",
            example_file: str = "inferenceComposer/triton.example.yaml",
            output_dir: str = "~/.inference/composer",
            stats_dir: str = "~/.trainer/stats",
    ):
        with open(prompt_file, 'r') as f:
            self.prompt_config = yaml.safe_load(f)
        self.input_tag = input_tag
        self.logger = logger
        self.queue = asyncio.Queue()
        self.recorder = Recorder()
        self.kbEvalClient = KbEvalClient()
        self.statsClient = StatsClient()
        self.codeExtractor = CodeExtractor()
        self.inference_client_config = inference_client_config
        self.inferenceClient = InferenceClient(config=self.inference_client_config)
        self.tokenizer = self.inferenceClient.tokenizer
        self.model_tag = self.inferenceClient.model_tag
        # self.output_dir = self._get_output_dir(output_dir)
        # self.output_dir.mkdir(parents=True, exist_ok=True)
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
            "__start_time__": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        if 'context_vars' in self.module_config:
            self.context_vars = self._process_context_vars(self.module_config['context_vars'], self.context_vars)

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

    def _process_variable(self, value: Any, context_vars: dict) -> Any:
        """Process a variable."""
        if isinstance(value, str):
            stripped_value = value.strip()
            if stripped_value.startswith('`') and stripped_value.endswith('`'):
                try:
                    # if the value is a python expression, evaluate it via python eval
                    return eval(stripped_value[1:-1], context_vars)
                except Exception as e:
                    logger.error(f"❌ [Composer] [{self.input_tag}] Error evaluating variable: [{type(e)}: {e}]\n{stripped_value} ")
                    raise e
            else:
                # if the value is a string, format it with the format
                return value.format(**context_vars)
        else:
            return value

    def _process_context_vars(self, context_config: dict, context_vars: dict, local_context_vars: dict = None) -> dict:
        """Process context variables."""
        for key, value in context_config.items():
            # recursively process the context variables
            if isinstance(value, dict):
                if local_context_vars is None:
                    context_vars[key] = self._process_context_vars(value, context_vars, local_context_vars={})
                else:
                    local_context_vars[key] = self._process_context_vars(value, context_vars, local_context_vars={})
            else:
                if local_context_vars is None:
                    context_vars[key] = self._process_variable(value, context_vars)
                else:
                    local_context_vars[key] = self._process_variable(value, context_vars)
        if local_context_vars is None:
            context_vars['__context__'] = context_vars
            return context_vars
        else:
            return local_context_vars

    def _process_input_vars(self, input_config: dict, context_vars: dict) -> dict:
        """Process input variables."""
        result = {}
        for key, value in input_config.items():
            result[key] = self._process_variable(value, context_vars)
        return result

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

    async def input_processor(self, block: ComposerBlock):
        """Process input variables."""
        for processor_config in self.module_config.get('input_processor', []):
            try:
                # get processor type
                processor_type = processor_config.get('processor', None)
                if not processor_type:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Input processor type not found in module [{self.module_file}] for processor [{processor_config}].")
                    continue
                # get processor name
                processor_name = processor_config.get('name', None)
                if not processor_name:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Input processor name not found in module [{self.module_file}] for processor [{processor_config}].")
                    continue

                if processor_type not in VALID_INPUT_PROCESSORS:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Invalid input processor type: [{processor_type}] in module [{self.module_file}] for processor [{processor_config}].")
                    continue

                logger.info(f"🔍 [Composer] [{self.input_tag}] Running input processor [{processor_type}] [{processor_name}]...")

                try:
                    # now we have a valid processor, process the context variables
                    context_vars = self.context_vars | {
                        "block": block,
                    }
                    context_vars['__context__'] = context_vars
                    if 'context_vars' in processor_config:
                        context_vars = self._process_context_vars(processor_config['context_vars'], context_vars)
                    
                    if 'query' not in processor_config:
                        logger.error(f"🔴 [Composer] [{self.input_tag}] Query not found in input processor [{processor_config}].")
                        continue

                    # now we have a valid processor, process the input variables
                    query = self._process_variable(processor_config['query'], context_vars)
                    if not query:
                        logger.error(f"🔴 [Composer] [{self.input_tag}] Query is empty in input processor [{processor_config}].")
                        continue

                except Exception as e:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Error running input processor [{processor_type}] [{processor_name}]: [{type(e)}: {e}]")
                    continue

                # now run the query
                try:
                    result = duckdb.sql(query)
                except Exception as e:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Error running query: [{query}] [{type(e)}: {e}]")
                    logger.error(traceback.format_exc())
                    continue

                # now we have a valid result, process the result
                result_df = result.df()
                if len(result_df) == 0:
                    logger.warning(f"🗑️ [Composer] [{self.input_tag}] Query returned no results: [{query}]")
                    continue

                logger.info(f"🔍 [Composer] [{self.input_tag}] Query returned [{len(result_df)}] results:\n[{result_df}]")

                enqueue_count = 0
                for idx, row in result_df.iterrows():
                    if "enqueue" in processor_config:
                        enqueue_config = processor_config.get('enqueue', {})

                        try:
                            # now we have a valid result, process the result
                            row_context_vars = context_vars | {
                                "row": row,
                            }
                            row_context_vars['__context__'] = row_context_vars
                            if 'context_vars' in enqueue_config:
                                enqueue_context_vars = self._process_context_vars(enqueue_config['context_vars'], row_context_vars)
                            else:
                                enqueue_context_vars = row_context_vars

                        except Exception as e:
                            logger.error(f"🔴 [Composer] [{self.input_tag}] Error preparing enqueue: [{enqueue_config}] [{type(e)}: {e}]")
                            continue

                        if "items" not in enqueue_config:
                            logger.error(f"🔴 [Composer] [{self.input_tag}] Item not found in enqueue config: {enqueue_config}")
                            continue

                        for item_config in enqueue_config.get('items', []):
                            # process each enqueue item
                            if "__repeat_count__" in item_config:
                                __repeat_count__ = self._process_variable(item_config['__repeat_count__'], enqueue_context_vars)
                            else:
                                __repeat_count__ = 1

                            for __repeat_idx__ in range(__repeat_count__):
                                enqueue_context_vars['__repeat_idx__'] = __repeat_idx__

                                try:                                
                                    item_vars = self._process_input_vars(item_config, enqueue_context_vars)
                                    await self.queue.put(item_vars)
                                except Exception as e:
                                    logger.error(f"🔴 [Composer] [{self.input_tag}] Error enqueueing item: [{item_config}] [{type(e)}: {e}]")
                                    continue

                                enqueue_count += 1

                logger.info(f"👌 [Composer] [{self.input_tag}] Input processor [{processor_type}] [{processor_name}] completed. Enqueued [{enqueue_count}] item(s).")

            except Exception as e:
                logger.error(f"🔴 [Composer] [{self.input_tag}] Input processor [{processor_type}] [{processor_name}] error: [{type(e)}: {e}]")
                logger.error(traceback.format_exc())
                continue

    def _process_log_eval(self, result: Any, context_vars: dict) -> dict:
        """Process log eval."""
        if 'logprobs' in result:
            context_vars['logprobs'] = result['logprobs']
        if 'usage' in result:
            context_vars['usage'] = result['usage']
        return context_vars

    
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
                    "input": item,
                } | item # add the input variables to the context variables
                context_vars['__context__'] = context_vars
                try:
                    # process additional context variables
                    if 'context_vars' in worker_config:
                        context_config = worker_config['context_vars']
                        context_vars = self._process_context_vars(context_config, context_vars)
                except Exception as e:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Error processing context variables: {worker_config['context_vars']} [{type(e)}: {e}]")
                    continue

                try:
                    if 'filter' in worker_config:
                        filter = self._process_variable(worker_config['filter'], context_vars)
                        if not filter:
                            logger.info(f"🔍 [Composer] [{self.input_tag}] Filter [{filter}] is false, skipping...")
                            continue
                except Exception as e:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Error processing filter: {worker_config['filter']} [{type(e)}: {e}]")
                    continue

                logger.info(f"🔍 [Composer] [{self.input_tag}] Running [worker {worker_id:02d}] [{worker_name}]...")

                try:
                    if '__loop_count__' in worker_config:
                        __loop_count__ = self._process_variable(worker_config['__loop_count__'], context_vars)
                    else:
                        __loop_count__ = 1
                except Exception as e:
                    logger.error(f"🔴 [Composer] [{self.input_tag}] Error processing loop count: {worker_config['__loop_count__']} [{type(e)}: {e}]")
                    continue

                step_context_vars = context_vars.copy() # create a copy of the context variables for the entire loop
                step_context_vars['__context__'] = step_context_vars
                for __loop_idx__ in range(__loop_count__):
                    # add context variables for the loop
                    step_context_vars['__loop_idx__'] = __loop_idx__
                    step_context_vars['__loop_count__'] = __loop_count__

                    # work through the steps
                    error_encountered = False
                    for step_config in worker_config['steps']:
                        try:
                            if 'endpoint' not in step_config:
                                logger.error(f"🔴 [Composer] [{self.input_tag}] Endpoint not found in step: {step_config}")
                                error_encountered = True
                                break

                            # get the endpoint instance and function
                            endpoint = step_config['endpoint'].split('.')
                            endpoint_class_name = endpoint[0]
                            endpoint_method_name = endpoint[1]
                            # get self.{endpoint_class}
                            endpoint_instance = getattr(self, endpoint_class_name)
                            # get self.{endpoint_class}.{endpoint_method}
                            endpoint_function = getattr(endpoint_instance, endpoint_method_name)

                        except Exception as e:
                            # assume each step depend on each other, always break the steps if current step fails
                            logger.error(f"🔴 [Composer] [{self.input_tag}] Error getting endpoint: {step_config} [{type(e)}: {e}]")
                            logger.error(traceback.format_exc())
                            error_encountered = True
                            break

                        try:
                            if 'context_vars' in step_config:
                                step_context_vars = self._process_context_vars(step_config['context_vars'], step_context_vars)

                        except Exception as e:
                            # assume each step depend on each other, always break the steps if current step fails
                            logger.error(f"🔴 [Composer] [{self.input_tag}] Error processing step context variables: {step_config['context_vars']} [{type(e)}: {e}]")
                            error_encountered = True
                            break

                        try:
                            # get the inputs
                            input_config = step_config.get('inputs', {})
                            input_vars = self._process_input_vars(input_config, step_context_vars)

                        except Exception as e:
                            # assume each step depend on each other, always break the steps if current step fails
                            logger.error(f"🔴 [Composer] [{self.input_tag}] Error processing inputs: {step_config} [{type(e)}: {e}]")
                            error_encountered = True
                            break

                        try:
                            # call the endpoint function
                            record_time = True if 'returns' in step_config else False
                            if record_time:
                                start_time = time.time()
                            # check if the endpoint function is async
                            if asyncio.iscoroutinefunction(endpoint_function):
                                result = await endpoint_function(**input_vars)
                            else:
                                result = endpoint_function(**input_vars)
                            if record_time:
                                end_time = time.time()
                                step_context_vars['__endpoint_time__'] = end_time - start_time

                        except Exception as e:
                            # assume each step depend on each other, always break the steps if current step fails
                            logger.error(f"🔴 [Composer] [{self.input_tag}] Error calling endpoint: {step_config} [{type(e)}: {e}]")
                            # log stack track only when actually calling the endpoint
                            logger.error(traceback.format_exc())
                            error_encountered = True
                            break

                        try:
                            if 'returns' in step_config:
                                returns_config = step_config['returns']
                                step_context_vars['__result__'] = result
                                # process error_if
                                if 'error_if' in returns_config:
                                    error_if = self._process_variable(returns_config['error_if'], step_context_vars)
                                    if error_if:
                                        logger.error(f"🔴 [Composer] [{self.input_tag}] Error in endpoint [{endpoint_class_name}.{endpoint_method_name}]: {result}")
                                        error_encountered = True
                                        break
                                # now we don't have any errors, process the return in context variables
                                if 'context_vars' in returns_config:
                                    step_context_vars = self._process_context_vars(returns_config['context_vars'], step_context_vars)
                                # process save_to
                                if 'save_to' in returns_config:
                                    for save_to_config in returns_config['save_to']:
                                        path = self._process_variable(save_to_config['path'], step_context_vars)
                                        data = self._process_variable(save_to_config['data'], step_context_vars)
                                        format = save_to_config['format'] if 'format' in save_to_config else 'text'
                                        self.recorder.save(path, data, format=format)
                                # process logging
                                if 'logging' in returns_config:
                                    log_config = returns_config['logging']
                                    try:
                                        log_method = getattr(self, log_config.get('method', None))
                                        if not log_method:
                                            logger.warning(f"🔴 [Composer] [{self.input_tag}] Logging method not found: {log_config}")
                                        # check if log_method is async
                                        if asyncio.iscoroutinefunction(log_method):
                                            await log_method(result, step_context_vars)
                                        else:
                                            log_method(result, step_context_vars)
                                    except Exception as e:
                                        logger.warning(f"🔴 [Composer] [{self.input_tag}] Error processing logging: {log_config} [{type(e)}: {e}]")

                        except Exception as e:
                            # assume each step depend on each other, always break the steps if current step fails
                            logger.error(f"🔴 [Composer] [{self.input_tag}] Error processing returns: [step_config={step_config}] [{type(e)}: {e}]")
                            logger.error(traceback.format_exc())
                            error_encountered = True
                            break

                    if error_encountered:
                        logger.error(f"🔴 [Composer] [{self.input_tag}] Error encountered in step: {step_config}")
                        break

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


async def composer_block(block: ComposerBlock, use_global_registry: bool = False):
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
        logger.info(f"🔍 [Composer] [{block.input_tag}] Starting block...")

        composerClient = ComposerClient(
            input_tag=block.input_tag,
            inference_client_config=config,
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
            globalWorkflow = GlobalWorkflow(prefix_tag=block.prefix_tag)
            await globalWorkflow.post_composer(block)

        logger.info(f"🎉 [Composer] [{block.input_tag}] Block completed")

    except Exception as e:
        logger.error(f"❌ [Composer] [{block.input_tag}] Error running block: [{e}] in [{traceback.format_exc()}]")
        logger.error(traceback.format_exc())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="TC_0.1.0_14B.h")
    parser.add_argument("--epoch_id", type=int, default=-1)
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_14B.h_001_01")
    parser.add_argument("--input_dir", type=str, default="~/.codeGenEval", help="Input directory containing Python files")
    parser.add_argument("--output_dir", type=str, default="~/.inference/composer", help="Output directory for the composer results")
    parser.add_argument("--provider", type=str, default="fireworks")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--model", type=str, default="deepseek-v3")  # most cost effective models are deepinfra-r1 and fireworks-v3
    parser.add_argument("--model_override", type=str, default=None)
    parser.add_argument("--parallel_workers", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--num_turns_per_generation", type=int, default=4)
    parser.add_argument("--use_global_queue", type=str, default=None) # this is the task_name of the global queue
    parser.add_argument("--proc_id", type=str, default=None)
    parser.add_argument("--module_file", type=str, default="inferenceComposer/exemplar.module.yaml")
    parser.add_argument("--prompt_file", type=str, default="inferenceComposer/codeGen.prompt.triton.yaml")
    parser.add_argument("--example_file", type=str, default="inferenceComposer/triton.example.yaml")
    args = parser.parse_args()

    if args.proc_id is not None:
        PROC_ID = args.proc_id
    else:
        PROC_ID = os.environ.get("PROC_ID", None)

    try:
        if args.use_global_queue:
            task_type = "inference.composer" # hard code for inferenceComposer
            task_name = args.use_global_queue # user configurable name
            QUEUE_NAME = f"{task_type}:{task_name}"
            # get the global registry
            global_reg_client = GlobalRegClient()
            # get the critiqueBlock from the global registry
            block_json = await global_reg_client.dequeue(QUEUE_NAME)
            # convert the block_json to a ComposerBlock object
            block = ComposerBlock(**block_json)
            # process the model override
            if task_name.startswith("codeGen"): # a hack for now. TODO: fix this
                model_override = await global_reg_client.get(f"{MODEL_OVERRIDE_KEY}")
                if model_override:
                    logger.info(f"🔍 [Composer] [{block.prefix_tag}] Using model override: [{model_override}]")
                    block.model_override = model_override
                    # update model_override in the global registry
                    if PROC_ID is None:
                        error_msg = f"❌ [Composer] [{block.prefix_tag}] Unable to get PROC_ID to update model_override [{model_override}]"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    else:
                        await global_reg_client.put(f"adapter.{QUEUE_NAME}.model_override.{PROC_ID}", model_override)
        else:
            block = ComposerBlock(
                prefix_tag=args.prefix_tag,
                epoch_id=args.epoch_id,
                block_id=args.block_id,
                input_tag=args.input_tag,
                provider_name=args.provider,
                model_name=args.model,
                module_file=args.module_file,
                prompt_file=args.prompt_file,
                example_file=args.example_file,
                num_samples=args.num_samples,
                num_generations=args.num_generations,
                num_turns_per_generation=args.num_turns_per_generation,
                parallel_workers=args.parallel_workers,
                model_override=args.model_override,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
            )
        # run the block
        await composer_block(block)

        if args.use_global_queue:
            globalWorkflow = GlobalWorkflow(prefix_tag=block.prefix_tag)
            await globalWorkflow.post_composer(args.use_global_queue, block)

    except Exception as e:
        logger.error(f"❌ [Composer] [{block.input_tag}] Error running block: [{e}]")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
