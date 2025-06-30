import re
import json
import os
import traceback
import asyncio
import torch
import httpx
import yaml
from pathlib import Path
from typing import Dict
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from transformers import TrainerCallback
from datetime import datetime
from logger import logger
import wandb

EVAL_TIMEOUT = 300 # seconds

max_prompt_length = 1024
max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

batch_size = 8
accumulation_steps = 1

num_generations = 8

model_name = "Qwen/Qwen3-4B"
# model_name = "Qwen/Qwen3-8B"
# model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"

model_tag = model_name
time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

reference_eval_cache = {}

time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

def _load_config(config_file: str = "grpo_unsloth_cuda.yaml") -> Dict:
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

config = _load_config()

def get_system_prompt() -> str:
    """Get system prompt from configuration."""
    prompts_config = config.get('prompts', {})
    reference_code = config.get('reference_code', '')
    generated_code = config.get('generated_code', '')
    return prompts_config.get('system_prompt', 'You are a helpful assistant.').format(reference_code=reference_code, generated_code=generated_code)

def get_user_prompt(source_code: str) -> str:
    """Get user prompt from configuration with source code substituted."""
    prompts_config = config.get('prompts', {})
    user_prompt_template = prompts_config.get('user_prompt', 'Analyze this code: {source_code}')
    return user_prompt_template.format(source_code=source_code)


# uncomment middle messages for 1-shot prompting
def get_kernel_bench_reference_code(split = "train") -> Dataset:
    data = load_dataset('dtadpole/kernel-bench', 'default')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': get_system_prompt()},
            {'role': 'user', 'content': get_user_prompt(x['reference_code'])}
        ],
        'reference_codes': x['reference_code'],
        'task_tags': x['task_tag'],
        'level_ids': x['level_id'],
        'source_ids': x['source_id'],
    })
    return data

dataset = get_kernel_bench_reference_code()



def extract_xml_think(text: str) -> str:
    if "<think>" not in text or "</think>" not in text:
        return ""
    # extract after the first <think>
    think = text.split("<think>")[-1]
    # extract before the last </think>
    think = think.split("</think>")[:-1]
    return think.strip()

def extract_xml_code(text: str) -> str:
    if "</think>" not in text:
        return ""
    code = text.split("</think>")[-1] # extract after the last </think>
    if "<code>" not in code or "</code>" not in code:
        return ""
    code = code.split("<code>")[-1]
    code = code.split("</code>")[0]
    return code.strip()

def extract_xml_explanation(text: str) -> str:
    if "</think>" not in text:
        return ""
    # extract after the last </think>
    after_think = text.split("</think>")[-1]
    if "<explanation>" not in after_think or "</explanation>" not in after_think:
        return ""
    explanation = after_think.split("<explanation>")[-1]
    explanation = explanation.split("</explanation>")[0]
    return explanation.strip()

def get_kb_eval_configs(config_file="grpo_unsloth_cuda.yaml") -> tuple[str, str]: # base_url, api_key
    config = _load_config(config_file)
    kb_eval_config = config.get('kbEval', {})
    base_url = kb_eval_config.get('base_url', 'http://10.12.0.204:5678')
    api_key_filepath = kb_eval_config.get('api_key_filepath', os.path.expanduser('~/.keys/kbeval.api.key'))
    with open(api_key_filepath, 'r') as f:
        api_key = f.read().strip()
    return base_url, api_key

async def kb_eval_reference_code(model_tag: str, task_tag: str, time_tag: str, reference_code: str) -> dict:
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        retry_count += 1
        try:
            base_url, api_key = get_kb_eval_configs()
            async with httpx.AsyncClient(timeout=EVAL_TIMEOUT) as client:
                response = await client.post(
                    f"{base_url}/kb_eval_ref",
                    json={
                        "model_tag": model_tag,
                        "task_tag": task_tag,
                        "time_tag": time_tag,
                        "reference_code": reference_code,
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=EVAL_TIMEOUT,
                )
                return response.json()
        except Exception as e:
            await asyncio.sleep(1)
            if retry_count >= max_retries:
                logger.error(f"❌ [{model_tag}] [{task_tag}] [{time_tag}] Error evaluating reference code: {e}")
                traceback.print_exc()
                break
    return None

async def kb_eval_generated_code(model_tag: str, task_tag: str, time_tag: str, eval_tag: str, reference_code: str, generated_code: str) -> dict:
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        retry_count += 1
        try:
            base_url, api_key = get_kb_eval_configs()
            async with httpx.AsyncClient(timeout=EVAL_TIMEOUT) as client:
                response = await client.post(
                    f"{base_url}/kb_eval",
                    json={
                        "model_tag": model_tag,
                        "task_tag": task_tag,
                        "time_tag": time_tag,
                        "eval_tag": eval_tag,
                        "reference_code": reference_code,
                        "generated_code": generated_code,
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=EVAL_TIMEOUT,
                )
                return response.json()
        except Exception as e:
            await asyncio.sleep(1)
            if retry_count >= max_retries:
                logger.error(f"❌ [{model_tag}] [{task_tag}] [{time_tag}] [{eval_tag}] Error evaluating generated code: {e}")
                traceback.print_exc()
                raise e

async def _eval_reference_task(reference_eval_cache: dict, model_tag: str, task_tag: str, time_tag: str, reference_code: str):
    reference_eval_cache[task_tag] = await kb_eval_reference_code(model_tag, task_tag, time_tag, reference_code)

async def _eval_generated_task(result_dict: dict, model_tag: str, task_tag: str, time_tag: str, eval_tag: str, reference_code: str, generated_code: str):
    if not generated_code:
        result_dict[f'{task_tag}_{eval_tag}'] = {
            'compiled': False,
            'correctness': False,
            'runtime': -1.0,
        }
        return
    
    try:
        result_dict[f'{task_tag}_{eval_tag}'] = await kb_eval_generated_code(model_tag, task_tag, time_tag, eval_tag, reference_code, generated_code)
    except Exception as e:
        result_dict[f'{task_tag}_{eval_tag}'] = {
            'compiled': False,
            'correctness': False,
            'runtime': -1.0,
            'metadata': {'process_error': str(e)},
        }
    

# Reward functions
async def async_kb_eval_reward_func(prompts, completions, reference_codes, task_tags, **kwargs) -> list[float]:
    global reference_eval_cache

    logger.info(f"🔍 [{model_tag}] [{time_tag}] [prompts] [{[prompt[0]['content'][:100] for prompt in prompts[:2]]}]") # log first 2 rows of each prompt[0]['content'], and 100 characters each
    logger.info(f"🔍 [{model_tag}] [{time_tag}] [completions] [{[completion[0]['content'][:100] for completion in completions[:2]]}]") # log first 2 rows of each completion[0]['content'], and 100 characters each
    logger.info(f"🔍 [{model_tag}] [{time_tag}] [reference_codes] [{[ref_code[:100] for ref_code in reference_codes[:2]]}") # log first 2 rows of reference_codes, and 100 characters each
    logger.info(f"🔍 [{model_tag}] [{time_tag}] [task_tags] [{task_tags}]") # log all the task_tags (this is a short string, let's log them all)

    reference_eval_local_keys = {}
    reference_eval_tasks = []
    for task_tag, reference_code in zip(task_tags, reference_codes):
        if task_tag not in reference_eval_cache and task_tag not in reference_eval_local_keys:
            reference_eval_local_keys[task_tag] = True # this will remove duplicate reference eval tasks
            ref_eval_task = asyncio.create_task(_eval_reference_task(reference_eval_cache, model_tag, task_tag, time_tag, reference_code))
            reference_eval_tasks.append(ref_eval_task)
    # wait for all reference eval tasks to complete
    await asyncio.gather(*reference_eval_tasks)

    reference_evals = [reference_eval_cache[task_tag] for task_tag in task_tags]

    responses = [completion[0]['content'] for completion in completions]
    # thoughts = [extract_xml_think(r) for r in responses]
    generated_codes = [extract_xml_code(r) for r in responses]
    explanations = [extract_xml_explanation(r) for r in responses]

    # create a list of evaluation tasks
    eval_tasks = []
    result_dict = {}
    for eval_id, (task_tag, reference_code, generated_code, explanation) in enumerate(zip(task_tags, reference_codes, generated_codes, explanations)):
        eval_tag = f"r{eval_id+1:02d}"
        eval_tasks.append(asyncio.create_task(_eval_generated_task(result_dict, model_tag, task_tag, time_tag, eval_tag, reference_code, generated_code)))
    # invoke eval tasks
    await asyncio.gather(*eval_tasks)

    scores = []
    log_results = {}
    for eval_id, (task_tag, generated_code, explanation, reference_eval) in enumerate(zip(task_tags, generated_codes, explanations, reference_evals)):
        eval_tag = f"r{eval_id+1:02d}"
        score = 0.0
        speed_up = 0.0
        generated_eval = result_dict[f'{task_tag}_{eval_tag}']
        if generated_eval['compiled'] == True:
            score += 0.1
        else:
            score -= 1.0
        if generated_eval['correctness'] == True:
            score += 0.3
        else:
            score -= 0.5
        if generated_eval['runtime'] > 0.0:
            generated_runtime = generated_eval['runtime']
            reference_runtime = reference_eval['runtime']
            speed_up = reference_runtime / generated_runtime
            score += speed_up * 2.0
        else:
            score -= 1.0
        scores.append(score)
        log_results[f'{task_tag}_{eval_tag}'] = {
            'score': score,
            'compiled': generated_eval['compiled'],
            'correctness': generated_eval['correctness'],
            'runtime': generated_eval['runtime'],
            'ref_runtime': reference_eval['runtime'],
            'speed_up': speed_up,
        }

    # logger.info(f"🔍 [{model_tag}] [{task_tag}] [{time_tag}] eval results:")
    for key, value in log_results.items():
        emoji = '✅' if value['compiled'] and value['correctness'] else '❌'
        logger.info(f"{emoji} [{model_tag}] [{task_tag}] [{time_tag}] [{key.split('_')[-1]}]: {json.dumps(value)}")
    # return the scores
    return scores

# convert async reward function to sync
def kb_eval_reward_func(prompts, completions, reference_codes, task_tags, **kwargs) -> list[float]:
    return asyncio.run(async_kb_eval_reward_func(prompts, completions, reference_codes, task_tags, **kwargs))

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\n<code>.*?</code>\n<explanation>.*?</explanation>$"
    responses = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.3 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\n<code>.*?</code>\n<explanation>.*?</explanation>"
    responses = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.3 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.05
    if text.count("\n</think>\n") == 1:
        count += 0.05
    if text.count("<code>\n") == 1:
        count += 0.05
    if text.count("\n</code>\n") == 1:
        count += 0.05
    if text.count("\n<explanation>\n") == 1:
        count += 0.05
        count -= len(text.split("\n</explanation>\n")[-1])*0.001
    if text.count("\n</explanation>") == 1:
        count += 0.05
        count -= (len(text.split("\n</explanation>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 3e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.03,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = batch_size,
    gradient_accumulation_steps = accumulation_steps, # Increase to 4 for smoother training
    num_generations = num_generations, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 10,
    max_grad_norm = 0.3,
    loss_type="dr_grpo", # token-level loss
    epsilon=0.2,         # clip lower
    epsilon_high=0.28,   # clip higher
    delta = 1.8,         # two-sided confidence interval
    beta = 0.0,          # no kl-divergence
    report_to = "wandb", # Can use Weights & Biases
    output_dir = f"outputs_{time_tag}",
    run_name = f"{model_name}_{time_tag}",
)

class WandbChartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        # Define metrics and their step relationship
        if wandb.run:
            wandb.define_metric("global_step")
            wandb.define_metric("training/*", step_metric="global_step")
            wandb.define_metric("eval/*", step_metric="global_step")
            print("W&B charts configured")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and wandb.run:
            # Prepare metrics for charting
            chart_data = {"global_step": state.global_step}
            
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if key.startswith('train'):
                        update_key = key[len('train/'):]
                        chart_data[f"training/{update_key}"] = float(value)
                    elif key.startswith('eval'):
                        update_key = key[len('eval/'):]
                        chart_data[f"eval/{update_key}"] = float(value)
                    elif key.startswith('rewards/'):
                        # remove the first # of characters that are 'rewards/'
                        update_key = key[len('rewards/'):]
                        chart_data[f"rewards/{update_key}"] = float(value)
                    elif key.startswith('completions/'):
                        # remove the first # of characters that are 'completions/'
                        update_key = key[len('completions/'):]
                        chart_data[f"completions/{update_key}"] = float(value)
                    else:
                        chart_data[f"metrics/{key}"] = float(value)
            
            wandb.log(chart_data)
            print(f"Step {state.global_step}: Logged {len(chart_data)} metrics")

wandb.init(
    project="grpo-kb",
    name=f"kb-{time_tag}",
    tags=["grpo"],
    config={
        "model_name": model_name,
        # Add other hyperparameters
    } | training_args.to_dict()
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        kb_eval_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
    callbacks=[WandbChartCallback()]
)
trainer.train()
