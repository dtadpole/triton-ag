import copy
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import threading
import subprocess
import os
import sys
import json
import numpy as np
from pathlib import Path
import gc
from tqdm import tqdm
from pydantic import BaseModel
import hashlib
import multiprocessing as mp
import shutil
import time
import torch
import asyncio
from logger import logger
from kbEvalCli import eval_kernel_custom
from kbEvalUtil import KernelExecResult
from typing import DefaultDict, Dict, List, Optional, Any, Tuple, Callable
import argparse
import yaml
import re
from unsloth_parallel import ThreadSafeModelPool, load_pickle


GRPO_FOLDER = "grpo_artifacts/"
# Set multiprocessing start method (important for CUDA)
mp.set_start_method('spawn', force=True)


###### Copied from trainerBase to avoid direct import them and cause unsloth error
class TrainerStatus(BaseModel):
    """Running status of the trainer"""
    global_step: int = 0

class ModelConfig(BaseModel):
    """Configuration for model parameters"""
    name: str = 'gpt2'
    tokenizer_name: Optional[str] = None
    max_seq_length: int = 16384
    use_unsloth: bool = True
    use_gradient_checkpointing: str = "unsloth"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    compute_dtype: str = "bfloat16"

class OptimizerConfig(BaseModel):
    """Configuration for optimizer parameters"""
    optimizer_type: str = "AdamW"  # AdamW, Adam, SGD
    betas: tuple = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD
    nesterov: bool = False  # For SGD

class TrainingConfig(BaseModel):
    """Configuration for training parameters"""
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.000005
    block_size: int = 32
    max_steps: int = 100000
    save_steps: int = 20
    eval_steps: int = 20
    logging_steps: int = 1
    checkpoint_path: str = "~/.trainer"
    latest_checkpoint_name: Optional[str] = "checkpoint-latest"
    max_grad_norm: float = 0.1
    scheduler_type: str = "cosine"
    num_warmup_steps: int = 50
    dataloader_num_workers: int = 4
    loss_multiplier: float = 1.0
    seed: int = -1

class TrainerLoraConfig(BaseModel):
    """Configuration for LoRA parameters"""
    use_lora: bool = True
    rank: int = 64
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    bias: str = "none"

class LoggingConfig(BaseModel):
    """Configuration for logging parameters"""
    use_wandb: bool = True
    wandb_project: str = "kb_trainer"
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None

class TrainerConfig(BaseModel):
    """Main configuration class containing all training parameters"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    lora: TrainerLoraConfig = TrainerLoraConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str, override_yaml_path: Optional[str] = None) -> 'TrainerConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if override_yaml_path is not None:
            with open(override_yaml_path, 'r') as f:
                override_config_dict = yaml.safe_load(f)
            # do a recursive merge of the two dictionaries
            config_dict = merge_dicts(config_dict, override_config_dict)

        # Create config objects from sections
        model_config = ModelConfig()
        training_config = TrainingConfig()
        optimizer_config = OptimizerConfig()
        lora_config = TrainerLoraConfig()
        logging_config = LoggingConfig()

        # Update from YAML sections
        if 'model' in config_dict:
            model_data = config_dict['model']
            model_config = ModelConfig(
                name=model_data.get('name', 'gpt2'),
                tokenizer_name=model_data.get('tokenizer_name'),
                max_seq_length=model_data.get('max_seq_length', 1024),
                use_unsloth=model_data.get('use_unsloth', True),
                use_gradient_checkpointing=model_data.get('use_gradient_checkpointing', "unsloth"),
                load_in_4bit=model_data.get('load_in_4bit', False),
                load_in_8bit=model_data.get('load_in_8bit', False),
                compute_dtype=model_data.get('compute_dtype', 'bfloat16')
            )

        if 'training' in config_dict:
            training_data = config_dict['training']
            training_config = TrainingConfig(**training_data)

        if 'optimizer' in config_dict:
            optimizer_data = config_dict['optimizer']
            # Convert betas list to tuple if present
            if 'betas' in optimizer_data and isinstance(optimizer_data['betas'], list):
                optimizer_data['betas'] = tuple(optimizer_data['betas'])
            optimizer_config = OptimizerConfig(**optimizer_data)

        if 'lora' in config_dict:
            lora_data = config_dict['lora']
            lora_config = TrainerLoraConfig(
                use_lora=lora_data.get('use_lora', True),
                rank=lora_data.get('rank', 64),
                alpha=lora_data.get('alpha', 16),
                dropout=lora_data.get('dropout', 0.0),
                target_modules=lora_data.get('target_modules'),
                bias=lora_data.get('bias', 'none')
            )

        if 'logging' in config_dict:
            logging_data = config_dict['logging']
            logging_config = LoggingConfig(**logging_data)

        return cls(
            model=model_config,
            training=training_config,
            optimizer=optimizer_config,
            lora=lora_config,
            logging=logging_config
        )
###############################

class RewardConfig(BaseModel):
    """Result of a single generation"""
    speed_up_threshold: float = 1.3
    compile_reward: float = 0.1
    compile_penalty: float = -0.5
    correct_reward: float = 0.1
    correct_penalty: float = -0.5
    speed_up_reward: float = 0.5
    speed_up_penalty: float = -0.1

    @classmethod
    def from_yaml(cls, file_path: str) -> "RewardConfig":
        """Load GRPO configuration from YAML file"""
        with open(os.path.expanduser(file_path), 'r') as f:
            config = yaml.safe_load(f)
        reward_config = config.get('rewards', {})
        return cls(**reward_config)


class PromptManager:
    """
    A class to manage prompt templates for code writer LLMs.
    Handles system prompt generation with examples and user prompt generation.
    """

    def __init__(self, config_file: str):
        """
        Initialize the PromptManager with templates.

        Args:
            system_template: Template string for system prompt with placeholders
            user_template: Template string for user prompt with placeholders
        """
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.prompts = self.config.get('prompts', {})
        self.reference_generated_pairs = []
        current_reference_code = self.config.get('prompts', {}).get('examples', {}).get('reference_code', '')
        current_generated_code = self.config.get('prompts', {}).get('examples', {}).get('generated_code', '')
        self.reference_generated_pairs.append([current_reference_code, current_generated_code])

        current_reference_code2 = self.config.get('prompts', {}).get('examples', {}).get('reference_code2', '')
        current_generated_code2 = self.config.get('prompts', {}).get('examples', {}).get('generated_code2', '')
        self.reference_generated_pairs.append([current_reference_code2, current_generated_code2])
        self.task_reference_codes = []

    def get_system_prompt(self, random: bool=False, pair_index: int=0) -> str:
        """Get system prompt from with the existing system prompt template and different options of reference and generated code"""
        if random is True:
            pair_index = np.random.randint(len(self.reference_generated_pairs))
        reference_code = self.reference_generated_pairs[pair_index][0]
        generated_code = self.reference_generated_pairs[pair_index][1]
        prompts_config = self.config.get('prompts', {})
        return prompts_config.get('system_prompt', 'You are a helpful assistant.').format(reference_code=reference_code, generated_code=generated_code)

    def get_user_prompt(self, source_code: str) -> str:
        """Get user prompt from configuration with source code substituted."""
        prompts_config = self.config.get('prompts', {})
        user_prompt_template = prompts_config.get('user_prompt', 'Analyze this code: {source_code}')
        return user_prompt_template.format(source_code=source_code)

    def extract_generated_code(self, content:str):
        code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
        generated_code = code_blocks[-1].strip() if code_blocks else content.strip()
        return generated_code

    def extract_task_reference_codes(self, kbpath):
        if self.config.get("kbtasks", {}).get("select_level1", False):
            task_list = self.config.get("kbtasks", {}).get("level1_tasks", [])
            leve1_path = kbpath + "/level1/"
            level_1_task_files = {int(f_.split("_")[0]): os.path.join(leve1_path, f_) for f_ in os.listdir(leve1_path)}
            for task_id in task_list:
                with open(level_1_task_files[task_id], 'r', encoding='utf-8') as file:
                    self.task_reference_codes.append(file.read())

        if self.config.get("kbtasks", {}).get("select_level2", False):
            task_list = self.config.get("kbtasks", {}).get("level2_tasks", [])
            leve2_path = kbpath + "/level2/"
            level_2_task_files = {int(f_.split("_")[0]): os.path.join(leve2_path, f_) for f_ in os.listdir(leve2_path)}
            for task_id in task_list:
                with open(level_2_task_files[task_id], 'r', encoding='utf-8') as file:
                    self.task_reference_codes.append(file.read())


# This roll out is the bottle neck, should be optimized later
# async def rollout_policy_logproba(trainer: GRPOTrainer, prompts: List[str], num_completion_per_prompt: int=8, max_new_tokens: int=100, temperature: float=0.6) -> List[List[Dict]]:
#     """
#     Given the model (current policy) and prompts, roll out the chat completion for code writing and return the List of List of Reponses
#     The 1st list is prompts
#     The 2nd list is num_completion_per_prompt
#     for each response, it is a dictionary
#     sequence_id: id of the current result in num_completion_per_prompt
#     prompt_token_ids: prompt token ids
#     completion_token_ids: the completion token ids
#     completion_log_probs: the log prob of the completion tokens
#     text: the text format of the completion
#     """
#     response_groups = []
#     for prompt in tqdm(prompts):
#         inputs = trainer.tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
#         # Generate with token IDs and log probabilities
#         with torch.no_grad():
#             outputs = trainer.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=True,
#                 temperature=temperature,
#                 num_return_sequences=num_completion_per_prompt,  # Generate 8 different completions, it is much faster than generating 1 by 1
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 pad_token_id=trainer.tokenizer.eos_token_id if trainer.tokenizer.eos_token_id else trainer.tokenizer.pad_token_id,
#                 eos_token_id=trainer.tokenizer.eos_token_id,
#                 use_cache=True,
#                 num_beams=1
#             )
#         all_sequences = outputs.sequences  # Shape: [num_completion_per_prompt, sequence_length]
#         scores = outputs.scores  # List of tensors, each with shape [num_completion_per_prompt, vocab_size]

#         # Get new tokens for all sequences (remove input prompt)
#         input_length = inputs.input_ids.shape[1]
#         all_new_tokens = all_sequences[:, input_length:].cpu()  # Shape: [num_completion_per_prompt, new_tokens_length]

#         # process the socre to get the log prob
#         all_scores = torch.stack(scores, dim=0).cpu()  # Shape: [num_new_tokens, 8, vocab_size]

#         del scores # score is a big tensor
#         torch.cuda.empty_cache()
#         gc.collect()

#         all_log_probs = torch.log_softmax(all_scores, dim=-1).cpu()  # Shape: [num_new_tokens, 8, vocab_size]
#         del all_scores # all_scores score is a big tensor
#         torch.cuda.empty_cache()
#         gc.collect()

#         # Extract log probs for each sequence
#         results = []
#         for seq_idx in range(num_completion_per_prompt):
#             sequence_tokens = all_new_tokens[seq_idx]
#             # Get log probs for this specific sequence

#             # Remove padding tokens (including EOS used as padding)
#             # Find the first occurrence of EOS/pad token
#             pad_token_id = trainer.tokenizer.eos_token_id if trainer.tokenizer.eos_token_id else trainer.tokenizer.pad_token_id
#             eos_token_id = trainer.tokenizer.eos_token_id

#             # Find where to truncate (first EOS or pad token)
#             truncate_idx = len(sequence_tokens)  # Default to full length

#             for i, token_id in enumerate(sequence_tokens):
#                 if token_id == pad_token_id or (eos_token_id and token_id == eos_token_id):
#                     truncate_idx = i + 1  # Include the EOS token itself
#                     break

#             # Truncate tokens and corresponding log probs
#             valid_tokens = sequence_tokens[:truncate_idx]

#             # Get log probs for this specific sequence (only for valid tokens)
#             if len(valid_tokens) > 0:
#                 sequence_log_probs = all_log_probs[torch.arange(len(valid_tokens)), seq_idx, valid_tokens]
#             else:
#                 sequence_log_probs = torch.tensor([])

#             results.append({
#                 'sequence_id': seq_idx,
#                 'prompt_token_ids': inputs["input_ids"].cpu().tolist()[0], # all input ids are the same, save the 1st element
#                 'completion_token_ids': valid_tokens.tolist(),
#                 'completion_log_probs': sequence_log_probs.tolist(),
#                 'text': trainer.tokenizer.decode(sequence_tokens, skip_special_tokens=True),
#                 'original_length': len(sequence_tokens),  # For debugging
#                 'valid_length': len(valid_tokens),        # For debugging
#             })
#             del valid_tokens
#             del sequence_log_probs

#         response_groups.append(results)
#          # release GPU memory for inputs and outputs
#         del all_sequences
#         del outputs
#         del inputs
#         torch.cuda.empty_cache()
#         gc.collect()
#     return response_groups

def rollout_policy_logproba_sp(lora_adapter_path: str, prompts: List[str], result_queue: mp.Queue, base_config: TrainerConfig, num_completion_per_prompt: int=8, max_new_tokens: int=100, temperature: float=0.6, device: int=0, batch_size: int=2):
    """
    Given the model (current policy) and prompts, roll out the chat completion for code writing and return the List of List of Reponses
    The 1st list is prompts
    The 2nd list is num_completion_per_prompt
    Each response is a dictionary
    sequence_id: id of the current result in num_completion_per_prompt
    prompt_token_ids: prompt token ids
    completion_token_ids: the completion token ids
    completion_log_probs: the log prob of the completion tokens
    text: the text format of the completion
    """
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_adapter_path,  # LoRA adapter path
        max_seq_length=base_config.model.max_seq_length,
        dtype=getattr(torch, base_config.model.compute_dtype, torch.bfloat16),
        load_in_4bit=base_config.model.load_in_4bit,
        use_cache=True,
        device_map={"": device}
    )
    # swith to inference mode
    FastLanguageModel.for_inference(model)
    device_ = torch.device(device)

    n_prompts = len(prompts)
    n_batches = (n_prompts + batch_size - 1) // batch_size

    for batch_i in tqdm(range(n_batches)):
        current_prompts = prompts[batch_i * batch_size : (batch_i + 1) * batch_size]
        inputs = tokenizer(current_prompts, return_tensors="pt", padding=True, truncation=True).to(device_)
        batch_input_lengths = inputs.attention_mask.sum(dim=1)
        max_input_length = max(batch_input_lengths).item()
        # Generate with token IDs and log probabilities
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_completion_per_prompt,  # Generate 8 different completions, it is much faster than generating 1 by 1
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1
            )
        all_sequences = outputs.sequences  # Shape: [num_completion_per_prompt, sequence_length]
        scores = [s_.cpu() for s_ in outputs.scores]  # List of tensors, each with shape [num_completion_per_prompt, vocab_size]

        # process the socre to get the log prob
        all_scores = torch.stack(scores, dim=0).cpu()  # Shape: [num_new_tokens, n_prompts*num_completion_per_prompt, vocab_size]
        all_log_probs = torch.log_softmax(all_scores, dim=-1).cpu()  # Shape: [num_new_tokens, n_prompts*num_completion_per_prompt, vocab_size]
        del scores
        del all_scores
        torch.cuda.empty_cache()
        gc.collect()

        for cur_id, prompt in enumerate(current_prompts):
            input_length = batch_input_lengths[cur_id].item()
            prompt_start_idx = cur_id * num_completion_per_prompt
            prompt_end_idx = (cur_id + 1) * num_completion_per_prompt
            prompt_sequences = all_sequences[prompt_start_idx:prompt_end_idx]
            prompt_new_tokens = prompt_sequences[:, max_input_length:].cpu()
            prompt_log_probs = all_log_probs[:, prompt_start_idx:prompt_end_idx, :]
            # Extract log probs for each sequence
            results = []
            for seq_idx in range(num_completion_per_prompt):
                sequence_tokens = prompt_new_tokens[seq_idx]
                # Get log probs for this specific sequence
                # Remove padding tokens (including EOS used as padding)
                # Find the first occurrence of EOS/pad token
                pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                eos_token_id = tokenizer.eos_token_id

                # Find where to truncate (first EOS or pad token)
                truncate_idx = len(sequence_tokens)  # Default to full length

                for i, token_id in enumerate(sequence_tokens):
                    if token_id == pad_token_id or (eos_token_id and token_id == eos_token_id):
                        truncate_idx = i + 1  # Include the EOS token itself
                        break

                # Truncate tokens and corresponding log probs
                valid_tokens = sequence_tokens[:truncate_idx]

                # Get log probs for this specific sequence (only for valid tokens)
                if len(valid_tokens) > 0:
                    sequence_log_probs = prompt_log_probs[torch.arange(len(valid_tokens)), seq_idx, valid_tokens]
                else:
                    sequence_log_probs = torch.tensor([])

                assert len(valid_tokens) == len(sequence_log_probs), "Error: completion_token_ids and completion_log_probs have different length"
                results.append({
                    'sequence_id': seq_idx,
                    'prompt_token_ids': inputs["input_ids"][cur_id, :input_length].cpu().tolist(), # all input ids are the same, save the 1st element
                    'completion_token_ids': valid_tokens.tolist(),
                    'completion_log_probs': sequence_log_probs.tolist(),
                    'text': tokenizer.decode(sequence_tokens, skip_special_tokens=True),
                })
            result_queue.put((prompt, results))

         # release GPU memory for inputs and outputs
        del all_sequences
        del outputs
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
    return

def rollout_policy_logproba_mp(lora_adapter_path: str, prompts: List[str], base_config: TrainerConfig, devices: List[int], num_completion_per_prompt: int=8, max_new_tokens: int=100, temperature: float=0.6, batch_size: int=2) -> List[List[Dict]]:
    n_gpus = len(devices)
    task_splits = [prompts[i::n_gpus] for i in range(n_gpus)]
    result_queue = mp.Queue()
    processes = []

    # Start processes
    for i, tasks in enumerate(task_splits):
        p = mp.Process(target=rollout_policy_logproba_sp, args=(lora_adapter_path, tasks, result_queue, base_config, num_completion_per_prompt, max_new_tokens, temperature, devices[i], batch_size))
        p.start()
        processes.append(p)

    # Collect results
    results_prompt_dict = {}
    for _ in range(len(prompts)):  # We know how many results to expect
        result = result_queue.get()
        results_prompt_dict[result[0]] = result[1]

    # Wait for processes to finish
    for p in processes:
        p.join()
    torch.cuda.empty_cache()
    gc.collect()
    response_groups = [results_prompt_dict[prompt] for prompt in prompts]
    return response_groups


def rollout_policy_logproba_safeworker(prompts, pool, result_queue, batch_size=4, max_new_tokens=3000, temperature=0.6, num_completion_per_prompt=8):
    model_instance = pool.get_model()
    try:
        model = model_instance['model']
        tokenizer = model_instance['tokenizer']
        device_ = model.device
        n_prompts = len(prompts)
        n_batches = (n_prompts + batch_size - 1) // batch_size

        for batch_i in tqdm(range(n_batches)):
            current_prompts = prompts[batch_i * batch_size : (batch_i + 1) * batch_size]
            inputs = tokenizer(current_prompts, return_tensors="pt", padding=True, truncation=True).to(device_)
            batch_input_lengths = inputs.attention_mask.sum(dim=1)
            max_input_length = max(batch_input_lengths).item()
            # Generate with token IDs and log probabilities
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=num_completion_per_prompt,  # Generate 8 different completions, it is much faster than generating 1 by 1
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
            all_sequences = outputs.sequences  # Shape: [num_completion_per_prompt, sequence_length]
            scores = [s_.cpu() for s_ in outputs.scores]  # List of tensors, each with shape [num_completion_per_prompt, vocab_size]

            # process the socre to get the log prob
            all_scores = torch.stack(scores, dim=0).cpu()  # Shape: [num_new_tokens, n_prompts*num_completion_per_prompt, vocab_size]
            all_log_probs = torch.log_softmax(all_scores, dim=-1).cpu()  # Shape: [num_new_tokens, n_prompts*num_completion_per_prompt, vocab_size]
            del scores
            del all_scores
            torch.cuda.empty_cache()
            gc.collect()

            for cur_id, prompt in enumerate(current_prompts):
                input_length = batch_input_lengths[cur_id].item()
                prompt_start_idx = cur_id * num_completion_per_prompt
                prompt_end_idx = (cur_id + 1) * num_completion_per_prompt
                prompt_sequences = all_sequences[prompt_start_idx:prompt_end_idx]
                prompt_new_tokens = prompt_sequences[:, max_input_length:].cpu()
                prompt_log_probs = all_log_probs[:, prompt_start_idx:prompt_end_idx, :]
                # Extract log probs for each sequence
                results = []
                for seq_idx in range(num_completion_per_prompt):
                    sequence_tokens = prompt_new_tokens[seq_idx]
                    # Get log probs for this specific sequence
                    # Remove padding tokens (including EOS used as padding)
                    # Find the first occurrence of EOS/pad token
                    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                    eos_token_id = tokenizer.eos_token_id

                    # Find where to truncate (first EOS or pad token)
                    truncate_idx = len(sequence_tokens)  # Default to full length

                    for i, token_id in enumerate(sequence_tokens):
                        if token_id == pad_token_id or (eos_token_id and token_id == eos_token_id):
                            truncate_idx = i + 1  # Include the EOS token itself
                            break

                    # Truncate tokens and corresponding log probs
                    valid_tokens = sequence_tokens[:truncate_idx]

                    # Get log probs for this specific sequence (only for valid tokens)
                    if len(valid_tokens) > 0:
                        sequence_log_probs = prompt_log_probs[torch.arange(len(valid_tokens)), seq_idx, valid_tokens]
                    else:
                        sequence_log_probs = torch.tensor([])

                    assert len(valid_tokens) == len(sequence_log_probs), "Error: completion_token_ids and completion_log_probs have different length"
                    results.append({
                        'sequence_id': seq_idx,
                        'prompt_token_ids': inputs["input_ids"][cur_id, :input_length].cpu().tolist(), # all input ids are the same, save the 1st element
                        'completion_token_ids': valid_tokens.tolist(),
                        'completion_log_probs': sequence_log_probs.tolist(),
                        'text': tokenizer.decode(sequence_tokens, skip_special_tokens=True),
                    })
                result_queue.put((prompt, results))

            # release GPU memory for inputs and outputs
            del all_sequences
            del outputs
            del inputs
            torch.cuda.empty_cache()
            gc.collect()
        return
    finally:
        pool.return_model(model_instance)

def rollout_policy_logproba_safethreading(lora_adapter_path: str, prompts: List[str], base_config: TrainerConfig, devices: List[int], num_completion_per_prompt: int=8, max_new_tokens: int=100, temperature: float=0.6, batch_size: int=2) -> List[List[Dict]]:
    '''
    unsloth has issues with mp, and this is an safe multithreading option for unsloth
    '''
    start_time = time.time()
    n_gpus = len(devices)
    unsloth_configs = [
        {
        'model_name': lora_adapter_path,
        'max_seq_length': base_config.model.max_seq_length,
        'dtype': getattr(torch, base_config.model.compute_dtype, torch.bfloat16),
        'load_in_4bit': base_config.model.load_in_4bit,
        "use_cache": True,
        "device_map" :{"": device},
        }
        for device in devices
    ]

    pool = ThreadSafeModelPool(unsloth_configs, max_workers=n_gpus)
    task_splits = [prompts[i::n_gpus] for i in range(n_gpus)]
    result_queue = mp.Queue()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [executor.submit(rollout_policy_logproba_safeworker,
                                   task_prompts,
                                   pool,
                                   result_queue,
                                   batch_size=batch_size,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature,
                                   num_completion_per_prompt=num_completion_per_prompt)
                        for task_prompts in task_splits]
        _ = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Collect results
    results_prompt_dict = {}
    for _ in range(len(prompts)):  # We know how many results to expect
        result = result_queue.get()
        results_prompt_dict[result[0]] = result[1]
    # clean cache again
    del pool
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Delete the unslot model pool for next iterations ...")

    response_groups = [results_prompt_dict[prompt] for prompt in prompts]

    end_time = time.time()
    logger.info(f"roll out current policy takes {end_time - start_time} seconds")
    return response_groups


def run_process_with_retry(command: str, result_file: str, max_retries: int = 3) -> bool:
    """Run a subprocess with retry logic. Returns True if successful, False otherwise."""
    for attempt in range(max_retries):
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = process.communicate()

            if process.returncode == 0 and os.path.exists(result_file):
                return True
            else:
                print(f"Attempt {attempt + 1} failed for {result_file}: return code {process.returncode}")
                if stderr:
                    print(f"Error: {stderr}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        except Exception as e:
            print(f"Attempt {attempt + 1} exception for {result_file}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    print(f"All {max_retries} attempts failed for {result_file}")
    return False

def rollout_policy_logproba_shellprocess(lora_adapter_path: str, prompts: List[str], base_config, devices: List[int], num_completion_per_prompt: int=8, max_new_tokens: int=100, temperature: float=0.6, batch_size: int=2) -> List[List[Dict]]:
    start_time = time.time()
    n_gpus = len(devices)
    unsloth_configs = [
        {
        'model_name': lora_adapter_path,
        'max_seq_length': base_config.model.max_seq_length,
        'dtype': base_config.model.compute_dtype,
        'load_in_4bit': base_config.model.load_in_4bit,
        "use_cache": True,
        "device_map" :{"": device},
        }
        for device in devices
    ]
    task_splits = [prompts[i::n_gpus] for i in range(n_gpus)]

    # Store job info for parallel execution with retry
    jobs = []

    for i, task in enumerate(task_splits):
        config_file = f"temp/config_{i}.json"
        result_file = f"temp/result_{i}.pkl"
        prompts_file = f"temp/prompts_{i}.json"

        # dump config to disk
        with open(config_file, 'w') as f:
            json.dump(unsloth_configs[i], f)

        with open(prompts_file, 'w') as f:
            json.dump(task, f)

        command = f"python unsloth_parallel.py --unsloth_config_file {config_file} --prompts_file {prompts_file} --batch_size {batch_size} --max_new_tokens {max_new_tokens} --temperature {temperature} --num_completion_per_prompt {num_completion_per_prompt} --result_path {result_file}"

        jobs.append((command, result_file))

    # Run all jobs in parallel with retry logic
    def run_job(command, result_file, results_dict):
        success = run_process_with_retry(command, result_file, max_retries=3)
        results_dict[result_file] = success

    # Track results and threads
    results_dict = {}
    threads = []

    # Start all jobs
    for command, result_file in jobs:
        thread = threading.Thread(target=run_job, args=(command, result_file, results_dict))
        thread.start()
        threads.append(thread)

    # Wait for all jobs to complete
    for thread in threads:
        thread.join()

    # Collect results from successful jobs
    all_results = []
    failed_jobs = []

    for command, result_file in jobs:
        if results_dict.get(result_file, False) and os.path.exists(result_file):
            results = load_pickle(result_file)  # Assuming this function exists
            all_results.extend(results)
        else:
            failed_jobs.append(result_file)

    if failed_jobs:
        print(f"Warning: {len(failed_jobs)} jobs failed after all retries: {failed_jobs}")

    # Build response groups
    results_prompt_dict = {}
    for prompt, result in all_results:
        results_prompt_dict[prompt] = result

    response_groups = []
    for prompt in prompts:
        if prompt in results_prompt_dict:
            response_groups.append(results_prompt_dict[prompt])
        else:
            print(f"Warning: No results found for prompt: {prompt[:50]}...")
            response_groups.append([])  # or handle missing results as needed

    # remove generated files in the temp/ folder
    for filename in os.listdir("temp/"):
        file_path = os.path.join("temp/", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            logger.info(f"Removed: {file_path}")  # Uncomment if logger is available
    logger.info("All temp files removed successfully!")  # Uncomment if logger is available

    end_time = time.time()
    # logger.info(f"roll out current policy takes {end_time - start_time} seconds")  # Uncomment if logger is available
    print(f"Rollout completed in {end_time - start_time:.2f} seconds")

    return response_groups

def eval_kernel_reference_sp(
    run_tag: str,
    model_tag: str,
    task_tag: str,
    reference_codes: str,
    device: int,
    result_queue: mp.Queue,
    verbose: bool=False):
    args = argparse.Namespace(verbose=verbose)
    device_ = torch.device(device)
    build_dir = GRPO_FOLDER + f"/kbeval/{run_tag}/{model_tag}/{task_tag}/eval_tag"
    for reference_code in reference_codes:
        result_queue.put((reference_code, eval_kernel_custom(run_tag, model_tag, task_tag, run_tag, reference_code, "<string>", "", "<string>", device_, work_dir=build_dir, measure_reference=True)))

def eval_kernel_reference_mp(run_tag: str,
                             model_tag: str,
                             task_tag: str,
                             reference_codes: List[str],
                             devices: List[int]) -> Dict[str, KernelExecResult]:
    n_gpus = len(devices)
    task_splits = [reference_codes[i::n_gpus] for i in range(n_gpus)]

    result_queue = mp.Queue()
    processes = []

    # Start processes
    for i, tasks in enumerate(task_splits):
        p = mp.Process(target=eval_kernel_reference_sp, args=(run_tag, model_tag, task_tag, tasks, devices[i], result_queue))
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in range(len(reference_codes)):  # We know how many results to expect
        result = result_queue.get()
        results.append(result)

    # Wait for processes to finish
    for p in processes:
        p.join()
    torch.cuda.empty_cache()
    gc.collect()
    return {c: r for c, r in results}


def eval_kernel_reference_generate_sp(
    run_tag: str,
    model_tag: str,
    task_tag: str,
    eval_tag: str,
    code_tuples: List[Tuple[str, str]],
    device: int,
    result_queue: mp.Queue):
    args = argparse.Namespace(verbose=False)
    device_ = torch.device(device)
    build_dir = GRPO_FOLDER + f"/kbeval/{run_tag}/{model_tag}/{task_tag}/{eval_tag}"
    for reference_code, generated_code in code_tuples:
        gi = hash_string_short(generated_code, 5)
        build_dir_cur = build_dir + f"/{gi}/"
        result_queue.put(((reference_code, generated_code),
                           eval_kernel_custom(run_tag, model_tag, task_tag, run_tag, reference_code, "<string>", generated_code, "<string>", device_, work_dir=build_dir_cur, code_type="cuda", measure_reference=False)
                           )
                          )

def eval_kernel_reference_generate_mp(run_tag: str,
                             model_tag: str,
                             task_tag: str,
                             eval_tag: str,
                             code_tuples: List[Tuple[str, str]],
                             devices: List[int]) -> Dict[Tuple[str, str], KernelExecResult]:
    n_gpus = len(devices)
    task_splits = [code_tuples[i::n_gpus] for i in range(n_gpus)]

    result_queue = mp.Queue()
    processes = []

    # Start processes
    for i, tasks in enumerate(task_splits):
        p = mp.Process(target=eval_kernel_reference_generate_sp, args=(run_tag, model_tag, task_tag, eval_tag, tasks, devices[i], result_queue))
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in range(len(code_tuples)):
        result = result_queue.get()
        results.append(result)

    # Wait for processes to finish
    for p in processes:
        p.join()
    torch.cuda.empty_cache()
    gc.collect()
    return {c: r for c, r in results}


def code_evaluation_local(
            gerenated_code_dict: Dict[str, List[str]],
            run_tag: str,
            model_tag: str,
            task_tag: str,
            reward_config: RewardConfig,
            devices: List[int]=[5, 6, 7] # assuming all the devices are available
        ) -> Dict[str, Dict]:
    """
    Given a dictionary with reference_code as key, and generated_codes in the rolls out as values
    evaluate the reference code and the generated code
    return the rewards and eval results for each generated_code as a list of list
    """

    reference_runtime_cache = {}
    reference_codes = list(gerenated_code_dict.keys())
    reference_runtime_cache = eval_kernel_reference_mp(run_tag, model_tag, task_tag, reference_codes, devices)

    ref_gen_tuples = []
    for ri, (reference_code, generated_codes) in enumerate(gerenated_code_dict.items()):
        ref_gen_tuples.extend([(reference_code, g_) for g_ in generated_codes ])

    ref_gen_code_results = eval_kernel_reference_generate_mp(run_tag, model_tag, task_tag, "eval_tag", ref_gen_tuples, devices)

    eval_scores = {}
    for reference_code, generated_codes in gerenated_code_dict.items():
        reference_runtime = reference_runtime_cache[reference_code].runtime
        generate_code_results = [ref_gen_code_results[(reference_code, generated_code)] for generated_code in generated_codes]
        eval_scores[reference_code] = {}
        eval_scores[reference_code]["result"] = generate_code_results
        eval_scores[reference_code]["reward"] = [reward_function(g.compiled, g.correctness, g.runtime / reference_runtime, reward_config)
                    for g in generate_code_results]
    return eval_scores


def reward_function(compile_: bool, correct: bool, speed_up: float, reward_config: RewardConfig):
    total_reward = 0
    if compile_ is False:
        total_reward += reward_config.compile_penalty
        return total_reward
    else:
        total_reward += reward_config.compile_reward

    if correct is False:
        total_reward += reward_config.correct_penalty
        return total_reward
    else:
        total_reward += reward_config.correct_reward

    if speed_up < reward_config.speed_up_threshold:
        total_reward += reward_config.speed_up_penalty
        return total_reward
    else:
        total_reward += reward_config.speed_up_reward
    return total_reward


def hash_string_short(text: str, length: int = 8) -> str:
    """
    Create a shorter hash for cases where you need compact representation
    Uses SHA-256 but truncates to specified length
    """
    full_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return full_hash[:length]


# async def test_rollout_policy_logproba(args):

#     try:
#         base_config = TrainerConfig.from_yaml(args.base_config)
#         logger.info(f"✅ [GRPOTrainer] Base configuration loaded from {args.base_config}")
#     except Exception as e:
#         logger.error(f"❌ [GRPOTrainer] Failed to load base configuration: {e}")
#         sys.exit(1)

#     try:
#         grpo_config = GRPOConfig.from_yaml(args.grpo_config)
#         logger.info(f"✅ [GRPOTrainer] GRPO configuration loaded from {args.grpo_config}")
#     except Exception as e:
#         logger.error(f"❌ [GRPOTrainer] Failed to load GRPO configuration: {e}")
#         sys.exit(1)

#     try:
#         trainer = GRPOTrainer(args.run_tag, grpo_config, base_config) # trainer.model is the current policy
#         logger.info("✅ [GRPOTrainer] Trainer initialized")
#     except Exception as e:
#         logger.error(f"❌ [GRPOTrainer] Initialization failed: {e}")
#         sys.exit(1)

#     prompt_manger = PromptManager("grpo_iterations.yaml")
#     system_prompt = prompt_manger.get_system_prompt()
#     user_prompt = prompt_manger.get_user_prompt(prompt_manger.reference_generated_pairs[0][0])
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
#     prompt = trainer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     prompts = [prompt] * 2
#     response_groups = await rollout_policy_logproba(trainer, prompts, num_completion_per_prompt=2, max_new_tokens=1000)
#     assert len(response_groups) == 2
#     assert len(response_groups[0]) == 2
#     generated_texts = [[trainer.tokenizer.decode(response["completion_token_ids"]) for response in response_group] for response_group in response_groups]
#     logger.info(f"The test responses from base model {generated_texts}")


async def test_rollout_policy_logproba_shellprocess(args):
    from trainerGRPO import GRPOTrainer, GRPOConfig, GenerationResultGroup, GenerationResult, GenerationDataset
    from trainerBase import TrainerConfig

    try:
        base_config = TrainerConfig.from_yaml(args.base_config)
        logger.info(f"✅ [GRPOTrainer] Base configuration loaded from {args.base_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load base configuration: {e}")
        sys.exit(1)
    base_config.model.name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

    try:
        grpo_config = GRPOConfig.from_yaml(args.grpo_config)
        logger.info(f"✅ [GRPOTrainer] GRPO configuration loaded from {args.grpo_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load GRPO configuration: {e}")
        sys.exit(1)

    try:
        trainer = GRPOTrainer(args.run_tag, grpo_config, base_config) # trainer.model is the current policy
        logger.info("✅ [GRPOTrainer] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Initialization failed: {e}")
        sys.exit(1)

    prompt_manger = PromptManager("grpo_iterations.yaml")
    system_prompt = prompt_manger.get_system_prompt()
    user_prompt = prompt_manger.get_user_prompt(prompt_manger.reference_generated_pairs[0][0])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = trainer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * 2
    response_groups = rollout_policy_logproba_shellprocess("unsloth/Qwen3-8B-unsloth-bnb-4bit", prompts, base_config, devices=[1, 2], num_completion_per_prompt=8, max_new_tokens=1000, batch_size=4)
    assert len(response_groups) == 2
    assert len(response_groups[0]) == 8
    generated_texts = [[trainer.tokenizer.decode(response["completion_token_ids"]) for response in response_group] for response_group in response_groups]
    logger.info(f"The test responses from base model {generated_texts}")

async def test_rollout_policy_logproba_safethreading(args):
    from trainerGRPO import GRPOTrainer, GRPOConfig, GenerationResultGroup, GenerationResult, GenerationDataset
    from trainerBase import TrainerConfig

    try:
        base_config = TrainerConfig.from_yaml(args.base_config)
        logger.info(f"✅ [GRPOTrainer] Base configuration loaded from {args.base_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load base configuration: {e}")
        sys.exit(1)
    base_config.model.name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

    try:
        grpo_config = GRPOConfig.from_yaml(args.grpo_config)
        logger.info(f"✅ [GRPOTrainer] GRPO configuration loaded from {args.grpo_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load GRPO configuration: {e}")
        sys.exit(1)

    try:
        trainer = GRPOTrainer(args.run_tag, grpo_config, base_config) # trainer.model is the current policy
        logger.info("✅ [GRPOTrainer] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Initialization failed: {e}")
        sys.exit(1)

    prompt_manger = PromptManager("grpo_iterations.yaml")
    system_prompt = prompt_manger.get_system_prompt()
    user_prompt = prompt_manger.get_user_prompt(prompt_manger.reference_generated_pairs[0][0])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = trainer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * 2
    response_groups = rollout_policy_logproba_safethreading("unsloth/Qwen3-8B-unsloth-bnb-4bit", prompts, base_config, devices=[1, 2], num_completion_per_prompt=8, max_new_tokens=1000, batch_size=4)
    assert len(response_groups) == 2
    assert len(response_groups[0]) == 8
    generated_texts = [[trainer.tokenizer.decode(response["completion_token_ids"]) for response in response_group] for response_group in response_groups]
    logger.info(f"The test responses from base model {generated_texts}")

async def test_rollout_policy_logproba_mp(args):
    from trainerGRPO import GRPOTrainer, GRPOConfig, GenerationResultGroup, GenerationResult, GenerationDataset
    from trainerBase import TrainerConfig

    try:
        base_config = TrainerConfig.from_yaml(args.base_config)
        logger.info(f"✅ [GRPOTrainer] Base configuration loaded from {args.base_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load base configuration: {e}")
        sys.exit(1)

    try:
        grpo_config = GRPOConfig.from_yaml(args.grpo_config)
        logger.info(f"✅ [GRPOTrainer] GRPO configuration loaded from {args.grpo_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load GRPO configuration: {e}")
        sys.exit(1)

    try:
        trainer = GRPOTrainer(args.run_tag, grpo_config, base_config) # trainer.model is the current policy
        logger.info("✅ [GRPOTrainer] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Initialization failed: {e}")
        sys.exit(1)

    prompt_manger = PromptManager("grpo_iterations.yaml")
    system_prompt = prompt_manger.get_system_prompt()
    user_prompt = prompt_manger.get_user_prompt(prompt_manger.reference_generated_pairs[0][0])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = trainer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * 2
    response_groups = rollout_policy_logproba_mp(args.start_model_path, prompts, base_config, devices=[1, 2], num_completion_per_prompt=8, max_new_tokens=1000, batch_size=4)
    assert len(response_groups) == 2
    assert len(response_groups[0]) == 8
    generated_texts = [[trainer.tokenizer.decode(response["completion_token_ids"]) for response in response_group] for response_group in response_groups]
    logger.info(f"The test responses from base model {generated_texts}")

async def test_code_evaluation_local():
    prompt_manger = PromptManager("grpo_iterations.yaml")
    example_reference_code = prompt_manger.reference_generated_pairs[0][0]
    example_generated_code = prompt_manger.reference_generated_pairs[0][1]
    example_reference_code2 = prompt_manger.reference_generated_pairs[1][0]
    example_generated_code2 = prompt_manger.reference_generated_pairs[1][1]
    # print("****reference code***")
    # print(example_reference_code)
    # print("***generated code***")
    # print(example_generated_code)
    test_input_dict = {}
    test_input_dict[example_reference_code] = [example_generated_code, example_generated_code2]
    test_input_dict[example_reference_code2] = [example_generated_code2]
    reward_config = RewardConfig()
    final_evals = code_evaluation_local(test_input_dict, "test", "example", "test_task", reward_config)
    print("final evals", final_evals.values())
    assert len(final_evals) == 2
    assert len(final_evals[example_reference_code]["reward"]) == 2
    assert len(final_evals[example_reference_code2]["reward"]) == 1
    logger.info("finished test for evaluating example reference and generated codes")

async def test_prompt_manager(args):
    prompt_manger = PromptManager(args.online_grpo_config)
    prompt_manger.extract_task_reference_codes("kernel_bench/")
    for task_source_code in prompt_manger.task_reference_codes:
        logger.info(f"The task source code read from kb is {task_source_code}")

async def train_grpo(args: argparse.Namespace):
    # The training step follow the pseudo code:
    #  https://docs.google.com/document/d/1r5Dl6L5kAFmYaKY1eqONfMsUQmbZof-Pfdr4XPNCu58/edit?tab=t.0#bookmark=id.236s9naq0h9n

    # load config
    from trainerGRPO import GRPOTrainer, GRPOConfig, GenerationResultGroup, GenerationResult, GenerationDataset
    from trainerBase import TrainerConfig
    reward_config = RewardConfig.from_yaml(args.online_grpo_config)
    with open(os.path.expanduser(args.online_grpo_config), 'r') as f:
            online_grpo_config = yaml.safe_load(f)

    # Setup trainer
    try:
        base_config = TrainerConfig.from_yaml(args.base_config)
        logger.info(f"✅ [GRPOTrainer] Base configuration loaded from {args.base_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load base configuration: {e}")
        sys.exit(1)

    # overwrite the base parameters
    base_config.training.checkpoint_path = GRPO_FOLDER + "/lora_adapter/"
    base_config.model.name = args.start_model_path
    base_config.training.save_steps = 10e9 # save the checkpoint by epochs
    base_config.training.micro_batch_size = online_grpo_config["train"]["micro_batch_size"]
    base_config.training.gradient_accumulation_steps = online_grpo_config["train"]["gradient_accumulation_steps"]
    base_config.training.learning_rate = online_grpo_config["train"]["learning_rate"]
    base_config.training.max_steps = online_grpo_config["train"]["max_steps"]
    base_config.training.max_grad_norm = online_grpo_config["train"]["max_grad_norm"]
    base_config.training.scheduler_type = online_grpo_config["train"]["scheduler_type"]
    base_config.training.num_warmup_steps = online_grpo_config["train"]["num_warmup_steps"]


    try:
        grpo_config = GRPOConfig.from_yaml(args.grpo_config)
        logger.info(f"✅ [GRPOTrainer] GRPO configuration loaded from {args.grpo_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load GRPO configuration: {e}")
        sys.exit(1)

    grpo_config.beta = online_grpo_config["train"]["grpo_beta"]
    grpo_config.reference_model_update_steps = online_grpo_config["train"]["grpo_reference_model_update_steps"]
    grpo_config.clip_epsilon_lower = online_grpo_config["train"]["grpo_clip_epsilon_lower"]
    grpo_config.clip_epsilon_upper = online_grpo_config["train"]["grpo_clip_epsilon_upper"]

    try:
        trainer = GRPOTrainer(args.run_tag, grpo_config, base_config) # trainer.model is the current policy
        logger.info("✅ [GRPOTrainer] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Initialization failed: {e}")
        sys.exit(1)
    ref_device = online_grpo_config["train"]["grpo_reference_model_device"]
    trainer._deepcopy_reference_model(f"cuda:{ref_device}")

    # save config files into the checkpoint folder
    shutil.copy(args.base_config, trainer.checkpoint_path)
    shutil.copy(args.grpo_config, trainer.checkpoint_path)
    shutil.copy(args.online_grpo_config, trainer.checkpoint_path)

    # get the current epoch from the last check point if exists
    inference_model_path = args.start_model_path
    current_epoch = 0
    checkpoint_location = trainer.checkpoint_path / base_config.training.latest_checkpoint_name
    if trainer._checkpoint_exists(checkpoint_location):
        logger.info(f"Starting from the latest checkpoint {checkpoint_location}")
        inference_model_path = str(checkpoint_location)
        checkpoint_path = os.readlink(checkpoint_location)
        current_epoch = int(checkpoint_path.split("-")[1])

    # setup prompt manager
    logger.info("Set up prompt manager")
    prompt_manger = PromptManager(args.online_grpo_config)
    prompt_manger.extract_task_reference_codes("kernel_bench/")
    # Generate for the prompts
    logger.info("Generating all the prompts from selected tasks")
    all_prompts = []
    all_reference_codes = []
    for reference_code in prompt_manger.task_reference_codes:
        system_prompt = prompt_manger.get_system_prompt(pair_index=1)
        user_prompt = prompt_manger.get_user_prompt(reference_code)
        all_reference_codes.append(reference_code)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = trainer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)



    logger.info("Loading training config")
    # train parameters
    max_epochs = online_grpo_config.get("train", {}).get("max_epochs", 2)
    # inference parameters
    group_size = online_grpo_config.get("inference", {}).get("group_size", 4)
    max_new_tokens = online_grpo_config.get("inference", {}).get("max_new_tokens", 3000)
    inference_gpus = online_grpo_config.get("inference", {}).get("gpus", [5, 6])
    inferece_batch_size = online_grpo_config.get("inference", {}).get("batch_size", 4)

    kb_gpus = online_grpo_config.get("kbtasks", {}).get("kb_gpus", [5, 6])
    save_checkpoint_every_epochs = online_grpo_config.get("train", {}).get("save_checkpoint_every_epochs", 4)

    logger.info("Start training")

    for epoch in range(current_epoch, 1 + max_epochs):

        logger.info("Rolling out current policy ...")
        if isinstance(inference_model_path, Path):
            inference_model_path = str(inference_model_path)
        response_groups = rollout_policy_logproba_shellprocess(inference_model_path,
                                                                all_prompts,
                                                                base_config,
                                                                inference_gpus,
                                                                num_completion_per_prompt=group_size,
                                                                max_new_tokens=max_new_tokens,
                                                                batch_size=inferece_batch_size)
        reward_eval_input = {}
        evaluated_reference_codes = []
        evaluated_response_groups = []
        for reference_code, response_group in zip(all_reference_codes, response_groups):
            if response_group == []: ## The case roll out policy failed for the reference code
                continue
            evaluated_reference_codes.append(reference_code)
            evaluated_response_groups.append(response_group)
            reward_eval_input[reference_code] = [prompt_manger.extract_generated_code(current_gen["text"]) for current_gen in response_group]

        logger.info("Evaluating the generated codes and get rewards ...")
        # reward_dict has reward scores and the eval results. The eval results can be used for teacher comments
        reward_dict = code_evaluation_local(reward_eval_input, args.run_tag, f"model_{epoch}", "rollout", reward_config, devices=kb_gpus)

        #Create GRPO dataset
        rewards_groups = [reward_dict[reference_code]["reward"] for reference_code in evaluated_reference_codes]
        rewards_save_path = os.path.join(trainer.checkpoint_path, f"reward_{epoch}.json")
        with open(rewards_save_path, "w") as f:
            json.dump({"referece_codes": evaluated_reference_codes, "rewards": rewards_groups}, f, indent=4)

        result_groups = []
        for response_group, rewards in zip(evaluated_response_groups, rewards_groups):
            result_group = GenerationResultGroup(task_tag="{args.run_tage}_{epoch}", results=[])
            id = 0
            for response, reward in zip(response_group, rewards):
                id += 1
                prompt_token_ids = response["prompt_token_ids"]
                completion_token_ids = response["completion_token_ids"]
                completion_log_probs = response["completion_log_probs"]
                result = GenerationResult(
                    gen_tag=f"gen_{id:02d}",
                    reward=reward,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    completion_log_probs=completion_log_probs,
                )
                # add the result to the result group
                result_group.results.append(result)
            # add the result group to the generation dataset
            result_groups.append(result_group)
        dataset = GenerationDataset(result_groups)

        # Train the block:
        try:
            trainer.train_block(args.run_tag, dataset)
            logger.info("🎉 [GRPOTrainer] Training completed successfully!")
            if epoch % save_checkpoint_every_epochs == 0:
                trainer._save_checkpoint(epoch)
                inference_model_path = trainer.checkpoint_path / f"checkpoint-{epoch}"
        except Exception as e:
            logger.error(f"❌ [GRPOTrainer] Training failed: {e}")
            sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="Iterate Model by Online GRPO")
    parser.add_argument("--start_model_path", type=str, default="finetune_model_output/sft_t2/qwen3_32b/")
    parser.add_argument("--run_tag", type=str, default="v0.1_20250801_grpot4")
    parser.add_argument("--base-config", type=str, default="trainerBase.yaml")
    parser.add_argument("--grpo-config", type=str, default="trainerGRPO.yaml")
    parser.add_argument("--online-grpo-config", type=str, default="grpo_iterations.yaml")
    parser.add_argument("--test", action='store_true', help='Enable verbose output.')
    args = parser.parse_args()

    if args.test:
        logger.info("Running test only ... ")
        # await test_rollout_policy_logproba(args)
        # await test_rollout_policy_logproba_mp(args) # unsloth is not pickable
        # await test_rollout_policy_logproba_safethreading(args) # safethreading is still slow
        # await test_rollout_policy_logproba_shellprocess(args)
        await test_code_evaluation_local()
        # await test_prompt_manager(args)
        return

    await train_grpo(args)

if __name__ == "__main__":
    asyncio.run(main())
