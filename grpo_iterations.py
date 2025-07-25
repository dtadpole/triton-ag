import os
import sys
import numpy as np
from pydantic import BaseModel
import hashlib
import multiprocessing as mp
import unsloth
import torch
import asyncio
from logger import logger
from trainerGRPO import GRPOTrainer, GRPOConfig
from trainerBase import TrainerConfig
from kbEvalCli import compile_and_eval_kernel, eval_kernel_reference
from kbEvalTest.kbeval import KernelExecResult
from typing import DefaultDict, Dict, List, Optional, Any, Tuple
import argparse
import yaml
import re


GRPO_FOLDER = "grpo_artifacts/"
# Set multiprocessing start method (important for CUDA)
mp.set_start_method('spawn', force=True)

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


async def rollout_policy_logproba(trainer: GRPOTrainer, prompts: List[str], num_completion_per_prompt: int=8, max_new_tokens: int=100, temperature: float=0.6) -> List[List[Dict]]:
    """
    Given the model (current policy) and prompts, roll out the chat completion for code writing and return the dictionary
    prompt_ids
    prompt
    """
    response_groups = []
    for prompt in prompts:
        inputs = trainer.tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
        # Generate with token IDs and log probabilities
        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_completion_per_prompt,  # Generate 8 different completions
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=trainer.tokenizer.eos_token_id if trainer.tokenizer.eos_token_id else trainer.tokenizer.pad_token_id,
                eos_token_id=trainer.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1
            )
        all_sequences = outputs.sequences  # Shape: [num_completion_per_prompt, sequence_length]
        scores = outputs.scores  # List of tensors, each with shape [num_completion_per_prompt, vocab_size]

        # Get new tokens for all sequences (remove input prompt)
        input_length = inputs.input_ids.shape[1]
        all_new_tokens = all_sequences[:, input_length:]  # Shape: [num_completion_per_prompt, new_tokens_length]

        # process the socre to get the log prob
        all_scores = torch.stack(scores, dim=0)  # Shape: [num_new_tokens, 8, vocab_size]
        all_log_probs = torch.log_softmax(all_scores, dim=-1)  # Shape: [num_new_tokens, 8, vocab_size]

        # Extract log probs for each sequence
        results = []
        for seq_idx in range(num_completion_per_prompt):
            sequence_tokens = all_new_tokens[seq_idx]
            # Get log probs for this specific sequence
            sequence_log_probs = all_log_probs[torch.arange(len(sequence_tokens)), seq_idx, sequence_tokens]
            results.append({
                'sequence_id': seq_idx,
                'prompt_token_ids': inputs["input_ids"].cpu().tolist()[0], # all input ids are the same, save the 1st element
                'completion_token_ids': sequence_tokens.tolist(),
                'completion_log_probs': sequence_log_probs.tolist(),
                'text': trainer.tokenizer.decode(sequence_tokens, skip_special_tokens=True),
            })
        response_groups.append(results)
    torch.cuda.empty_cache()
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
    for reference_code in reference_codes:
        result_queue.put((reference_code, eval_kernel_reference(run_tag, model_tag, task_tag, reference_code, device_, args)))


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
                           compile_and_eval_kernel(run_tag, model_tag, task_tag, eval_tag, reference_code, generated_code, device_, build_dir_cur, args)
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


async def test_rollout_policy_logproba(args):

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
    response_groups = await rollout_policy_logproba(trainer, prompts, num_completion_per_prompt=2, max_new_tokens=1000)
    assert len(response_groups) == 2
    assert len(response_groups[0]) == 2
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
    # Setup trainer
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

    # load config
    reward_config = RewardConfig.from_yaml(args.online_grpo_config)
    with open(os.path.expanduser(args.online_grpo_config), 'r') as f:
            online_grpo_config = yaml.safe_load(f)

    logger.info("Loading training config")
    max_epochs = online_grpo_config.get("train", {}).get("max_epochs", 2)
    group_size = online_grpo_config.get("train", {}).get("group_size", 4)
    max_new_tokens = online_grpo_config.get("train", {}).get("max_new_tokens", 1500)
    reference_update_interval = online_grpo_config.get("train", {}).get("reference_update_interval", 1)

    #####
    # TODO check reference model set up. Assuming current reference model is the frozen version of model but not a softlink to model.
    logger.info("Start training")
    for epoch in range(1, 1 + max_epochs):

        logger.info("Rolling out current policy ...")
        response_groups = await rollout_policy_logproba(trainer, all_prompts, group_size, max_new_tokens=max_new_tokens)
        reward_eval_input = {}
        for reference_code, response_group in zip(all_reference_codes, response_groups):
            reward_eval_input[reference_code] = [prompt_manger.extract_generated_code(current_gen["text"]) for current_gen in response_group]

        logger.info("Evaluating the generated codes and get rewards ...")
        reward_dict = code_evaluation_local(reward_eval_input, args.run_tag, f"model_{epoch}", "rollout", reward_config, devices=[5, 6])



async def main():
    parser = argparse.ArgumentParser(description="Iterate Model by Online GRPO")
    parser.add_argument("--start_model_path", type=str, default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--run_tag", type=str, default="v0.1_20250714_050308")
    parser.add_argument("--base-config", type=str, default="trainerBase.yaml")
    parser.add_argument("--grpo-config", type=str, default="trainerGRPO.yaml")
    parser.add_argument("--online-grpo-config", type=str, default="grpo_iterations.yaml")
    parser.add_argument("--test", action='store_true', help='Enable verbose output.')
    args = parser.parse_args()

    if args.test:
        await test_rollout_policy_logproba(args)
        await test_code_evaluation_local()
        await test_prompt_manager(args)
        return

    await train_grpo(args)


if __name__ == "__main__":
    asyncio.run(main())
