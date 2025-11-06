import torch
import numpy as np
from transformers import AutoTokenizer
from operator import itemgetter
from itertools import groupby
import scipy.stats as stats
from scipy.optimize import brentq
import math

def grpo_compute_rewards(
    query_result: list[dict],
    gamma: float = 0.5,
    debug: bool = False,
) -> dict[str, list[dict]]:
    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of rows
    results_by_gen = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_tag")),
            key=itemgetter("task_tag","gen_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, gen_tag)\n')
    for key, value in results_by_gen.items():
        trajectory_reward = 0
        # for each list, reverse iterate the list, and accumulate discounted reward for earlier turns
        for i, turn in enumerate(reversed(value)):
            correctness_reward = 0.3 if turn["correctness"] else 0.0
            speedup_reward = (turn["ref_runtime"] / turn["runtime"]) if turn["runtime"] > 0 and turn['ref_runtime'] > 0 else 0.0 # could be noisy
            step_reward = correctness_reward + speedup_reward
            trajectory_reward = step_reward + gamma * trajectory_reward
            turn["reward_items"] = {
                "correctness": correctness_reward,
                "speedup": speedup_reward,
                "step_reward": step_reward,
                "trajectory_reward": trajectory_reward,
            }
            turn['reward'] = trajectory_reward
            turn['turn_only_tag'] = turn['turn_tag'].split('_')[-1]
        # debug message prints reward for a trajectory
        if debug:
            print(key, '=>', [f'{turn["reward"]:.2f}' for turn in value])

    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of generations
    results_by_turn_only = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_only_tag","gen_tag")),
            key=itemgetter("task_tag","turn_only_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, turn_only_tag)\n')
    groups = {}
    for key, value in results_by_turn_only.items():
        # if all the rewards in the group for different gen_tag are 0, then ignore the group
        if all(gen["reward"] == 0.0 for gen in value):
            continue
        if len(value) == 1:
            # ignore single item groups
            continue
        # otherwise, create a group with prompt, logprobs, (including the task_tag and turn_only_tag)
        group = [{
            "task_tag": item["task_tag"],
            "gen_tag": item["gen_tag"],
            "turn_only_tag": item["turn_only_tag"],
            "turn_tag": item["turn_tag"],
            "reward": item["reward"],
            "reward_items": item["reward_items"],
            "prompt": item["prompt"],
            "logprobs": item["logprobs"],
            "logp_server_prompt_ids": item["prompt_ids"],
            "logp_server_completion_ids": item["completion_ids"],
            "logp_server_input_ids": item["input_ids"],
            "logp_server_logps": item["logps"],
            "runtime": item["runtime"],
            "checkpoint_name": item["metadata"]["model_override"].split("/")[-1] if "model_override" in item["metadata"] and item["metadata"]["model_override"] else None,
            "hint": item.get("hint", ""),  # Add hint from query_result if it exists
        } for item in value]
        # add the group to the groups dict
        groups[key] = group

        if debug:
            print(key, '=>', [f'{gen["reward"]:.2f}' for gen in value])

    return groups


def grpo_compute_rewards_v2(
    query_result: list[dict],
    gamma: float = 0.5,
    speedup_threahold: float = 1.8, # code-gen specific parameter
    improvement_bonus: float = 0.1, # code-gen specific parameter
    debug: bool = False,
) -> dict[str, list[dict]]:
    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of rows
    results_by_gen = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_tag")),
            key=itemgetter("task_tag","gen_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, gen_tag)\n')
    for key, value in results_by_gen.items():
        previous_max = 0.3
        previous_max_speedup = 0
        # for each list, iteration from first to last, and if current step_reward is better than previous best, give an extra reward
        for i, turn in enumerate(value):
            correctness_reward = 0.3 if turn["correctness"] else 0.0
            speedup_ = (turn["ref_runtime"] / turn["runtime"]) if turn["runtime"] > 0 and turn['ref_runtime'] > 0 else 0.0 # could be noisy
            if speedup_ >= speedup_threahold:
                speedup_reward = 0.3
            else:
                speedup_reward = 0
            step_reward = correctness_reward + speedup_reward
            if previous_max != -1 and (step_reward > previous_max or speedup_ > previous_max_speedup >= speedup_threahold): # reward incremental speed up
                previous_max = max(previous_max, step_reward)
                previous_max_speedup = max(previous_max_speedup, speedup_)
                step_reward += improvement_bonus
            trajectory_reward = step_reward
            turn["reward_items"] = {
                "correctness": correctness_reward,
                "speedup": speedup_reward,
                "step_reward": step_reward,
                "trajectory_reward": trajectory_reward,
            }
            turn['reward'] = trajectory_reward
            turn['turn_only_tag'] = turn['turn_tag'].split('_')[-1]
        # debug message prints reward for a trajectory
        if debug:
            print(key, '=>', [f'{turn["reward"]:.2f}' for turn in value])

    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of generations
    results_by_turn_only = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_only_tag","gen_tag")),
            key=itemgetter("task_tag","turn_only_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, turn_only_tag)\n')
    groups = {}
    for key, value in results_by_turn_only.items():
        # if all the rewards in the group for different gen_tag are 0, then ignore the group
        if all(gen["reward"] == 0.0 for gen in value):
            continue
        if len(value) == 1:
            # ignore single item groups
            continue
        # ignore the case when the reward contrast is not large enough, such the case all generations are full scores
        max_reward_ = max(gen["reward"] for gen in value)
        min_reward_ = min(gen["reward"] for gen in value)
        if (max_reward_ - min_reward_) < 0.01:
            continue
        # otherwise, create a group with prompt, logprobs, (including the task_tag and turn_only_tag)
        group = [{
            "task_tag": item["task_tag"],
            "gen_tag": item["gen_tag"],
            "turn_only_tag": item["turn_only_tag"],
            "turn_tag": item["turn_tag"],
            "reward": item["reward"],
            "reward_items": item["reward_items"],
            "prompt": item["prompt"],
            "logprobs": item["logprobs"],
            "logp_server_prompt_ids": item["prompt_ids"],
            "logp_server_completion_ids": item["completion_ids"],
            "logp_server_input_ids": item["input_ids"],
            "logp_server_logps": item["logps"],
            "runtime": item["runtime"],
            "checkpoint_name": item["metadata"]["model_override"].split("/")[-1] if "model_override" in item["metadata"] and item["metadata"]["model_override"] else None,
        } for item in value]
        # add the group to the groups dict
        groups[key] = group

        if debug:
            print(key, '=>', [f'{gen["reward"]:.2f}' for gen in value])

    return groups


def grpo_compute_rewards_v3(
    query_result: list[dict],
    gamma: float = 0.5,
    speedup_threahold: float = 1.3, # code-gen specific parameter
    improvement_bonus: float = 0.1, # code-gen specific parameter
    debug: bool = False,
) -> dict[str, list[dict]]:
    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of rows
    results_by_gen = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_tag")),
            key=itemgetter("task_tag","gen_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, gen_tag)\n')
    for key, value in results_by_gen.items():
        previous_max = 0.3
        previous_max_speedup = 9999
        # for each list, iteration from first to last, and if current step_reward is better than previous best, give an extra reward
        for i, turn in enumerate(value):
            correctness_reward = 0.3 if turn["correctness"] else 0.0
            speedup_test_result = speedup_threshold_alpha(turn["ref_runtime_stats"], turn["runtime_stats"], alpha=0.05) if turn["runtime"] > 0 and turn['ref_runtime'] > 0 else {} # could be noisy
            speedup_ = speedup_test_result["max_speedup_threshold"] if "max_speedup_threshold" in speedup_test_result else -1
            if speedup_ > 0:
                speedup_reward = min(0.3, speedup_ / speedup_threahold * 0.3)
            else:
                speedup_reward = 0
            step_reward = correctness_reward + speedup_reward
            if i > 0 and (step_reward > previous_max or speedup_ > previous_max_speedup * 1.2): # reward incremental speed up
                previous_max = max(previous_max, step_reward)
                previous_max_speedup = max(previous_max_speedup, speedup_)
                step_reward += improvement_bonus
            trajectory_reward = step_reward
            turn["reward_items"] = {
                "correctness": correctness_reward,
                "speedup": speedup_reward,
                "step_reward": step_reward,
                "trajectory_reward": trajectory_reward,
            }
            turn['reward'] = trajectory_reward
            turn['turn_only_tag'] = turn['turn_tag'].split('_')[-1]
        # debug message prints reward for a trajectory
        if debug:
            print(key, '=>', [f'{turn["reward"]:.2f}' for turn in value])

    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of generations
    results_by_turn_only = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_only_tag","gen_tag")),
            key=itemgetter("task_tag","turn_only_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, turn_only_tag)\n')
    groups = {}
    for key, value in results_by_turn_only.items():
        # if all the rewards in the group for different gen_tag are 0, then ignore the group
        if all(gen["reward"] == 0.0 for gen in value):
            continue
        if len(value) == 1:
            # ignore single item groups
            continue
        # ignore the case when the reward contrast is not large enough, such the case all generations are full scores
        max_reward_ = max(gen["reward"] for gen in value)
        min_reward_ = min(gen["reward"] for gen in value)
        if (max_reward_ - min_reward_) < 0.01:
            continue
        # otherwise, create a group with prompt, logprobs, (including the task_tag and turn_only_tag)
        group = [{
            "task_tag": item["task_tag"],
            "gen_tag": item["gen_tag"],
            "turn_only_tag": item["turn_only_tag"],
            "turn_tag": item["turn_tag"],
            "reward": item["reward"],
            "reward_items": item["reward_items"],
            "prompt": item["prompt"],
            "logprobs": item["logprobs"],
            "logp_server_prompt_ids": item["prompt_ids"],
            "logp_server_completion_ids": item["completion_ids"],
            "logp_server_input_ids": item["input_ids"],
            "logp_server_logps": item["logps"],
            "runtime": item["runtime"],
            "checkpoint_name": item["metadata"]["model_override"].split("/")[-1] if "model_override" in item["metadata"] and item["metadata"]["model_override"] else None,
        } for item in value]
        # add the group to the groups dict
        groups[key] = group

        if debug:
            print(key, '=>', [f'{gen["reward"]:.2f}' for gen in value])

    return groups


# V4 use non-linear continuous speedup reward to encourage more speedup
def grpo_compute_rewards_v4(
    query_result: list[dict],
    gamma: float = 0.5,
    speedup_threahold: float = 1.3, # code-gen specific parameter
    improvement_bonus: float = 0.1, # code-gen specific parameter
    debug: bool = False,
) -> dict[str, list[dict]]:
    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of rows
    results_by_gen = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_tag")),
            key=itemgetter("task_tag","gen_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, gen_tag)\n')
    for key, value in results_by_gen.items():
        previous_max_speedup = 0
        # for each list, iteration from first to last, and if current step_reward is better than previous best, give an extra reward
        for i, turn in enumerate(value):
            correctness_reward = 0.3 if turn["correctness"] else 0.0
            speedup_test_result = speedup_threshold_alpha(turn["ref_runtime_stats"], turn["runtime_stats"], alpha=0.05) if turn["runtime"] > 0 and turn['ref_runtime'] > 0 else {} # could be noisy
            speedup_ = speedup_test_result["max_speedup_threshold"] if "max_speedup_threshold" in speedup_test_result else -1
            if speedup_ > 0:
                speedup_reward = min(0.3, (speedup_ / speedup_threahold)**6 * 0.3)
            else:
                speedup_reward = 0
            step_reward = correctness_reward + speedup_reward
            if i > 0 and (previous_max_speedup >= speedup_threahold and speedup_ > previous_max_speedup * 1.5): # reward incremental speed up
                step_reward += improvement_bonus
            previous_max_speedup = max(previous_max_speedup, speedup_)
            trajectory_reward = step_reward
            turn["reward_items"] = {
                "correctness": correctness_reward,
                "speedup": speedup_reward,
                "step_reward": step_reward,
                "trajectory_reward": trajectory_reward,
            }
            turn['reward'] = trajectory_reward
            turn['turn_only_tag'] = turn['turn_tag'].split('_')[-1]
        # debug message prints reward for a trajectory
        if debug:
            print(key, '=>', [f'{turn["reward"]:.2f}' for turn in value])

    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of generations
    results_by_turn_only = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_only_tag","gen_tag")),
            key=itemgetter("task_tag","turn_only_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, turn_only_tag)\n')
    groups = {}
    for key, value in results_by_turn_only.items():
        # if all the rewards in the group for different gen_tag are 0, then ignore the group
        if all(gen["reward"] == 0.0 for gen in value):
            continue
        if len(value) == 1:
            # ignore single item groups
            continue
        # ignore the case when the reward contrast is not large enough, such the case all generations are full scores
        max_reward_ = max(gen["reward"] for gen in value)
        min_reward_ = min(gen["reward"] for gen in value)
        if (max_reward_ - min_reward_) < 0.01:
            continue
        # otherwise, create a group with prompt, logprobs, (including the task_tag and turn_only_tag)
        group = [{
            "task_tag": item["task_tag"],
            "gen_tag": item["gen_tag"],
            "turn_only_tag": item["turn_only_tag"],
            "turn_tag": item["turn_tag"],
            "reward": item["reward"],
            "reward_items": item["reward_items"],
            "prompt": item["prompt"],
            "logprobs": item["logprobs"],
            "logp_server_prompt_ids": item["prompt_ids"],
            "logp_server_completion_ids": item["completion_ids"],
            "logp_server_input_ids": item["input_ids"],
            "logp_server_logps": item["logps"],
            "runtime": item["runtime"],
            "checkpoint_name": item["metadata"]["model_override"].split("/")[-1] if "model_override" in item["metadata"] and item["metadata"]["model_override"] else None,
        } for item in value]
        # add the group to the groups dict
        groups[key] = group

        if debug:
            print(key, '=>', [f'{gen["reward"]:.2f}' for gen in value])

    return groups


def grpo_compute_rewards_v5(
    query_result: list[dict],
    gamma: float = 0.5,
    speedup_threahold: float = 1.3, # code-gen specific parameter
    improvement_bonus: float = 0.2, # code-gen specific parameter
    debug: bool = False,
) -> dict[str, list[dict]]:
    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of rows
    results_by_gen = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_tag")),
            key=itemgetter("task_tag","gen_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, gen_tag)\n')
    for key, value in results_by_gen.items():
        previous_max_speedup = 0
        had_improvement = False
        # for each list, iteration from first to last, and if current step_reward is better than previous best, give an extra reward
        for i, turn in enumerate(value):
            correctness_reward = 0.3 if turn["correctness"] else 0.0
            try: # in case the reference runtime is not available caused by the kb eval error
                speedup_test_result = speedup_threshold_alpha(turn["ref_runtime_stats"], turn["runtime_stats"], alpha=0.05) if turn["runtime"] > 0 and turn['ref_runtime'] > 0 else {} # could be noisy
                speedup_ = speedup_test_result["max_speedup_threshold"] if "max_speedup_threshold" in speedup_test_result else 0
            except:
                speedup_ = (turn["ref_runtime"] / turn["runtime"]) if turn["runtime"] > 0 and turn['ref_runtime'] > 0 else 0.0 # could be noisy
            speedup_reward = min(0.3, (speedup_ / speedup_threahold)**4 * 0.3)

            step_reward = correctness_reward + speedup_reward
            if i > 0 and (speedup_ > previous_max_speedup >= speedup_threahold) and had_improvement is False: # reward incremental speed up
                step_reward += improvement_bonus
                had_improvement = True
            previous_max_speedup = max(previous_max_speedup, speedup_)
            trajectory_reward = step_reward
            turn["reward_items"] = {
                "correctness": correctness_reward,
                "speedup": speedup_reward,
                "step_reward": step_reward,
                "trajectory_reward": trajectory_reward,
            }
            turn['reward'] = trajectory_reward
            turn['turn_only_tag'] = turn['turn_tag'].split('_')[-1]
        # debug message prints reward for a trajectory
        if debug:
            print(key, '=>', [f'{turn["reward"]:.2f}' for turn in value])

    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of generations
    results_by_turn_only = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_only_tag","gen_tag")),
            key=itemgetter("task_tag","turn_only_tag")
        )
    }
    if debug:
        print('\nProcessing trajectory (by task_tag, turn_only_tag)\n')
    groups = {}
    for key, value in results_by_turn_only.items():
        # if all the rewards in the group for different gen_tag are 0, then ignore the group
        if all(gen["reward"] == 0.0 for gen in value):
            continue
        if len(value) == 1:
            # ignore single item groups
            continue
        # ignore the case when the reward contrast is not large enough, such the case all generations are full scores
        max_reward_ = max(gen["reward"] for gen in value)
        min_reward_ = min(gen["reward"] for gen in value)
        if (max_reward_ - min_reward_) < 0.01:
            continue
        # otherwise, create a group with prompt, logprobs, (including the task_tag and turn_only_tag)
        group = [{
            "task_tag": item["task_tag"],
            "gen_tag": item["gen_tag"],
            "turn_only_tag": item["turn_only_tag"],
            "turn_tag": item["turn_tag"],
            "reward": item["reward"],
            "reward_items": item["reward_items"],
            "prompt": item["prompt"],
            "logprobs": item["logprobs"],
            "logp_server_prompt_ids": item["prompt_ids"],
            "logp_server_completion_ids": item["completion_ids"],
            "logp_server_input_ids": item["input_ids"],
            "logp_server_logps": item["logps"],
            "runtime": item["runtime"],
            "checkpoint_name": item["metadata"]["model_override"].split("/")[-1] if "model_override" in item["metadata"] and item["metadata"]["model_override"] else None,
        } for item in value]
        # add the group to the groups dict
        groups[key] = group

        if debug:
            print(key, '=>', [f'{gen["reward"]:.2f}' for gen in value])

    return groups



def grpo_compute_advantages(
    groups: dict[str, list[dict]],
    reward_scale: bool = True,
    reward_epsilon: float = 1e-3,
    reward_noise: float = 1e-2,
    debug: bool = False,
):
    """Compute advantages for the generated tokens"""
    if debug:
        print('\nComputing advantages\n')
    for key, group in groups.items():
        # calculate mean and stdev of the rewards
        rewards = np.array([result["reward"] for result in group])
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        # whether to scale the rewards
        if reward_scale:
            advantages = (rewards - mean_reward) / (std_reward + reward_epsilon)
        else:
            advantages = rewards - mean_reward
        # add noise to the advantages
        advantages = advantages + np.random.normal(0, reward_noise, size=advantages.shape)
        # add the advantages to the group
        for gen, advantage in zip(group, advantages):
            gen["advantage"] = advantage
        if debug:
            print(key, '=>', [f'{gen["advantage"]:.2f}' for gen in group])

    return groups


def grpo_group_to_dataset(
    group: list[dict],
    tokenizer: AutoTokenizer,
):
    from logger import logger

    group_dataset = []
    for result in group:
        vllm_prompt_ids = tokenizer.encode(result["prompt"])
        vllm_completion_ids = [logprob['token_id'] for logprob in result["logprobs"]]
        vllm_completion_log_probs = [logprob['logprob'] for logprob in result["logprobs"]]
        vllm_input_ids = torch.tensor(vllm_prompt_ids + vllm_completion_ids)
        vllm_attention_mask = torch.ones_like(vllm_input_ids)
        logp_server_prompt_ids = torch.tensor(result["logp_server_prompt_ids"])
        logp_server_completion_ids = torch.tensor(result["logp_server_completion_ids"])
        logp_server_input_ids = torch.tensor(result["logp_server_input_ids"])
        logp_server_attention_mask = torch.ones_like(logp_server_input_ids)
        logp_server_logps = result["logp_server_logps"]
        # check if the prompt ids length are different
        if len(vllm_prompt_ids) != len(logp_server_prompt_ids):
            logger.error(f"len(vllm_prompt_ids) [{len(vllm_prompt_ids)}] != len(logp_server_prompt_ids): [{len(logp_server_prompt_ids)}]")
            continue # skip the group if the prompt ids length are different
        # calculate the number of prompt ids that are different
        diff_count_prompt_ids = sum(1 for i, j in zip(vllm_prompt_ids, logp_server_prompt_ids) if i != j)
        if diff_count_prompt_ids > 0:
            logger.error(f"vllm_prompt_ids != logp_server_prompt_ids: [{diff_count_prompt_ids}/{len(vllm_prompt_ids)} tokens different]")
            continue # skip the group if the prompt ids are different
        # check if the completion ids length are different
        if len(vllm_completion_ids) != len(logp_server_completion_ids):
            logger.error(f"len(vllm_completion_ids) [{len(vllm_completion_ids)}] != len(logp_server_completion_ids): [{len(logp_server_completion_ids)}]")
            continue # skip the group if the completion ids length are different
        # calculate the number of completion ids that are different
        diff_count_completion_ids = sum(1 for i, j in zip(vllm_completion_ids, logp_server_completion_ids) if i != j)
        if diff_count_completion_ids > 0:
            logger.error(f"vllm_completion_ids != logp_server_completion_ids: [{diff_count_completion_ids}/{len(vllm_completion_ids)} tokens different]")
            continue # skip the group if the prompt ids are different
        # check if logps length are different
        if len(vllm_completion_log_probs) != len(logp_server_logps) - len(logp_server_prompt_ids) + 1:
            logger.error(f"len(vllm_completion_log_probs) [{len(vllm_completion_log_probs)}] != len(logp_server_logps) - len(logp_server_prompt_ids) + 1: [{len(logp_server_logps) - len(logp_server_prompt_ids) + 1}]")
            # logger.error(f"len(vllm_input_ids): [{len(vllm_input_ids)}], len(logp_server_input_ids): [{len(logp_server_input_ids)}]")
            # logger.error(f"len(vllm_completion_ids): [{len(vllm_completion_ids)}], len(logp_server_completion_ids): [{len(logp_server_completion_ids)}]")
            # logger.error(f"len(vllm_prompt_ids): [{len(vllm_prompt_ids)}], len(logp_server_prompt_ids): [{len(logp_server_prompt_ids)}]")
            continue # skip the group if the logps length are different
        # calculate the number of logps that are different
        # diff_count_logps = sum(1 for i, j in zip(vllm_completion_log_probs, logp_server_logps) if abs(i-j) / abs(i+j) > 1e-2 and abs(i-j) > 1e-2)
        # if diff_count_logps > 0:
        #     logger.warning(f"vllm_completion_log_probs != logp_server_logps: [{diff_count_logps}/{min_logps_len} tokens different]")
        group_dataset.append({
            'task_tag': result["task_tag"],
            'turn_tag': result["turn_tag"],
            'reward': result["reward"],
            'runtime': result["runtime"],
            'checkpoint_name': result["checkpoint_name"],
            'reward_items': result["reward_items"],
            'advantage': result["advantage"],
            'vllm_prompt_ids': vllm_prompt_ids,
            'vllm_completion_ids': vllm_completion_ids,
            'vllm_completion_log_probs': vllm_completion_log_probs,
            'vllm_input_ids': vllm_input_ids,
            'vllm_attention_mask': vllm_attention_mask,
            'logp_server_prompt_ids': logp_server_prompt_ids,
            'logp_server_completion_ids': logp_server_completion_ids,
            'logp_server_input_ids': logp_server_input_ids,
            'logp_server_logps': logp_server_logps,
            'logp_server_attention_mask': logp_server_attention_mask,
            'input_ids': logp_server_input_ids,
            'attention_mask': logp_server_attention_mask,
        })
    return group_dataset


def speedup_alpha(ref_dict, test_dict, speedup_threshold=1.3, alpha=0.05):
    """
    Test if test has a statistically significant speedup over reference.

    Speedup is defined as: speedup = test_mean / ref_mean

    H0: speedup <= speedup_threshold
    H1: speedup > speedup_threshold (test is faster by at least the threshold)

    Parameters:
    -----------
    ref_dict : dict
        Reference measurement: {'mean': float, 'std': float, 'min': float, 'max': float, 'num_trials': int}
    test_dict : dict
        Test measurement: {'mean': float, 'std': float, 'min': float, 'max': float, 'num_trials': int}
    speedup_threshold : float
        Minimum speedup ratio to test for (default=1.2 for 20% speedup)
    alpha : float
        Significance level (default=0.05)

    Returns:
    --------
    dict : {
        'has_speedup': bool,
        'observed_speedup': float,
        'p_value': float,
        't_statistic': float,
        'ci_lower': float,  # Lower bound of confidence interval for speedup
        'ci_upper': float   # Upper bound of confidence interval for speedup
    }
    """

    mean_ref = float(ref_dict['mean'])
    std_ref = float(ref_dict['std'])
    n_ref = int(ref_dict['num_trials'])

    mean_test = float(test_dict['mean'])
    std_test = float(test_dict['std'])
    n_test = float(test_dict['num_trials'])

    # Observed speedup
    observed_speedup = mean_test / mean_ref

    # We want to test if mean_test / mean_ref > speedup_threshold
    # Equivalently: mean_test > speedup_threshold * mean_ref
    # Or: mean_test - speedup_threshold * mean_ref > 0

    # Difference in means
    mean_diff = mean_test - speedup_threshold * mean_ref

    # Standard error of the difference
    # Var(aX - bY) = a²·Var(X) + b²·Var(Y) for independent X, Y
    # Here: Var(mean_test - threshold·mean_ref)
    var_diff = (std_test**2 / n_test) + (speedup_threshold**2 * std_ref**2 / n_ref)
    se_diff = math.sqrt(var_diff)

    # T-statistic
    t_statistic = mean_diff / se_diff

    # Degrees of freedom (Welch-Satterthwaite approximation)
    s_test_sq = std_test**2 / n_test
    s_ref_sq = std_ref**2 / n_ref
    df = ((s_test_sq + speedup_threshold**2 * s_ref_sq)**2 /
          (s_test_sq**2 / (n_test - 1) + (speedup_threshold**2 * s_ref_sq)**2 / (n_ref - 1)))

    # One-tailed p-value (testing if speedup > threshold)
    p_value = 1 - stats.t.cdf(t_statistic, df)

    # Has statistically significant speedup?
    has_speedup = p_value < alpha

    # Confidence interval for the speedup ratio
    # Using delta method approximation: Var(X/Y) ≈ (1/μ_Y)²·Var(X) + (μ_X/μ_Y²)²·Var(Y)
    var_ratio = (1/mean_ref)**2 * (std_test**2/n_test) + (mean_test/mean_ref**2)**2 * (std_ref**2/n_ref)
    se_ratio = math.sqrt(var_ratio)

    # CI for speedup (using normal approximation for large samples)
    z_critical = stats.norm.ppf(1 - alpha/2)  # Two-tailed for CI
    ci_lower = observed_speedup - z_critical * se_ratio
    ci_upper = observed_speedup + z_critical * se_ratio

    return {
        'has_speedup': has_speedup,
        'observed_speedup': observed_speedup,
        'speedup_threshold': speedup_threshold,
        'p_value': p_value,
        't_statistic': t_statistic,
        'degrees_of_freedom': df,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'alpha': alpha
    }


def speedup_threshold_alpha(ref_dict, test_dict, alpha=0.05):
    """
    Find the maximum speedup threshold that is statistically significant.

    This finds the largest speedup_threshold such that we can still reject:
    H0: speedup <= speedup_threshold at significance level alpha.

    In other words, this returns the lower bound of the confidence interval
    for the speedup ratio.

    Parameters:
    -----------
    ref_dict : dict
        Reference measurement: {'mean': float, 'std': float, 'min': float, 'max': float, 'num_trials': int}
    test_dict : dict
        Test measurement: {'mean': float, 'std': float, 'min': float, 'max': float, 'num_trials': int}
    alpha : float
        Significance level (default=0.05)

    Returns:
    --------
    dict : {
        'max_speedup_threshold': float,  # Maximum threshold that can be claimed
        'observed_speedup': float,       # Actual observed speedup
        'ci_lower': float,               # Lower bound of CI (same as max_speedup_threshold)
        'ci_upper': float                # Upper bound of CI
    }
    """

    mean_ref = float(ref_dict['mean'])
    std_ref = float(ref_dict['std'])
    n_ref = int(ref_dict['num_trials'])

    mean_test = float(test_dict['mean'])
    std_test = float(test_dict['std'])
    n_test = int(test_dict['num_trials'])

    # Observed speedup
    observed_speedup = mean_test / mean_ref

    # The maximum speedup threshold is the lower bound of the confidence interval
    # We need to find the threshold where p-value = alpha
    # This is equivalent to finding the lower bound of a one-sided CI

    def p_value_for_threshold(threshold):
        """Calculate p-value for a given threshold"""
        mean_diff = mean_test - threshold * mean_ref
        var_diff = (std_test**2 / n_test) + (threshold**2 * std_ref**2 / n_ref)
        se_diff = math.sqrt(var_diff)
        t_stat = mean_diff / se_diff

        # Degrees of freedom
        s_test_sq = std_test**2 / n_test
        s_ref_sq = std_ref**2 / n_ref
        df = ((s_test_sq + threshold**2 * s_ref_sq)**2 /
              (s_test_sq**2 / (n_test - 1) + (threshold**2 * s_ref_sq)**2 / (n_ref - 1)))

        return 1 - stats.t.cdf(t_stat, df)

    # Find the threshold where p-value = alpha using binary search
    # The maximum threshold is where p-value exactly equals alpha
    try:
        # Search between a reasonable range
        # Lower bound: some fraction of observed speedup
        # Upper bound: observed speedup (p-value = 0.5 at this point)
        lower_bound = max(0.01, observed_speedup * 0.5)
        upper_bound = observed_speedup

        # Find the threshold where p-value = alpha
        max_threshold = brentq(lambda t: p_value_for_threshold(t) - alpha,
                               lower_bound, upper_bound, xtol=1e-6)
    except ValueError:
        # If no solution found, use a fallback method
        max_threshold = observed_speedup * 0.9  # Conservative estimate

    # Calculate confidence interval for the speedup ratio
    var_ratio = (1/mean_ref)**2 * (std_test**2/n_test) + (mean_test/mean_ref**2)**2 * (std_ref**2/n_ref)
    se_ratio = math.sqrt(var_ratio)

    # One-sided CI: lower bound
    t_critical = stats.t.ppf(1 - alpha, n_test + n_ref - 2)
    ci_lower = observed_speedup - t_critical * se_ratio
    ci_lower = max(0, ci_lower)  # Can't be negative

    # Two-sided CI for reference
    t_critical_2sided = stats.t.ppf(1 - alpha/2, n_test + n_ref - 2)
    ci_upper = observed_speedup + t_critical_2sided * se_ratio

    return {
        'max_speedup_threshold': max_threshold,
        'observed_speedup': observed_speedup,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'alpha': alpha
    }


if __name__ == "__main__":
    import duckdb
    import argparse
    # add parent directory to path
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    sql = f"""
            WITH refs AS(
                SELECT
                    regexp_replace(filename, '/[^/]*$', '') AS task_id,
                    runtime_stats
                FROM read_json_auto('{args.input_dir}/**/reference_eval.json', filename = true)
            ),
            evals AS (
                SELECT
                    filename AS eval_file,
                    regexp_replace(filename, '_generated_eval\.json$', '') AS stem,
                    regexp_replace(filename, '/[^/]*$', '') AS task_id,
                    compiled, correctness, runtime, reference_runtime as ref_runtime, runtime_stats
                FROM read_json_auto('{args.input_dir}/**/*_generated_eval.json', filename = true)
            ),
            evals_join AS (
                SELECT e.*, r.runtime_stats as ref_runtime_stats
                FROM evals e
                LEFT JOIN refs r USING (task_id)
            ),
            comps AS (
                SELECT
                    filename AS completion_file,
                    regexp_replace(filename, '_completion\.json$', '') AS stem,
                    prompt, completion, logprobs, metadata,
                    metadata->>'task_tag' AS task_tag, metadata->>'gen_tag' AS gen_tag, metadata->>'turn_tag' AS turn_tag
                FROM read_json_auto('{args.input_dir}/**/*_completion.json', filename = true)
            ),
            logps AS (
                SELECT
                    filename AS logps_file,
                    regexp_replace(filename, '_logps\.json$', '') AS stem,
                    prompt_ids, completion_ids, input_ids, logps
                FROM read_json_auto('{args.input_dir}/**/*_logps.json', filename = true)
            )
            SELECT
                e.compiled, e.correctness, e.runtime, e.ref_runtime, e.runtime_stats, e.ref_runtime_stats,
                c.prompt, c.completion, c.logprobs, c.metadata,
                e.eval_file, c.completion_file,
                l.prompt_ids, l.completion_ids, l.input_ids, l.logps,
                c.task_tag, c.gen_tag, c.turn_tag
            FROM evals_join e
            JOIN comps c USING (stem)
            JOIN logps l USING (stem)
            ORDER BY task_tag, gen_tag, turn_tag
    """
    result = duckdb.sql(sql)
    print(result)

    groups = grpo_compute_rewards_v5(result.df().to_dict(orient="records"), debug=args.debug)
    groups = grpo_compute_advantages(groups, debug=args.debug)
    # for key, value in groups.items():
    #     print(f'\n{key}:')
    #     print('    => rewards: ', [f'{gen["reward"]:.2f}' for gen in value])
    #     print('    => advantages: ', [f'{gen["advantage"]:.2f}' for gen in value])
    #     print('    => len(logprobs): ', [f'{len(gen["logprobs"])}' for gen in value])

    group_datasets = {key: grpo_group_to_dataset(value, tokenizer) for key, value in groups.items()}
    idx = 0
    for key, value in group_datasets.items():
        idx += 1
        print(f'\n[{idx:02d}] {key}:')
        print('    => rewards: ', [f'{gen["reward"]:.2f}' for gen in value])
        print('    => advantages: ', [f'{gen["advantage"]:.2f}' for gen in value])
        print('    => runtime: ', [f'{gen["runtime"]:.2f}' for gen in value])
        print('    => checkpoint_number: ', [f'{gen["checkpoint_name"].split("-")[-1] if gen["checkpoint_name"] else None}' for gen in value if "checkpoint_name" in gen])
        print('    => len(vllm_prompt_ids): ', [f'{len(gen["vllm_prompt_ids"])}' for gen in value if gen["vllm_prompt_ids"]])
        print('    => len(vllm_completion_ids): ', [f'{len(gen["vllm_completion_ids"])}' for gen in value if gen["vllm_completion_ids"]])
        print('    => len(vllm_completion_log_probs): ', [f'{len(gen["vllm_completion_log_probs"])}' for gen in value if gen["vllm_completion_log_probs"]])
        print('    => len(vllm_input_ids): ', [f'{len(gen["vllm_input_ids"])}' for gen in value if "vllm_input_ids" in gen])
        print('    => len(vllm_attention_mask): ', [f'{len(gen["vllm_attention_mask"])}' for gen in value if "vllm_attention_mask" in gen])
        print('    => len(logp_server_prompt_ids): ', [f'{len(gen["logp_server_prompt_ids"])}' for gen in value if "logp_server_prompt_ids" in gen])
        print('    => len(logp_server_completion_ids): ', [f'{len(gen["logp_server_completion_ids"])}' for gen in value if "logp_server_completion_ids" in gen])
        print('    => len(logp_server_input_ids): ', [f'{len(gen["logp_server_input_ids"])}' for gen in value if "logp_server_input_ids" in gen])
        print('    => len(logp_server_logps): ', [f'{len(gen["logp_server_logps"])}' for gen in value if "logp_server_logps" in gen])
        print('    => len(logp_server_attention_mask): ', [f'{len(gen["logp_server_attention_mask"])}' for gen in value if "logp_server_attention_mask" in gen])


    # Test case 1: Clear speedup
    ref = {
        'mean': 100,
        'std': 10,
        'min': 80,
        'max': 120,
        'num_trials': 10
    }

    test = {
        'mean': 130,  # 30% faster
        'std': 12,
        'min': 110,
        'max': 150,
        'num_trials': 10
    }

    result1 = speedup_alpha(ref, test, speedup_threshold=1.2, alpha=0.05)

    print("="*70)
    print("Test Case 1: Clear Speedup")
    print("="*70)
    print(f"Reference: mean={ref['mean']}, std={ref['std']}, n={ref['num_trials']}")
    print(f"Test: mean={test['mean']}, std={test['std']}, n={test['num_trials']}")
    print(f"\nTesting for speedup > {result1['speedup_threshold']:.2f}x")
    print(f"Observed speedup: {result1['observed_speedup']:.3f}x")
    print(f"95% CI: [{result1['ci_lower']:.3f}, {result1['ci_upper']:.3f}]")
    print(f"T-statistic: {result1['t_statistic']:.4f}")
    print(f"P-value: {result1['p_value']:.4f}")
    print(f"Has significant speedup? {result1['has_speedup']}")

    # Test case 2: Marginal speedup
    ref2 = {
        'mean': 100,
        'std': 10,
        'min': 80,
        'max': 120,
        'num_trials': 10
    }

    test2 = {
        'mean': 115,  # 15% faster, but threshold is 20%
        'std': 12,
        'min': 95,
        'max': 135,
        'num_trials': 10
    }

    result2 = speedup_alpha(ref2, test2, speedup_threshold=1.2, alpha=0.05)

    print("\n" + "="*70)
    print("Test Case 2: Marginal Speedup (below threshold)")
    print("="*70)
    print(f"Reference: mean={ref2['mean']}, std={ref2['std']}, n={ref2['num_trials']}")
    print(f"Test: mean={test2['mean']}, std={test2['std']}, n={test2['num_trials']}")
    print(f"\nTesting for speedup > {result2['speedup_threshold']:.2f}x")
    print(f"Observed speedup: {result2['observed_speedup']:.3f}x")
    print(f"95% CI: [{result2['ci_lower']:.3f}, {result2['ci_upper']:.3f}]")
    print(f"T-statistic: {result2['t_statistic']:.4f}")
    print(f"P-value: {result2['p_value']:.4f}")
    print(f"Has significant speedup? {result2['has_speedup']}")

    # Test case 3: Large sample size
    ref3 = {
        'mean': 100,
        'std': 10,
        'min': 70,
        'max': 130,
        'num_trials': 100  # More trials
    }

    test3 = {
        'mean': 122,  # 22% faster
        'std': 12,
        'min': 90,
        'max': 150,
        'num_trials': 100  # More trials
    }

    result3 = speedup_alpha(ref3, test3, speedup_threshold=1.2, alpha=0.05)

    print("\n" + "="*70)
    print("Test Case 3: Larger Sample Size")
    print("="*70)
    print(f"Reference: mean={ref3['mean']}, std={ref3['std']}, n={ref3['num_trials']}")
    print(f"Test: mean={test3['mean']}, std={test3['std']}, n={test3['num_trials']}")
    print(f"\nTesting for speedup > {result3['speedup_threshold']:.2f}x")
    print(f"Observed speedup: {result3['observed_speedup']:.3f}x")
    print(f"95% CI: [{result3['ci_lower']:.3f}, {result3['ci_upper']:.3f}]")
    print(f"T-statistic: {result3['t_statistic']:.4f}")
    print(f"P-value: {result3['p_value']:.4f}")
    print(f"Has significant speedup? {result3['has_speedup']}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("The function tests if test has a statistically significant speedup")
    print("over reference by at least the specified threshold.")
    print("\nSpeedup = test_mean / ref_mean")
    print("H0: speedup <= threshold")
    print("H1: speedup > threshold")



    # Test the new function: speedup_threshold_alpha
    alpha = 0.05
    print("\n\n" + "="*70)
    print("NEW FUNCTION: speedup_threshold_alpha")
    print("="*70)

    # Example 1
    max_result1 = speedup_threshold_alpha(ref, test, alpha=0.05)
    print("\nExample 1:")
    print(f"Reference: mean={ref['mean']}, std={ref['std']}, n={ref['num_trials']}")
    print(f"Test: mean={test['mean']}, std={test['std']}, n={test['num_trials']}")
    print(f"\nObserved speedup: {max_result1['observed_speedup']:.3f}x")
    print(f"Maximum claimable speedup threshold: {max_result1['max_speedup_threshold']:.3f}x")
    print(f"95% CI: [{max_result1['ci_lower']:.3f}, {max_result1['ci_upper']:.3f}]")
    print(f"\nInterpretation: You can claim with 95% confidence that the speedup")
    print(f"is at least {max_result1['max_speedup_threshold']:.3f}x")

    # Verify with speedup_alpha
    verify1 = speedup_alpha(ref, test,
                           speedup_threshold=max_result1['max_speedup_threshold'],
                           alpha=0.05)
    print(f"\nVerification (should be borderline significant):")
    print(f"  P-value at max threshold: {verify1['p_value']:.4f} (should be ≈{alpha})")
    print(f"  Has speedup: {verify1['has_speedup']}")

    # Example 2 - marginal case
    max_result2 = speedup_threshold_alpha(ref2, test2, alpha=0.05)
    print("\n" + "-"*70)
    print("\nExample 2 (marginal speedup):")
    print(f"Reference: mean={ref2['mean']}, std={ref2['std']}, n={ref2['num_trials']}")
    print(f"Test: mean={test2['mean']}, std={test2['std']}, n={test2['num_trials']}")
    print(f"\nObserved speedup: {max_result2['observed_speedup']:.3f}x")
    print(f"Maximum claimable speedup threshold: {max_result2['max_speedup_threshold']:.3f}x")
    print(f"95% CI: [{max_result2['ci_lower']:.3f}, {max_result2['ci_upper']:.3f}]")

    # Example 3 - large sample
    max_result3 = speedup_threshold_alpha(ref3, test3, alpha=0.05)
    print("\n" + "-"*70)
    print("\nExample 3 (large sample size):")
    print(f"Reference: mean={ref3['mean']}, std={ref3['std']}, n={ref3['num_trials']}")
    print(f"Test: mean={test3['mean']}, std={test3['std']}, n={test3['num_trials']}")
    print(f"\nObserved speedup: {max_result3['observed_speedup']:.3f}x")
    print(f"Maximum claimable speedup threshold: {max_result3['max_speedup_threshold']:.3f}x")
    print(f"95% CI: [{max_result3['ci_lower']:.3f}, {max_result3['ci_upper']:.3f}]")
    print(f"\nNote: Larger sample size gives tighter CI, closer to observed speedup")
