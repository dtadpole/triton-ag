import torch
import numpy as np
from transformers import AutoTokenizer
from operator import itemgetter
from itertools import groupby
import scipy.stats as stats
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
        } for item in value]
        # add the group to the groups dict
        groups[key] = group

        if debug:
            print(key, '=>', [f'{gen["reward"]:.2f}' for gen in value])

    return groups


def grpo_compute_rewards_v2(
    query_result: list[dict],
    gamma: float = 0.5,
    speedup_threahold: float = 1.8,
    improvement_bonus: float = 0.1,
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
    speedup_threahold: float = 1.3,
    improvement_bonus: float = 0.1,
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
                speedup_reward = 0.3 * (speedup_ / speedup_threahold) * gamma # give partial credit
            step_reward = correctness_reward + speedup_reward
            if previous_max != -1 and (step_reward > previous_max or speedup_ > previous_max_speedup * 1.2): # reward incremental speed up
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


def is_mean_larger(mean, std, min_val, max_val, num_trials, fixed_value, alpha=0.05):
    """
    Performs a one-sample t-test to determine if the sample mean is
    statistically significantly larger than a fixed value.

    Parameters:
    -----------
    mean : float
        Sample mean
    std : float
        Sample standard deviation
    min_val : float
        Minimum value in sample (not used in calculation, for reference)
    max_val : float
        Maximum value in sample (not used in calculation, for reference)
    num_trials : int
        Number of samples/trials
    fixed_value : float
        The hypothesized population mean to test against
    alpha : float, optional
        Significance level (default=0.05)

    Returns:
    --------
    dict : Dictionary containing:
        - 'is_larger': bool, True if mean is statistically larger
        - 't_statistic': float, the calculated t-statistic
        - 'p_value': float, one-tailed p-value
        - 'critical_value': float, critical t-value at given alpha
        - 'effect_size': float, Cohen's d effect size
    """

    # Calculate t-statistic
    # t = (sample_mean - hypothesized_mean) / (std_error)
    # where std_error = std / sqrt(n)
    std_error = std / math.sqrt(num_trials)
    t_statistic = (mean - fixed_value) / std_error

    # Degrees of freedom
    df = num_trials - 1

    # One-tailed p-value (testing if mean > fixed_value)
    p_value = 1 - stats.t.cdf(t_statistic, df)

    # Critical value for one-tailed test
    critical_value = stats.t.ppf(1 - alpha, df)

    # Effect size (Cohen's d)
    cohens_d = (mean - fixed_value) / std

    # Determine if statistically significant
    is_larger = p_value < alpha

    return {
        'is_larger': is_larger,
        't_statistic': t_statistic,
        'p_value': p_value,
        'critical_value': critical_value,
        'effect_size': cohens_d,
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
        WITH evals AS (
        SELECT
            filename AS eval_file,
            regexp_replace(filename, '_generated_eval\\.json$', '') AS stem,
            compiled, correctness, runtime, reference_runtime as ref_runtime
        FROM read_json_auto('{args.input_dir}/**/*_generated_eval.json', filename = true)
        ),
        comps AS (
        SELECT
            filename AS completion_file,
            regexp_replace(filename, '_completion\\.json$', '') AS stem,
            prompt, completion, logprobs, metadata,
            metadata->>'task_tag' AS task_tag, metadata->>'gen_tag' AS gen_tag, metadata->>'turn_tag' AS turn_tag
        FROM read_json_auto('{args.input_dir}/**/*_completion.json', filename = true)
        ),
        logps AS (
            SELECT
                filename AS logps_file,
                regexp_replace(filename, '_logps\\.json$', '') AS stem,
                prompt_ids, completion_ids, input_ids, logps
            FROM read_json_auto('{args.input_dir}/**/*_logps.json', filename = true)
        )
        SELECT
            e.compiled, e.correctness, e.runtime, e.ref_runtime,
            c.prompt, c.completion, c.logprobs, c.metadata,
            e.eval_file, c.completion_file,
            l.prompt_ids, l.completion_ids, l.input_ids, l.logps,
            c.task_tag, c.gen_tag, c.turn_tag
        FROM evals e
        JOIN comps c USING (stem)
        JOIN logps l USING (stem)
        ORDER BY task_tag, gen_tag, turn_tag
    """
    result = duckdb.sql(sql)
    print(result)

    groups = grpo_compute_rewards(result.df().to_dict(orient="records"), debug=args.debug)
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

    result = is_mean_larger(
        mean=105,
        std=10,
        min_val=85,
        max_val=125,
        num_trials=10,
        fixed_value=100,
        alpha=0.05
    )
    print(f"Is mean statistically larger? {result['is_larger']}")
    print(f"T-statistic: {result['t_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Critical value: {result['critical_value']:.4f}")
    print(f"Effect size (Cohen's d): {result['effect_size']:.4f}")
