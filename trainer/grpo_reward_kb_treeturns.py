"""
GRPO reward calculation for tree-based multi-turn conversations.
This version is specifically for scenarios where selection files exist.
"""

import math
from itertools import groupby
from operator import itemgetter

import numpy as np
import scipy.stats as stats
import torch
from scipy.optimize import brentq
from transformers import AutoTokenizer


def grpo_compute_rewards_v5_treeturns(
    query_result: list[dict],
    gamma: float = 0.5,
    speedup_threahold: float = 1.3,  # code-gen specific parameter
    improvement_bonus: float = 0.2,  # code-gen specific parameter
    debug: bool = False,
) -> dict[str, list[dict]]:
    """
    Compute rewards for tree-based multi-turn conversations.

    The improvement bonus is based on the selected conversation from the previous turn,
    not comparing within the same generation trajectory.

    For turn 0: No improvement bonus (first turn)
    For turn > 0: Compare against selected conversation's speedup from previous turn
    """
    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of rows
    results_by_gen = {
        k: list(g)
        for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag", "turn_tag")),
            key=itemgetter("task_tag", "gen_tag"),
        )
    }
    if debug:
        print("\nProcessing trajectory (by task_tag, gen_tag) - Tree Turns Mode\n")
    for key, value in results_by_gen.items():
        had_improvement = False
        # for each list, iteration from first to last, and if current step_reward is better than previous best, give an extra reward
        for i, turn in enumerate(value):
            correctness_reward = 0.3 if turn["correctness"] else 0.0
            try:  # in case the reference runtime is not available caused by the kb eval error
                speedup_test_result = (
                    speedup_threshold_alpha(
                        turn["ref_runtime_stats"], turn["runtime_stats"], alpha=0.05
                    )
                    if turn["runtime"] > 0 and turn["ref_runtime"] > 0
                    else {}
                )  # could be noisy
                speedup_ = (
                    speedup_test_result["max_speedup_threshold"]
                    if "max_speedup_threshold" in speedup_test_result
                    else 0
                )
            except:
                speedup_ = (
                    (turn["ref_runtime"] / turn["runtime"])
                    if turn["runtime"] > 0 and turn["ref_runtime"] > 0
                    else 0.0
                )  # could be noisy
            speedup_reward = min(0.3, (speedup_ / speedup_threahold) ** 4 * 0.3)

            step_reward = correctness_reward + speedup_reward

            # Calculate improvement bonus based on selected conversation from previous turn
            # IMPORTANT: Only award improvement bonus if selection file exists for this turn
            # If selection file is missing, the turn must start from the beginning (no tree structure)
            if i > 0 and had_improvement is False:
                # Check if selection data exists - if not, no improvement bonus possible
                # This handles the case where the selection file is missing for a turn
                has_selection_data = (
                    turn.get("selected_turn_tag") is not None
                    and turn.get("selected_runtime_stats") is not None
                )

                if not has_selection_data:
                    # No selection file for this turn - cannot award improvement bonus
                    # The generation must start from the beginning
                    if debug:
                        print(f"  [Turn {i}] No selection data - skipping improvement bonus")
                else:
                    # Get the selected conversation's speedup for comparison
                    selected_speedup = 0
                    if turn.get("ref_runtime_stats"):
                        try:
                            selected_speedup_test_result = speedup_threshold_alpha(
                                turn["ref_runtime_stats"],
                                turn["selected_runtime_stats"],
                                alpha=0.05,
                            )
                            selected_speedup = selected_speedup_test_result.get(
                                "max_speedup_threshold", 0
                            )
                        except:
                            # Fallback to simple speedup calculation
                            if (
                                turn.get("selected_runtime")
                                and turn["selected_runtime"] > 0
                                and turn.get("ref_runtime")
                                and turn["ref_runtime"] > 0
                            ):
                                selected_speedup = (
                                    turn["ref_runtime"] / turn["selected_runtime"]
                                )

                    # Award improvement bonus if current speedup is better than selected speedup
                    if speedup_ > selected_speedup >= speedup_threahold:
                        step_reward += improvement_bonus
                        had_improvement = True
                        if debug:
                            print(f"  [Turn {i}] Improvement bonus awarded: speedup {speedup_:.2f} > selected {selected_speedup:.2f}")

            trajectory_reward = step_reward
            turn["reward_items"] = {
                "correctness": correctness_reward,
                "speedup": speedup_reward,
                "step_reward": step_reward,
                "trajectory_reward": trajectory_reward,
            }
            turn["reward"] = trajectory_reward
            turn["turn_only_tag"] = turn["turn_tag"].split("_")[-1]
        # debug message prints reward for a trajectory
        if debug:
            print(key, "=>", [f'{turn["reward"]:.2f}' for turn in value])

    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of generations
    results_by_turn_only = {
        k: list(g)
        for k, g in groupby(
            sorted(
                query_result, key=itemgetter("task_tag", "turn_only_tag", "gen_tag")
            ),
            key=itemgetter("task_tag", "turn_only_tag"),
        )
    }
    if debug:
        print("\nProcessing trajectory (by task_tag, turn_only_tag)\n")
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
        group = [
            {
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
                "checkpoint_name": (
                    item["metadata"]["model_override"].split("/")[-1]
                    if "model_override" in item["metadata"]
                    and item["metadata"]["model_override"]
                    else None
                ),
            }
            for item in value
        ]
        # add the group to the groups dict
        groups[key] = group

        if debug:
            print(key, "=>", [f'{gen["reward"]:.2f}' for gen in value])

    return groups


def grpo_compute_advantages(
    groups: dict[str, list[dict]],
    reward_scale: bool = True,
    reward_epsilon: float = 1e-3,
    reward_noise: float = 1e-2,
    debug: bool = False,
    weight_more_max_reward: bool = False,  # weight more max reward
    weight_more_max_reward_scale: float = 1.0,  # weight more max reward scale
):
    """Compute advantages for the generated tokens"""
    if debug:
        print("\nComputing advantages\n")
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

        if weight_more_max_reward:
            argmax_index = np.where(rewards == max(rewards))
            advantages[argmax_index] *= weight_more_max_reward_scale

        # add noise to the advantages
        advantages = advantages + np.random.normal(
            0, reward_noise, size=advantages.shape
        )
        # add the advantages to the group
        for gen, advantage in zip(group, advantages):
            gen["advantage"] = advantage
        if debug:
            print(key, "=>", [f'{gen["advantage"]:.2f}' for gen in group])

    return groups


def grpo_group_to_dataset(
    group: list[dict],
    tokenizer: AutoTokenizer,
):
    from logger import logger

    group_dataset = []
    for result in group:
        vllm_prompt_ids = tokenizer.encode(result["prompt"])
        vllm_completion_ids = [logprob["token_id"] for logprob in result["logprobs"]]
        vllm_completion_log_probs = [
            logprob["logprob"] for logprob in result["logprobs"]
        ]
        vllm_input_ids = torch.tensor(vllm_prompt_ids + vllm_completion_ids)
        vllm_attention_mask = torch.ones_like(vllm_input_ids)
        logp_server_prompt_ids = torch.tensor(result["logp_server_prompt_ids"])
        logp_server_completion_ids = torch.tensor(result["logp_server_completion_ids"])
        logp_server_input_ids = torch.tensor(result["logp_server_input_ids"])
        logp_server_attention_mask = torch.ones_like(logp_server_input_ids)
        logp_server_logps = result["logp_server_logps"]
        # check if the prompt ids length are different
        if len(vllm_prompt_ids) != len(logp_server_prompt_ids):
            logger.error(
                f"len(vllm_prompt_ids) [{len(vllm_prompt_ids)}] != len(logp_server_prompt_ids): [{len(logp_server_prompt_ids)}]"
            )
            continue  # skip the group if the prompt ids length are different
        # calculate the number of prompt ids that are different
        diff_count_prompt_ids = sum(
            1 for i, j in zip(vllm_prompt_ids, logp_server_prompt_ids) if i != j
        )
        if diff_count_prompt_ids > 0:
            logger.error(
                f"vllm_prompt_ids != logp_server_prompt_ids: [{diff_count_prompt_ids}/{len(vllm_prompt_ids)} tokens different]"
            )
            continue  # skip the group if the prompt ids are different
        # check if the completion ids length are different
        if len(vllm_completion_ids) != len(logp_server_completion_ids):
            logger.error(
                f"len(vllm_completion_ids) [{len(vllm_completion_ids)}] != len(logp_server_completion_ids): [{len(logp_server_completion_ids)}]"
            )
            continue  # skip the group if the completion ids length are different
        # calculate the number of completion ids that are different
        diff_count_completion_ids = sum(
            1 for i, j in zip(vllm_completion_ids, logp_server_completion_ids) if i != j
        )
        if diff_count_completion_ids > 0:
            logger.error(
                f"vllm_completion_ids != logp_server_completion_ids: [{diff_count_completion_ids}/{len(vllm_completion_ids)} tokens different]"
            )
            continue  # skip the group if the prompt ids are different
        # check if logps length are different
        if (
            len(vllm_completion_log_probs)
            != len(logp_server_logps) - len(logp_server_prompt_ids) + 1
        ):
            logger.error(
                f"len(vllm_completion_log_probs) [{len(vllm_completion_log_probs)}] != len(logp_server_logps) - len(logp_server_prompt_ids) + 1: [{len(logp_server_logps) - len(logp_server_prompt_ids) + 1}]"
            )
            continue  # skip the group if the logps length are different
        group_dataset.append(
            {
                "task_tag": result["task_tag"],
                "turn_tag": result["turn_tag"],
                "reward": result["reward"],
                "runtime": result["runtime"],
                "checkpoint_name": result["checkpoint_name"],
                "reward_items": result["reward_items"],
                "advantage": result["advantage"],
                "vllm_prompt_ids": vllm_prompt_ids,
                "vllm_completion_ids": vllm_completion_ids,
                "vllm_completion_log_probs": vllm_completion_log_probs,
                "vllm_input_ids": vllm_input_ids,
                "vllm_attention_mask": vllm_attention_mask,
                "logp_server_prompt_ids": logp_server_prompt_ids,
                "logp_server_completion_ids": logp_server_completion_ids,
                "logp_server_input_ids": logp_server_input_ids,
                "logp_server_logps": logp_server_logps,
                "logp_server_attention_mask": logp_server_attention_mask,
                "input_ids": logp_server_input_ids,
                "attention_mask": logp_server_attention_mask,
            }
        )
    return group_dataset


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

    mean_ref = float(ref_dict["mean"])
    std_ref = float(ref_dict["std"])
    n_ref = int(ref_dict["num_trials"])

    mean_test = float(test_dict["mean"])
    std_test = float(test_dict["std"])
    n_test = int(test_dict["num_trials"])

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
        df = (s_test_sq + threshold**2 * s_ref_sq) ** 2 / (
            s_test_sq**2 / (n_test - 1) + (threshold**2 * s_ref_sq) ** 2 / (n_ref - 1)
        )

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
        max_threshold = brentq(
            lambda t: p_value_for_threshold(t) - alpha,
            lower_bound,
            upper_bound,
            xtol=1e-6,
        )
    except ValueError:
        # If no solution found, use a fallback method
        max_threshold = observed_speedup * 0.9  # Conservative estimate

    # Calculate confidence interval for the speedup ratio
    var_ratio = (1 / mean_ref) ** 2 * (std_test**2 / n_test) + (
        mean_test / mean_ref**2
    ) ** 2 * (std_ref**2 / n_ref)
    se_ratio = math.sqrt(var_ratio)

    # One-sided CI: lower bound
    t_critical = stats.t.ppf(1 - alpha, n_test + n_ref - 2)
    ci_lower = observed_speedup - t_critical * se_ratio
    ci_lower = max(0, ci_lower)  # Can't be negative

    # Two-sided CI for reference
    t_critical_2sided = stats.t.ppf(1 - alpha / 2, n_test + n_ref - 2)
    ci_upper = observed_speedup + t_critical_2sided * se_ratio

    return {
        "max_speedup_threshold": max_threshold,
        "observed_speedup": observed_speedup,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
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
    parser.add_argument("--input_tag", type=str, required=True, help="Input tag (e.g., cudacoder_treeturns_gspo_qwen32b.t01_000_04)")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"\n{'='*70}")
    print(f"GRPO Tree Turns Test")
    print(f"{'='*70}")
    print(f"Input directory: {args.input_dir}")
    print(f"Input tag: {args.input_tag}")
    print(f"Tokenizer: {args.tokenizer_name}")
    print(f"Debug mode: {args.debug}")
    print(f"{'='*70}\n")

    input_dir = args.input_dir
    input_tag = args.input_tag
    sql = f"""
            WITH refs AS(
    SELECT
        regexp_replace(filename, '/[^/]*$', '') AS task_id,
        runtime_stats
    FROM read_json_auto('{input_dir}/{input_tag}/**/reference_eval.json', filename = true)
),
evals AS (
    SELECT
        filename AS eval_file,
        regexp_replace(filename, '_generated_eval\.json$', '') AS stem,
        regexp_replace(filename, '/[^/]*$', '') AS task_id,
        compiled, correctness, runtime, reference_runtime as ref_runtime, runtime_stats
    FROM read_json_auto('{input_dir}/{input_tag}/**/*_generated_eval.json', filename = true)
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
    FROM read_json_auto('{input_dir}/{input_tag}/**/*_completion.json', filename = true)
),
logps AS (
    SELECT
        filename AS logps_file,
        regexp_replace(filename, '_logps\.json$', '') AS stem,
        prompt_ids, completion_ids, input_ids, logps
    FROM read_json_auto('{input_dir}/{input_tag}/**/*_logps.json', filename = true)
),
selections AS (
    SELECT
        regexp_replace(filename, '/[^/]*$', '') AS task_id,
        regexp_replace(filename, '.*/t(\d+)_selected_from_t(\d+)\.json$', '\1') AS current_turn,
        turn_tag AS selected_turn_tag,
        generated_eval_path
    FROM read_json_auto('{input_dir}/{input_tag}/**/*_selected_from_*.json', filename = true)
),
selected_evals AS (
    SELECT
        s.task_id,
        s.current_turn,
        s.selected_turn_tag,
        e.runtime AS selected_runtime,
        e.runtime_stats AS selected_runtime_stats,
        e.correctness AS selected_correctness
    FROM selections s
    LEFT JOIN evals e
    ON regexp_extract(s.generated_eval_path, 'shared(.*)$', 1)  = regexp_extract(e.eval_file, 'shared(.*)$', 1)
)
SELECT
    e.compiled, e.correctness, e.runtime, e.ref_runtime, e.runtime_stats, e.ref_runtime_stats,
    c.prompt, c.completion, c.logprobs, c.metadata,
    e.eval_file, c.completion_file,
    l.prompt_ids, l.completion_ids, l.input_ids, l.logps,
    c.task_tag, c.gen_tag, c.turn_tag,
    se.selected_turn_tag, se.selected_runtime, se.selected_runtime_stats, se.selected_correctness
FROM evals_join e
JOIN comps c USING (stem)
JOIN logps l USING (stem)
LEFT JOIN selected_evals se ON (e.task_id = se.task_id AND regexp_replace(c.turn_tag, '.*_(t\d+)$', '\1') = se.current_turn)
ORDER BY task_tag, gen_tag, turn_tag
    """

    print("Executing SQL query...")
    result = duckdb.sql(sql)
    print(f"Query returned {len(result.df())} rows\n")

    # Print the full DataFrame to eyeball the results
    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    print("\nFull DataFrame (first 100 rows):")
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 20, 'display.width', 150):
        print(result.df().head(100))

    # Show sample of data
    df = result.df()
    print(f"Sample columns: {list(df.columns)[:10]}...")
    print(f"Number of unique tasks: {df['task_tag'].nunique()}")
    print(f"Number of unique generations: {df['gen_tag'].nunique()}")
    print(f"Number of unique turns: {df['turn_tag'].nunique()}")

    # Check if selection data exists
    has_selection_data = df['selected_turn_tag'].notna().any()
    print(f"\nSelection data found: {has_selection_data}")
    if has_selection_data:
        print(f"  - Rows with selection data: {df['selected_turn_tag'].notna().sum()}")
        print(f"  - Sample selected turn tags: {df['selected_turn_tag'].dropna().unique()[:5]}")
    print()

    print("Computing rewards (Tree Turns Mode)...")
    groups = grpo_compute_rewards_v5_treeturns(
        result.df().to_dict(orient="records"), debug=args.debug
    )
    print(f"Generated {len(groups)} groups\n")

    print("Computing advantages...")
    groups = grpo_compute_advantages(groups, debug=args.debug)

    print("Creating datasets...")
    group_datasets = {
        key: grpo_group_to_dataset(value, tokenizer) for key, value in groups.items()
    }

    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"{'='*70}\n")

    idx = 0
    for key, value in group_datasets.items():
        idx += 1
        print(f"\n[{idx:02d}] {key}:")
        print("    => rewards: ", [f'{gen["reward"]:.2f}' for gen in value])
        print("    => advantages: ", [f'{gen["advantage"]:.2f}' for gen in value])
        print("    => runtime: ", [f'{gen["runtime"]:.2f}' for gen in value])
        print("    => reward_items: ", [gen["reward_items"] for gen in value])
        print(
            "    => checkpoint_number: ",
            [
                f'{gen["checkpoint_name"].split("-")[-1] if gen["checkpoint_name"] else None}'
                for gen in value
                if "checkpoint_name" in gen
            ],
        )
        print(
            "    => len(vllm_prompt_ids): ",
            [
                f'{len(gen["vllm_prompt_ids"])}'
                for gen in value
                if gen["vllm_prompt_ids"]
            ],
        )
        print(
            "    => len(vllm_completion_ids): ",
            [
                f'{len(gen["vllm_completion_ids"])}'
                for gen in value
                if gen["vllm_completion_ids"]
            ],
        )
        print(
            "    => len(logp_server_input_ids): ",
            [
                f'{len(gen["logp_server_input_ids"])}'
                for gen in value
                if "logp_server_input_ids" in gen
            ],
        )

    print(f"\n{'='*70}")
    print(f"Test Complete - Total groups: {len(group_datasets)}")
    print(f"{'='*70}\n")
