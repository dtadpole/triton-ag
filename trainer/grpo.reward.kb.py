import torch
import numpy as np
from transformers import AutoTokenizer
from operator import itemgetter
from itertools import groupby

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
            speedup_reward = (turn["ref_runtime"] / turn["runtime"]) if turn["runtime"] > 0 and turn['ref_runtime'] > 0 else 0.0
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
    group_dataset = []
    for result in group:
        prompt_token_ids = tokenizer.encode(result["prompt"])
        completion_token_ids = [logprob['token_id'] for logprob in result["logprobs"]]
        completion_log_probs = [logprob['logprob'] for logprob in result["logprobs"]]
        input_ids = torch.tensor(prompt_token_ids + completion_token_ids)
        attention_mask = torch.ones_like(input_ids)
        group_dataset.append({
            'task_tag': result["task_tag"],
            'turn_tag': result["turn_tag"],
            'reward': result["reward"],
            'reward_items': result["reward_items"],
            'advantage': result["advantage"],
            'prompt_token_ids': prompt_token_ids,
            'completion_token_ids': completion_token_ids,
            'completion_log_probs': completion_log_probs,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }) 
    return group_dataset


if __name__ == "__main__":
    import duckdb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--input_dir", type=str, default="~/.codeGenEval/TC_0.1.0_14B.m_005_05/")
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
        )
        SELECT
            e.compiled, e.correctness, e.runtime, e.ref_runtime,
            c.prompt, c.completion, c.logprobs, c.metadata,
            e.eval_file, c.completion_file,
            c.task_tag, c.gen_tag, c.turn_tag
        FROM evals e
        JOIN comps c USING (stem)
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
    for key, value in group_datasets.items():
        print(f'\n{key}:')
        print('    => rewards: ', [f'{gen["reward"]:.2f}' for gen in value])
        print('    => advantages: ', [f'{gen["advantage"]:.2f}' for gen in value])
        print('    => len(prompt_token_ids): ', [f'{len(gen["prompt_token_ids"])}' for gen in value if gen["prompt_token_ids"]])
        print('    => len(completion_token_ids): ', [f'{len(gen["completion_token_ids"])}' for gen in value if gen["completion_token_ids"]])
        print('    => len(completion_log_probs): ', [f'{len(gen["completion_log_probs"])}' for gen in value if gen["completion_log_probs"]])
        print('    => len(input_ids): ', [f'{len(gen["input_ids"])}' for gen in value if "input_ids" in gen])
        print('    => len(attention_mask): ', [f'{len(gen["attention_mask"])}' for gen in value if "attention_mask" in gen])
