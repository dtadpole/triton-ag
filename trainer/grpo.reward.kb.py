import pandas as pd
from transformers import AutoTokenizer
from operator import itemgetter
from itertools import groupby

def grpo_kb_reward(query_result: list[dict], gamma: float = 0.5, tokenizer: AutoTokenizer = None, debug: bool = False) -> dict[str, list[dict]]:
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
        # otherwise, create a group with prompt, logprobs, (including the task_tag and turn_only_tag)
        group = [{
            "task_tag": item["task_tag"],
            "gen_tag": item["gen_tag"],
            "turn_only_tag": item["turn_only_tag"],
            "reward": item["reward"],
            "reward_items": item["reward_items"],
            "prompt": item["prompt"],
            "prompt_ids": tokenizer.encode(item["prompt"]) if tokenizer else None,
            "logprobs": item["logprobs"],
        } for item in value]
        # add the group to the groups dict
        groups[key] = group

        if debug:
            print(key, '=>', [f'{gen["reward"]:.2f}' for gen in value])

    return groups


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

    groups = grpo_kb_reward(result.df().to_dict(orient="records"), tokenizer=tokenizer, debug=args.debug)
    for key, value in groups.items():
        print(f'\n{key}:')
        print('    => rewards: ', [f'{gen["reward"]:.2f}' for gen in value])
        print('    => len(logprobs): ', [f'{len(gen["logprobs"])}' for gen in value])
        print('    => len(prompt_ids): ', [f'{len(gen["prompt_ids"])}' for gen in value if gen["prompt_ids"]])
