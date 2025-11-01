import pandas as pd
import asyncio
import numpy as np
import os
import json
from trainerUtil import softmax_temperature_sampling
from util import TRAINER_DIR


def read_json(json_path):
    with open(json_path) as f:
        res = json.load(f)
    return res


def simple_reward(result_line, speedup_threahold=1.3):
    reward = 0
    reward += 0.3 if result_line["correctness"] else 0

    if "ref_runtime" in result_line and "runtime" in result_line:
        speedup_ = (result_line["ref_runtime"] / result_line["runtime"]) \
                        if result_line["runtime"] > 0 and result_line['ref_runtime'] > 0 else 0.0
    elif "speedup" in result_line:
        speedup_ = result_line["speedup"]
    else:
        speedup_ = 0
    if speedup_ > 0:
        reward += min(0.3, (speedup_ / speedup_threahold)**4 * 0.3) ## Non-linear speed up reward
    return reward


def sample_based_on_reward(query_result,
                           block,
                           model_tag,
                           stats_dir=TRAINER_DIR + "/stats",
                           speedup_threahold=1.3,
                           recent_n_logs=100,
                           reward_average_n=5,
                           temperature=1.0,
                           min_exploration=0.04):
    """
    Sample based on the reward the model received along the way.
    The more reward the model received, the easier it is considered and less likely to be sampled
    there is a min_exploration buget to balance the exploration
    """
    all_tasks_df = query_result.df()
    all_tasks_df["task_tag"] = all_tasks_df["kb_filename"].apply(lambda x: '_'.join('_'.join(x.split('/')[-2:]).split('_')[:5]))
    prefix_tag = block.prefix_tag
    n_samples = block.num_samples
    category = "generated"

    rewards = []
    for task_tag in all_tasks_df["task_tag"].values:
        file_path = os.path.join(
                stats_dir,
                prefix_tag,
                f"{model_tag}",
                f"{task_tag}",
                f"{category}.jsonl"
            )
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            rewards_ = []
            for line in lines[-recent_n_logs:]:
                data = json.loads(line)
                if data is None:
                    logger.warning(f"[{os.getpid()}] Ignore null JSON line in [{file_path}]: [{line}]")
                    continue
                rewards_.append(simple_reward(data))
            rewards.append(np.mean(sorted(rewards_)[-reward_average_n:]))
        else:
            rewards.append(0)
    all_tasks_df["reward"] = rewards
    sampled_df = softmax_temperature_sampling(all_tasks_df,
                                              all_tasks_df,
                                              task_id_col='task_tag',
                                              reward_col='reward',
                                              temperature=temperature,
                                              min_exploration=min_exploration,
                                              n_samples=n_samples,
                                              )
    return sampled_df


def sample_based_on_list(query_result, task_list_file="inference/other_tasks.json"):
    """
    Sample based on the tasks list file
    """
    all_tasks_df = query_result.df()
    all_tasks_df["task_tag"] = all_tasks_df["kb_filename"].apply(lambda x: '_'.join('_'.join(x.split('/')[-2:]).split('_')[:5]))
    task_data = read_json(task_list_file)
    if "tasks" in task_data:
        target_tasks = task_data["tasks"]
    else:
        raise ValueError(f"[{task_list_file}] does not contain tasks key")
    sampled_df = all_tasks_df[all_tasks_df["task_tag"].apply(lambda x: x in target_tasks)]
    return sampled_df
