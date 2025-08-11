import torch
import numpy as np
from typing import Callable
from transformers import AutoTokenizer
from operator import itemgetter
from itertools import groupby
import random
from collections import defaultdict

def rft_bucket_by_reward(
    query_result: list[dict],
    return_top_percentile: float = 0.3,
    debug_bucket_count: int = 8,
    debug: bool = False,
) -> dict[str, list[dict]]:
    # for each task_tag, group by gen_tag, and return a list of turns, each turn is a sorted list of rows
    results_by_task = {
        k: list(g) for k, g in groupby(
            sorted(query_result, key=itemgetter("task_tag","turn_tag")),
            key=itemgetter("task_tag")
        )
    }
    if debug:
        print('\nProcessing rewards (by task_tag)\n')

    # results_by_task_sorted is a list of tuples, each tuple is (task_tag, list of turns)
    results_by_task_top_bucket: dict[str, list[dict]] = defaultdict(list)
    results_by_task_bottom_bucket: dict[str, list[dict]] = defaultdict(list)
    for key, value in results_by_task.items():
        # print the turn_tag, as well as compiled, correctness, runtime value array by percentile, using bucket_count
        value.sort(key=lambda d: (
            -int(d["compiled"]),
            -int(d["correctness"]),
            d["runtime"] if d["runtime"] > 0 else float('inf'),
            random.random()
        ))
        # put everything into the bucket
        # if everything is False, False, and negative runtime, then skip the task
        if all(item["compiled"] == False and item["correctness"] == False and item["runtime"] < 0 for item in value):
            continue
        # split the list into two halves
        results_by_task_top_bucket[key] = value[:int(len(value) * return_top_percentile)]
        results_by_task_bottom_bucket[key] = value[int(len(value) * return_top_percentile):]
        # print the buckets
        if debug:
            print(f'\n{key}:')
            for i in range(debug_bucket_count):
                print(f'    => {i*100//debug_bucket_count}%: {[f'{item["turn_tag"]} {item["compiled"]} {item["correctness"]} {item["runtime"]}' for item in value[len(value)*i//debug_bucket_count:len(value)*(i+1)//debug_bucket_count]]}')

    return results_by_task_top_bucket, results_by_task_bottom_bucket

def rft_bucket_to_dataset(
    bucket: list[dict],
    tokenizer: AutoTokenizer,
    format_conversation: Callable,
    discard_long_conversations: bool = True,
    max_seq_length: int = 16384,
):
    rft_dataset = []

    for task_tag, bucket in bucket.items():
        for result in bucket:
            messages = result["messages"]
            formatted_data = format_conversation(
                messages,
                tokenizer,
                mask_non_assistant_tokens=True,
                mask_non_last_assistant_tokens=True,
            )
            
            # Convert tensors to lists for the data collator
            input_ids = formatted_data['input_ids'].flatten().tolist()
            attention_mask = formatted_data['attention_mask'].flatten().tolist()
            labels = formatted_data['labels'].flatten().tolist()

            if discard_long_conversations and len(input_ids) > max_seq_length:
                continue

            rft_dataset.append({
                'task_tag': result["task_tag"],
                'turn_tag': result["turn_tag"],
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'compiled': result["compiled"],
                'correctness': result["correctness"],
                'runtime': result["runtime"],
            })

    return rft_dataset

if __name__ == "__main__":
    import duckdb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--return_top_percentile", type=float, default=0.3)
    parser.add_argument("--input_dir", type=str, default="~/.codeGenEval/TC_0.1.0_14B.m_005_05")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_bucket_count", type=int, default=8)
    parser.add_argument("--print_bottom_bucket", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    sql = f"""
        WITH evals AS (
            SELECT
                filename AS eval_file,
                regexp_replace(filename, '_generated_eval\\.json$', '') AS stem,
                compiled, correctness, runtime
            FROM read_json_auto('{args.input_dir}/**/*_generated_eval.json', filename = true)
        ),
        convos AS (
            SELECT
                filename AS conversation_file,
                regexp_replace(filename, '_conversation\\.json$', '') AS stem,
                messages, metadata,
                metadata->>'task_tag' AS task_tag, metadata->>'turn_tag' AS turn_tag
            FROM read_json_auto('{args.input_dir}/**/*_conversation.json', filename = true)
        )
        SELECT
            e.compiled, e.correctness, e.runtime,
            c.messages, c.metadata,
            e.eval_file, c.conversation_file,
            c.task_tag, c.turn_tag
        FROM evals e
        JOIN convos c USING (stem)
        ORDER BY task_tag, compiled DESC, correctness DESC, runtime ASC
    """
    result = duckdb.sql(sql)
    print(result)

    top_bucket, bottom_bucket = rft_bucket_by_reward(
        result.df().to_dict(orient="records"),
        return_top_percentile=args.return_top_percentile,
        debug_bucket_count=args.debug_bucket_count,
    )
    for key, value in top_bucket.items():
        print(f'\nTop bucket of [{key}]:')
        print('    => compiled: ', [f'{gen["compiled"]}' for gen in value])
        print('    => correctness: ', [f'{gen["correctness"]}' for gen in value])
        print('    => runtime: ', [f'{gen["runtime"]}' for gen in value])
        print('    => turn_tag: ', [f'{gen["turn_tag"]}' for gen in value])
        print('    => messages: ', [f'{len(gen["messages"])}' for gen in value])
    
    if args.print_bottom_bucket:
        for key, value in bottom_bucket.items():
            print(f'\nBottom bucket of [{key}]:')
            print('    => compiled: ', [f'{gen["compiled"]}' for gen in value])
            print('    => correctness: ', [f'{gen["correctness"]}' for gen in value])
            print('    => runtime: ', [f'{gen["runtime"]}' for gen in value])
            print('    => turn_tag: ', [f'{gen["turn_tag"]}' for gen in value])
            print('    => messages: ', [f'{len(gen["messages"])}' for gen in value])

    import sys
    import os
    from pathlib import Path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    # print(sys.path)
    from trainerUtil import format_conversation

    rft_dataset = rft_bucket_to_dataset(
        top_bucket,
        tokenizer,
        format_conversation,
        discard_long_conversations=True,
        max_seq_length=16384,
    )

    for item in rft_dataset:
        print(f"[{item['task_tag']} {item['turn_tag']}] => [input_ids={len(item['input_ids'])}] [labels={len([l for l in item['labels'] if l != -100])} [attn_mask={len([m for m in item['attention_mask'] if m != 0])}] [{item['compiled']} {item['correctness']} {item['runtime']}]")
