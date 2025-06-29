import os
import pandas as pd
from datasets import load_dataset
import regex as re
from typing import List, Dict, Any
import argparse
import predibase
import traceback
from predibase import Predibase, GRPOConfig, RewardFunctionsConfig, RewardFunction, SamplingParamsConfig


# Check if the output is properly formatted
def format_reward_func(prompt: str, completion: str, example: dict[str, str]) -> int:
    # Imported packages must be inside each reward function
    import re

    reward = 0
    try:
        # Add synthetic <think> as it's already part of the prompt and prefilled
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Check if the format matches expected pattern:
        # <think> content </think> followed by <answer> content </answer>
        regex = (
            r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\n"
            r"<answer>\s*([\s\S]*?)\s*<\/answer>$"
        )

        # Search for the regex in the completion
        match = re.search(regex, completion, re.DOTALL)
        if match is not None and len(match.groups()) == 2:
            reward = 1.0
    except Exception:
        pass

    print(f"Format reward: {reward}")
    return reward

# Check if the output contains the correct answer
def equation_reward_func(prompt: str, completion: str, example: dict[str, str]) -> int:
    # Imported packages must be inside each reward function
    import re
    import ast

    reward = 0.0
    try:
        # add synthetic <think> as its already part of the prompt and prefilled
        # for the assistant to more easily match the regex
        completion = "<think>" + completion
        match = re.search(r"<answer>\s*([\s\S]*?)\s*<\/answer>", completion)
        if not match:
            print("No answer found in completion. Equation reward: 0.0")
            return 0.0

        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

        # Convert the example["nums"] to a list if it's a string
        # This is common for columns like lists in datasets
        if isinstance(example["nums"], str):
            example["nums"] = ast.literal_eval(example["nums"])

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(example["nums"]):
            print("Numbers used in equation not the same as in example. Equation reward: 0.0")
            return 0.0

        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
            print("Equation contains invalid characters. Equation reward: 0.0")
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(example["target"])) < 1e-5:
            reward = 1.0
        else:
            print("Equation is incorrect. Equation reward: 0.0")
            return 0.0

    except Exception:
        pass

    print(f"Equation reward: {reward}")
    return reward

template = """<|im_start|>system
You are a helpful assistant. You first think about the reasoning process step by step and then provide the user with an answer.<|im_end|>
<|im_start|>user
Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and parentheses, and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""

def format_row(row):
    return template.format(
        nums=row["nums"],
        target=row["target"]
    )

def grpo(pb, args):
    # Create an adapter repository
    repo = pb.repos.create(
        name="my-adapter-repo",
        description="My first adapter repo",
        exists_ok=True,
    )

    # Load the dataset
    dataset = load_dataset("predibase/countdown")
    train_df = pd.DataFrame(dataset["train"])
    eval_df = pd.DataFrame(dataset["test"])

    train_df["prompt"] = train_df.apply(format_row, axis=1)
    eval_df["prompt"] = eval_df.apply(format_row, axis=1)

    print("--------")
    print("train_df")
    print(train_df.head(5))
    print(eval_df.iloc[0]["prompt"])
    print("--------")
    print("eval_df")
    print(eval_df.head(5))
    print(eval_df.iloc[0]["prompt"])
    print("--------")

    train_df.to_json("./countdown_train.jsonl", lines=True, orient="records")

    # Upload a dataset
    try:
        dataset = pb.datasets.from_file("./countdown_train.jsonl", name="countdown_train")
    except Exception as e:
        # print exception and stacktrack
        # print(f"Error uploading dataset: {e}")
        # print(traceback.format_exc())
        dataset = pb.datasets.get("countdown_train")
    
    sampling_config = SamplingParamsConfig(
        temperature=0.5,
        top_p=0.95,
        top_k=40,
        max_tokens=2048,
    )

    # Launch the finetuning job!
    adapter = pb.adapters.create(
        config=GRPOConfig(
            base_model="qwen3-8b",
            reward_fns=RewardFunctionsConfig(
                functions={
                    "format": RewardFunction.from_callable(format_reward_func),
                    "answer": RewardFunction.from_callable(equation_reward_func),
                }
            ),
            target_modules=[
                'q_proj', 'v_proj', 'k_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ],
            train_steps=200,
            num_generations=16,
            sampling_params=sampling_config,
        ),
        dataset="countdown_train",
        repo=repo,
        description="Countdown!"
    )


def test_generate(args):
    # Connect to a shared deployment
    client = pb.deployments.client("qwen3-32b")

    # Generate text
    if args.no_stream:
        response = client.generate(
            "What are some popular tourist spots in San Francisco?",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(response.generated_text)
    else:
        for response in client.generate_stream(
            "What are some popular tourist spots in San Francisco?",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ):
            if not response.token.special:
                print(response.token.text, sep="", end="", flush=True)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--no_stream", action="store_true", default=False)
    argparse.add_argument("--max_new_tokens", type=int, default=256)
    argparse.add_argument("--temperature", type=float, default=0.6)
    argparse.add_argument("--test", action="store_true", default=False)
    args = argparse.parse_args()    

    # read token from $HOME/.keys/pbase.api.key
    with open(os.path.expanduser("~/.keys/pbase.api.key"), "r") as f:
        api_token = f.read().strip()
    pb = Predibase(api_token=api_token)

    # Get a list of available models
    available_models = pb.deployments.list()

    if args.test:
        test_generate(args)
    else:
        grpo(pb, args)
