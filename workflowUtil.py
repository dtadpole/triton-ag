import os
import yaml
import asyncio
import traceback
from logger import logger
from datetime import datetime
from pydantic import BaseModel, Field
from util import INFERENCE_DIR, TRAINER_DIR


MODEL_OVERRIDE_KEY = "adapter.model_override"

class WorkflowSyncBlock(BaseModel):
    queue_type: str = Field(default="sync")
    queue_name: str
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    module_file: str = Field(default="workflow/sync.module.vllm.yaml")
    context: dict = Field(default={})

class TrainerBlock(BaseModel):
    queue_type: str = Field(default="trainer")
    queue_name: str
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default=INFERENCE_DIR + "/output")
    output_dir: str = Field(default=TRAINER_DIR)
    context: dict = Field(default={})

class InferenceBlock(BaseModel):
    queue_type: str = Field(default="inference")
    queue_name: str
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str

    model_name: str
    num_samples: int = Field(default=20)
    num_generations: int = Field(default=8)
    num_turns_per_generation: int = Field(default=4)
    parallel_workers: int = Field(default=32)
    vllm_providers: list[str] = Field(default=["local"])
    logp_providers: list[str] = Field(default=["local"])
    kbeval_providers: list[str] = Field(default=["local"])
    module_file: str = Field(default="inference/codeGen.module.vllm+logp.yaml")
    prompt_file: str = Field(default="inference/triton.prompt.yaml")
    example_file: str = Field(default="inference/triton.example.yaml")
    input_dir: str = Field(default="~/KernelBench/KernelBench")
    output_dir: str = Field(default="~/.inference/output")
    sft_dir: str = Field(default="~/.inference/sft")
    context: dict = Field(default_factory=dict)
    hint_length_prob: float = Field(default=0) # ADD THIS TO UFT, the hint length prob. https://arxiv.org/pdf/2505.16984



def merge_dicts(a: dict, b: dict) -> dict:
    """
    Return a new dict that is a recursive merge of a and b.
    Keys in b override keys in a. If both values are dicts, merge them recursively.
    """
    result = a.copy()
    for key, b_val in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(b_val, dict):
            result[key] = merge_dicts(result[key], b_val)
        else:
            result[key] = b_val
    return result


def deep_format(obj, env, *, strict: bool = True):
    """
    Recursively format all *string values* in nested dict/list/tuple/set
    using str.format_map(env). Dict KEYS are left untouched.

    strict=True  -> raise on missing key/index/attribute
    strict=False -> leave that string unchanged
    """
    def fmt(s: str) -> str:
        if strict:
            return s.format_map(env)
        try:
            return s.format_map(env)
        except (KeyError, IndexError, AttributeError):
            return s

    if isinstance(obj, str):
        return fmt(obj)
    if isinstance(obj, dict):
        return {k: deep_format(v, env, strict=strict) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_format(v, env, strict=strict) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_format(v, env, strict=strict) for v in obj)
    if isinstance(obj, set):
        return {deep_format(v, env, strict=strict) for v in obj}
    return obj


if __name__ == "__main__":
    import json
    import random
    obj = {
        "prefix_tag": "test_{prefix_tag}",
        "epoch_id": "{epoch_id:03d}",
        "block_id": "{block_id:02d}",
        "run_tag": "test_{prefix_tag}_{epoch_id:03d}_{block_id:02d}",
        "context": {
            "prefix_tag": "test_{prefix_tag}",
            "epoch_id": "{epoch_id:03d}",
            "block_id": "{block_id:02d}",
        },
    }
    print(json.dumps(deep_format(obj, {
        "prefix_tag": "auto",
        "epoch_id": random.randint(0, 1000),
        "block_id": random.randint(0, 100),
    }), indent=4))
