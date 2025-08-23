import os
import yaml
import asyncio
import traceback
from logger import logger
from datetime import datetime
from pydantic import BaseModel, Field

MODEL_OVERRIDE_KEY = "adapter.model_override"

class WorkflowSyncBlock(BaseModel):
    name: str
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    module_file: str = Field(default="workflow/sync.module.vllm.yaml")
    context_vars: dict = Field(default={})

class TrainerBlock(BaseModel):
    name: str
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default="~/.inference/output")
    output_dir: str = Field(default="~/.trainer")
    context_vars: dict = Field(default={})

class InferenceBlock(BaseModel):
    name: str
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
    context: dict = Field(default={})

def get_prefix_tag(prefix_tag:str="auto", config_path:str="globalWorkflow.yaml"):
    if prefix_tag == "auto":
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        loaded_prefix_tag = yaml_data.get("global", {}).get("prefix_tag", f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # write to config_path
        return loaded_prefix_tag
    else:
        return prefix_tag
