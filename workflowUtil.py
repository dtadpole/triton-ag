import os
import yaml
import asyncio
import traceback
from logger import logger
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from util import INFERENCE_DIR, TRAINER_DIR

REG_PORT_FILE = ".reg.port"
REG_DIR = ".reg"

MODEL_OVERRIDE_KEY = "adapter.model_override"

class TrainerSFTBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default=INFERENCE_DIR + "/exemplar")
    output_dir: str = Field(default=TRAINER_DIR)

class TrainerRFTBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default=INFERENCE_DIR + "/codeGenEval")
    output_dir: str = Field(default=TRAINER_DIR)

class TrainerGRPOBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default=INFERENCE_DIR + "/codeGenEval")
    output_dir: str = Field(default=TRAINER_DIR)

class CodeGenEvalBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    provider_name: str
    model_name: str
    num_samples: int
    num_generations: int
    num_turns_per_generation: int = Field(default=4)
    parallel_tasks: int = Field(default=24)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/triton-ag/kernel_bench")
    output_dir: str = Field(default="~/.codeGenEval")
    template: str = Field(default="triton.1")
    logprobs: bool = Field(default=True)

class ExemplarBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    num_generations: int
    parallel_tasks: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.exemplar")
    template: str = Field(default="triton.1")

class CritiqueBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    parallel_tasks: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.critique")

class ReflectionBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    parallel_tasks: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.reflection")

class ComposerBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    num_samples: int
    num_generations: int
    num_turns_per_generation: int = Field(default=4)
    parallel_workers: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    module_file: str = Field(default="inference/codeGen.module.yaml")
    prompt_file: str = Field(default="inference/triton.prompt.yaml")
    example_file: str = Field(default="inference/triton.example.yaml")
    input_dir: str = Field(default=INFERENCE_DIR + "/composer")
    output_dir: str = Field(default=INFERENCE_DIR + "/composer")

def get_prefix_tag(prefix_tag:str="auto", config_path:str="globalWorkflow.yaml"):
    if prefix_tag == "auto":
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        loaded_prefix_tag = yaml_data.get("global", {}).get("prefix_tag", f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # write to config_path
        return loaded_prefix_tag
    else:
        return prefix_tag

def get_global_registry_dir(prefix_tag:str="auto", trainer_dir:str=TRAINER_DIR):
    prefix_tag = get_prefix_tag(prefix_tag)
    return os.path.join(os.path.expanduser(trainer_dir), prefix_tag, REG_DIR)

def get_global_registry_port(prefix_tag:str="auto", trainer_dir:str=TRAINER_DIR):
    prefix_tag = get_prefix_tag(prefix_tag)
    port_file = os.path.join(get_global_registry_dir(prefix_tag, trainer_dir), REG_PORT_FILE)
    if os.path.exists(port_file):
        with open(port_file, "r") as f:
            return int(f.read())
    else:
        return None
