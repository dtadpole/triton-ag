import os
import yaml
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

REG_PORT_FILE = ".reg.port"
REG_DIR = ".reg"

MODEL_OVERRIDE_KEY = "adapter.codeGenEval.model_override"

class TrainerSFTBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default="~/.exemplar")
    output_dir: str = Field(default="~/.trainer")
    test_mode: bool = Field(default=False)

class TrainerRFTBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.trainer")
    test_mode: bool = Field(default=False)

class TrainerGRPOBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.trainer")
    test_mode: bool = Field(default=False)

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
    test_mode: bool = Field(default=False)

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
    test_mode: bool = Field(default=False)

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
    test_mode: bool = Field(default=False)

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
    test_mode: bool = Field(default=False)

class ComposerBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    module_file: str
    prompt_file: str
    example_file: str
    num_samples: int
    num_generations: int
    num_turns_per_generation: int = Field(default=4)
    parallel_workers: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/.inference/composer")
    output_dir: str = Field(default="~/.inference/composer")
    test_mode: bool = Field(default=False)

def get_prefix_tag(prefix_tag:str="auto", config_path:str="globalWorkflow.yaml"):
    if prefix_tag == "auto":
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        return yaml_data.get("global", {}).get("prefix_tag", f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        return prefix_tag

def get_global_registry_dir(prefix_tag:str="auto", trainer_dir:str="~/.trainer"):
    return os.path.join(os.path.expanduser(trainer_dir), prefix_tag, REG_DIR)

def get_global_registry_port(prefix_tag:str="auto", trainer_dir:str="~/.trainer"):
    port_file = os.path.join(get_global_registry_dir(prefix_tag, trainer_dir), REG_PORT_FILE)
    if os.path.exists(port_file):
        with open(port_file, "r") as f:
            return int(f.read())
    else:
        return None

class GlobalUtils:
    _instance = None # class variable to store the instance

    # singleton pattern
    def __new__(cls):
        if cls._instance is None:
            # If no instance exists, create one using the superclass's __new__
            cls._instance = super(GlobalUtils, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # __init__ will be called every time, but only the first time will
        # actually initialize the instance if we add a flag.
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.fastapi = FastAPI()

    def fastapi(self):
        return self.fastapi
