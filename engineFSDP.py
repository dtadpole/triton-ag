import os
import asyncio
import sys
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from transformers import (
    get_scheduler,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftConfig,
    PeftModel,
)
from typing import Dict, List, Optional, Any, Callable
import time
import traceback
from pathlib import Path
import math
from datetime import datetime
import random
import numpy as np
from torch.utils.data import Subset
from pydantic import BaseModel
from tqdm import tqdm
import wandb
import yaml
import argparse
from logger import logger
from trainerUtil import SimpleCollator, merge_dicts
from trainerBase import BaseTrainer, TrainerConfig, TrainerStatus, TextDataset

TRAINING_STATUS_FILE = "training_status.json"
ADAPTER_MODEL_FILE = "adapter_model.safetensors"
CHECKPOINT_READY_FILE = "checkpoint.ready"
TRAINING_STATE_FILE = "training_state.pt"
DEEPSPEED_TAG = "ds"

class FSDPTrainer(BaseTrainer):
    """Base trainer for Hugging Face models with step-by-step training implementation"""

    def __init__(self, prefix_tag: str, config: TrainerConfig, status: Optional[TrainerStatus] = None):
        # TODO
        pass

async def main():
    pass

if __name__ == "__main__":
    # run main async
    asyncio.run(main())
