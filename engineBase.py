import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import wandb
import yaml
from logger import logger
from pydantic import BaseModel
from torch.utils.data import Dataset
from trainerUtil import SimpleCollator
from workflowUtil import merge_dicts

class TrainerStatus(BaseModel):
    """Running status of the trainer"""

    global_step: int = 0


class EngineModelConfig(BaseModel):
    """Configuration for model parameters"""

    name: str = "gpt2"
    tokenizer_name: Optional[str] = None
    max_seq_length: int = 16384
    engine: str = "deepspeed"  # deepspeed, unsloth, fsdp, etc
    deepspeed_config_path: str = "engineDeepspeed.json"
    use_gradient_checkpointing: bool = True
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    compute_dtype: str = "bfloat16"
    trust_remote_code: bool = True


class EngineOptimizerConfig(BaseModel):
    """Configuration for optimizer parameters"""

    optimizer_type: str = "AdamW"  # AdamW, Adam, SGD
    betas: tuple = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD
    nesterov: bool = False  # For SGD


class EngineTrainingConfig(BaseModel):
    """Configuration for training parameters"""

    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.000005
    block_size: int = 32
    max_steps: int = 100000
    save_steps: int = 20
    eval_steps: int = 20
    retain_steps: int = 1000
    logging_steps: int = 1
    checkpoint_path: str = "~/.trainer"
    latest_checkpoint_name: Optional[str] = "checkpoint-latest"
    max_grad_norm: float = 0.1
    scheduler_type: str = "cosine"
    num_warmup_steps: int = 50
    dataloader_num_workers: int = 1
    loss_multiplier: float = 1.0
    keep_checkpoint_num: int = 5
    seed: int = -1


class EngineLoraConfig(BaseModel):
    """Configuration for LoRA parameters"""

    use_lora: bool = True
    rank: int = 64
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Optional[List[str] | str] = None
    target_parameters: Optional[List[str] | str] = None
    modules_to_save: Optional[List[str] | str] = None
    bias: str = "none"


class LoggingConfig(BaseModel):
    """Configuration for logging parameters"""

    use_wandb: bool = True
    wandb_project: str = "kb_trainer"
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None


class EngineConfig(BaseModel):
    """Main configuration class containing all training parameters"""

    model: EngineModelConfig = EngineModelConfig()
    training: EngineTrainingConfig = EngineTrainingConfig()
    optimizer: EngineOptimizerConfig = EngineOptimizerConfig()
    lora: EngineLoraConfig = EngineLoraConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_yaml(
        cls, yaml_path: str, override_yaml_path: Optional[str] = None
    ) -> "EngineConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if override_yaml_path is not None:
            with open(override_yaml_path, "r") as f:
                override_config_dict = yaml.safe_load(f)
            # do a recursive merge of the two dictionaries
            config_dict = merge_dicts(config_dict, override_config_dict)

        # Create config objects from sections
        model_config = EngineModelConfig()
        training_config = EngineTrainingConfig()
        optimizer_config = EngineOptimizerConfig()
        lora_config = EngineLoraConfig()
        logging_config = LoggingConfig()

        # Update from YAML sections
        if "model" in config_dict:
            model_data = config_dict["model"]
            model_config = EngineModelConfig(
                name=model_data.get("name", "gpt2"),
                tokenizer_name=model_data.get("tokenizer_name"),
                max_seq_length=model_data.get("max_seq_length", 1024),
                use_gradient_checkpointing=model_data.get(
                    "use_gradient_checkpointing", "unsloth"
                ),
                load_in_4bit=model_data.get("load_in_4bit", False),
                load_in_8bit=model_data.get("load_in_8bit", False),
                compute_dtype=model_data.get("compute_dtype", "bfloat16"),
            )

        if "training" in config_dict:
            training_data = config_dict["training"]
            training_config = EngineTrainingConfig(**training_data)

        if "optimizer" in config_dict:
            optimizer_data = config_dict["optimizer"]
            # Convert betas list to tuple if present
            if "betas" in optimizer_data and isinstance(optimizer_data["betas"], list):
                optimizer_data["betas"] = tuple(optimizer_data["betas"])
            optimizer_config = EngineOptimizerConfig(**optimizer_data)

        if "lora" in config_dict:
            lora_data = config_dict["lora"]
            lora_config = EngineLoraConfig(
                use_lora=lora_data.get("use_lora", True),
                rank=lora_data.get("rank", 64),
                alpha=lora_data.get("alpha", 16),
                dropout=lora_data.get("dropout", 0.0),
                target_modules=lora_data.get("target_modules", []),
                modules_to_save=lora_data.get("modules_to_save", []),
                bias=lora_data.get("bias", "none"),
            )

        if "logging" in config_dict:
            logging_data = config_dict["logging"]
            logging_config = LoggingConfig(**logging_data)

        return cls(
            model=model_config,
            training=training_config,
            optimizer=optimizer_config,
            lora=lora_config,
            logging=logging_config,
        )


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize the text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            # padding='max_length',
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze(),
        }


class EngineBase(ABC):

    @classmethod
    def create_engine(
        cls,
        prefix_tag: str,
        config: EngineConfig,
        status: Optional[TrainerStatus] = None,
        inference_mode=False,
    ):
        if config.model.engine == "deepspeed":
            from engineDeepspeed import EngineDeepspeed

            return EngineDeepspeed(
                prefix_tag, config, status=status, inference_mode=inference_mode
            )
        elif config.model.engine == "unsloth":
            from engineUnsloth import EngineUnsloth

            return EngineUnsloth(
                prefix_tag, config, status=status, inference_mode=inference_mode
            )
        elif config.model.engine == "fsdp":
            from engineFSDP import EngineFSDP

            return EngineFSDP(
                prefix_tag, config, status=status, inference_mode=inference_mode
            )

    """Base training engine for Hugging Face models with step-by-step training implementation"""

    def __init__(
        self,
        prefix_tag: str,
        config: EngineConfig,
        status: Optional[TrainerStatus] = None,
        inference_mode=False,
    ):
        self.inference_mode = inference_mode
        self.prefix_tag = prefix_tag
        self.config = config
        self.status = status if status is not None else TrainerStatus()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Setup output directory
        self.checkpoint_path = (
            Path(os.path.expanduser(self.config.training.checkpoint_path))
            / self.prefix_tag
        )
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def short_name(self):
        return "base"

    def _update_config(self, config: EngineConfig):
        """Update config"""
        if type(config) != EngineConfig:
            raise ValueError(
                f"❌ [{self.__class__.__name__}] Invalid config type: [{type(config)}]"
            )
        self.config = config

    def _print_model_info(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"🛳️ [{self.__class__.__name__}] loaded - Trainable: [{trainable_params:,}/{total_params:,}] ({100 * trainable_params / total_params:.1f}%)"
        )

    @abstractmethod
    def _backward_step(self, loss: torch.Tensor):
        """Execute a backward step"""
        pass

    @abstractmethod
    def _optimization_step(self):
        """Execute optimization step with gradient clipping"""
        pass

    @abstractmethod
    def _get_current_lr(self):
        """Get current learning rate"""
        pass

    @abstractmethod
    def _base_model(self):
        """Get base model"""
        pass

    @abstractmethod
    def _lora_model(self):
        """Get LoRA model"""
        pass

    @abstractmethod
    def _checkpoint_exists(self, checkpoint_path: str) -> bool:
        """Check if checkpoint exists"""
        pass

    @abstractmethod
    def _load_checkpoint(self, checkpoint_location: str):
        """Load checkpoint for resuming training"""
        pass

    @abstractmethod
    def _save_checkpoint(self, step: int, callback: Optional[Callable] = None):
        """Save checkpoint for resuming training"""
        pass

    def _update_latest_checkpoint_link(self, checkpoint_path: Path):
        """Update latest checkpoint link"""
        latest_path = self.checkpoint_path / self.config.training.latest_checkpoint_name

        if (
            dist.is_initialized() and dist.get_rank() == 0
        ) or not dist.is_initialized():
            # Remove existing link/directory
            if latest_path.exists():
                if latest_path.is_symlink():
                    latest_path.unlink()
                else:
                    shutil.rmtree(latest_path)

            # Create symlink or copy
            try:
                latest_path.symlink_to(checkpoint_path.name)
            except OSError:
                shutil.copytree(checkpoint_path, latest_path)

    def _cleanup_checkpoint(self, checkpoint_path: Path):
        """Cleanup checkpoint"""
        if (
            dist.is_initialized() and dist.get_rank() == 0
        ) or not dist.is_initialized():
            # check all the folders under checkpoint_path
            # retain only the latest {self.config.training.keep_checkpoint_num} checkpoints
            checkpoints = list(checkpoint_path.glob("checkpoint-*"))
            # remove checkpoint-latest
            checkpoints.remove(
                checkpoint_path / self.config.training.latest_checkpoint_name
            )
            checkpoints.sort(key=lambda x: int(x.name.split("-")[1].split(".")[0]))
            for checkpoint in checkpoints[: -self.config.training.keep_checkpoint_num]:
                checkpoint_num = int(checkpoint.name.split("-")[1].split(".")[0])
                if checkpoint_num % self.config.training.retain_steps == 0:
                    logger.info(
                        f"🔍 [{self.__class__.__name__}] Retaining checkpoint: {checkpoint}"
                    )
                    continue
                else:
                    logger.info(
                        f"🔍 [{self.__class__.__name__}] Removing checkpoint: {checkpoint}"
                    )
                    shutil.rmtree(checkpoint)

    def _setup_logging(self):
        """Setup logging and tracking"""
        if (
            dist.is_initialized() and dist.get_rank() == 0
        ) or not dist.is_initialized():
            # Initialize logging
            self.config.logging.wandb_run_id = self.prefix_tag
            self.config.logging.wandb_run_name = (
                self.prefix_tag + "_" + datetime.now().strftime("%m%d")
            )
            if self.config.logging.use_wandb:
                wandb.init(
                    project=self.config.logging.wandb_project,
                    id=self.config.logging.wandb_run_id,
                    name=self.config.logging.wandb_run_name,
                    config=self.config.model_dump(),
                    resume="allow",
                )
                logger.info(f"📊 [{self.__class__.__name__}] W&B logging enabled")

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics"""
        if (
            dist.is_initialized() and dist.get_rank() == 0
        ) or not dist.is_initialized():
            step = self.status.global_step
            if step % self.config.training.logging_steps == 0:
                # format metrics into a string with .4f format
                formatted_metrics = {k: f"{v:.4f}" for k, v in metrics.items()}
                logger.info(
                    f"🔍 [{self.__class__.__name__}] [G-Step={step}] {formatted_metrics}"
                )

                if self.config.logging.use_wandb:
                    wandb.log(metrics, step=step)


def create_sample_training_dataset(
    tokenizer, size: int = 100, max_length: int = 512
) -> TextDataset:
    """Create a sample dataset for CLI training"""
    # Sample conversations for training
    conversations = [
        "Human: What is artificial intelligence?\nAssistant: Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.",
        "Human: How do neural networks work?\nAssistant: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections.",
        "Human: What is machine learning?\nAssistant: Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.",
        "Human: Explain deep learning.\nAssistant: Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
        "Human: What are transformers in AI?\nAssistant: Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data, revolutionizing natural language processing.",
        "Human: How does training work?\nAssistant: Training involves feeding data to a model, calculating errors, and adjusting parameters to minimize those errors through backpropagation.",
        "Human: What is fine-tuning?\nAssistant: Fine-tuning is the process of taking a pre-trained model and further training it on a specific task or dataset to improve its performance on that particular task.",
        "Human: What is LoRA?\nAssistant: LoRA (Low-Rank Adaptation) is a technique that fine-tunes large language models efficiently by updating only a small number of parameters while keeping most of the model frozen.",
    ]

    # Repeat to reach desired size
    repeated_conversations = (conversations * (size // len(conversations) + 1))[:size]
    return TextDataset(repeated_conversations, tokenizer, max_length)
