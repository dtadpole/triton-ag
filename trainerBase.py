# Set environment variables for optimal performance
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import Unsloth first for optimal performance
from unsloth import FastLanguageModel
import unsloth
import asyncio
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from transformers import (
    get_scheduler,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)
import bitsandbytes as bnb
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
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


class TrainerStatus(BaseModel):
    """Running status of the trainer"""
    global_step: int = 0

class ModelConfig(BaseModel):
    """Configuration for model parameters"""
    name: str = 'gpt2'
    tokenizer_name: Optional[str] = None
    max_seq_length: int = 16384
    use_unsloth: bool = True
    use_gradient_checkpointing: str = "unsloth"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    compute_dtype: str = "bfloat16"

class OptimizerConfig(BaseModel):
    """Configuration for optimizer parameters"""
    optimizer_type: str = "AdamW"  # AdamW, Adam, SGD
    betas: tuple = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD
    nesterov: bool = False  # For SGD

class TrainingConfig(BaseModel):
    """Configuration for training parameters"""
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.000005
    block_size: int = 32
    max_steps: int = 100000
    save_steps: int = 20
    eval_steps: int = 20
    logging_steps: int = 1
    checkpoint_path: str = "~/.trainer"
    latest_checkpoint_name: Optional[str] = "checkpoint-latest"
    max_grad_norm: float = 0.1
    scheduler_type: str = "cosine"
    num_warmup_steps: int = 50
    dataloader_num_workers: int = 4
    loss_multiplier: float = 1.0
    seed: int = -1

class TrainerLoraConfig(BaseModel):
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

class TrainerConfig(BaseModel):
    """Main configuration class containing all training parameters"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    lora: TrainerLoraConfig = TrainerLoraConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str, override_yaml_path: Optional[str] = None) -> 'TrainerConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if override_yaml_path is not None:
            with open(override_yaml_path, 'r') as f:
                override_config_dict = yaml.safe_load(f)
            # do a recursive merge of the two dictionaries
            config_dict = merge_dicts(config_dict, override_config_dict)

        # Create config objects from sections
        model_config = ModelConfig()
        training_config = TrainingConfig()
        optimizer_config = OptimizerConfig()
        lora_config = TrainerLoraConfig()
        logging_config = LoggingConfig()

        # Update from YAML sections
        if 'model' in config_dict:
            model_data = config_dict['model']
            model_config = ModelConfig(
                name=model_data.get('name', 'gpt2'),
                tokenizer_name=model_data.get('tokenizer_name'),
                max_seq_length=model_data.get('max_seq_length', 1024),
                use_unsloth=model_data.get('use_unsloth', True),
                use_gradient_checkpointing=model_data.get('use_gradient_checkpointing', "unsloth"),
                load_in_4bit=model_data.get('load_in_4bit', False),
                load_in_8bit=model_data.get('load_in_8bit', False),
                compute_dtype=model_data.get('compute_dtype', 'bfloat16')
            )

        if 'training' in config_dict:
            training_data = config_dict['training']
            training_config = TrainingConfig(**training_data)

        if 'optimizer' in config_dict:
            optimizer_data = config_dict['optimizer']
            # Convert betas list to tuple if present
            if 'betas' in optimizer_data and isinstance(optimizer_data['betas'], list):
                optimizer_data['betas'] = tuple(optimizer_data['betas'])
            optimizer_config = OptimizerConfig(**optimizer_data)

        if 'lora' in config_dict:
            lora_data = config_dict['lora']
            lora_config = TrainerLoraConfig(
                use_lora=lora_data.get('use_lora', True),
                rank=lora_data.get('rank', 64),
                alpha=lora_data.get('alpha', 16),
                dropout=lora_data.get('dropout', 0.0),
                target_modules=lora_data.get('target_modules', []),
                modules_to_save=lora_data.get('modules_to_save', []),
                bias=lora_data.get('bias', 'none')
            )

        if 'logging' in config_dict:
            logging_data = config_dict['logging']
            logging_config = LoggingConfig(**logging_data)

        return cls(
            model=model_config,
            training=training_config,
            optimizer=optimizer_config,
            lora=lora_config,
            logging=logging_config
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
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }


class BaseTrainer:
    """Base trainer for Hugging Face models with step-by-step training implementation"""

    def __init__(self, prefix_tag: str, config: TrainerConfig, status: Optional[TrainerStatus] = None, base_trainer = None):
        self.prefix_tag = prefix_tag
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if base_trainer is None:
            self.trainer_status = status if status is not None else TrainerStatus()
        else:
            self.trainer_status = base_trainer.trainer_status

        # Set random seeds for reproducibility
        self._set_seed()

        # Initialize model and tokenizer
        if base_trainer is None:
            self.base_model, self.tokenizer = self._setup_model_and_tokenizer()
        else:
            self.base_model = base_trainer.base_model
            self.tokenizer = base_trainer.tokenizer

        # Setup LoRA if enabled
        if base_trainer is None:
            if self.config.lora.use_lora:
                self.model = self._setup_lora()
            else:
                self.model = self.base_model
        else:
            self.model = base_trainer.model

        # Initialize optimizer and scheduler
        if base_trainer is None:
            self.optimizer, self.scheduler = self._setup_optimizer_and_scheduler()
        else:
            self.optimizer = base_trainer.optimizer
            self.scheduler = base_trainer.scheduler

        # Initialize data collator
        self.data_collator = SimpleCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
        )

        # Setup output directory
        self.checkpoint_path = Path(os.path.expanduser(config.training.checkpoint_path)) / self.prefix_tag

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.config.logging.wandb_run_id = self.prefix_tag
        self.config.logging.wandb_run_name = self.prefix_tag + "_" + datetime.now().strftime("%m%d_%H%M%S")
        if base_trainer is None:
            self._setup_logging()
        else:
            self.config.logging.wandb_run_id = base_trainer.config.logging.wandb_run_id
            self.config.logging.wandb_run_name = base_trainer.config.logging.wandb_run_name

        logger.info(f"🔍 [{self.__class__.__name__}] Config: {self.config.model_dump_json()}")

        # Load checkpoint if specified
        if base_trainer is None and config.training.latest_checkpoint_name:
            checkpoint_location = self.checkpoint_path / config.training.latest_checkpoint_name
            if self._checkpoint_exists(checkpoint_location):
                self._load_checkpoint(checkpoint_location)
            else:
                logger.warning(f"⚠️ [{self.__class__.__name__}] Checkpoint not found: {checkpoint_location} - Starting fresh training")

    def short_name(self):
        return 'base'

    def _update_config(self, config: TrainerConfig):
        """Update config"""
        self.config = config

    def _set_seed(self):
        """Set random seeds for reproducibility"""
        if self.config.training.seed is not None and self.config.training.seed >= 0:
            random.seed(self.config.training.seed)
            np.random.seed(self.config.training.seed)
            torch.manual_seed(self.config.training.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.training.seed)
            logger.info(f"🎲 [{self.__class__.__name__}] Random seed set to: {self.config.training.seed}")
        else:
            logger.info(f"🎲 [{self.__class__.__name__}] Using random seed (no fixed seed set)")

    def _setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer using Unsloth"""
        logger.info(f"🚀 [{self.__class__.__name__}] Loading model: {self.config.model.name}")

        if self.config.model.use_unsloth:
            # Load model with Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model.name,
                dtype=getattr(torch, self.config.model.compute_dtype, torch.bfloat16),
                max_seq_length=self.config.model.max_seq_length,
                load_in_4bit=self.config.model.load_in_4bit,
                load_in_8bit=self.config.model.load_in_8bit,
                full_finetuning=self.config.model.full_finetuning,
                use_gradient_checkpointing=self.config.model.use_gradient_checkpointing,
                # token="hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
            )
        else:
            config = AutoConfig.from_pretrained(self.config.model.name)
            config.max_position_embeddings = self.config.model.max_seq_length
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name,
                config=config,
                torch_dtype=getattr(torch, self.config.model.compute_dtype, torch.bfloat16),
                device_map="auto",
            )
            model.gradient_checkpointing_enable()
            tokenizer = AutoTokenizer.from_pretrained(self.config.model.name if self.config.model.tokenizer_name is None else self.config.model.tokenizer_name)

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Special handling for Qwen models
        is_qwen_model = "qwen" in self.config.model.name.lower()
        if is_qwen_model:
            # Ensure Qwen-specific tokenizer settings
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is None:
                logger.warning("⚠️ [{self.__class__.__name__}] Qwen model missing chat template, this may cause generation issues")

            # Set trust_remote_code for Qwen models if needed
            if hasattr(tokenizer, 'trust_remote_code'):
                tokenizer.trust_remote_code = True

            logger.info(f"🛠️ [{self.__class__.__name__}] Qwen tokenizer configured with chat template: {hasattr(tokenizer, 'chat_template')}")

        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"📜 [{self.__class__.__name__}] Model loaded - Total: {total_params:,}, Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

        return model, tokenizer

    def _setup_lora(self):
        """Setup LoRA configuration using Unsloth"""
        logger.info(f"🛠️ [{self.__class__.__name__}] Setting up LoRA (rank={self.config.lora.rank}, alpha={self.config.lora.alpha})")

        if self.config.model.use_unsloth:
        # Apply LoRA with Unsloth
            lora_model = FastLanguageModel.get_peft_model(
                self.base_model,
                r=self.config.lora.rank,
                target_modules=self.config.lora.target_modules,
                target_parameters=self.config.lora.target_parameters,
                modules_to_save=self.config.lora.modules_to_save,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                bias=self.config.lora.bias,
                use_gradient_checkpointing=self.config.model.use_gradient_checkpointing,
                random_state=self.config.training.seed if self.config.training.seed is not None and self.config.training.seed >= 0 else random.randint(0,2**31),
                use_rslora=False,  # Use regular LoRA
                loftq_config=None,
                # autocast_adapter_dtype=getattr(torch, self.config.model.compute_dtype, torch.bfloat16),
            )
        else:
            lora_model = get_peft_model(self.base_model, TrainerLoraConfig(
                r=self.config.lora.rank,
                lora_alpha=self.config.lora.alpha,
                target_modules=self.config.lora.target_modules,
                target_parameters=self.config.lora.target_parameters,
                lora_dropout=self.config.lora.dropout,
                bias=self.config.lora.bias,
                task_type=TaskType.CAUSAL_LM,
                # autocast_adapter_dtype=False,
            ))

        # Log LoRA information
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())

        logger.info(f"🛠️ [{self.__class__.__name__}] LoRA applied - Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        if trainable_params == 0:
            logger.error(f"❌ [{self.__class__.__name__}] No trainable parameters found!")
            raise RuntimeError("No trainable parameters found after LoRA application")

        return lora_model

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Group parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.optimizer.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Create optimizer based on type (using bitsandbytes paged optimizers)
        optimizer_type = self.config.optimizer.optimizer_type.lower()
        if optimizer_type == "adamw":
            # self.optimizer = bnb.optim.PagedAdamW(
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        elif optimizer_type == "adam":
            # self.optimizer = bnb.optim.PagedAdam(
            optimizer = optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        elif optimizer_type == "sgd":
            # Fall back to standard SGD as bitsandbytes doesn't have paged SGD
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                momentum=self.config.optimizer.momentum,
                nesterov=self.config.optimizer.nesterov
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer.optimizer_type}")

        # Setup learning rate scheduler
        scheduler = get_scheduler(
            self.config.training.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.config.training.num_warmup_steps,
            num_training_steps=self.config.training.max_steps,
        )

        logger.info(f"⚙️ [{self.__class__.__name__}] Paged optimizer ({optimizer_type}) and scheduler initialized")

        return optimizer, scheduler

    def _checkpoint_exists(self, checkpoint_path: str) -> bool:
        """Check if checkpoint exists"""
        checkpoint_path_obj = Path(checkpoint_path)

        # Check for training_state.pt file
        if checkpoint_path_obj.is_dir():
            return (checkpoint_path_obj / "training_state.pt").exists()
        else:
            return checkpoint_path_obj.exists()

    def _load_checkpoint(self, checkpoint_location: str):
        """Load checkpoint for resuming training"""
        logger.info(f"🔄 [{self.__class__.__name__}] Loading checkpoint from: {checkpoint_location}")

        # Get the training state file path
        checkpoint_path_obj = Path(checkpoint_location)
        training_state_path = checkpoint_path_obj / "training_state.pt" if checkpoint_path_obj.is_dir() else checkpoint_path_obj

        # Load checkpoint
        checkpoint = torch.load(training_state_path, map_location='cpu')

        # Load model state
        if self.config.lora.use_lora and 'lora_state_dict' in checkpoint:
            set_peft_model_state_dict(self.model, checkpoint['lora_state_dict'])
        elif not self.config.lora.use_lora and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning(f"⚠️ [{self.__class__.__name__}] Model state not found or incompatible in checkpoint")

        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.trainer_status.global_step = checkpoint.get('global_step', 0)

        logger.info(f"📜 [{self.__class__.__name__}] Checkpoint loaded - Step: [{self.trainer_status.global_step}]")

    def _save_checkpoint(self, step: int, callback: Optional[Callable] = None):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_path / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        checkpoint_state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.trainer_status.global_step,
            'config': self.config.model_dump(),
        }

        if self.config.lora.use_lora:
            checkpoint_state['lora_state_dict'] = get_peft_model_state_dict(self.model)
        else:
            checkpoint_state['model_state_dict'] = self.model.state_dict()

        torch.save(checkpoint_state, checkpoint_path / "training_state.pt")

        # Save config
        with open(checkpoint_path / "training_config.yaml", 'w') as f:
            yaml.dump(self.config.model_dump(), f, default_flow_style=False)

        # Update latest checkpoint link
        self._update_latest_checkpoint_link(checkpoint_path)

        # Callback
        if callback:
            try:
                logger.info(f"🔍 [{self.__class__.__name__}] Running callback: {callback}")
                callback(checkpoint_path)
                logger.info(f"🔍 [{self.__class__.__name__}] Callback completed")
            except Exception as e:
                logger.error(f"❌ [{self.__class__.__name__}] Error in callback: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.info(f"🔍 [{self.__class__.__name__}] No callback provided")

        logger.info(f"💾 [{self.__class__.__name__}] Checkpoint saved: {checkpoint_path}")

    def _update_latest_checkpoint_link(self, checkpoint_path: Path):
        """Update latest checkpoint link"""
        latest_path = self.checkpoint_path / self.config.training.latest_checkpoint_name

        # Remove existing link/directory
        if latest_path.exists():
            if latest_path.is_symlink():
                latest_path.unlink()
            else:
                import shutil
                shutil.rmtree(latest_path)

        # Create symlink or copy
        try:
            latest_path.symlink_to(checkpoint_path.name)
        except OSError:
            import shutil
            shutil.copytree(checkpoint_path, latest_path)

    def _setup_logging(self):
        """Setup logging and tracking"""
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                id=self.config.logging.wandb_run_id,
                name=self.config.logging.wandb_run_name,
                config=self.config.model_dump(),
                resume="allow",
            )
            logger.info(f"📊 [{self.__class__.__name__}] W&B logging enabled")

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch"""
        # Move tensors to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        return loss

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step"""
        self.model.train()

        # Compute loss
        loss = self._compute_loss(batch)

        # Scale loss for gradient accumulation
        loss = loss * self.config.training.loss_multiplier / self.config.training.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return loss.item()

    def _optimization_step(self):
        """Execute optimization step with gradient clipping"""
        # Clip gradients
        grad_norm = clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        # logger.info(f"🔍 [{self.__class__.__name__}] Grad norm: [{grad_norm:.4f}] max grad norm: [{self.config.training.max_grad_norm:.2f}]")

        # Update parameters
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad()

        return grad_norm

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics"""
        step = self.trainer_status.global_step
        if step % self.config.training.logging_steps == 0:
            # format metrics into a string with .4f format
            formatted_metrics = {k: f"{v:.4f}" for k, v in metrics.items()}
            logger.info(f"🔍 [{self.__class__.__name__}] [G-Step={step}] {formatted_metrics}")

            if self.config.logging.use_wandb:
                wandb.log(metrics, step=step)

    async def train_block(
        self,
        run_tag: str,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callback: Optional[Callable] = None,
    ):
        """Train the model for one block"""
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.micro_batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.data_collator
        )

        logger.info(f"👉 [{self.__class__.__name__}] [{run_tag}] Block started with [{len(dataloader)}] micro batches, Initial global step: [{self.trainer_status.global_step}]")

        start_time = time.time()
        accumulated_loss = 0.0

        # Create progress bar
        progress_bar = tqdm(
            total=len(dataloader),
            desc=run_tag,
            initial=0
        )

        for batch_idx, batch in enumerate(dataloader):
            # Training step
            step_loss = self._train_step(batch)
            accumulated_loss += step_loss

            # Optimization step (only after accumulation)
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                grad_norm = self._optimization_step()

                # Calculate average loss
                avg_loss = accumulated_loss / self.config.training.gradient_accumulation_steps
                accumulated_loss = 0.0

                # Update step counter
                self.trainer_status.global_step += 1
                progress_bar.update(1)
                await asyncio.sleep(0.1)

                # Log metrics
                current_lr = self.scheduler.get_last_lr()[0]
                self._log_metrics({
                    "train/loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": grad_norm,
                    f"train_{self.short_name()}/loss": avg_loss,
                    f"train_{self.short_name()}/learning_rate": current_lr,
                    f"train_{self.short_name()}/grad_norm": grad_norm,
                }, self.trainer_status.global_step)

                # Save checkpoint
                if self.trainer_status.global_step % self.config.training.save_steps == 0:
                    self._save_checkpoint(self.trainer_status.global_step, callback=callback)

                # Evaluation
                if eval_dataset and self.trainer_status.global_step % self.config.training.eval_steps == 0:
                    self._evaluate(eval_dataset)

                # Check if training is complete
                if self.trainer_status.global_step >= self.config.training.max_steps:
                    break

        try:
            # always save checkpoint at the end of the block
            self._save_checkpoint(self.trainer_status.global_step)
        except Exception as e:
            logger.error(f"❌ [{self.__class__.__name__}] [{run_tag}] Failed to save checkpoint: {e}")

        total_time = time.time() - start_time
        logger.info(f"🎉 [{self.__class__.__name__}] [{run_tag}] Block completed in [{total_time:.1f}s] - Final global step: [{self.trainer_status.global_step}]")
        progress_bar.close()
        await asyncio.sleep(0.1)

    def _evaluate(self, eval_dataset: Dataset):
        """Evaluate the model on evaluation dataset"""
        logger.info(f"📊 [{self.__class__.__name__}] Running evaluation...")

        self.model.eval()
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.training.micro_batch_size,
            shuffle=False,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.data_collator
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_eval_loss = total_loss / num_batches
        perplexity = math.exp(avg_eval_loss)

        logger.info(f"📊 [{self.__class__.__name__}] Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")

        if self.config.logging.use_wandb:
            wandb.log({
                "eval/loss": avg_eval_loss,
                "eval/perplexity": perplexity,
                "eval/step": self.trainer_status.global_step
            })

        self.model.train()

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate text using the trained model"""
        self.model.eval()

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Store input length for proper output extraction
        input_length = inputs['input_ids'].shape[1]

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Extract only the generated tokens (excluding input)
        generated_tokens = outputs[0][input_length:]

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text.strip()

async def _train_loop(prefix_tag: str, trainer: BaseTrainer, dataset: Dataset, eval_dataset: Optional[Dataset] = None):
    """Main training loop"""
    logger.info(f"🏋️ [BaseTrainer] Starting training - Total steps: [{trainer.config.training.max_steps}], Batch size: [{trainer.config.training.micro_batch_size}], Block size: [{trainer.config.training.block_size}]")

    # run with asyncio task
    loop = asyncio.get_event_loop()

    # Training loop
    epoch_id = 0
    block_id = 0
    while trainer.trainer_status.global_step < trainer.config.training.max_steps:

        run_tag = f"{prefix_tag}_{epoch_id:03d}_{block_id:02d}"

        # train the model on the block
        await trainer.train_block(run_tag, dataset, eval_dataset)

        # increment the epoch id
        block_id += 1
        if block_id >= trainer.config.training.block_size:
            epoch_id += 1
            block_id = 0

        # check if training is complete
        if trainer.trainer_status.global_step >= trainer.config.training.max_steps:
            break

    # Final checkpoint
    trainer._save_checkpoint(trainer.trainer_status.global_step)

    if trainer.config.logging.use_wandb:
        wandb.finish()

async def train_async(prefix_tag: str, trainer: BaseTrainer, train_dataset: Dataset, eval_dataset: Dataset) -> bool:
    """Train the model asynchronously"""
    # Run sync function in thread pool
    try:
        await _train_loop(prefix_tag=prefix_tag, trainer=trainer, dataset=train_dataset, eval_dataset=eval_dataset)
        logger.info("🎉 Training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def create_sample_training_dataset(tokenizer, size: int = 100, max_length: int = 512) -> TextDataset:
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


async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train a model using BaseTrainer")
    parser.add_argument("--prefix-tag", type=str, default="v0.1.a",
                       help="Prefix tag for the training run")
    parser.add_argument("--config", type=str, default="trainerBase.yaml",
                       help="Path to configuration YAML file")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Override model name")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Override maximum training steps")

    args = parser.parse_args()

    # Load configuration
    try:
        config = TrainerConfig.from_yaml(args.config)
        logger.info(f"📜 Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Apply command line overrides
    if args.model_name:
        config.model.name = args.model_name
    if args.max_steps:
        config.training.max_steps = args.max_steps

    # Display configuration summary
    logger.info(f"📊 Training Config - Model: {config.model.name}, Steps: {config.training.max_steps}, Batch: {config.training.micro_batch_size}, LR: {config.training.learning_rate}")
    logger.info(f"⚙️ Optimizer Config - Type: {config.optimizer.optimizer_type}, Weight Decay: {config.optimizer.weight_decay}")
    if config.lora.use_lora:
        logger.info(f"🎯 LoRA Config - Rank: {config.lora.rank}, Alpha: {config.lora.alpha}")

    # Initialize trainer
    try:
        trainer = BaseTrainer(args.prefix_tag, config)
        logger.info("✅ Trainer initialized")
    except Exception as e:
        logger.error(f"❌ Trainer initialization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

    # Create datasets
    train_dataset = create_sample_training_dataset(trainer.tokenizer, size=20, max_length=trainer.config.model.max_seq_length)
    eval_dataset = create_sample_training_dataset(trainer.tokenizer, size=2, max_length=trainer.config.model.max_seq_length)

    logger.info(f"📊 Dataset created - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Start training
    success = await train_async(args.prefix_tag, trainer, train_dataset, eval_dataset)
    if not success:
        logger.error("❌ Training failed")
        sys.exit(1)

    try:
        logger.info("🎉 Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

    # Test generation
    logger.info("🤖 Testing text generation...")
    test_prompts = [
        "What is the future of AI?",
        "How can I improve my programming skills?",
        "Explain machine learning in simple terms.",
    ]

    for prompt in test_prompts:
        try:
            logger.info(f"💬 Testing prompt: {prompt}")
            generated = trainer.generate_text(prompt, max_length=50, temperature=0.7)
            logger.info(f"🤖 Generated: {generated}")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"❌ Generation failed for prompt '{prompt}': {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info("🎯 Training completed!")


if __name__ == "__main__":
    # run main async
    asyncio.run(main())
