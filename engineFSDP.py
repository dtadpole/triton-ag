import argparse
import asyncio
import copy
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as tdc
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.optim as optim
import wandb
import yaml
from engineBase import (
    create_sample_training_dataset,
    EngineBase,
    EngineConfig,
    TrainerStatus,
)
from logger import logger
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
    PeftConfig,
    PeftModel,
    set_peft_model_state_dict,
    TaskType,
)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from trainerUtil import SimpleCollator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    LocalStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig, OptimStateDictConfig
from torch.distributed.checkpoint import save as tdc_save, load as tdc_load


class MetadataWrapper:
    """Wrapper class to make simple values compatible with TDC"""
    def __init__(self, global_step: int, config: dict):
        self.global_step = global_step
        self.config = config
    
    def state_dict(self):
        return {
            "global_step": self.global_step,
            "config": self.config,
        }
    
    def load_state_dict(self, state_dict):
        self.global_step = state_dict.get("global_step", 0)
        self.config = state_dict.get("config", {})
import functools

FSDP_TRAINER_STATE_FILE = "training_state.pt"
FSDP_TAG = "fsdp"


def get_transformer_layer_cls(model_name: str):
    """Get the transformer layer class for auto wrapping"""
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
        from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        
        # Map model names to their decoder layer classes
        layer_map = {
            "llama": LlamaDecoderLayer,
            "mistral": MistralDecoderLayer,
            "qwen3": Qwen3DecoderLayer,  # Qwen3 models
            "qwen": Qwen2DecoderLayer,   # Qwen2/Qwen2.5 models
            "gemma": GemmaDecoderLayer,
            "phi": Phi3DecoderLayer,
        }
        
        model_name_lower = model_name.lower()
        for key, layer_cls in layer_map.items():
            if key in model_name_lower:
                return layer_cls
                
        # Default to LlamaDecoderLayer for most models
        return LlamaDecoderLayer
    except ImportError:
        logger.warning("Could not import transformer layer classes, using default wrapping")
        return None


class EngineFSDP(EngineBase):
    """FSDP trainer for Hugging Face models with step-by-step training implementation"""

    def __init__(
        self,
        prefix_tag: str,
        config: EngineConfig,
        status: Optional[TrainerStatus] = None,
        inference_mode: bool = False,
    ):
        super().__init__(prefix_tag, config, status, inference_mode)
        
        # Initialize distributed training
        if (
            os.environ.get("LOCAL_RANK") is not None
            and os.environ.get("WORLD_SIZE") is not None
        ):
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(
                f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
            )
            self.use_distributed = True
        else:
            self.rank = 0
            self.world_size = 1
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.use_distributed = False

        self.status = status if status is not None else TrainerStatus()

        # Initialize model and tokenizer
        self.base_model, self.tokenizer = self._setup_model_and_tokenizer()

        # Setup LoRA if enabled
        if self.config.lora.use_lora:
            self.model, self.lora_cfg = self._setup_lora()
        else:
            self.model = self.base_model
            self.lora_cfg = None

        # Wrap model with FSDP
        self.model = self._setup_fsdp()

        if not self.inference_mode:
            # Initialize optimizer and scheduler
            self.optimizer, self.scheduler = self._setup_optimizer_and_scheduler()

            # Initialize data collator
            self.data_collator = SimpleCollator(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=8,
            )

            logger.info(
                f"🔍 [{self.__class__.__name__}-{self.rank}] Config: {self.config.model_dump_json()}"
            )

            # Load checkpoint if specified
            if config.training.latest_checkpoint_name:
                checkpoint_location = (
                    self.checkpoint_path / config.training.latest_checkpoint_name
                )
                if self._checkpoint_exists(checkpoint_location):
                    self._load_checkpoint(checkpoint_location)
                else:
                    logger.warning(
                        f"⚠️ [{self.__class__.__name__}-{self.rank}] Checkpoint not found: {checkpoint_location} - Starting fresh training"
                    )

    def short_name(self):
        return "fsdp"

    def _base_model(self):
        return self.base_model

    def _lora_model(self):
        return self.model

    def _setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer"""
        logger.info(
            f"🚀 [{self.__class__.__name__}-{self.rank}] Loading model: {self.config.model.name}"
        )

        config = AutoConfig.from_pretrained(self.config.model.name)
        config.max_position_embeddings = self.config.model.max_seq_length
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            config=config,
            torch_dtype=getattr(torch, self.config.model.compute_dtype, torch.bfloat16),
            trust_remote_code=self.config.model.trust_remote_code,
        )
        model.config.use_cache = False  # required with grad ckpt
        if self.config.model.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name
            if self.config.model.tokenizer_name is None
            else self.config.model.tokenizer_name
        )

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Move model to the correct device
        model = model.to(self.device)
        logger.info(
            f"🔄 [{self.__class__.__name__}-{self.rank}] Model moved to device: {self.device}"
        )

        self._print_model_info(model)

        return model, tokenizer

    def _setup_lora(self):
        """Setup LoRA configuration"""
        logger.info(
            f"🌸 [{self.__class__.__name__}-{self.rank}] Setting up LoRA (rank={self.config.lora.rank}, alpha={self.config.lora.alpha})"
        )

        lora_cfg = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.lora.target_modules,
        )

        # Load checkpoint if specified
        checkpoint_location = (
            self.checkpoint_path / self.config.training.latest_checkpoint_name
        )
        if self._checkpoint_exists(checkpoint_location):
            lora_model = PeftModel.from_pretrained(
                self.base_model, checkpoint_location, is_trainable=True
            )
            logger.info(
                f"🌸 [{self.__class__.__name__}-{self.rank}] Loaded LoRA checkpoint: {checkpoint_location}"
            )
        else:
            logger.warning(
                f"⚠️ [{self.__class__.__name__}-{self.rank}] LoRA checkpoint not found: {checkpoint_location} - Starting fresh training..."
            )
            lora_model = get_peft_model(self.base_model, lora_cfg)

        # Move LoRA model to the correct device
        lora_model = lora_model.to(self.device)
        logger.info(
            f"🔄 [{self.__class__.__name__}-{self.rank}] LoRA model moved to device: {self.device}"
        )

        self._print_model_info(lora_model)
        return lora_model, lora_cfg

    def _setup_fsdp(self):
        """Setup FSDP wrapping"""
        logger.info(
            f"🔄 [{self.__class__.__name__}-{self.rank}] Setting up FSDP..."
        )

        # If not using distributed training, just return the model as-is
        if not self.use_distributed:
            logger.info(
                f"🔄 [{self.__class__.__name__}-{self.rank}] Single GPU mode - skipping FSDP wrapping"
            )
            return self.model

        # Get transformer layer class for auto wrapping
        transformer_layer_cls = get_transformer_layer_cls(self.config.model.name)
        
        # Create auto wrap policy
        if transformer_layer_cls:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_layer_cls},
            )
        else:
            auto_wrap_policy = None

        # Setup mixed precision
        mixed_precision = MixedPrecision(
            param_dtype=getattr(torch, self.config.model.compute_dtype, torch.bfloat16),
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Wrap model with FSDP
        fsdp_model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=self.device,
            use_orig_params=True,  # Enable for better optimizer state compatibility
        )

        logger.info(
            f"🔄 [{self.__class__.__name__}-{self.rank}] FSDP setup completed"
        )

        return fsdp_model

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Group parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.optimizer.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Create optimizer based on type
        optimizer_type = self.config.optimizer.optimizer_type.lower()
        if optimizer_type == "adamw":
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps,
            )
        elif optimizer_type == "adam":
            optimizer = optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps,
            )
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                momentum=self.config.optimizer.momentum,
                nesterov=self.config.optimizer.nesterov,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.config.optimizer.optimizer_type}"
            )

        # Setup learning rate scheduler
        scheduler = get_scheduler(
            self.config.training.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.config.training.num_warmup_steps,
            num_training_steps=self.config.training.max_steps,
        )

        logger.info(
            f"⚙️ [{self.__class__.__name__}-{self.rank}] Optimizer ({optimizer_type}) and scheduler initialized"
        )

        return optimizer, scheduler

    def _save_config_and_tokenizer(self, checkpoint_path: Path):
        """Save training config and tokenizer (rank 0 only)"""
        if self.rank == 0:
            # Save config
            with open(checkpoint_path / "training_config.yaml", "w") as f:
                yaml.dump(self.config.model_dump(), f, default_flow_style=False)
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_path)

    def _handle_checkpoint_cleanup_and_callback(self, checkpoint_path: Path, callback: Optional[Callable] = None):
        """Handle checkpoint cleanup, link updates, and callback execution (rank 0 only)"""
        if self.rank == 0:
            # Update latest checkpoint link
            self._update_latest_checkpoint_link(checkpoint_path)
            # Clean up old checkpoints
            self._cleanup_checkpoint(self.checkpoint_path)

            # Execute callback if provided
            if callback:
                try:
                    logger.info(
                        f"🔍 [{self.__class__.__name__}-{self.rank}] Running callback: {callback}"
                    )
                    callback(checkpoint_path)
                    logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] Callback completed")
                except Exception as e:
                    logger.error(f"❌ [{self.__class__.__name__}-{self.rank}] Error in callback: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] No callback provided")

    def _checkpoint_exists(self, checkpoint_path: str) -> bool:
        """Check if checkpoint exists"""
        checkpoint_path_obj = Path(checkpoint_path)
        
        if self.use_distributed:
            # For TDC checkpoints, check if the checkpoint directory exists
            # TDC creates its own internal structure
            return (
                checkpoint_path_obj.exists() and
                checkpoint_path_obj.is_dir()
            )
        else:
            # For single GPU checkpoints, check for the training state file
            return (checkpoint_path_obj / FSDP_TRAINER_STATE_FILE).exists()

    def _load_checkpoint(self, checkpoint_location: str):
        """Load checkpoint for resuming training using TDC"""
        logger.info(
            f"🔄 [{self.__class__.__name__}-{self.rank}] Loading checkpoint from: {checkpoint_location}"
        )

        checkpoint_path_obj = Path(checkpoint_location)
        
        if self.use_distributed:
            # Use TDC for distributed loading
            self._load_checkpoint_tdc(checkpoint_path_obj)
        else:
            # Use regular loading for single GPU
            self._load_checkpoint_single_gpu(checkpoint_path_obj)

    def _load_checkpoint_tdc(self, checkpoint_path_obj: Path):
        """Load checkpoint using TDC for distributed training"""
        try:
            # TDC automatically handles state dict type selection
            # Include all training state in the distributed checkpoint
            # Use MetadataWrapper to make simple values compatible with TDC
            metadata_wrapper = MetadataWrapper(global_step=0, config={})
            
            # Get state dict using the correct API that handles FSDP properly
            model_state_dict, optim_state_dict = get_state_dict(
                model=self.model,
                optimizers=self.optimizer,
            )
            
            # Combine with other state
            state_dict = {
                "model": model_state_dict,
                "optimizer": optim_state_dict,
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "metadata": metadata_wrapper,
            }
            
            tdc.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_path_obj,
            )
            
            # Set state dict for each individual object after loading
            set_state_dict(
                model=self.model,
                optimizers=self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
            )
            
            # Load scheduler state if available
            if self.scheduler and state_dict.get("scheduler") is not None:
                self.scheduler.load_state_dict(state_dict["scheduler"])
            
            # Extract metadata from the wrapper
            self.status.global_step = metadata_wrapper.global_step
            logger.info(f"📊 Loaded global_step from TDC metadata: {self.status.global_step}")
            logger.info("✅ TDC with FSDP: Optimizer state properly restored using TDC APIs")

            # Synchronize all ranks after loading
            if self.use_distributed:
                dist.barrier()

            logger.info(
                f"📜 [{self.__class__.__name__}-{self.rank}] TDC Checkpoint loaded - Step: [{self.status.global_step}]"
            )

        except Exception as e:
            logger.error(f"❌ [{self.__class__.__name__}-{self.rank}] Failed to load TDC checkpoint: {e}")
            logger.error(traceback.format_exc())
            raise

    def _load_checkpoint_single_gpu(self, checkpoint_path_obj: Path):
        """Load checkpoint for single GPU training"""
        training_state_path = (
            checkpoint_path_obj / FSDP_TRAINER_STATE_FILE
            if checkpoint_path_obj.is_dir()
            else checkpoint_path_obj
        )

        # Load checkpoint
        checkpoint = torch.load(training_state_path, map_location="cpu")

        # Load model state
        if self.config.lora.use_lora and "lora_state_dict" in checkpoint:
            set_peft_model_state_dict(self.model, checkpoint["lora_state_dict"])
        elif not self.config.lora.use_lora and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning(
                f"⚠️ [{self.__class__.__name__}-{self.rank}] Model state not found or incompatible in checkpoint"
            )

        # Load optimizer and scheduler state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.status.global_step = checkpoint.get("global_step", 0)

        logger.info(
            f"📜 [{self.__class__.__name__}-{self.rank}] Checkpoint loaded - Step: [{self.status.global_step}]"
        )

    def _save_checkpoint(self, step: int, callback: Optional[Callable] = None):
        """Save training checkpoint using TDC"""
        checkpoint_path = self.checkpoint_path / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        if self.use_distributed:
            # Use TDC for distributed saving
            self._save_checkpoint_tdc(checkpoint_path, step, callback)
        else:
            # Use regular saving for single GPU
            self._save_checkpoint_single_gpu(checkpoint_path, step, callback)

    def _save_checkpoint_tdc(self, checkpoint_path: Path, step: int, callback: Optional[Callable] = None):
        """Save checkpoint using TDC for distributed training"""
        try:
            # Save config and tokenizer (only on rank 0)
            self._save_config_and_tokenizer(checkpoint_path)

            # TDC requires ALL ranks to participate in save operation
            # Use MetadataWrapper to make simple values compatible with TDC
            metadata_wrapper = MetadataWrapper(
                global_step=self.status.global_step,
                config=self.config.model_dump()
            )
            
            # Get state dict using the correct API that handles FSDP properly
            model_state_dict, optim_state_dict = get_state_dict(
                model=self.model,
                optimizers=self.optimizer,
            )
            
            # Combine with other state
            state_dict = {
                "model": model_state_dict,
                "optimizer": optim_state_dict,
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "metadata": metadata_wrapper,
            }
            
            # All ranks participate in TDC save
            tdc.save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_path,
            )

            # Save LoRA model (only on rank 0) if using LoRA
            if self.config.lora.use_lora and self.rank == 0:
                self.model.save_pretrained(checkpoint_path)

            # Handle checkpoint cleanup and callback
            self._handle_checkpoint_cleanup_and_callback(checkpoint_path, callback)

            # Synchronize all ranks after saving
            if self.use_distributed:
                dist.barrier()

            logger.info(
                f"💾 [{self.__class__.__name__}-{self.rank}] TDC Checkpoint saved: {checkpoint_path}"
            )

        except Exception as e:
            logger.error(f"❌ [{self.__class__.__name__}-{self.rank}] Failed to save TDC checkpoint: {e}")
            logger.error(traceback.format_exc())
            raise

    def _save_checkpoint_single_gpu(self, checkpoint_path: Path, step: int, callback: Optional[Callable] = None):
        """Save checkpoint for single GPU training"""
        # Save config and tokenizer
        self._save_config_and_tokenizer(checkpoint_path)

        # Save model state
        if self.config.lora.use_lora:
            lora_state_dict = get_peft_model_state_dict(self.model)
            # Save LoRA model
            self.model.save_pretrained(checkpoint_path)
        else:
            model_state_dict = self.model.state_dict()
            # Save model
            torch.save(model_state_dict, checkpoint_path / "pytorch_model.bin")

        # Save training state
        checkpoint_state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.status.global_step,
            "config": self.config.model_dump(),
        }

        if self.config.lora.use_lora:
            checkpoint_state["lora_state_dict"] = get_peft_model_state_dict(self.model)
        else:
            checkpoint_state["model_state_dict"] = self.model.state_dict()

        # Save training state
        torch.save(checkpoint_state, checkpoint_path / FSDP_TRAINER_STATE_FILE)

        # Handle checkpoint cleanup and callback
        self._handle_checkpoint_cleanup_and_callback(checkpoint_path, callback)

        logger.info(
            f"💾 [{self.__class__.__name__}-{self.rank}] Checkpoint saved: {checkpoint_path}"
        )

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch"""
        # Move tensors to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        return loss

    def _backward_step(self, loss: torch.Tensor):
        """Execute a backward step"""
        loss.backward()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step"""
        self.model.train()

        # Compute loss
        loss = self._compute_loss(batch)

        # Scale loss for gradient accumulation
        loss = (
            loss
            * self.config.training.loss_multiplier
            / self.config.training.gradient_accumulation_steps
        )

        # Backward pass
        self._backward_step(loss)

        return loss.item()

    def _optimization_step(self):
        """Execute optimization step with gradient clipping"""
        # Clip gradients
        grad_norm = clip_grad_norm_(
            self.model.parameters(), self.config.training.max_grad_norm
        )

        # Update parameters
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad()

        return grad_norm

    def _get_current_lr(self):
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0]

    async def train_block(
        self,
        run_tag: str,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callback: Optional[Callable] = None,
    ):
        """Train the model for one block"""
        # Create data loader
        if self.use_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = None

        def worker_init_fn(worker_id):
            # Make dataloader workers deterministic but distinct per rank/worker
            base_seed = 1234
            seed = base_seed + self.rank * 10_000 + worker_id
            torch.manual_seed(seed)

        # Create dataloader
        batch_size = (
            self.config.training.micro_batch_size
            * self.config.training.gradient_accumulation_steps
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),  # Only shuffle if not using distributed sampler
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.data_collator,
            worker_init_fn=worker_init_fn,
        )

        logger.info(
            f"👉 [{self.__class__.__name__}-{self.rank}] [{run_tag}] Block started with [{len(dataloader)}] micro batches, Initial global step: [{self.status.global_step}]"
        )

        start_time = time.time()
        accumulated_loss = 0.0

        # Create progress bar (only on rank 0)
        if self.rank == 0:
            progress_bar = tqdm(total=len(dataloader), desc=run_tag, initial=0)

        for batch_idx, batch in enumerate(dataloader):
            # Training step
            step_loss = self._train_step(batch)
            accumulated_loss += step_loss

            # Optimization step (only after accumulation)
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                grad_norm = self._optimization_step()

                # Calculate average loss
                avg_loss = (
                    accumulated_loss / self.config.training.gradient_accumulation_steps
                )
                accumulated_loss = 0.0

                # Update step counter
                self.status.global_step += 1
                if self.rank == 0:
                    progress_bar.update(1)
                await asyncio.sleep(0.1)

                # Log metrics
                current_lr = self._get_current_lr()
                self._log_metrics(
                    {
                        "train/loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": grad_norm,
                        f"train_{self.short_name()}/loss": avg_loss,
                        f"train_{self.short_name()}/learning_rate": current_lr,
                        f"train_{self.short_name()}/grad_norm": grad_norm,
                    },
                    self.status.global_step,
                )

                # Save checkpoint
                if (
                    self.status.global_step % self.config.training.save_steps == 0
                    and self.rank == 0
                ):
                    self._save_checkpoint(self.status.global_step, callback=callback)

                # Evaluation
                if (
                    eval_dataset
                    and self.status.global_step % self.config.training.eval_steps == 0
                    and self.rank == 0
                ):
                    self._evaluate(eval_dataset)

                # Check if training is complete
                if self.status.global_step >= self.config.training.max_steps:
                    break

        try:
            # Always save checkpoint at the end of the block
            if self.rank == 0:
                self._save_checkpoint(self.status.global_step)
        except Exception as e:
            logger.error(
                f"❌ [{self.__class__.__name__}-{self.rank}] [{run_tag}] Failed to save checkpoint: {e}"
            )

        total_time = time.time() - start_time
        logger.info(
            f"🎉 [{self.__class__.__name__}-{self.rank}] [{run_tag}] Block completed in [{total_time:.1f}s] - Final global step: [{self.status.global_step}]"
        )

        if self.rank == 0:
            progress_bar.close()

        await asyncio.sleep(0.1)

    def _evaluate(self, eval_dataset: Dataset):
        """Evaluate the model on evaluation dataset"""
        logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Running evaluation...")

        self.model.eval()
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.training.micro_batch_size,
            shuffle=False,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.data_collator,
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

        logger.info(
            f"📊 [{self.__class__.__name__}-{self.rank}] Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}"
        )

        if self.config.logging.use_wandb:
            wandb.log(
                {
                    "eval/loss": avg_eval_loss,
                    "eval/perplexity": perplexity,
                    "eval/step": self.status.global_step,
                }
            )

        self.model.train()

    def generate_text(
        self, prompt: str, max_length: int = 100, temperature: float = 0.7
    ) -> str:
        """Generate text using the trained model"""
        self.model.eval()

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

        # Move inputs to the same device as the model
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # Store input length for proper output extraction
        input_length = inputs["input_ids"].shape[1]

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
                pad_token_id=(
                    self.tokenizer.eos_token_id
                    if self.tokenizer.eos_token_id
                    else self.tokenizer.pad_token_id
                ),
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Extract only the generated tokens (excluding input)
        generated_tokens = outputs[0][input_length:]

        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text.strip()


async def _train_loop(
    prefix_tag: str,
    trainer: EngineBase,
    dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
):
    """Main training loop"""
    logger.info(
        f"🏋️ [{trainer.__class__.__name__}-{trainer.rank}] Starting training - Total steps: [{trainer.config.training.max_steps}], Batch size: [{trainer.config.training.micro_batch_size}], Block size: [{trainer.config.training.block_size}]"
    )

    # Training loop
    epoch_id = 0
    block_id = 0
    while trainer.status.global_step < trainer.config.training.max_steps:

        run_tag = f"{prefix_tag}_{epoch_id:03d}_{block_id:02d}"

        # Train the model on the block
        await trainer.train_block(run_tag, dataset, eval_dataset)

        # Increment the epoch id
        block_id += 1
        if block_id >= trainer.config.training.block_size:
            epoch_id += 1
            block_id = 0

        # Check if training is complete
        if trainer.status.global_step >= trainer.config.training.max_steps:
            break

    # Final checkpoint
    if trainer.rank == 0:
        trainer._save_checkpoint(trainer.status.global_step)

    if trainer.config.logging.use_wandb:
        wandb.finish()


async def train_async(
    prefix_tag: str, trainer: EngineBase, train_dataset: Dataset, eval_dataset: Dataset
) -> bool:
    """Train the model asynchronously"""
    try:
        await _train_loop(
            prefix_tag=prefix_tag,
            trainer=trainer,
            dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        logger.info("🎉 Training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train a model using EngineFSDP")
    parser.add_argument(
        "--prefix-tag",
        type=str,
        default="auto.fsdp",
        help="Prefix tag for the training run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="engineBase.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="Override model name"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override maximum training steps"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = EngineConfig.from_yaml(args.config)
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
    logger.info(
        f"📊 Training Config - Model: {config.model.name}, Steps: {config.training.max_steps}, Batch: {config.training.micro_batch_size}, LR: {config.training.learning_rate}"
    )
    logger.info(
        f"⚙️ Optimizer Config - Type: {config.optimizer.optimizer_type}, Weight Decay: {config.optimizer.weight_decay}"
    )
    if config.lora.use_lora:
        logger.info(
            f"🎯 LoRA Config - Rank: {config.lora.rank}, Alpha: {config.lora.alpha}"
        )

    # Initialize trainer
    try:
        trainer = EngineFSDP(args.prefix_tag, config)
        logger.info(f"✅ [{trainer.__class__.__name__}-{trainer.rank}] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ Trainer initialization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

    # Create datasets
    train_dataset = create_sample_training_dataset(
        trainer.tokenizer, size=100, max_length=trainer.config.model.max_seq_length
    )
    eval_dataset = create_sample_training_dataset(
        trainer.tokenizer, size=2, max_length=trainer.config.model.max_seq_length
    )

    logger.info(
        f"📊 [{trainer.__class__.__name__}-{trainer.rank}] Dataset created - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}"
    )

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
    if trainer.rank == 0:
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
    # Run main async
    asyncio.run(main())
