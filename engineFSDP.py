import argparse
import asyncio
import hashlib
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
from torch.optim.lr_scheduler import LRScheduler
from torch.distributed.checkpoint.stateful import Stateful
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
    fully_shard,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig, OptimStateDictConfig
from torch.distributed.checkpoint import save as tdc_save, load as tdc_load
import functools

FSDP_TRAINER_STATE_FILE = "training_state.pt"
FSDP_TAG = "fsdp"

def setup_distributed():
    """Initialize distributed process group"""
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    
    return rank, world_size, device


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


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: optim.Optimizer,
        scheduler: Optional[LRScheduler]=None,
        global_step: Optional[int]=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = global_step

    def state_dict(self):
        rank = dist.get_rank()
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        # logger.info(f"🔍 [{self.__class__.__name__}-{rank}] Getting state dict...")
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        # logger.info(f"🔍 [{self.__class__.__name__}-{rank}] Got model state dict.")
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"]
        )
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.global_step is not None:
            self.global_step = state_dict["global_step"]


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

        self.rank, self.world_size, self.device = setup_distributed()
        logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] Distributed training initialized [Rank: {self.rank}, World Size: {self.world_size}, Device: {self.device}]")

        # Initialize model and tokenizer
        self.base_model, self.tokenizer = self._setup_model_and_tokenizer()

        self.model = self.base_model
        self.lora_cfg = None

        # Wrap model with FSDP
        self.model = self._setup_fsdp()
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] Model has [{num_params:,}] parameters")

        # future for async saving of checkpoint
        self.save_checkpoint_future = None
        # debug fsdp save and load
        self.fsdp_debug = os.getenv("FSDP_DEBUG", "false").lower() == "true"

        if not self.inference_mode:
            # setup logging
            self._setup_logging()
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

    def _setup_fsdp(self):
        """Setup FSDP wrapping"""
        logger.info(
            f"🔄 [{self.__class__.__name__}-{self.rank}] Setting up FSDP..."
        )

        # Get transformer layer class for auto wrapping
        transformer_layer_cls = get_transformer_layer_cls(self.config.model.name)
        
        # Create auto wrap policy
        if transformer_layer_cls:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_layer_cls},
            )
            logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] Using transformer auto wrap policy with {transformer_layer_cls}")
        else:
            # Fallback: use size-based wrapping
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=100_000,  # Wrap modules with at least 100k parameters
            )
            logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] Using size-based auto wrap policy (min 100k params)")

        # Setup mixed precision
        mixed_precision = MixedPrecision(
            param_dtype=getattr(torch, self.config.model.compute_dtype, torch.bfloat16),
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Debug: Check model structure before FSDP
        total_params_before = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] Model parameters before FSDP: [{total_params_before:,}]")
        
        # Wrap model with FSDP
        fsdp_model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=self.device,
            use_orig_params=True,  # Enable for better optimizer state compatibility
        )

        # fsdp_model = fully_shard(self.model)

        # Debug: Check model structure after FSDP
        total_params_after = sum(p.numel() for p in fsdp_model.parameters())
        logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] Model parameters after FSDP: [{total_params_after:,}]")
        
        # Debug: Check if model is actually FSDP wrapped
        is_fsdp_wrapped = hasattr(fsdp_model, '_fsdp_wrapped_module')
        logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] Model is FSDP wrapped: [{is_fsdp_wrapped}]")

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
        
        # For TDC checkpoints, check if the checkpoint directory exists
        # TDC creates its own internal structure
        return (
            checkpoint_path_obj.exists() and
            checkpoint_path_obj.is_dir()
        )

    def _load_checkpoint(self, checkpoint_location: str):
        """Load checkpoint for resuming training using TDC"""
        logger.info(
            f"🔄 [{self.__class__.__name__}-{self.rank}] Loading checkpoint from: {checkpoint_location}"
        )

        if self.fsdp_debug:
            self_model_state_hash = get_model_state_hash(self.model)
            self_optimizer_state_hash = get_optimizer_state_hash(self.optimizer)
            self_scheduler_state_hash = get_scheduler_state_hash(self.scheduler)

            logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Before load, self model state hash: [{self_model_state_hash}]")
            logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Before load, self optimizer state hash: [{self_optimizer_state_hash}]")
            logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Before load, self scheduler state hash: [{self_scheduler_state_hash}]")

        start_time = time.time()
        checkpoint_path_obj = Path(checkpoint_location)
        
        """Load checkpoint using TDC for distributed training"""
        try:
            app_state = AppState(self.model, self.optimizer, scheduler=self.scheduler, global_step=self.status.global_step)
            state_dict = { "app": app_state }
            tdc.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_path_obj,
            )

            self.status.global_step = app_state.global_step
            logger.info(f"📊 Loaded global_step from TDC metadata: [{self.status.global_step}]")
            logger.info(f"✅ TDC with FSDP: Optimizer state properly restored using TDC APIs in [{time.time() - start_time:.2f}s]")

            if self.fsdp_debug:
                app_model_state_hash = get_model_state_hash(app_state.model)
                app_optimizer_state_hash = get_optimizer_state_hash(app_state.optimizer)
                app_scheduler_state_hash = get_scheduler_state_hash(app_state.scheduler)

                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Loaded app model state hash: [{app_model_state_hash}]")
                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Loaded App optimizer state hash: [{app_optimizer_state_hash}]")
                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Loaded App scheduler state hash: [{app_scheduler_state_hash}]")

                self_model_state_hash = get_model_state_hash(self.model)
                self_optimizer_state_hash = get_optimizer_state_hash(self.optimizer)
                self_scheduler_state_hash = get_scheduler_state_hash(self.scheduler)

                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] After load, self model state hash: [{self_model_state_hash}]")
                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] After load, self optimizer state hash: [{self_optimizer_state_hash}]")
                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] After load, self scheduler state hash: [{self_scheduler_state_hash}]")
            
            # Synchronize all ranks after loading
            dist.barrier()

            logger.info(
                f"📜 [{self.__class__.__name__}-{self.rank}] TDC Checkpoint loaded - Step: [{self.status.global_step}] in [{time.time() - start_time:.2f}s]"
            )

        except Exception as e:
            logger.error(f"❌ [{self.__class__.__name__}-{self.rank}] Failed to load TDC checkpoint: {e}")
            logger.error(traceback.format_exc())
            raise

    def _save_checkpoint(self, step: int, callback: Optional[Callable] = None):
        """Save training checkpoint using TDC"""
        checkpoint_path = self.checkpoint_path / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # waits for checkpointing to finish if one exists, avoiding queuing more then one checkpoint request at a time
        if self.save_checkpoint_future is not None:
            self.save_checkpoint_future.result()

        """Save checkpoint using TDC for distributed training"""
        try:
            start_time = time.time()
            # Save config and tokenizer (only on rank 0)
            if self.rank == 0:
                self._save_config_and_tokenizer(checkpoint_path)

            # TDC requires ALL ranks to participate in save operation
            # Use AppState to make simple values compatible with TDC
            app_state = AppState(self.model, self.optimizer, scheduler=self.scheduler, global_step=self.status.global_step)
            state_dict = { "app": app_state }

            if self.fsdp_debug:
                self_model_state_hash = get_model_state_hash(self.model)
                self_optimizer_state_hash = get_optimizer_state_hash(self.optimizer)
                self_scheduler_state_hash = get_scheduler_state_hash(self.scheduler)

                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Before save, self model state hash: [{self_model_state_hash}]")
                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Before save, self optimizer state hash: [{self_optimizer_state_hash}]")
                logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Before save, self scheduler state hash: [{self_scheduler_state_hash}]")

            # Ensure all ranks are synchronized before TDC save
            dist.barrier()
            logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] synchronized all ranks before TDC save.")

            def checkpoint_cleanup_and_callback(*args, **kwargs):
                logger.info(
                    f"💾 [{self.__class__.__name__}-{self.rank}] TDC Checkpoint saved: [{checkpoint_path}] in [{time.time() - start_time:.2f}s]"
                )
                if self.rank == 0:
                    callback_start_time = time.time()
                    self._handle_checkpoint_cleanup_and_callback(checkpoint_path, callback)
                    logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] handled checkpoint cleanup and callback in [{time.time() - callback_start_time:.2f}s]")

            def prepend_callback_cf(fut, cb):
                # Register normally first (so it’s wrapped as expected)
                fut.add_done_callback(cb)
                # Danger: private internals ahead
                # _callbacks is a list of callables; _condition is a threading.Condition
                with fut._condition:              # private API
                    fut._done_callbacks.insert(0, fut._done_callbacks.pop())
                    logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] prepended callback to future [{fut}] with [{len(fut._done_callbacks)}] callbacks")

            # All ranks participate in TDC save
            try:
                async_save_start_time = time.time()
                self.save_checkpoint_future = tdc.async_save(
                    state_dict=state_dict,
                    checkpoint_id=checkpoint_path,
                )
                prepend_callback_cf(self.save_checkpoint_future, checkpoint_cleanup_and_callback)
                logger.info(f"💾 [{self.__class__.__name__}-{self.rank}] async TDC will save checkpoint to: [{checkpoint_path}] in [{time.time() - async_save_start_time:.2f}s]")
            except Exception as tdc_error:
                logger.error(f"❌ [{self.__class__.__name__}-{self.rank}] TDC save failed: {tdc_error} in [{time.time() - async_save_start_time:.2f}s]")
                # Synchronize all ranks even if save failed
                dist.barrier()
                raise tdc_error

            # Synchronize all ranks after saving
            # dist.barrier()
            # logger.info(f"🔍 [{self.__class__.__name__}-{self.rank}] synchronized all ranks after async TDC saving.")

        except Exception as e:
            logger.error(f"❌ [{self.__class__.__name__}-{self.rank}] Failed to save TDC checkpoint: {e} in [{time.time() - async_save_start_time:.2f}s]")
            logger.error(traceback.format_exc())
            raise

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
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )

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
                if self.status.global_step % self.config.training.save_steps == 0:
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
            # Always save checkpoint at the end of the block (only if save_steps is reached)
            if self.status.global_step % self.config.training.save_steps == 0:
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

def get_model_state_hash(model):
    """Get hash of model parameters for consistency checking in FSDP"""
    hasher = hashlib.md5()
    rank = dist.get_rank()
    
    # For FSDP models, we need to use the state dict approach to get consistent hashing
    # across all ranks. Each rank will hash its own sharded parameters, but we need
    # to ensure consistent ordering and handling.
    
    # Get the model state dict - this gives us the sharded parameters for this rank
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model.state_dict()
        
        # Count parameters for logging - handle both regular tensors and ShardedTensors
        num_params = 0
        for p in model_state_dict.values():
            # Check for ShardedTensor first (it's a subclass of torch.Tensor)
            if hasattr(p, 'local_shards'):  # This is a ShardedTensor
                num_params += p.local_shards()[0].tensor.numel()
            elif isinstance(p, torch.Tensor):  # This is a regular tensor
                num_params += p.numel()
        logger.info(f"🔍 Rank {rank}: Model has [{num_params:,}] parameters in state dict")
        
        # Sort by parameter names for consistent ordering across ranks
        # Each rank will hash its own sharded parameters
        for param_name in sorted(model_state_dict.keys()):
            param_tensor = model_state_dict[param_name]
            
            # Skip non-tensor entries (like metadata)
            if not isinstance(param_tensor, torch.Tensor) and not hasattr(param_tensor, 'size'):
                continue
            
            # For ShardedTensor, we need to get the local shard
            if hasattr(param_tensor, 'local_shards') and param_tensor.local_shards():
                # This is a ShardedTensor - get the local shard data
                local_shard = param_tensor.local_shards()[0].tensor
                tensor_to_hash = local_shard
            elif isinstance(param_tensor, torch.Tensor):
                # This is a regular tensor
                tensor_to_hash = param_tensor
            else:
                continue
                
            # Convert to float32 if needed for consistent hashing
            if tensor_to_hash.dtype == torch.bfloat16:
                tensor_to_hash = tensor_to_hash.float()
            elif tensor_to_hash.dtype == torch.float16:
                tensor_to_hash = tensor_to_hash.float()
                
            # Move to CPU and hash the tensor data
            hasher.update(tensor_to_hash.cpu().numpy().tobytes())
    
    return hasher.hexdigest()


def get_optimizer_state_hash(optimizer):
    """Get hash of optimizer state for consistency checking"""
    hasher = hashlib.md5()
    
    # Get optimizer state dict
    optim_state = optimizer.state_dict()
    
    # Convert to JSON string for hashing (handles nested structures)
    state_str = json.dumps(optim_state, sort_keys=True, default=str)
    hasher.update(state_str.encode())
    
    return hasher.hexdigest()


def get_scheduler_state_hash(scheduler):
    """Get hash of scheduler state for consistency checking"""
    if scheduler is None:
        return "no_scheduler"
    
    hasher = hashlib.md5()
    
    # Get scheduler state dict
    scheduler_state = scheduler.state_dict()
    
    # Convert to JSON string for hashing
    state_str = json.dumps(scheduler_state, sort_keys=True, default=str)
    hasher.update(state_str.encode())
    
    return hasher.hexdigest()


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
        "--sample-size", type=int, default=128, help="Override sample size"
    )
    parser.add_argument(
        "--sample-test-size", type=int, default=2, help="Override sample test size"
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
        trainer.tokenizer, size=args.sample_size, max_length=trainer.config.model.max_seq_length
    )
    eval_dataset = create_sample_training_dataset(
        trainer.tokenizer, size=args.sample_test_size, max_length=trainer.config.model.max_seq_length
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
    # rank, world_size, device = setup_distributed()
    # logger.info(f"🔍 [{__name__}-{rank}] Distributed training initialized")
    # logger.info(f"🔍 [{__name__}-{rank}] World size: {world_size}, Rank: {rank}, Device: {device}")
    # Run main async
    asyncio.run(main())
