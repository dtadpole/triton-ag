import argparse
import asyncio
import copy
import gc
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
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
from pydantic import BaseModel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from tqdm import tqdm
import wandb
import yaml
import argparse
from logger import logger
from trainerUtil import SimpleCollator
from workflowUtil import merge_dicts
from engineBase import EngineBase, EngineConfig, TrainerStatus, create_sample_training_dataset

TRAINING_STATUS_FILE = "training_status.json"
ADAPTER_MODEL_FILE = "adapter_model.safetensors"
CHECKPOINT_READY_FILE = "checkpoint.ready"
TRAINING_STATE_FILE = "training_state.pt"
DEEPSPEED_TAG = "ds"


class EngineDeepspeed(EngineBase):
    """Base trainer for Hugging Face models with step-by-step training implementation"""

    def __init__(
        self,
        prefix_tag: str,
        config: EngineConfig,
        status: Optional[TrainerStatus] = None,
        inference_mode: bool = False,
    ):
        super().__init__(prefix_tag, config, status, inference_mode)
        # for now keep these with deepspeed init
        if (
            os.environ.get("LOCAL_RANK") is not None
            and os.environ.get("WORLD_SIZE") is not None
        ):
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(
                f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.rank = 0
            self.world_size = 1
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        deepspeed.init_distributed(dist_backend="nccl")
        deepspeed_config_path = Path(
            os.path.expanduser(self.config.model.deepspeed_config_path)
        )
        if not deepspeed_config_path.exists():
            raise FileNotFoundError(
                f"Deepspeed config file not found: {deepspeed_config_path}"
            )
        with open(deepspeed_config_path, "r") as f:
            self.deepspeed_config = json.load(f)

        self.status = status if status is not None else TrainerStatus()

        # Initialize model and tokenizer
        self.base_model, self.tokenizer = self._setup_model_and_tokenizer()

        # Setup LoRA if enabled
        if self.config.lora.use_lora:
            self.model, self.lora_cfg = self._setup_lora()
        else:
            self.model = self.base_model
            self.lora_cfg = None

        before_init = time.time()
        self.engine, _, _, _ = deepspeed.initialize(
            config=self.deepspeed_config,
            model=self.model,
            model_parameters=self.model.parameters(),
        )
        self.device = self.engine.device
        after_init = time.time()
        logger.info(
            f"🔍 [{self.__class__.__name__}-{self.rank}] Deepspeed initialized in [{after_init - before_init:.1f}s]"
        )

        if not self.inference_mode:
            # Initialize data collator
            self.data_collator = SimpleCollator(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=8,
            )
            logger.info(
                f"🔍 [{self.__class__.__name__}-{self.rank}] Config: {self.config.model_dump_json()}"
            )

            # if checkpoint exists, load it
            checkpoint_location = (
                self.checkpoint_path / self.config.training.latest_checkpoint_name
            )
            if self._checkpoint_exists(checkpoint_location):
                self._load_checkpoint(checkpoint_location)

    def short_name(self):
        return "ds"

    def _base_model(self):
        return self.base_model

    def _lora_model(self):
        return self.engine.module

    def _setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer using Unsloth"""
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
            # device_map="auto",
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

        self._print_model_info(model)

        return model, tokenizer

    def _setup_lora(self):
        """Setup LoRA configuration using Unsloth"""
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

        self._print_model_info(lora_model)
        return lora_model, lora_cfg

    def _checkpoint_exists(self, checkpoint_path: str) -> bool:
        """Check if checkpoint exists"""
        checkpoint_path_obj = Path(checkpoint_path)
        for i in range(self.world_size):
            if not (
                checkpoint_path_obj / f"{CHECKPOINT_READY_FILE}.{self.world_size}.{i}"
            ).exists():
                return False
        return True

    def _load_checkpoint(self, checkpoint_location: str):
        """Initialize from checkpoint"""
        start_time = time.time()
        logger.info(
            f"🔄 [{self.__class__.__name__}-{self.rank}] Loading checkpoint from: {checkpoint_location}"
        )

        if self.world_size > 1:
            dist.barrier()
        # load training status always
        training_status_path = checkpoint_location / TRAINING_STATUS_FILE
        with open(training_status_path, "r") as f:
            loaded_status = json.load(f)
            self.status = TrainerStatus.model_validate(loaded_status)
            logger.info(
                f"🔍 [{self.__class__.__name__}-{self.rank}] Loaded training status: {self.status.model_dump()}"
            )

        # load optimizer state and scheduler state if using lora (lora model state is already loaded)
        if self.config.lora.use_lora:
            opt_state = torch.load(
                f"{checkpoint_location}/optim_rank{self.engine.global_rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
            # Build a list with only *this* rank's shard set
            shard_list = [None] * dist.get_world_size()
            shard_list[self.engine.global_rank] = opt_state
            self.engine.optimizer.load_state_dict(shard_list)
            logger.info(
                f"🔍 [{self.__class__.__name__}-{self.rank}] Loaded optimizer state from {checkpoint_location}"
            )
            # load scheduler state if using lora
            if hasattr(self.engine, "lr_scheduler") and os.path.exists(
                f"{checkpoint_location}/scheduler.pt"
            ):
                sch_state = torch.load(
                    f"{checkpoint_location}/scheduler.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                self.engine.lr_scheduler.load_state_dict(sch_state)
                logger.info(
                    f"🔍 [{self.__class__.__name__}-{self.rank}] Loaded scheduler state from {checkpoint_location}"
                )
        else:
            # load full model only if not using lora
            self.engine.load_checkpoint(checkpoint_location, tag=DEEPSPEED_TAG)
            logger.info(
                f"🔍 [{self.__class__.__name__}-{self.rank}] Loaded full model state"
            )
        if self.world_size > 1:
            dist.barrier()

        logger.info(
            f"📜 [{self.__class__.__name__}-{self.rank}] Checkpoint loaded - G-Step: [{self.status.global_step}] in [{time.time() - start_time:.1f}s]"
        )

    def _save_checkpoint(self, step: int, callback: Optional[Callable] = None):
        """Save training checkpoint"""
        start_time = time.time()
        checkpoint_path = self.checkpoint_path / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        if self.world_size > 1:
            dist.barrier()
        if self.engine.global_rank == 0:
            # Save config
            with open(checkpoint_path / "training_config.yaml", "w") as f:
                yaml.dump(self.config.model_dump(), f, default_flow_style=False)
            # save tokenizer
            self.tokenizer.save_pretrained(checkpoint_path)
            # save training status
            training_status_path = checkpoint_path / TRAINING_STATUS_FILE
            with open(training_status_path, "w") as f:
                json.dump(self.status.model_dump(), f)

        # directly save optimizer state and lora state if using lora
        if self.config.lora.use_lora:
            torch.save(
                self.engine.optimizer.state_dict(),
                f"{checkpoint_path}/optim_rank{self.engine.global_rank}.pt",
            )
            # Scheduler is small & identical on all ranks → save on rank 0
            if (
                hasattr(self.engine, "lr_scheduler")
                and self.engine.lr_scheduler is not None
                and self.engine.global_rank == 0
            ):
                torch.save(
                    self.engine.lr_scheduler.state_dict(),
                    f"{checkpoint_path}/scheduler.pt",
                )
            # now handle the lora parameters
            if int(self.engine.zero_optimization_stage()) == 3:
                # lora_params = [p for n, p in self.engine.module.named_parameters() if "lora" in n.lower()]
                lora_params = get_peft_model_state_dict(self.engine.module)
                num_lora_params = sum(p.numel() for p in lora_params)
                logger.info(
                    f"🔍 [{self.__class__.__name__}-{self.rank}] Gathering lora parameters: {num_lora_params:,}"
                )
                ctx = deepspeed.zero.GatheredParameters(lora_params, modifier_rank=0)
                with ctx:
                    # if self.world_size > 1: dist.barrier()
                    # only save one copy of the full lora model
                    peft_sd = (
                        lora_params  # get_peft_model_state_dict(self.engine.module)
                    )
                    peft_sd_cpu = {k: v.clone().cpu() for k, v in peft_sd.items()}
                    num_peft_params = sum(p.numel() for p in peft_sd_cpu.values())
                    logger.info(
                        f"🔍 [{self.__class__.__name__}-{self.rank}] Peft state dict: {num_peft_params:,}"
                    )
                    if self.engine.global_rank == 0:
                        # add barrier to ensure we have peft_sd on all ranks
                        base_model_copy = copy.deepcopy(self.base_model)
                        temp_peft_model = get_peft_model(
                            base_model_copy, self.lora_cfg, adapter_name="temp"
                        )  # on cpu
                        temp_peft_model.load_state_dict(
                            peft_sd_cpu, strict=False
                        )  # on cpu
                        temp_peft_model.save_pretrained(
                            checkpoint_path, safe_serialization=True
                        )
                        logger.info(
                            f"🔍 [{self.__class__.__name__}-{self.rank}] Saved lora model to {checkpoint_path}"
                        )
                        # clean up
                        del base_model_copy
                        del temp_peft_model
                        gc.collect()
                        torch.cuda.empty_cache()
                    # change adapter back to default
                    self.engine.module.set_adapter("default")
            else:
                self.engine.module.save_pretrained(checkpoint_path)
        else:
            self.engine.save_checkpoint(checkpoint_path, tag=DEEPSPEED_TAG)
        if self.world_size > 1:
            dist.barrier()

        # touch checkpoint ready file
        (
            checkpoint_path / f"{CHECKPOINT_READY_FILE}.{self.world_size}.{self.rank}"
        ).touch()

        # callback only on rank 0
        if self.engine.global_rank == 0:
            # Update latest checkpoint link
            self._update_latest_checkpoint_link(checkpoint_path)
            # clean up old checkpoints
            self._cleanup_checkpoint(self.checkpoint_path)

            # Callback
            if callback:
                try:
                    logger.info(
                        f"🔍 [{self.__class__.__name__}-{self.rank}] Running callback: {callback}.."
                    )
                    callback(checkpoint_path)
                    logger.info(
                        f"🔍 [{self.__class__.__name__}-{self.rank}] Callback completed."
                    )
                except Exception as e:
                    logger.error(
                        f"❌ [{self.__class__.__name__}-{self.rank}] Error in callback: [{type(e).__name__}: {e}]"
                    )
                    logger.error(traceback.format_exc())
            else:
                logger.info(
                    f"🔍 [{self.__class__.__name__}-{self.rank}] No callback provided"
                )

        logger.info(
            f"💾 [{self.__class__.__name__}-{self.rank}] Checkpoint saved: {checkpoint_path} in [{time.time() - start_time:.1f}s]"
        )

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch"""
        # Move tensors to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.engine(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        return loss

    def _backward_step(self, loss: torch.Tensor):
        """Execute a backward step"""
        self.engine.backward(loss)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step"""
        self.engine.train()

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
            self.engine.module.parameters(), self.config.training.max_grad_norm
        )
        # logger.info(f"🔍 [{self.__class__.__name__}] Grad norm: [{grad_norm:.4f}] max grad norm: [{self.config.training.max_grad_norm:.2f}]")

        self.engine.step()
        self.engine.zero_grad()

        return grad_norm

    def _get_current_lr(self):
        """Get current learning rate"""
        return self.engine.get_lr()[0]

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
            # make dataloader workers deterministic but distinct per rank/worker
            base_seed = 1234
            seed = base_seed + self.rank * 10_000 + worker_id
            torch.manual_seed(seed)

        # create dataloader
        batch_size = (
            self.config.training.micro_batch_size
            * self.config.training.gradient_accumulation_steps
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
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

        # Create progress bar
        if self.engine.global_rank == 0:
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
                if self.engine.global_rank == 0:
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
                    and self.engine.global_rank == 0
                ):
                    self._save_checkpoint(self.status.global_step, callback=callback)

                # Evaluation
                if (
                    eval_dataset
                    and self.config.training.eval_steps > 0
                    and self.status.global_step % self.config.training.eval_steps == 0
                ):
                    self._evaluate(eval_dataset)

                # Check if training is complete
                if self.status.global_step >= self.config.training.max_steps:
                    break

        try:
            # always save checkpoint at the end of the block
            self._save_checkpoint(self.status.global_step)
        except Exception as e:
            logger.error(
                f"❌ [{self.__class__.__name__}-{self.rank}] [{run_tag}] Failed to save checkpoint: [{type(e).__name__}: {e}]"
            )

        total_time = time.time() - start_time
        logger.info(
            f"🎉 [{self.__class__.__name__}-{self.rank}] [{run_tag}] Block completed in [{total_time:.1f}s] - Final global step: [{self.status.global_step}]"
        )

        if self.engine.global_rank == 0:
            progress_bar.close()

        await asyncio.sleep(0.1)

    def _evaluate(self, eval_dataset: Dataset):
        """Evaluate the model on evaluation dataset"""
        logger.info(f"📊 [{self.__class__.__name__}-{self.rank}] Running evaluation...")

        self.engine.eval()
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

        self.engine.train()

    def generate_text(
        self, prompt: str, max_length: int = 100, temperature: float = 0.7
    ) -> str:
        """Generate text using the trained model"""
        self.model.eval()

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

    # run with asyncio task
    loop = asyncio.get_event_loop()

    # Training loop
    epoch_id = 0
    block_id = 0
    while trainer.status.global_step < trainer.config.training.max_steps:

        run_tag = f"{prefix_tag}_{epoch_id:03d}_{block_id:02d}"

        # train the model on the block
        await trainer.train_block(run_tag, dataset, eval_dataset)

        # increment the epoch id
        block_id += 1
        if block_id >= trainer.config.training.block_size:
            epoch_id += 1
            block_id = 0

        # check if training is complete
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
    # Run sync function in thread pool
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
    parser = argparse.ArgumentParser(description="Train a model using EngineDeepspeed")
    parser.add_argument(
        "--prefix-tag",
        type=str,
        default="auto.deepspeed",
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
    args, _ = parser.parse_known_args()  # still robust to extras

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

    # Initialize trainer
    try:
        trainer = EngineDeepspeed(args.prefix_tag, config)
        logger.info(
            f"✅ [{trainer.__class__.__name__}-{trainer.rank}] Trainer initialized"
        )
    except Exception as e:
        logger.error(
            f"❌ [{trainer.__class__.__name__}] Trainer initialization failed: {e}"
        )
        logger.error(
            f" [{trainer.__class__.__name__}] Traceback: {traceback.format_exc()}"
        )
        sys.exit(1)

    # Create datasets
    train_dataset = create_sample_training_dataset(trainer.tokenizer, size=100, max_length=trainer.config.model.max_seq_length)
    # eval_dataset = create_sample_training_dataset(trainer.tokenizer, size=2, max_length=trainer.config.model.max_seq_length)

    logger.info(f"📊 [{trainer.__class__.__name__}-{trainer.rank}] Dataset created - Train: {len(train_dataset)}")

    # Start training
    success = await train_async(args.prefix_tag, trainer, train_dataset, None)
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
