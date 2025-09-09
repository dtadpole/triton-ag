# Set environment variables for optimal performance
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import math
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import bitsandbytes as bnb
import numpy as np
import torch
import torch.optim as optim

# Import Unsloth first for optimal performance
import unsloth
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
    set_peft_model_state_dict,
    TaskType,
)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from trainerUtil import SimpleCollator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from unsloth import FastLanguageModel


UNSLOTH_TRAINER_STATE_FILE = "training_state.pt"


class EngineUnsloth(EngineBase):
    """Base trainer for Hugging Face models with step-by-step training implementation"""

    def __init__(
        self,
        prefix_tag: str,
        config: EngineConfig,
        status: Optional[TrainerStatus] = None,
        inference_mode: bool = False,
    ):
        super().__init__(prefix_tag, config, status, inference_mode)

        # Initialize model and tokenizer
        self.base_model, self.tokenizer = self._setup_model_and_tokenizer()

        # Setup LoRA if enabled
        if self.config.lora.use_lora:
            self.model = self._setup_lora()
        else:
            self.model = self.base_model

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
                f"🔍 [{self.__class__.__name__}] Config: {self.config.model_dump_json()}"
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
                        f"⚠️ [{self.__class__.__name__}] Checkpoint not found: {checkpoint_location} - Starting fresh training"
                    )

    def short_name(self):
        return "unsloth"

    def _base_model(self):
        return self.base_model

    def _lora_model(self):
        return self.model

    def _setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer using Unsloth"""
        logger.info(
            f"🚀 [{self.__class__.__name__}] Loading model: {self.config.model.name}"
        )

        # Load model with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.name,
            dtype=getattr(torch, self.config.model.compute_dtype, torch.bfloat16),
            max_seq_length=self.config.model.max_seq_length,
            load_in_4bit=self.config.model.load_in_4bit,
            load_in_8bit=self.config.model.load_in_8bit,
            full_finetuning=self.config.model.full_finetuning,
            use_gradient_checkpointing=(
                "unsloth" if self.config.model.use_gradient_checkpointing else None
            ),
            # token="hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Log model information
        self._print_model_info(model)

        return model, tokenizer

    def _setup_lora(self):
        """Setup LoRA configuration using Unsloth"""
        logger.info(
            f"🛠️ [{self.__class__.__name__}] Setting up LoRA (rank={self.config.lora.rank}, alpha={self.config.lora.alpha})"
        )

        # Apply LoRA with Unsloth
        lora_model = FastLanguageModel.get_peft_model(
            self.base_model,
            r=self.config.lora.rank,
            target_modules=self.config.lora.target_modules,
            modules_to_save=self.config.lora.modules_to_save,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            use_gradient_checkpointing=self.config.model.use_gradient_checkpointing,
            random_state=(
                self.config.training.seed
                if self.config.training.seed is not None
                and self.config.training.seed >= 0
                else random.randint(0, 2**31)
            ),
            use_rslora=False,  # Use regular LoRA
            loftq_config=None,
            # autocast_adapter_dtype=getattr(torch, self.config.model.compute_dtype, torch.bfloat16),
        )

        # Log LoRA information
        self._print_model_info(lora_model)

        return lora_model

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

        # Create optimizer based on type (using bitsandbytes paged optimizers)
        optimizer_type = self.config.optimizer.optimizer_type.lower()
        if optimizer_type == "adamw":
            # self.optimizer = bnb.optim.PagedAdamW(
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps,
            )
        elif optimizer_type == "adam":
            # self.optimizer = bnb.optim.PagedAdam(
            optimizer = optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.training.learning_rate,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps,
            )
        elif optimizer_type == "sgd":
            # Fall back to standard SGD as bitsandbytes doesn't have paged SGD
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
            f"⚙️ [{self.__class__.__name__}] Paged optimizer ({optimizer_type}) and scheduler initialized"
        )

        return optimizer, scheduler

    def _checkpoint_exists(self, checkpoint_path: str) -> bool:
        """Check if checkpoint exists"""
        checkpoint_path_obj = Path(checkpoint_path)
        return (checkpoint_path_obj / UNSLOTH_TRAINER_STATE_FILE).exists()

    def _load_checkpoint(self, checkpoint_location: str):
        """Load checkpoint for resuming training"""
        logger.info(
            f"🔄 [{self.__class__.__name__}] Loading checkpoint from: {checkpoint_location}"
        )

        # Get the training state file path
        checkpoint_path_obj = Path(checkpoint_location)
        training_state_path = (
            checkpoint_path_obj / "training_state.pt"
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
                f"⚠️ [{self.__class__.__name__}] Model state not found or incompatible in checkpoint"
            )

        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.status.global_step = checkpoint.get("global_step", 0)

        logger.info(
            f"📜 [{self.__class__.__name__}] Checkpoint loaded - Step: [{self.status.global_step}]"
        )

    def _save_checkpoint(self, step: int, callback: Optional[Callable] = None):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_path / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

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

        torch.save(checkpoint_state, checkpoint_path / "training_state.pt")

        # Save config
        with open(checkpoint_path / "training_config.yaml", "w") as f:
            yaml.dump(self.config.model_dump(), f, default_flow_style=False)

        # Update latest checkpoint link
        self._update_latest_checkpoint_link(checkpoint_path)
        # clean up old checkpoints
        self._cleanup_checkpoint(self.checkpoint_path)

        # Callback
        if callback:
            try:
                logger.info(
                    f"🔍 [{self.__class__.__name__}] Running callback: {callback}"
                )
                callback(checkpoint_path)
                logger.info(f"🔍 [{self.__class__.__name__}] Callback completed")
            except Exception as e:
                logger.error(f"❌ [{self.__class__.__name__}] Error in callback: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.info(f"🔍 [{self.__class__.__name__}] No callback provided")

        logger.info(
            f"💾 [{self.__class__.__name__}] Checkpoint saved: {checkpoint_path}"
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
        # logger.info(f"🔍 [{self.__class__.__name__}] Grad norm: [{grad_norm:.4f}] max grad norm: [{self.config.training.max_grad_norm:.2f}]")

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
        save_at_end: bool = False,
    ):
        """Train the model for one block"""
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.micro_batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

        logger.info(
            f"👉 [{self.__class__.__name__}] [{run_tag}] Block started with [{len(dataloader)}] micro batches, Initial global step: [{self.status.global_step}]"
        )

        start_time = time.time()
        accumulated_loss = 0.0

        # Create progress bar
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
                    and self.config.training.eval_steps > 0
                    and self.status.global_step % self.config.training.eval_steps == 0
                ):
                    self._evaluate(eval_dataset)

                # Check if training is complete
                if self.status.global_step >= self.config.training.max_steps:
                    break

        try:
            # always save checkpoint at the end of the block
            if save_at_end:
                if self.status.global_step % self.config.training.save_steps == 0:
                    # if end of block happens to be the save step, we don't need to save, because we already saved at the save step
                    pass
                else:
                    self._save_checkpoint(self.status.global_step)
        except Exception as e:
            logger.error(
                f"❌ [{self.__class__.__name__}] [{run_tag}] Failed to save checkpoint: {e}"
            )

        total_time = time.time() - start_time
        logger.info(
            f"🎉 [{self.__class__.__name__}] [{run_tag}] Block completed in [{total_time:.1f}s] - Final global step: [{self.status.global_step}]"
        )
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
            f"📊 [{self.__class__.__name__}] Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}"
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
        f"🏋️ [BaseTrainer] Starting training - Total steps: [{trainer.config.training.max_steps}], Batch size: [{trainer.config.training.micro_batch_size}], Block size: [{trainer.config.training.block_size}]"
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
    parser = argparse.ArgumentParser(description="Train a model using EngineUnsloth")
    parser.add_argument(
        "--prefix-tag",
        type=str,
        default="auto.unsloth",
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
        trainer = EngineUnsloth(args.prefix_tag, config)
        logger.info("✅ Trainer initialized")
    except Exception as e:
        logger.error(f"❌ Trainer initialization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

    # Create datasets
    train_dataset = create_sample_training_dataset(
        trainer.tokenizer, size=20, max_length=trainer.config.model.max_seq_length
    )
    eval_dataset = create_sample_training_dataset(
        trainer.tokenizer, size=2, max_length=trainer.config.model.max_seq_length
    )

    logger.info(
        f"📊 Dataset created - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}"
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
