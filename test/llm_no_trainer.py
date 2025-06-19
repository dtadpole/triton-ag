#!/usr/bin/env python3
"""
Manual LLM Training Script without HuggingFace Trainer
Implements custom training loop with full control over the training process.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import time
import math
import gc

# Import custom utilities
from util import logger
from data_util import load_experiences, create_dataset, CustomDataCollatorWithMasking

# Try to import unsloth for optimized model loading
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False
    logger.warning("Unsloth not available, using standard transformers")


class ManualLLMTrainer:
    """Manual LLM training implementation without HuggingFace Trainer."""
    
    def __init__(self, config_path: str = "finetune.yaml"):
        """Initialize the trainer with configuration."""
        self.config_path = config_path
        self.load_config()
        
        # Setup distributed training
        self.setup_distributed()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Device setup
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        
        self.log_main(f"Initialized trainer on device: {self.device}")
        
    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract commonly used config values
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.lora_config = self.config['lora']
        self.data_config = self.config['data']
        
    def setup_distributed(self):
        """Setup distributed training if available."""
        self.is_distributed = "RANK" in os.environ
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if self.is_distributed:
            try:
                dist.init_process_group(backend="nccl")
                self.log_main(f"Distributed training initialized - Rank: {self.local_rank}, World Size: {self.world_size}")
            except Exception as e:
                logger.error(f"Failed to initialize distributed training: {e}")
                self.is_distributed = False
                self.world_size = 1
                
    def log_main(self, message: str, level: str = "info"):
        """Log message only on main process."""
        if self.local_rank == 0:
            getattr(logger, level)(message)
            
    def setup_model_and_tokenizer(self):
        """Setup the model and tokenizer."""
        self.log_main(f"Loading model: {self.model_config['name']}")
        
        if USE_UNSLOTH:
            self._setup_with_unsloth()
        else:
            self._setup_with_transformers()
            
        # Setup LoRA
        self._setup_lora()
        
        # Setup for distributed training
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            
        self.log_main("Model and tokenizer setup complete")
        
    def _setup_with_unsloth(self):
        """Setup model using Unsloth for optimization."""
        device_map = {"": self.local_rank} if self.is_distributed else "auto"
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config['name'],
            max_seq_length=self.model_config['max_seq_length'],
            dtype=self.model_config['dtype'],
            load_in_4bit=self.model_config['load_in_4bit'],
            trust_remote_code=True,
            device_map=device_map
        )
        
    def _setup_with_transformers(self):
        """Setup model using standard transformers."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['name'],
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        device_map = {"": self.local_rank} if self.is_distributed else "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['name'],
            torch_dtype=torch.float16 if self.model_config['dtype'] is None else self.model_config['dtype'],
            device_map=device_map,
            trust_remote_code=True,
            load_in_4bit=self.model_config.get('load_in_4bit', False)
        )
        
    def _setup_lora(self):
        """Setup LoRA configuration."""
        if USE_UNSLOTH:
            # Use Unsloth's LoRA setup
            use_gradient_checkpointing = "unsloth" if not self.is_distributed else True
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_config['r'],
                target_modules=self.lora_config['target_modules'],
                lora_alpha=self.lora_config['alpha'],
                lora_dropout=self.lora_config['dropout'],
                bias=self.lora_config['bias'],
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
        else:
            # Use standard PEFT LoRA setup
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['alpha'],
                lora_dropout=self.lora_config['dropout'],
                target_modules=self.lora_config['target_modules'],
                bias=self.lora_config['bias']
            )
            
            self.model = get_peft_model(self.model, peft_config)
            
        # Enable gradient checkpointing if specified
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            
        # Print trainable parameters
        self._print_trainable_parameters()
        
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_param = 0
        
        for param in self.model.parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        self.log_main(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.4f}"
        )
        
    def setup_data(self):
        """Setup training data."""
        self.log_main("Setting up training data...")
        
        # Load experiences
        experiences = load_experiences(self.data_config['local_dir'])
        
        # Create dataset
        train_dataset = create_dataset(
            experiences,
            self.tokenizer,
            self.model_config['max_seq_length'],
            self.local_rank
        )
        
        # Create data collator
        data_collator = CustomDataCollatorWithMasking(
            tokenizer=self.tokenizer,
            mlm=False,
            ignore_index=-100
        )
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_config['per_device_batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=data_collator,
        )
        
        self.log_main(f"Training dataset size: {len(train_dataset)}")
        self.log_main(f"Number of batches per epoch: {len(self.train_dataloader)}")
        
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Get model parameters
        if self.is_distributed:
            model_params = self.model.module.parameters()
        else:
            model_params = self.model.parameters()
            
        # Setup optimizer
        if self.training_config['optim'] == 'adamw':
            self.optimizer = AdamW(
                model_params,
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.training_config['optim'] == 'paged_adamw_8bit':
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.PagedAdamW8bit(
                    model_params,
                    lr=self.training_config['learning_rate'],
                    weight_decay=self.training_config['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to AdamW")
                self.optimizer = AdamW(
                    model_params,
                    lr=self.training_config['learning_rate'],
                    weight_decay=self.training_config['weight_decay']
                )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config['optim']}")
            
        # Calculate total training steps
        steps_per_epoch = len(self.train_dataloader)
        if self.training_config['max_steps'] > 0:
            self.total_steps = self.training_config['max_steps']
            self.num_epochs = math.ceil(self.total_steps / steps_per_epoch)
        else:
            self.num_epochs = self.training_config['num_train_epochs']
            self.total_steps = steps_per_epoch * self.num_epochs
            
        # Setup learning rate scheduler
        if self.training_config['lr_scheduler_type'] == 'cosine':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.training_config['warmup_steps'],
                num_training_steps=self.total_steps
            )
        elif self.training_config['lr_scheduler_type'] == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.training_config['warmup_steps'],
                num_training_steps=self.total_steps
            )
        else:
            # No scheduler
            self.scheduler = None
            
        self.log_main(f"Total training steps: {self.total_steps}")
        self.log_main(f"Number of epochs: {self.num_epochs}")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / self.training_config['gradient_accumulation_steps']
        
        # Backward pass
        loss.backward()
        
        return {'loss': loss.item() * self.training_config['gradient_accumulation_steps']}
        
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        # Gradient clipping
        if hasattr(self.training_config, 'max_grad_norm'):
            max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
            if self.is_distributed:
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Zero gradients
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
    def save_checkpoint(self, output_dir: str, is_best: bool = False):
        """Save model checkpoint."""
        if self.local_rank != 0:
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if self.is_distributed:
            model_to_save = self.model.module
        else:
            model_to_save = self.model
            
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(output_dir, 'training_state.bin'))
        
        if is_best:
            # Save as best model
            best_dir = os.path.join(output_dir, 'best')
            os.makedirs(best_dir, exist_ok=True)
            model_to_save.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)
            
        self.log_main(f"Checkpoint saved to {output_dir}")
        
    def train(self):
        """Main training loop."""
        self.log_main("Starting training...")
        
        # Setup everything
        self.setup_model_and_tokenizer()
        self.setup_data()
        self.setup_optimizer_and_scheduler()
        
        # Training loop
        total_loss = 0.0
        steps_since_log = 0
        gradient_accumulation_steps = self.training_config['gradient_accumulation_steps']
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.log_main(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
                
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=self.local_rank != 0
            )
            
            for step, batch in enumerate(progress_bar):
                # Training step
                step_outputs = self.train_step(batch)
                step_loss = step_outputs['loss']
                
                total_loss += step_loss
                epoch_loss += step_loss
                steps_since_log += 1
                epoch_steps += 1
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    
                    # Logging
                    if self.global_step % self.training_config['logging_steps'] == 0:
                        avg_loss = total_loss / steps_since_log
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.training_config['learning_rate']
                        
                        self.log_main(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, "
                            f"lr={current_lr:.2e}, "
                            f"epoch={epoch + 1}"
                        )
                        
                        # Reset logging counters
                        total_loss = 0.0
                        steps_since_log = 0
                        
                    # Checkpointing
                    if self.global_step % self.training_config['save_steps'] == 0:
                        checkpoint_dir = os.path.join(
                            self.training_config['output_dir'],
                            f"checkpoint-{self.global_step}"
                        )
                        is_best = avg_loss < self.best_loss if steps_since_log == 0 else False
                        if is_best:
                            self.best_loss = avg_loss
                            
                        self.save_checkpoint(checkpoint_dir, is_best)
                        
                        # Clean up old checkpoints
                        self._cleanup_checkpoints()
                        
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': step_loss,
                    'lr': current_lr if 'current_lr' in locals() else self.training_config['learning_rate']
                })
                
                # Check if we've reached max steps
                if (self.training_config['max_steps'] > 0 and 
                    self.global_step >= self.training_config['max_steps']):
                    self.log_main(f"Reached max steps ({self.training_config['max_steps']})")
                    break
                    
            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            self.log_main(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Early stopping check or max steps reached
            if (self.training_config['max_steps'] > 0 and 
                self.global_step >= self.training_config['max_steps']):
                break
                
        # Final checkpoint
        final_dir = os.path.join(self.training_config['output_dir'], 'final')
        self.save_checkpoint(final_dir)
        
        self.log_main("Training completed!")
        
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints based on save_total_limit."""
        if self.local_rank != 0:
            return
            
        save_total_limit = self.training_config.get('save_total_limit', 3)
        if save_total_limit <= 0:
            return
            
        output_dir = Path(self.training_config['output_dir'])
        checkpoints = [
            d for d in output_dir.iterdir() 
            if d.is_dir() and d.name.startswith('checkpoint-')
        ]
        
        if len(checkpoints) <= save_total_limit:
            return
            
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-save_total_limit]:
            import shutil
            shutil.rmtree(checkpoint)
            self.log_main(f"Removed old checkpoint: {checkpoint}")


def main():
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manual LLM Training")
    parser.add_argument("--config", default="finetune.yaml", help="Path to config file")
    parser.add_argument("--output_dir", help="Override output directory")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ManualLLMTrainer(config_path=args.config)
    
    # Override output dir if specified
    if args.output_dir:
        trainer.training_config['output_dir'] = args.output_dir
        
    # Run training
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.log_main("Training interrupted by user")
    except Exception as e:
        trainer.log_main(f"Training failed with error: {e}", level="error")
        raise
    finally:
        # Cleanup distributed training
        if trainer.is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
