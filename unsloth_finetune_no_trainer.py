#!/usr/bin/env python3
"""
Manual Fine-tuning with Unsloth and Hugging Face Models
No Trainer class - manual training loop implementation
"""

import os
import yaml
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.distributed as dist
from tqdm import tqdm
import time
from pathlib import Path

# Import unsloth first for optimizations
from unsloth import FastLanguageModel

# Import local utilities
from util import logger
from data_util import load_experiences, create_dataset, CustomDataCollatorWithMasking


def setup_distributed():
    """Initialize distributed training if running in distributed mode."""
    if "RANK" in os.environ:
        try:
            dist.init_process_group(backend="nccl", timeout=torch.distributed.default_pg_timeout)
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            
            # Test communication to ensure DDP is working
            if dist.is_initialized():
                test_tensor = torch.tensor([local_rank], dtype=torch.float32).cuda()
                dist.all_reduce(test_tensor)
                logger.info(f"DDP initialized successfully on rank {local_rank}")
            
            return local_rank
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            logger.info("Falling back to single GPU training")
            return 0
    return 0


class ManualUnslothTrainer:
    """Manual training loop implementation with Unsloth optimization."""
    
    def __init__(self, config_path="finetune.yaml"):
        self.local_rank = setup_distributed()
        self.is_distributed = "RANK" in os.environ
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
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
        self.training_loss = []
        
        # Set device
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        
    def log_main(self, message, level="info"):
        """Log message only on main process."""
        if self.local_rank == 0:
            getattr(logger, level)(message)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer using Unsloth."""
        model_config = self.config['model']
        
        self.log_main(f"Loading model: {model_config['name']}")
        
        # Configure device mapping
        if self.config['gpu'].get('single_gpu', False):
            device_map = {"": 0}
        elif self.is_distributed or self.config['gpu'].get('data_parallel', False):
            device_map = {"": self.local_rank}
        else:
            device_map = "auto"
        
        # Load model and tokenizer with Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['name'],
            max_seq_length=model_config['max_seq_length'],
            dtype=model_config['dtype'],
            load_in_4bit=model_config['load_in_4bit'],
            trust_remote_code=True,
            device_map=device_map
        )
        
        # Setup LoRA
        lora_config = self.config['lora']
        use_gradient_checkpointing = "unsloth" if not self.is_distributed else True
        
        if self.is_distributed:
            self.log_main("Using standard gradient checkpointing for multi-GPU compatibility")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config['r'],
            target_modules=lora_config['target_modules'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            bias=lora_config['bias'],
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Enable training mode
        self.model.train()
        
        # Setup DDP if distributed
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], find_unused_parameters=False
            )
        
        self.log_main("Model setup complete")
    
    def setup_dataset(self):
        """Setup training dataset and dataloader."""
        # Load experiences
        experiences = load_experiences(self.config['data']['local_dir'])
        
        # Create dataset
        self.train_dataset = create_dataset(
            experiences, 
            self.tokenizer,
            self.config['model']['max_seq_length'],
            self.local_rank
        )
        
        # Create data collator with dynamic padding
        self.collate_fn = CustomDataCollatorWithMasking(
            tokenizer=self.tokenizer,
            mlm=False,
            ignore_index=-100,
            max_length=self.config['model']['max_seq_length'],
            pad_to_multiple_of=8,  # For efficiency on modern GPUs
            use_dynamic_padding=True  # Enable dynamic length padding
        )
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['per_device_batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        
        self.log_main(f"Created dataset with {len(self.train_dataset)} examples")
        self.log_main(f"Dataloader has {len(self.train_dataloader)} batches")
        self.log_main("Using dynamic length batching for improved efficiency")
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Get model parameters (handle DDP case)
        if self.is_distributed:
            model_params = self.model.module.parameters()
        else:
            model_params = self.model.parameters()
        
        # Setup optimizer
        if training_config['optim'] == 'adamw':
            self.optimizer = AdamW(
                model_params,
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif training_config['optim'] == 'paged_adamw_8bit':
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.PagedAdamW8bit(
                    model_params,
                    lr=training_config['learning_rate'],
                    weight_decay=training_config['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to AdamW")
                self.optimizer = AdamW(
                    model_params,
                    lr=training_config['learning_rate'],
                    weight_decay=training_config['weight_decay']
                )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optim']}")
        
        # Calculate total training steps
        steps_per_epoch = len(self.train_dataloader)
        
        if training_config['max_steps'] > 0:
            self.total_steps = training_config['max_steps']
            self.num_epochs = math.ceil(self.total_steps / steps_per_epoch)
        else:
            self.num_epochs = training_config['num_train_epochs']
            self.total_steps = steps_per_epoch * self.num_epochs
        
        # Setup learning rate scheduler
        if training_config['lr_scheduler_type'] == 'cosine':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=training_config['warmup_steps'],
                num_training_steps=self.total_steps
            )
        elif training_config['lr_scheduler_type'] == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=training_config['warmup_steps'],
                num_training_steps=self.total_steps
            )
        else:
            # Default to linear warmup with cosine annealing
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=training_config['warmup_steps'],
                num_training_steps=self.total_steps
            )
        
        self.log_main(f"Setup optimizer: {training_config['optim']}")
        self.log_main(f"Total training steps: {self.total_steps}")
        self.log_main(f"Number of epochs: {self.num_epochs}")
    
    def compute_loss(self, batch):
        """Compute loss for a batch."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    def train_step(self, batch):
        """Execute one training step."""
        self.model.train()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Handle gradient accumulation
        if self.config['training']['gradient_accumulation_steps'] > 1:
            loss = loss / self.config['training']['gradient_accumulation_steps']
        
        # Backward pass
        loss.backward()
        
        return loss.item()
    
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        # Gradient clipping
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
    
    def save_checkpoint(self, output_dir: str, is_best: bool = False):
        """Save model checkpoint."""
        if self.local_rank != 0:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the model (handle DDP case)
        model_to_save = self.model.module if self.is_distributed else self.model
        
        # Save with FastLanguageModel for Unsloth compatibility
        try:
            model_to_save.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
            
            # Save training state
            checkpoint = {
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_loss': self.best_loss,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'config': self.config
            }
            
            torch.save(checkpoint, output_path / 'training_state.pt')
            
            if is_best:
                # Copy to best model directory
                best_path = output_path.parent / f"{output_path.name}_best"
                best_path.mkdir(parents=True, exist_ok=True)
                model_to_save.save_pretrained(str(best_path))
                self.tokenizer.save_pretrained(str(best_path))
            
            logger.info(f"Checkpoint saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def train(self):
        """Main training loop."""
        self.log_main("Starting manual training loop")
        
        training_config = self.config['training']
        gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        logging_steps = training_config['logging_steps']
        save_steps = training_config['save_steps']
        output_dir = training_config['output_dir']
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            step_loss = 0.0
            
            # Create progress bar for main process
            if self.local_rank == 0:
                pbar = tqdm(
                    self.train_dataloader, 
                    desc=f"Epoch {epoch+1}/{self.num_epochs}",
                    leave=True
                )
            else:
                pbar = self.train_dataloader
            
            for step, batch in enumerate(pbar):
                start_time = time.time()
                
                # Training step
                loss = self.train_step(batch)
                step_loss += loss
                epoch_loss += loss
                
                # Optimizer step (after accumulation)
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Update progress bar
                    if self.local_rank == 0:
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else training_config['learning_rate']
                        pbar.set_postfix({
                            'loss': f"{step_loss/gradient_accumulation_steps:.4f}",
                            'lr': f"{current_lr:.2e}",
                            'step': self.global_step
                        })
                    
                    # Logging
                    if self.global_step % logging_steps == 0:
                        avg_loss = step_loss / gradient_accumulation_steps
                        self.training_loss.append(avg_loss)
                        
                        self.log_main(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, "
                            f"lr={current_lr:.2e}, "
                            f"time={time.time()-start_time:.2f}s"
                        )
                    
                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        checkpoint_dir = f"{output_dir}/checkpoint-{self.global_step}"
                        is_best = avg_loss < self.best_loss
                        if is_best:
                            self.best_loss = avg_loss
                        self.save_checkpoint(checkpoint_dir, is_best=is_best)
                    
                    # Reset step loss
                    step_loss = 0.0
                    
                    # Check if we've reached max steps
                    if training_config['max_steps'] > 0 and self.global_step >= training_config['max_steps']:
                        break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            self.log_main(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save end-of-epoch checkpoint
            checkpoint_dir = f"{output_dir}/checkpoint-epoch-{epoch+1}"
            is_best = avg_epoch_loss < self.best_loss
            if is_best:
                self.best_loss = avg_epoch_loss
            self.save_checkpoint(checkpoint_dir, is_best=is_best)
            
            # Early exit if max steps reached
            if training_config['max_steps'] > 0 and self.global_step >= training_config['max_steps']:
                break
        
        # Final checkpoint
        final_dir = f"{output_dir}/final"
        self.save_checkpoint(final_dir)
        
        self.log_main("Training completed!")
        self.log_main(f"Best loss: {self.best_loss:.4f}")
        self.log_main(f"Total steps: {self.global_step}")
    
    def test_model(self, prompt: str):
        """Test the model with a prompt."""
        self.log_main("Testing model...")
        
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.log_main(f"Input: {prompt}")
        self.log_main(f"Output: {response}")
        
        return response


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manual Fine-tuning with Unsloth")
    parser.add_argument("--config", default="finetune.yaml", help="Config file path")
    parser.add_argument("--test-prompt", type=str, help="Test the model with a prompt after training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ManualUnslothTrainer(config_path=args.config)
    
    # Setup all components
    trainer.setup_model_and_tokenizer()
    trainer.setup_dataset()
    trainer.setup_optimizer_and_scheduler()
    
    # Start training
    trainer.train()
    
    # Test model if prompt provided
    if args.test_prompt:
        trainer.test_model(args.test_prompt)
    
    # Cleanup distributed training
    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 