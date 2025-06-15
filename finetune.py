#!/usr/bin/env python3
"""
Simplified Qwen Fine-tuning with Unsloth
Clean and efficient script for fine-tuning Qwen models.
"""

# Import unsloth first for optimizations
from unsloth import FastLanguageModel

import os
import yaml
from torch.utils.data import DataLoader
import torch
import tqdm
import torch.distributed as dist
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import math
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


class QwenUnslothTrainer:
    """Simplified Qwen fine-tuning with Unsloth."""
    
    def __init__(self, config_path="finetune.yaml"):
        self.local_rank = setup_distributed()
        self.is_distributed = "RANK" in os.environ
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        self.max_seq_length = self.config['model']['max_seq_length']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
       
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
        
        self.log_main("Model setup complete")

    def setup_dataset(self):
        # Create dataset
        experiences = load_experiences(self.config['data']['local_dir'])
        self.train_dataset = create_dataset(
            experiences, 
            self.tokenizer,
            self.max_seq_length,
            self.local_rank
        )
        self.collate_fn = CustomDataCollatorWithMasking(
            tokenizer=self.tokenizer,
            mlm=False,
            ignore_index=-100
        )

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

    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Get model parameters
        if self.is_distributed:
            model_params = self.model.module.parameters()
        else:
            model_params = self.model.parameters()
            
        # Setup optimizer
        if self.config['training']['optim'] == 'adamw':
            self.optimizer = AdamW(
                model_params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config['training']['optim'] == 'paged_adamw_8bit':
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.PagedAdamW8bit(
                    model_params,
                    lr=self.config['training']['learning_rate'],
                    weight_decay=self.config['training']['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to AdamW")
                self.optimizer = AdamW(
                    model_params,
                    lr=self.config['training']['learning_rate'],
                    weight_decay=self.config['training']['weight_decay']
                )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optim']}")
            
        # Calculate total training steps
        steps_per_epoch = len(self.train_dataloader)
        if self.config['training']['max_steps'] > 0:
            self.total_steps = self.config['training']['max_steps']
            self.num_epochs = math.ceil(self.total_steps / steps_per_epoch)
        else:
            self.num_epochs = self.config['training']['num_train_epochs']
            self.total_steps = steps_per_epoch * self.num_epochs
            
        # Setup learning rate scheduler
        if self.config['training']['lr_scheduler_type'] == 'cosine':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config['training']['warmup_steps'],
                num_training_steps=self.total_steps
            )
        elif self.config['training']['lr_scheduler_type'] == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config['training']['warmup_steps'],
                num_training_steps=self.total_steps
            )
        else:
            # No scheduler
            self.scheduler = None
            
        self.log_main(f"Total training steps: {self.total_steps}")
        self.log_main(f"Number of epochs: {self.num_epochs}")

    def train_step(self, batch):
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
        loss = loss / self.config['training']['gradient_accumulation_steps']
        
        # Backward pass
        loss.backward()
        
        return {'loss': loss.item() * self.config['training']['gradient_accumulation_steps']}
        
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        # Gradient clipping
        max_grad_norm = self.config['training'].get('max_grad_norm', 0.5)
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

    def train(self):
        """Main training loop."""
        self.log_main("Starting training...")

        # Setup everything
        self.setup_model_and_tokenizer()
        self.setup_dataset()
        self.setup_optimizer_and_scheduler()
        
        # Training loop
        total_loss = 0.0
        steps_since_log = 0
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.log_main(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
                
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm.tqdm(
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
                    if self.global_step % self.config['training']['logging_steps'] == 0:
                        avg_loss = total_loss / steps_since_log
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config['training']['learning_rate']
                        
                        self.log_main(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, "
                            f"lr={current_lr:.2e}, "
                            f"epoch={epoch + 1}"
                        )
                        
                        # Reset logging counters
                        total_loss = 0.0
                        steps_since_log = 0
                        
                    # Checkpointing
                    if self.global_step % self.config['training']['save_steps'] == 0:
                        checkpoint_dir = os.path.join(
                            self.config['training']['output_dir'],
                            f"checkpoint-{self.global_step}"
                        )
                        is_best = avg_loss < self.best_loss if 'avg_loss' in locals() else False
                        if is_best:
                            self.best_loss = avg_loss
                            
                        self.save_checkpoint(checkpoint_dir, is_best)
                        
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config['training']['learning_rate']
                progress_bar.set_postfix({
                    'loss': step_loss,
                    'lr': current_lr
                })
                
                # Check if we've reached max steps
                if (self.config['training']['max_steps'] > 0 and 
                    self.global_step >= self.config['training']['max_steps']):
                    self.log_main(f"Reached max steps ({self.config['training']['max_steps']})")
                    break
                    
            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            self.log_main(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Early stopping check or max steps reached
            if (self.config['training']['max_steps'] > 0 and 
                self.global_step >= self.config['training']['max_steps']):
                break
                
        # Final checkpoint
        if not self.is_distributed or self.local_rank == 0:
            output_dir = self.config['training']['output_dir']
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self.log_main(f"Training completed! Model saved to {output_dir}")
    
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

    def test_model(self, prompt):
        """Test the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            self.log_main("Model not loaded! Run train() first.", "error")
            return None
        
        if self.is_distributed and self.local_rank != 0:
            return "Inference skipped on non-zero rank"
        
        FastLanguageModel.for_inference(self.model)
        device = f"cuda:{self.local_rank}" if self.is_distributed else "cuda"
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    use_cache=True,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.batch_decode(outputs)[0]
            return response[len(prompt):].strip()
        except Exception as e:
            self.log_main(f"Error during inference: {e}", "error")
            return f"Inference failed: {e}"


def main():
    """Main function to run the fine-tuning process."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if local_rank == 0:
        logger.info("Starting Qwen fine-tuning with Unsloth...")
    
    trainer = QwenUnslothTrainer()
    trainer.train()
    
    # Test model
    if local_rank == 0:
        test_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n"
        logger.info("Testing the fine-tuned model...")
        response = trainer.test_model(test_prompt)
        logger.info(f"Model response: {response}")


if __name__ == "__main__":
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    main()
