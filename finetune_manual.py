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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import math
import time
from util import logger
from data_util import ExperienceDataset, SimpleDataCollator


def setup_distributed():
    """Initialize distributed training if running in distributed mode."""
    if "RANK" in os.environ:
        try:
            dist.init_process_group(backend="nccl", timeout=torch.distributed.default_pg_timeout)
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            
            # Test communication to ensure distributed training is working
            if dist.is_initialized():
                test_tensor = torch.tensor([local_rank], dtype=torch.float32).cuda()
                dist.all_reduce(test_tensor)
                logger.info(f"FSDP distributed training initialized successfully on rank {local_rank}")
            
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

    def get_model_for_save(self):
        """Get the underlying model for saving, handling PEFT, DDP, and FSDP cases."""
        if isinstance(self.model, FSDP):
            # This is an FSDP wrapped model - access the wrapped module
            return self.model._fsdp_wrapped_module
        elif hasattr(self.model, 'module'):
            # This is a DDP wrapped model
            return self.model.module
        else:
            # This is likely a PEFT model or regular model
            return self.model
    
    def get_model_parameters(self):
        """Get model parameters, handling PEFT, DDP, and FSDP cases."""
        # For FSDP and PEFT models, self.model.parameters() works directly
        return self.model.parameters()

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer using Unsloth."""
        model_config = self.config['model']
        
        self.log_main(f"Loading model: {model_config['name']}")
        
        # Configure device mapping
        gpu_config = self.config.get('gpu', {})
        if gpu_config.get('single_gpu', False):
            device_map = {"": 0}
        elif self.is_distributed or gpu_config.get('data_parallel', False):
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
            self.log_main("Using standard gradient checkpointing for FSDP compatibility")
        
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
        
        # Apply FSDP if distributed training
        if self.is_distributed:
            self.log_main("Wrapping model with FSDP")
            
            # Get FSDP configuration from config
            fsdp_config = self.config.get('fsdp', {})
            
            # Configure FSDP mixed precision
            mixed_precision_policy = None
            if fsdp_config.get('mixed_precision', True):
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
                self.log_main("FSDP mixed precision enabled (bfloat16)")
            
            sharding_strategy = getattr(ShardingStrategy, fsdp_config.get('sharding_strategy', 'FULL_SHARD'))
            
            # Wrap model with FSDP
            self.model = FSDP(
                self.model,
                mixed_precision=mixed_precision_policy,
                sharding_strategy=sharding_strategy,
                device_id=self.local_rank,
                use_orig_params=True,  # Important for PEFT compatibility
                sync_module_states=True,  # Ensure all ranks have same initial state
            )
            
            # Apply activation checkpointing if enabled
            if fsdp_config.get('activation_checkpointing', True):
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    CheckpointImpl,
                    apply_activation_checkpointing,
                )
                
                # Apply activation checkpointing to transformer layers
                def check_fn(submodule):
                    # Apply checkpointing to transformer blocks/layers
                    return any(name in submodule.__class__.__name__.lower() 
                             for name in ['block', 'layer', 'decoder'])
                
                apply_activation_checkpointing(
                    self.model,
                    checkpoint_wrapper_fn=lambda x: checkpoint_wrapper(x, CheckpointImpl.NO_REENTRANT),
                    check_fn=check_fn,
                )
                self.log_main("FSDP activation checkpointing enabled")
            
            self.log_main(f"Model wrapped with FSDP using {sharding_strategy} sharding strategy")
        
        self.log_main("Model setup complete")

    def setup_dataset(self):
        # Get data configuration
        data_config = self.config.get('data', {})
        data_dir = data_config.get('processed_dir', 'finetune_processed_experiences')
        
        # Create dataset using max_seq_length from model config
        self.train_dataset = ExperienceDataset(
            data_dir=data_dir,
            max_length=self.max_seq_length
        )
        
        # Create simple data collator
        collator_config = data_config.get('collator', {})
        self.collate_fn = SimpleDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=collator_config.get('pad_to_multiple_of', 8)
        )

        # Create dataloader
        dataloader_config = data_config.get('dataloader', {})
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['per_device_batch_size'],
            shuffle=dataloader_config.get('shuffle', True),
            num_workers=dataloader_config.get('num_workers', 4),
            pin_memory=dataloader_config.get('pin_memory', True),
            drop_last=dataloader_config.get('drop_last', True),
            collate_fn=self.collate_fn,
        )
        
        self.log_main(f"Dataloader has {len(self.train_dataloader)} batches")
        self.log_main(f"Created dataset with {len(self.train_dataset)} examples")

    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Get model parameters
        model_params = self.get_model_parameters()
        
        # Get optimizer configuration
        optim_config = training_config.get('optimizer_config', {})
        betas = optim_config.get('betas', (0.9, 0.999))
        eps = optim_config.get('eps', 1e-8)
            
        # Setup optimizer
        if training_config['optim'] == 'adamw':
            self.optimizer = AdamW(
                model_params,
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=betas,
                eps=eps
            )
        elif training_config['optim'] == 'paged_adamw_8bit':
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.PagedAdamW8bit(
                    model_params,
                    lr=training_config['learning_rate'],
                    weight_decay=training_config['weight_decay'],
                    betas=betas,
                    eps=eps
                )
            except ImportError:
                self.log_main("bitsandbytes not available, falling back to AdamW", "warning")
                self.optimizer = AdamW(
                    model_params,
                    lr=training_config['learning_rate'],
                    weight_decay=training_config['weight_decay'],
                    betas=betas,
                    eps=eps
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
        scheduler_type = training_config['lr_scheduler_type']
        if scheduler_type in ['cosine', 'linear']:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=training_config['warmup_steps'],
                num_training_steps=self.total_steps
            )
        else:
            # No scheduler
            self.scheduler = None
            
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
        training_config = self.config['training']
        
        # Gradient clipping
        max_grad_norm = training_config.get('max_grad_norm', 0.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.get_model_parameters(), max_grad_norm)
        
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
        
        self.log_main("Starting manual training loop")
        
        training_config = self.config['training']
        training_start_time = time.time()
        gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        logging_steps = training_config['logging_steps']
        save_steps = training_config['save_steps']
        output_dir = training_config['output_dir']
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            step_loss = 0.0
            epoch_start_time = time.time()
            
            # Create progress bar for main process
            if self.local_rank == 0:
                pbar = tqdm.tqdm(
                    self.train_dataloader, 
                    desc=f"Epoch {epoch+1}/{self.num_epochs}",
                    leave=True,
                    dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                )
            else:
                pbar = self.train_dataloader
            
            for step, batch in enumerate(pbar):
                step_start_time = time.time()
                
                # Training step
                loss = self.train_step(batch)
                step_loss += loss
                epoch_loss += loss
                
                # Optimizer step (after accumulation)
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    
                    # Calculate current learning rate and metrics
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else training_config['learning_rate']
                    avg_loss = step_loss / gradient_accumulation_steps
                    step_time = time.time() - step_start_time
                    
                    # Update progress bar with detailed info
                    if self.local_rank == 0:
                        # Calculate ETA and throughput
                        steps_remaining = self.total_steps - self.global_step
                        eta_seconds = steps_remaining * step_time if step_time > 0 else 0
                        eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
                        
                        pbar.set_postfix({
                            'loss': f"{avg_loss:.4f}",
                            'lr': f"{current_lr:.2e}",
                            'step': f"{self.global_step}/{self.total_steps}",
                            'time': f"{step_time:.1f}s",
                            'ETA': eta_str
                        })
                    
                    # Logging
                    if self.global_step % logging_steps == 0:
                        self.log_main(
                            f"Step {self.global_step}/{self.total_steps}: "
                            f"loss={avg_loss:.4f}, lr={current_lr:.2e}, "
                            f"time={step_time:.2f}s, epoch={epoch+1}/{self.num_epochs}"
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
                        if self.local_rank == 0:
                            pbar.set_description(f"Epoch {epoch+1}/{self.num_epochs} [MAX STEPS REACHED]")
                        break
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            
            # Update progress bar description for completed epoch
            if self.local_rank == 0:
                pbar.set_description(f"Epoch {epoch+1}/{self.num_epochs} [COMPLETED]")
                pbar.close()
            
            self.log_main(
                f"Epoch {epoch+1}/{self.num_epochs} completed in {epoch_time/60:.1f}m. "
                f"Average loss: {avg_epoch_loss:.4f}, Best loss: {self.best_loss:.4f}"
            )
            
            # Save end-of-epoch checkpoint
            checkpoint_dir = f"{output_dir}/checkpoint-epoch-{epoch+1}"
            is_best = avg_epoch_loss < self.best_loss
            if is_best:
                self.best_loss = avg_epoch_loss
                self.log_main(f"New best loss: {self.best_loss:.4f}")
            self.save_checkpoint(checkpoint_dir, is_best=is_best)
            
            # Early exit if max steps reached
            if training_config['max_steps'] > 0 and self.global_step >= training_config['max_steps']:
                self.log_main(f"Reached maximum steps ({training_config['max_steps']}). Stopping training.")
                break
        
        # Final checkpoint
        final_dir = f"{output_dir}/final"
        self.save_checkpoint(final_dir)
        
        # Training summary
        total_training_time = time.time() - training_start_time
        self.log_main("=" * 60)
        self.log_main("Training completed!")
        self.log_main(f"Total training time: {total_training_time/3600:.2f} hours")
        self.log_main(f"Total steps completed: {self.global_step}")
        self.log_main(f"Epochs completed: {self.epoch + 1}")
        self.log_main(f"Best loss achieved: {self.best_loss:.4f}")
        self.log_main(f"Final checkpoint saved to: {final_dir}")
        self.log_main("=" * 60)
    
    def save_checkpoint(self, output_dir: str, is_best: bool = False):
        """Save model checkpoint."""
        if self.local_rank != 0:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle FSDP model saving
        if isinstance(self.model, FSDP):
            # For FSDP, we need to use state_dict with full_state_dict
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state_dict = self.model.state_dict()
            
            # Get the underlying model for saving
            model_to_save = self.get_model_for_save()
            
            try:
                # Save model state dict manually for FSDP
                torch.save(model_state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
                
                # Save model config and tokenizer
                model_to_save.save_pretrained(output_dir, state_dict=model_state_dict)
                self.tokenizer.save_pretrained(output_dir)
                
                self.log_main(f"FSDP model checkpoint saved to {output_dir}")
                
            except Exception as e:
                self.log_main(f"Error saving FSDP checkpoint: {e}", "error")
                # Fallback to regular saving
                try:
                    model_to_save = self.get_model_for_save()
                    model_to_save.save_pretrained(output_dir)
                    self.tokenizer.save_pretrained(output_dir)
                    self.log_main(f"Fallback checkpoint saved to {output_dir}")
                except Exception as e2:
                    self.log_main(f"Fallback save also failed: {e2}", "error")
                    return
        else:
            # Regular model saving (non-FSDP)
            model_to_save = self.get_model_for_save()
            
            try:
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                self.log_main(f"Checkpoint saved to {output_dir}")
                
            except Exception as e:
                self.log_main(f"Error saving checkpoint: {e}", "error")
                return
        
        # Save training state (common for both FSDP and regular)
        try:
            checkpoint = {
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_loss': self.best_loss,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'config': self.config
            }
            
            torch.save(checkpoint, os.path.join(output_dir, 'training_state.pt'))
            
            if is_best:
                # Copy to best model directory
                best_dir = output_dir + "_best"
                os.makedirs(best_dir, exist_ok=True)
                
                if isinstance(self.model, FSDP):
                    # Copy FSDP model files
                    import shutil
                    for file in ['pytorch_model.bin', 'config.json', 'tokenizer.json', 'tokenizer_config.json']:
                        src = os.path.join(output_dir, file)
                        if os.path.exists(src):
                            shutil.copy2(src, best_dir)
                else:
                    model_to_save = self.get_model_for_save()
                    model_to_save.save_pretrained(best_dir)
                    self.tokenizer.save_pretrained(best_dir)
                
                self.log_main(f"Best model saved to {best_dir}")
                
        except Exception as e:
            self.log_main(f"Error saving training state: {e}", "error")

    def test_model(self, prompt=None):
        """Test the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            self.log_main("Model not loaded! Run train() first.", "error")
            return None
        
        if self.is_distributed and self.local_rank != 0:
            return "Inference skipped on non-zero rank"
        
        # Use default prompt if none provided
        if prompt is None:
            test_config = self.config.get('test', {})
            prompt = test_config.get('default_prompt', 
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n")
        
        FastLanguageModel.for_inference(self.model)
        device = f"cuda:{self.local_rank}" if self.is_distributed else "cuda"
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        
        # Get generation config
        test_config = self.config.get('test', {})
        generation_config = test_config.get('generation', {})
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config.get('max_new_tokens', 512),
                    use_cache=generation_config.get('use_cache', True),
                    temperature=generation_config.get('temperature', 0.7),
                    do_sample=generation_config.get('do_sample', True),
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
        logger.info("Testing the fine-tuned model...")
        response = trainer.test_model()  # Uses default prompt from config
        logger.info(f"Model response: {response}")


if __name__ == "__main__":
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    main()
