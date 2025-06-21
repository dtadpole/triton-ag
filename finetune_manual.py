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
import gc
import psutil
from datetime import datetime
from pathlib import Path
from util import logger
from data_util import ExperienceDataset, SimpleDataCollator
from huggingface_hub import HfApi, create_repo


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
        self.output_path = None

        self.max_seq_length = self.config['model']['max_seq_length']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Memory and performance tracking
        self.initial_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
        self.peak_memory = 0
        self.training_start_time = None
       
    def log_main(self, message, level="info"):
        """Log message only on main process."""
        if self.local_rank == 0:
            getattr(logger, level)(message)
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            return total - allocated
        return 0.0
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _format_time(self, seconds: float) -> str:
        """Format time in a readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m{seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"
    
    def _get_memory_info(self) -> dict:
        """Get comprehensive memory information."""
        info = {}
        if torch.cuda.is_available():
            info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3 
            info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info['gpu_free'] = info['gpu_total'] - info['gpu_allocated']
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        info['cpu_used'] = memory_info.rss / 1024**3  # GB
        info['cpu_available'] = psutil.virtual_memory().available / 1024**3  # GB
        
        return info
    
    def _get_model_size_mb(self) -> float:
        """Calculate model size in MB."""
        try:
            if self.model is None:
                return 0.0
                
            model_to_check = self.get_model_for_save()
            param_size = 0
            for param in model_to_check.parameters():
                param_size += param.nelement() * param.element_size()
            
            buffer_size = 0
            for buffer in model_to_check.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
        except Exception:
            return 0.0
    
    def _calculate_average_speed(self) -> float:
        """Calculate average training speed in samples per second."""
        try:
            if not hasattr(self, 'train_dataset') or self.training_start_time is None:
                return 0.0
                
            total_time = time.time() - self.training_start_time
            total_samples = self.global_step * self.config['training']['per_device_batch_size'] * self.config['training']['gradient_accumulation_steps']
            
            if total_time > 0:
                return total_samples / total_time
            return 0.0
        except Exception:
            return 0.0

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
        self.log_main("=" * 80)
        self.log_main("STARTING FINE-TUNING")
        self.log_main("=" * 80)

        # Setup everything
        self.setup_model_and_tokenizer()
        self.setup_dataset()
        self.setup_optimizer_and_scheduler()
        
        # Initialize training metrics
        self.training_start_time = time.time()
        training_config = self.config['training']
        gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        logging_steps = training_config['logging_steps']
        save_steps = training_config['save_steps']
        output_dir = training_config['output_dir']
        
        # Memory and performance tracking
        start_memory = self._get_memory_info()
        if self.local_rank == 0:
            self.log_main(f"Initial GPU memory: {start_memory.get('gpu_allocated', 0):.2f} GB")
            self.log_main(f"Available GPU memory: {start_memory.get('gpu_free', 0):.2f} GB")
            self.log_main(f"Training Configuration:")
            self.log_main(f"  • Total steps: {self.total_steps}")
            self.log_main(f"  • Epochs: {self.num_epochs}")
            self.log_main(f"  • Batch size: {training_config['per_device_batch_size']}")
            self.log_main(f"  • Gradient accumulation: {gradient_accumulation_steps}")
            self.log_main(f"  • Learning rate: {training_config['learning_rate']}")
            self.log_main(f"  • Max sequence length: {self.max_seq_length}")
            self.log_main("=" * 80)
        
        # Create overall training progress bar
        if self.local_rank == 0:
            overall_pbar = tqdm.tqdm(
                total=self.total_steps,
                desc="Training Progress",
                leave=True,
                dynamic_ncols=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]',
                colour='cyan',
                position=0,
                unit='step'
            )
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            step_loss = 0.0
            epoch_start_time = time.time()
            samples_processed = 0
            
            # Epoch header
            if self.local_rank == 0:
                self.log_main(f"📊 EPOCH {epoch+1}/{self.num_epochs}")
                self.log_main("-" * 80)
            
            # Create enhanced epoch progress bar like finetune_unsloth.py
            if self.local_rank == 0:
                pbar = tqdm.tqdm(
                    self.train_dataloader, 
                    desc=f"Epoch {epoch+1}/{self.num_epochs}",
                    leave=False,
                    dynamic_ncols=True,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                    colour='green',
                    disable=False,
                    unit='batch',
                    position=1,
                    miniters=1,
                    mininterval=0.5
                )
            else:
                pbar = self.train_dataloader
            
            for step, batch in enumerate(pbar):
                step_start_time = time.time()
                
                # Training step
                loss = self.train_step(batch)
                step_loss += loss
                epoch_loss += loss
                
                # Count samples processed
                batch_size = batch['input_ids'].size(0)
                samples_processed += batch_size
                
                # Update peak memory tracking
                current_memory = self._get_gpu_memory()
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # Optimizer step (after accumulation)
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    
                    # Calculate metrics
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else training_config['learning_rate']
                    avg_loss = step_loss / gradient_accumulation_steps
                    step_time = time.time() - step_start_time
                    
                    # Calculate throughput
                    effective_batch_size = batch_size * gradient_accumulation_steps
                    samples_per_sec = effective_batch_size / step_time if step_time > 0 else 0
                    
                    # Calculate ETA
                    steps_remaining = self.total_steps - self.global_step
                    if step_time > 0 and steps_remaining > 0:
                        eta_seconds = steps_remaining * step_time
                        eta_str = self._format_time(eta_seconds)
                    else:
                        eta_str = "N/A"
                    
                    # Calculate progress percentage
                    progress_pct = (self.global_step / self.total_steps) * 100 if self.total_steps > 0 else 0
                    
                    # Update progress bars with real-time metrics
                    if self.local_rank == 0:
                        elapsed_time = time.time() - self.training_start_time
                        memory_info = self._get_memory_info()
                        
                        # Update epoch progress bar postfix
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.1e}',
                            'gpu': f'{memory_info.get("gpu_allocated", 0):.1f}GB',
                            'speed': f'{samples_per_sec:.1f}s/s'
                        })
                        
                        # Update overall progress bar
                        overall_pbar.update(1)
                        overall_pbar.set_postfix({
                            'epoch': f'{epoch+1}/{self.num_epochs}',
                            'loss': f'{avg_loss:.4f}',
                            'best': f'{self.best_loss:.4f}',
                            'lr': f'{current_lr:.1e}',
                            'gpu': f'{memory_info.get("gpu_allocated", 0):.1f}GB'
                        })
                    
                    # Additional detailed logging at intervals (less frequent to avoid cluttering progress bars)
                    if self.global_step % (logging_steps * 10) == 0 and self.local_rank == 0:
                        elapsed_time = time.time() - self.training_start_time
                        memory_info = self._get_memory_info()
                        
                        # Print a newline to separate from progress bars
                        print()
                        detailed_log = (
                            f"📊 Step {self.global_step:,}/{self.total_steps:,}: "
                            f"Elapsed: {self._format_time(elapsed_time)} | "
                            f"GPU Memory: {memory_info.get('gpu_allocated', 0):.1f}GB/{memory_info.get('gpu_total', 0):.1f}GB | "
                            f"Peak Memory: {self.peak_memory:.1f}GB | "
                            f"CPU Memory: {memory_info.get('cpu_used', 0):.1f}GB"
                        )
                        self.log_main(detailed_log)
                    
                    # Save checkpoint with enhanced logging
                    if self.global_step % save_steps == 0:
                        checkpoint_dir = f"{output_dir}/checkpoint-{self.global_step}"
                        is_best = avg_loss < self.best_loss
                        if is_best:
                            self.best_loss = avg_loss
                            self.log_main(f"💫 NEW BEST LOSS: {self.best_loss:.4f} (Step {self.global_step})")
                        
                        self.log_main(f"💾 Saving checkpoint: {checkpoint_dir}")
                        self.save_checkpoint(checkpoint_dir, is_best=is_best)
                        
                        # Memory cleanup after checkpoint
                        self._cleanup_memory()
                    
                    # Reset step loss
                    step_loss = 0.0
                    
                    # Check if we've reached max steps
                    if training_config['max_steps'] > 0 and self.global_step >= training_config['max_steps']:
                        if self.local_rank == 0:
                            self.log_main(f"🛑 Maximum steps reached at step {self.global_step}")
                        break
            
            # End of epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            total_elapsed = time.time() - self.training_start_time
            
            # Close progress bar for completed epoch
            if self.local_rank == 0:
                pbar.close()
            
            # Enhanced epoch summary
            memory_info = self._get_memory_info()
            samples_per_epoch = len(self.train_dataset)
            epoch_throughput = samples_per_epoch / epoch_time if epoch_time > 0 else 0
            
            self.log_main("-" * 80)
            self.log_main(f"✅ EPOCH {epoch+1}/{self.num_epochs} COMPLETED")
            self.log_main(f"   Duration: {self._format_time(epoch_time)}")
            self.log_main(f"   Average Loss: {avg_epoch_loss:.4f}")
            self.log_main(f"   Best Loss: {self.best_loss:.4f}")
            self.log_main(f"   Samples Processed: {samples_processed:,}")
            self.log_main(f"   Throughput: {epoch_throughput:.1f} samples/s")
            self.log_main(f"   GPU Memory: {memory_info.get('gpu_allocated', 0):.1f}GB (Peak: {self.peak_memory:.1f}GB)")
            self.log_main(f"   Total Elapsed: {self._format_time(total_elapsed)}")
            self.log_main("-" * 80)
            
            # Save end-of-epoch checkpoint
            checkpoint_dir = f"{output_dir}/checkpoint-epoch-{epoch+1}"
            is_best = avg_epoch_loss < self.best_loss
            if is_best:
                self.best_loss = avg_epoch_loss
                self.log_main(f"💫 NEW BEST EPOCH LOSS: {self.best_loss:.4f}")
            
            self.save_checkpoint(checkpoint_dir, is_best=is_best)
            
            # Early exit if max steps reached
            if training_config['max_steps'] > 0 and self.global_step >= training_config['max_steps']:
                self.log_main(f"🛑 Reached maximum steps ({training_config['max_steps']:,}). Stopping training.")
                # Close progress bars before breaking
                if self.local_rank == 0:
                    pbar.close()
                break
        
        # Final checkpoint
        final_dir = f"{output_dir}/final"
        self.log_main(f"💾 Saving final checkpoint: {final_dir}")
        self.save_checkpoint(final_dir)
        
        # Store the output path for potential upload
        self.output_path = final_dir
        
        # Close overall progress bar
        if self.local_rank == 0:
            overall_pbar.close()
        
        # Final cleanup and summary
        self._cleanup_memory()
        total_training_time = time.time() - self.training_start_time
        final_memory = self._get_memory_info()
        
        # Enhanced training summary
        self.log_main("=" * 80)
        self.log_main("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        self.log_main("=" * 80)
        self.log_main(f"📊 TRAINING STATISTICS:")
        self.log_main(f"   • Total Duration: {self._format_time(total_training_time)}")
        self.log_main(f"   • Steps Completed: {self.global_step:,}/{self.total_steps:,}")
        self.log_main(f"   • Epochs Completed: {self.epoch + 1}/{self.num_epochs}")
        self.log_main(f"   • Best Loss: {self.best_loss:.4f}")
        self.log_main("")
        self.log_main(f"💾 MODEL ARTIFACTS:")
        self.log_main(f"   • Final Checkpoint: {final_dir}")
        self.log_main(f"   • Model Size: {self._get_model_size_mb():.1f} MB")
        self.log_main("")
        self.log_main(f"🖥️  SYSTEM PERFORMANCE:")
        self.log_main(f"   • Peak GPU Memory: {self.peak_memory:.1f}GB")
        self.log_main(f"   • Final GPU Memory: {final_memory.get('gpu_allocated', 0):.1f}GB")
        self.log_main(f"   • Average Speed: {self._calculate_average_speed():.1f} samples/s")
        self.log_main("")
        if torch.cuda.is_available():
            self.log_main(f"🚀 GPU INFO:")
            self.log_main(f"   • Device: {torch.cuda.get_device_name()}")
            self.log_main(f"   • Memory Used: {final_memory.get('gpu_allocated', 0):.1f}GB / {final_memory.get('gpu_total', 0):.1f}GB")
        self.log_main("=" * 80)
    
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
            self.log_main("❌ Model not loaded! Run train() first.", "error")
            return None
        
        if self.is_distributed and self.local_rank != 0:
            return "Inference skipped on non-zero rank"
        
        self.log_main("🧪 TESTING FINE-TUNED MODEL")
        self.log_main("-" * 60)
        
        # Use default prompt if none provided
        if prompt is None:
            test_config = self.config.get('test', {})
            prompt = test_config.get('default_prompt', 
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n")
        
        self.log_main(f"📝 Test Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Enable inference mode - handle quantized models
        try:
            FastLanguageModel.for_inference(self.model)
        except Exception as e:
            self.log_main(f"⚠️  Fast inference mode failed, using regular mode: {e}", "warning")
            # For quantized models, we might need to use regular mode
            self.model.eval()
        
        device = f"cuda:{self.local_rank}" if self.is_distributed else "cuda"
        
        # Get generation config
        test_config = self.config.get('test', {})
        generation_config = test_config.get('generation', {})
        max_tokens = generation_config.get('max_new_tokens', 512)
        temperature = generation_config.get('temperature', 0.7)
        
        self.log_main(f"⚙️  Generation Config: max_tokens={max_tokens}, temperature={temperature}")
        
        try:
            inference_start = time.time()
            inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
            
            with torch.no_grad():
                try:
                    # Try with unsloth's fast generation first
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        use_cache=generation_config.get('use_cache', True),
                        temperature=temperature,
                        do_sample=generation_config.get('do_sample', True),
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                except Exception as gen_e:
                    self.log_main(f"⚠️  Fast generation failed, trying alternative approach: {gen_e}", "warning")
                    # Fallback to more basic generation without some optimization parameters
                    try:
                        outputs = self.model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs.get('attention_mask'),
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            do_sample=generation_config.get('do_sample', True),
                            use_cache=False,  # Disable cache for quantized models
                            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id
                        )
                    except Exception as gen_e2:
                        self.log_main(f"⚠️  Alternative generation also failed, trying minimal approach: {gen_e2}", "warning")
                        # Last resort: minimal generation parameters
                        outputs = self.model.generate(
                            inputs['input_ids'],
                            max_new_tokens=min(max_tokens, 256),  # Limit tokens
                            do_sample=False,  # Use greedy decoding
                            use_cache=False,
                            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id
                        )
            
            inference_time = time.time() - inference_start
            response = self.tokenizer.batch_decode(outputs)[0]
            generated_text = response[len(prompt):].strip()
            
            # Calculate generation metrics
            output_tokens = len(self.tokenizer.encode(generated_text))
            tokens_per_sec = output_tokens / inference_time if inference_time > 0 else 0
            
            self.log_main("-" * 60)
            self.log_main("✅ INFERENCE COMPLETED")
            self.log_main(f"   • Generation Time: {inference_time:.2f}s")
            self.log_main(f"   • Output Tokens: {output_tokens}")
            self.log_main(f"   • Speed: {tokens_per_sec:.1f} tokens/s")
            self.log_main("-" * 60)
            self.log_main("🤖 MODEL RESPONSE:")
            self.log_main(f"{generated_text}")
            self.log_main("-" * 60)
            
            return generated_text
            
        except Exception as e:
            self.log_main(f"❌ Error during inference: {e}", "error")
            import traceback
            self.log_main(f"Traceback: {traceback.format_exc()}", "error")
            return f"Inference failed: {e}"

    def upload_to_huggingface(self, model_path: str = None, repo_name: str = None):
        """Upload the fine-tuned model to Hugging Face Hub."""
        if self.local_rank != 0:
            return  # Only upload from rank 0
            
        # Get configuration for HF upload
        upload_config = self.config.get('huggingface', {})
        if not upload_config.get('upload', False):
            self.log_main("Hugging Face upload is disabled")
            return
        
        if model_path is None:
            model_path = self.output_path or f"{self.config['training']['output_dir']}/final"
        
        if repo_name is None:
            repo_name = upload_config.get('repo_name')
            
        if not repo_name:
            self.log_main("No Hugging Face repository name specified", "error")
            return
            
        model_path = Path(model_path)
        
        try:
            self.log_main(f"Uploading model to Hugging Face: {repo_name}")
            
            # Initialize HF API
            api = HfApi()
            
            # Create repository if it doesn't exist
            try:
                create_repo(
                    repo_id=repo_name,
                    token=upload_config.get('token'),
                    private=upload_config.get('private', False),
                    exist_ok=True
                )
                self.log_main(f"Repository {repo_name} is ready")
            except Exception as e:
                self.log_main(f"Repository creation/check failed: {e}", "warning")
            
            # Upload all files in the model directory
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_name,
                token=upload_config.get('token'),
                commit_message=f"Upload fine-tuned model - {upload_config.get('commit_message', 'Fine-tuned model')}",
                ignore_patterns=["*.git*", "__pycache__", "*.pyc", "training_state.pt"]
            )
            
            self.log_main(f"Successfully uploaded model to https://huggingface.co/{repo_name}")
            
            # Create a model card if specified
            if upload_config.get('create_model_card', True):
                self._create_model_card(api, repo_name, upload_config)
                
        except Exception as e:
            self.log_main(f"Error uploading to Hugging Face: {e}", "error")
            self.log_main("Make sure you have:")
            self.log_main("1. Set your HF_TOKEN environment variable or specify token in config")
            self.log_main("2. Have write access to the repository")
            self.log_main("3. Installed huggingface_hub: pip install huggingface_hub")
    
    def _create_model_card(self, api: HfApi, repo_name: str, upload_config: dict):
        """Create a model card for the uploaded model."""
        try:
            model_card_content = f"""---
library_name: peft
base_model: {self.config['model']['name']}
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
- triton-ag
- unsloth
- lora
---

# {repo_name}

This model is a fine-tuned version of [{self.config['model']['name']}](https://huggingface.co/{self.config['model']['name']}) using Unsloth and LoRA.

## Model Details

- **Base Model:** {self.config['model']['name']}
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Max Sequence Length:** {self.config['model']['max_seq_length']}
- **Training Examples:** {len(self.train_dataset) if hasattr(self, 'train_dataset') else 'N/A'}
- **LoRA Rank:** {self.config['lora']['r']}
- **LoRA Alpha:** {self.config['lora']['alpha']}

## Training Configuration

- **Epochs:** {self.config['training']['num_train_epochs']}
- **Learning Rate:** {self.config['training']['learning_rate']}
- **Batch Size:** {self.config['training']['per_device_batch_size']}
- **Gradient Accumulation Steps:** {self.config['training']['gradient_accumulation_steps']}
- **Best Loss:** {self.best_loss:.4f}

## Usage

```python
from unsloth import FastLanguageModel
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{repo_name}",
    max_seq_length={self.config['model']['max_seq_length']},
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Format your prompt
messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Your question here"}}
]

formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# Generate
inputs = tokenizer(formatted_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Data

This model was fine-tuned on processed conversation experiences for improved performance on specific tasks.

## Limitations

- This is a LoRA adapter that requires the base model to function
- Performance may vary depending on the specific use case
- The model inherits any limitations from the base model

## Framework Versions

- Unsloth: 2025.6.1
- Transformers: 4.52.4
- PyTorch: 2.7.0
- PEFT: Latest

"""
            
            # Upload model card
            api.upload_file(
                path_or_fileobj=model_card_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_name,
                token=upload_config.get('token'),
                commit_message="Add model card"
            )
            
            self.log_main("Model card created successfully")
            
        except Exception as e:
            self.log_main(f"Failed to create model card: {e}", "warning")


def main():
    """Main function to run the fine-tuning process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen models using Unsloth with manual training loop")
    parser.add_argument(
        "--config", 
        type=str, 
        default="finetune.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test the model, skip training"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Test prompt for model testing"
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        default=True,
        help="Upload model to Hugging Face Hub after training"
    )
    parser.add_argument(
        "--hf-repo-user",
        type=str,
        default="dtadpole",
        help="Hugging Face repository user name"
    )
    parser.add_argument(
        "--hf-repo-model-name",
        type=str,
        default="KernelCoder",
        help="Hugging Face repository model name (e.g., 'KernelCoder')"
    )
    parser.add_argument(
        "--hf-create-model-card",
        action="store_true",
        default=True,
        help="Create a model card for the uploaded model"
    )
    
    args = parser.parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if local_rank == 0:
        logger.info("Starting Qwen fine-tuning with Unsloth...")
    
    trainer = QwenUnslothTrainer(config_path=args.config)
    
    # Override HF settings from command line arguments
    if args.upload_to_hf:
        trainer.config.setdefault('huggingface', {})
        trainer.config['huggingface']['upload'] = True
        if args.hf_repo_user and args.hf_repo_model_name:
            # find model name from config, extract the model size, from the last part of the model name
            model_size = trainer.config['model']['name'].split('-')[1:]
            model_tag = f"{args.hf_repo_model_name}-{'-'.join(model_size)}"
            time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
            trainer.config['huggingface']['repo_name'] = f"{args.hf_repo_user}/{model_tag}_{time_tag}"
            trainer.config['huggingface']['create_model_card'] = args.hf_create_model_card
    
    if args.test_only:
        # Only test the model
        trainer.setup_model_and_tokenizer()
        if local_rank == 0:
            logger.info("Testing the fine-tuned model...")
            response = trainer.test_model(prompt=args.prompt)
            logger.info(f"Model response: {response}")
    else:
        # Run full training
        trainer.train()
        
        # Upload to Hugging Face (if configured)
        if args.upload_to_hf:
            trainer.upload_to_huggingface()
        
        # Test model
        if local_rank == 0:
            logger.info("Testing the fine-tuned model...")
            response = trainer.test_model(prompt=args.prompt)  # Uses default prompt from config
            logger.info(f"Model response: {response}")


if __name__ == "__main__":
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    main()
