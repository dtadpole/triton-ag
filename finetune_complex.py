#!/usr/bin/env python3
"""
Qwen Fine-tuning with Unsloth
Clean and efficient script for fine-tuning Qwen models using Unsloth optimization.
"""

# Import unsloth first for optimizations
from unsloth import FastLanguageModel

import os
import json
import yaml
import torch
import torch.distributed as dist
from transformers import TrainingArguments
from trl import SFTTrainer
from util import logger
from data_util import create_dataset, CustomDataCollatorWithMasking


def setup_distributed():
    """Initialize distributed training if running in distributed mode."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def is_distributed():
    """Check if we're running in distributed mode."""
    return "RANK" in os.environ


class QwenUnslothTrainer:
    """Fine-tune Qwen models using Unsloth with distributed support."""
    
    def __init__(self, config_path="finetune.yaml"):
        """Initialize trainer with configuration."""
        self.local_rank = setup_distributed()
        self.is_distributed = is_distributed()
        self.config = self._load_config(config_path)
        self._validate_cuda()
        self.tokenizer = None
        self.model = None

    def _log_main(self, message, level="info"):
        """Log message only on main process (rank 0)."""
        if self.local_rank == 0:
            getattr(logger, level)(message)

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {config_path} not found.")

    def _get_config_value(self, *keys, default=None):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def _validate_cuda(self):
        """Validate CUDA availability."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
        self._log_main(f"Available CUDA devices: {torch.cuda.device_count()}")

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer using Unsloth."""
        model_name = self._get_config_value('model', 'name')
        max_seq_length = int(self._get_config_value('model', 'max_seq_length'))
        
        self._log_main(f"Loading model: {model_name}")
        
        model_kwargs = {
            'model_name': model_name,
            'max_seq_length': max_seq_length,
            'dtype': self._get_config_value('model', 'dtype'),
            'load_in_4bit': bool(self._get_config_value('model', 'load_in_4bit')),
            'trust_remote_code': True,
        }
        
        if self.is_distributed:
            model_kwargs['device_map'] = {"": self.local_rank}
        else:
            model_kwargs['device_map'] = "auto"
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        self._setup_lora()
        
        mode = "distributed" if self.is_distributed else "single GPU"
        self._log_main(f"Model setup complete ({mode})")

    def _setup_lora(self):
        """Configure and apply LoRA using Unsloth."""
        lora_config = self.config['lora']
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=int(lora_config['r']),
            target_modules=lora_config['target_modules'],
            lora_alpha=int(lora_config['alpha']),
            lora_dropout=float(lora_config['dropout']),
            bias=lora_config['bias'],
            use_gradient_checkpointing=lora_config.get('use_gradient_checkpointing', 'unsloth'),
            random_state=int(lora_config.get('random_state', 3407)),
            use_rslora=bool(lora_config.get('use_rslora', False)),
            loftq_config=lora_config.get('loftq_config'),
        )

    def get_training_arguments(self):
        """Get training arguments from configuration."""
        train_config = self.config['training']
        
        # Calculate effective batch size
        per_device_batch_size = int(train_config['per_device_batch_size'])
        gradient_accumulation_steps = int(train_config['gradient_accumulation_steps'])
        
        if self.is_distributed:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            effective_batch_size = per_device_batch_size * world_size * gradient_accumulation_steps
        else:
            effective_batch_size = per_device_batch_size * gradient_accumulation_steps
        
        self._log_main(f"Effective batch size: {effective_batch_size}")
        
        training_args = {
            'output_dir': train_config['output_dir'],
            'per_device_train_batch_size': per_device_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'warmup_steps': int(train_config['warmup_steps']),
            'max_steps': int(train_config['max_steps']),
            'learning_rate': float(train_config['learning_rate']),
            'logging_steps': int(train_config['logging_steps']),
            'optim': train_config['optim'],
            'weight_decay': float(train_config['weight_decay']),
            'lr_scheduler_type': train_config['lr_scheduler_type'],
            'seed': int(train_config['seed']),
            'save_steps': int(train_config['save_steps']),
            'save_total_limit': int(train_config['save_total_limit']),
            'report_to': None,
            'remove_unused_columns': False,
        }
        
        # Gradient clipping
        max_grad_norm = train_config.get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            training_args['max_grad_norm'] = float(max_grad_norm)
            self._log_main(f"Gradient clipping enabled: max_grad_norm={max_grad_norm}")
        
        # Mixed precision
        training_args['fp16'] = not torch.cuda.is_bf16_supported()
        training_args['bf16'] = torch.cuda.is_bf16_supported()
        
        return TrainingArguments(**training_args)

    def train(self):
        """Train the model."""
        mode = "distributed" if self.is_distributed else "single GPU"
        self._log_main(f"Starting fine-tuning ({mode})...")
        
        self.setup_model_and_tokenizer()
        data_dir = self._get_config_value('data', 'local_dir')
        max_length = int(self._get_config_value('model', 'max_seq_length'))
        train_dataset = create_dataset(data_dir, max_length, self.local_rank)
        training_args = self.get_training_arguments()
        train_config = self.config['training']
        
        # Create trainer
        trainer_kwargs = {
            'model': self.model,
            'train_dataset': train_dataset,
            'dataset_text_field': "text",
            'max_seq_length': int(self._get_config_value('model', 'max_seq_length')),
            'dataset_num_proc': int(train_config.get('dataset_num_proc', 2)),
            'packing': bool(train_config.get('packing', False)),
            'args': training_args,
        }
        
        # Add custom data collator if masking is enabled
        if train_config.get('use_custom_loss_masking', True):
            data_collator = CustomDataCollatorWithMasking(
                tokenizer=self.tokenizer,
                mlm=False,
                ignore_index=-100
            )
            trainer_kwargs['data_collator'] = data_collator
            self._log_main("Using custom loss masking")
        
        trainer = SFTTrainer(**trainer_kwargs)
        
        if self.local_rank == 0:
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            self._log_main(f"GPU memory before training: {memory_gb:.2f} GB")
        
        trainer.train()
        
        # Save model only on rank 0 for distributed training
        if not self.is_distributed or self.local_rank == 0:
            output_dir = self._get_config_value('training', 'output_dir')
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            self._log_main(f"Training completed! Model saved to {output_dir}")

    def test_model(self, prompt):
        """Test the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            self._log_main("Model not loaded! Run train() first.", "error")
            return None
        
        # Only test on rank 0 for distributed training
        if self.is_distributed and self.local_rank != 0:
            return "Inference skipped on non-zero rank"
        
        FastLanguageModel.for_inference(self.model)
        device = f"cuda:{self.local_rank}" if self.is_distributed else "cuda"
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    use_cache=True,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.batch_decode(outputs)[0]
            return response[len(prompt):].strip()
        except Exception as e:
            self._log_main(f"Error during inference: {e}", "error")
            return f"Inference failed: {e}"


def main():
    """Main function to run the fine-tuning process."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = is_distributed()
    
    if local_rank == 0:
        mode = "distributed" if is_dist else "single GPU"
        logger.info(f"Starting Qwen fine-tuning with Unsloth ({mode})...")
    
    trainer = QwenUnslothTrainer()
    trainer.train()
    
    # Test model only on rank 0
    if not is_dist or local_rank == 0:
        test_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n"
        logger.info("Testing the fine-tuned model...")
        response = trainer.test_model(test_prompt)
        logger.info(f"Model response: {response}")


if __name__ == "__main__":
    main()
