#!/usr/bin/env python3
"""
Simplified Qwen Fine-tuning with Unsloth
Clean and efficient script for fine-tuning Qwen models.
"""

# Import unsloth first for optimizations
from unsloth import FastLanguageModel

import os
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

    def get_training_arguments(self):
        """Get training arguments from configuration."""
        train_config = self.config['training']
        
        # Base arguments
        args = {
            'num_train_epochs': train_config['num_train_epochs'],
            'output_dir': train_config['output_dir'],
            'per_device_train_batch_size': train_config['per_device_batch_size'],
            'gradient_accumulation_steps': train_config['gradient_accumulation_steps'],
            'warmup_steps': train_config['warmup_steps'],
            'max_steps': train_config['max_steps'],
            'learning_rate': train_config['learning_rate'],
            'logging_steps': train_config['logging_steps'],
            'optim': train_config['optim'],
            'weight_decay': train_config['weight_decay'],
            'lr_scheduler_type': train_config['lr_scheduler_type'],
            'seed': train_config['seed'],
            'save_steps': train_config['save_steps'],
            'save_total_limit': train_config['save_total_limit'],
            'report_to': None,
            'remove_unused_columns': False,
        }
        
        # Mixed precision
        args['fp16'] = not torch.cuda.is_bf16_supported()
        args['bf16'] = torch.cuda.is_bf16_supported()
        
        # DDP-specific configurations
        if self.is_distributed:
            args.update({
                'ddp_find_unused_parameters': False,
                'ddp_bucket_cap_mb': 25,
                'dataloader_pin_memory': False
            })
            self.log_main("Applied DDP-specific configurations for stability")
        
        return TrainingArguments(**args)

    def train(self):
        """Train the model."""
        self.log_main("Starting fine-tuning...")
        
        self.setup_model_and_tokenizer()
        
        # Create dataset
        max_length = self.config['model']['max_seq_length']
        train_dataset = create_dataset(
            self.config['data']['local_dir'], 
            max_length, 
            self.local_rank, 
            self.tokenizer
        )
        
        # Setup trainer
        trainer_kwargs = {
            'model': self.model,
            'train_dataset': train_dataset,
            'dataset_text_field': "text",
            'max_seq_length': max_length,
            'dataset_num_proc': 2,
            'packing': False,
            'args': self.get_training_arguments(),
        }
        
        # Add custom data collator if enabled
        if self.config['training'].get('use_custom_loss_masking', True):
            trainer_kwargs['data_collator'] = CustomDataCollatorWithMasking(
                tokenizer=self.tokenizer,
                mlm=False,
                ignore_index=-100
            )
            self.log_main("Using custom loss masking")
        
        trainer = SFTTrainer(**trainer_kwargs)
        
        # Handle DDP static graph
        if self.is_distributed and hasattr(trainer.model, 'module'):
            try:
                trainer.model._set_static_graph()
                self.log_main("Set static graph for DDP compatibility")
            except AttributeError:
                self.log_main("Static graph not available, continuing with standard DDP")
        
        # Train and save
        trainer.train()
        
        if not self.is_distributed or self.local_rank == 0:
            output_dir = self.config['training']['output_dir']
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            self.log_main(f"Training completed! Model saved to {output_dir}")

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
    main() 