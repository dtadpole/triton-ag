#!/usr/bin/env python3
"""
Optimized Unsloth Fine-tuning Script for Local Experiences
Enhanced with memory management, auto-tuning, and performance optimizations.
"""

import os
import sys
from datetime import datetime
import yaml
import torch
import time
import gc
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings

# Import unsloth first for optimizations
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch.distributed as dist

# Import local utilities
from util import logger
from data_util import ExperienceDataset, SimpleDataCollator
from huggingface_hub import HfApi, create_repo


class OptimizedUnslothFineTuner:
    """Optimized fine-tuning using unsloth with memory management and auto-tuning."""
    
    def __init__(self, config_path: str = "finetune.yaml"):
        """Initialize the fine-tuner with configuration."""
        self.config_path = config_path
        self.config = None
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        # Setup distributed training if available
        self.local_rank = self._setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Memory management
        self.initial_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
        self.peak_memory = 0
        
        self._load_config()
        self._optimize_config()
        
    def _setup_distributed(self) -> int:
        """Initialize distributed training if running in distributed mode."""
        if "RANK" in os.environ:
            try:
                dist.init_process_group(backend="nccl")
                local_rank = int(os.environ["LOCAL_RANK"])
                torch.cuda.set_device(local_rank)
                logger.info(f"DDP initialized successfully on rank {local_rank}")
                return local_rank
            except Exception as e:
                logger.error(f"Failed to initialize distributed training: {e}")
                logger.info("Falling back to single GPU training")
                return 0
        return 0
    
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
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            if self.local_rank == 0:
                logger.info(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _optimize_config(self):
        """Auto-optimize configuration based on available resources."""
        if not torch.cuda.is_available():
            return
            
        available_memory = self._get_available_gpu_memory()
        if self.local_rank == 0:
            logger.info(f"Available GPU memory: {available_memory:.2f} GB")
        
        # Auto-adjust sequence length based on memory
        max_seq = self.config['model']['max_seq_length']
        if available_memory < 20 and max_seq > 16384:
            new_max_seq = 16384
            logger.warning(f"Limited GPU memory detected. Reducing max_seq_length from {max_seq} to {new_max_seq}")
            self.config['model']['max_seq_length'] = new_max_seq
        elif available_memory < 15 and max_seq > 8192:
            new_max_seq = 8192
            logger.warning(f"Very limited GPU memory detected. Reducing max_seq_length from {max_seq} to {new_max_seq}")
            self.config['model']['max_seq_length'] = new_max_seq
        elif available_memory < 10 and max_seq > 4096:
            new_max_seq = 4096
            logger.warning(f"Extremely limited GPU memory detected. Reducing max_seq_length from {max_seq} to {new_max_seq}")
            self.config['model']['max_seq_length'] = new_max_seq
        
        # Auto-adjust batch size based on memory and sequence length
        current_batch_size = self.config['training']['per_device_batch_size']
        seq_len = self.config['model']['max_seq_length']
        
        # Estimate memory usage and adjust batch size
        if seq_len > 16384 and available_memory < 20:
            if current_batch_size > 1:
                new_batch_size = 1
                logger.warning(f"Reducing batch size from {current_batch_size} to {new_batch_size} due to long sequences")
                self.config['training']['per_device_batch_size'] = new_batch_size
                # Increase gradient accumulation to maintain effective batch size
                self.config['training']['gradient_accumulation_steps'] *= current_batch_size
        
        # Optimize number of workers based on CPU count
        cpu_count = psutil.cpu_count()
        optimal_workers = min(4, max(0, cpu_count // 2))
        self.config['training']['num_workers'] = optimal_workers
        
        if self.local_rank == 0:
            logger.info("Configuration auto-optimization complete")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer using Unsloth with optimizations."""
        model_config = self.config['model']
        
        if self.local_rank == 0:
            logger.info(f"Loading model: {model_config['name']}")
            logger.info(f"Max sequence length: {model_config['max_seq_length']}")
        
        # Configure device mapping
        if self.config['gpu'].get('single_gpu', False):
            device_map = {"": 0}
        elif "RANK" in os.environ or self.config['gpu'].get('data_parallel', False):
            device_map = {"": self.local_rank}
        else:
            device_map = "auto"
        
        try:
            # Load model and tokenizer with Unsloth
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_config['name'],
                max_seq_length=model_config['max_seq_length'],
                dtype=model_config.get('dtype'),
                load_in_4bit=model_config.get('load_in_4bit', True),
                trust_remote_code=True,
                device_map=device_map,
                use_cache=False,  # Disable cache to save memory during training
            )
            
            # Setup LoRA with optimizations
            lora_config = self.config['lora']
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_config['r'],
                target_modules=lora_config['target_modules'],
                lora_alpha=lora_config['alpha'],
                lora_dropout=lora_config['dropout'],
                bias=lora_config['bias'],
                use_gradient_checkpointing="unsloth",  # Use unsloth's efficient checkpointing
                random_state=self.config['training'].get('seed', 3407),
                use_rslora=False,
                loftq_config=None,
            )
            
            # Memory cleanup after model loading
            self._cleanup_memory()
            
            if self.local_rank == 0:
                current_memory = self._get_gpu_memory()
                logger.info(f"Model loaded. GPU memory usage: {current_memory:.2f} GB")
                logger.info("Model and tokenizer setup complete")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._cleanup_memory()
            raise
    
    def setup_dataset(self):
        """Setup training dataset with optimizations."""
        # Load experiences from processed directory
        data_dir = self.config['data']['processed_dir']
        if self.local_rank == 0:
            logger.info(f"Loading experiences from: {data_dir}")
        
        try:
            # Create dataset using data_util
            self.dataset = ExperienceDataset(
                data_dir=data_dir,
                max_length=self.config['model']['max_seq_length']
            )
            
            if self.local_rank == 0:
                logger.info(f"Created dataset with {len(self.dataset)} examples")
                
        except Exception as e:
            logger.error(f"Error setting up dataset: {e}")
            raise
    
    def setup_trainer(self):
        """Setup the SFTTrainer with optimizations."""
        training_config = self.config['training']
        
        # Create output directory
        output_dir = Path(training_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup data collator with optimizations
        data_collator = SimpleDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8
        )
        
        # Optimize training arguments
        num_workers = training_config.get('num_workers', 0)
        
        # Setup training arguments with optimizations
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            warmup_steps=training_config['warmup_steps'],
            max_steps=training_config.get('max_steps', -1),
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            max_grad_norm=training_config.get('max_grad_norm', 0.5),
            optim=training_config.get('optim', 'paged_adamw_8bit'),  # Use memory-efficient optimizer
            seed=training_config.get('seed', 3407),
            dataloader_pin_memory=True,
            dataloader_num_workers=num_workers,
            fp16=not torch.cuda.get_device_capability()[0] >= 8,  # Use fp16 for older GPUs
            bf16=torch.cuda.get_device_capability()[0] >= 8,      # Use bf16 for newer GPUs
            group_by_length=True,
            ddp_find_unused_parameters=False,
            report_to=None,  # Disable wandb/tensorboard
            gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
            dataloader_drop_last=True,   # Drop incomplete batches
            remove_unused_columns=False,  # Keep all columns for custom collator
            prediction_loss_only=True,   # Only compute loss, not predictions
        )
        
        try:
            # Create trainer
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.dataset,
                data_collator=data_collator,
                args=training_args,
                max_seq_length=self.config['model']['max_seq_length'],
                dataset_text_field="text",  # This will be ignored since we use custom collator
                packing=False,  # Disable packing since we use custom data processing
            )
            
            if self.local_rank == 0:
                logger.info("Trainer setup complete")
                current_memory = self._get_gpu_memory()
                logger.info(f"Memory usage after trainer setup: {current_memory:.2f} GB")
                
        except Exception as e:
            logger.error(f"Error setting up trainer: {e}")
            self._cleanup_memory()
            raise
    
    def train(self):
        """Run the fine-tuning process with monitoring."""
        if self.local_rank == 0:
            logger.info("Starting fine-tuning...")
            start_time = time.time()
            start_memory = self._get_gpu_memory()
        
        try:
            # Monitor memory during training
            if self.local_rank == 0:
                logger.info(f"Pre-training memory usage: {start_memory:.2f} GB")
            
            # Run training
            self.trainer.train()
            
            if self.local_rank == 0:
                end_time = time.time()
                end_memory = self._get_gpu_memory()
                duration = end_time - start_time
                
                logger.info(f"Fine-tuning completed in {duration:.2f} seconds")
                logger.info(f"Final memory usage: {end_memory:.2f} GB")
                logger.info(f"Peak memory usage: {max(end_memory, self.peak_memory):.2f} GB")
                
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self._cleanup_memory()
            raise
    
    def save_model(self, output_path: Optional[str] = None):
        """Save the fine-tuned model with optimizations."""
        time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        if output_path is None:
            output_path = os.path.join(self.config['training']['output_dir'], self.config['model']['name'], time_tag)
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        
        if self.local_rank == 0:
            logger.info(f"Saving model to {output_path}")
        
        try:
            # Save using unsloth's efficient method
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
            
            # Save configuration for reproducibility
            config_save_path = output_path / "training_config.yaml"
            with open(config_save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            if self.local_rank == 0:
                logger.info("Model saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def upload_to_huggingface(self, model_path: Optional[str] = None, repo_name: str = None):
        """Upload the fine-tuned model to Hugging Face Hub."""
        if self.local_rank != 0:
            return  # Only upload from rank 0
            
        # Get configuration for HF upload
        upload_config = self.config.get('huggingface', {})
        if not upload_config.get('upload', False):
            logger.info("Hugging Face upload is disabled")
            return
        
        if model_path is None:
            model_path = self.output_path
        
        if repo_name is None:
            repo_name = upload_config.get('repo_name')
            
        if not repo_name:
            logger.error("No Hugging Face repository name specified")
            return
            
        model_path = Path(model_path)
        
        try:
            logger.info(f"Uploading model to Hugging Face: {repo_name}")
            
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
                logger.info(f"Repository {repo_name} is ready")
            except Exception as e:
                logger.warning(f"Repository creation/check failed: {e}")
            
            # Upload all files in the model directory
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_name,
                token=upload_config.get('token'),
                commit_message=f"Upload fine-tuned model - {upload_config.get('commit_message', 'Fine-tuned model')}",
                ignore_patterns=["*.git*", "__pycache__", "*.pyc"]
            )
            
            logger.info(f"Successfully uploaded model to https://huggingface.co/{repo_name}")
            
            # Create a model card if specified
            if upload_config.get('create_model_card', True):
                self._create_model_card(api, repo_name, upload_config)
                
        except Exception as e:
            logger.error(f"Error uploading to Hugging Face: {e}")
            logger.error("Make sure you have:")
            logger.error("1. Set your HF_TOKEN environment variable or specify token in config")
            logger.error("2. Have write access to the repository")
            logger.error("3. Installed huggingface_hub: pip install huggingface_hub")
    
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
- **Training Examples:** {len(self.dataset) if hasattr(self, 'dataset') else 'N/A'}
- **LoRA Rank:** {self.config['lora']['r']}
- **LoRA Alpha:** {self.config['lora']['alpha']}

## Training Configuration

- **Epochs:** {self.config['training']['num_train_epochs']}
- **Learning Rate:** {self.config['training']['learning_rate']}
- **Batch Size:** {self.config['training']['per_device_batch_size']}
- **Gradient Accumulation Steps:** {self.config['training']['gradient_accumulation_steps']}

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
            
            logger.info("Model card created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create model card: {e}")
    
    def test_model(self, prompt: str = "What is machine learning?", max_length: int = 256):
        """Test the fine-tuned model with optimizations."""
        if self.local_rank == 0:
            logger.info("Testing fine-tuned model...")
            
            try:
                # Enable fast inference
                FastLanguageModel.for_inference(self.model)
                
                # Format prompt in chat format
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                
                # Apply chat template
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Tokenize and generate
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        use_cache=True,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Test prompt: {prompt}")
                logger.info(f"Model response: {response}")
                
            except Exception as e:
                logger.error(f"Error during model testing: {e}")
    
    def run_full_pipeline(self):
        """Run the complete optimized fine-tuning pipeline."""
        try:
            # Setup components
            self.setup_model_and_tokenizer()
            self.setup_dataset()
            self.setup_trainer()
            
            # Run training
            self.train()
            
            # Save model
            self.save_model()
            
            # Upload to Hugging Face (if configured)
            self.upload_to_huggingface()
            
            # Test the model
            if self.local_rank == 0:
                self.test_model()
            
            # Final cleanup
            self._cleanup_memory()
                
        except Exception as e:
            logger.error(f"Error during fine-tuning pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._cleanup_memory()
            sys.exit(1)


def main():
    """Main entry point with enhanced argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Fine-tune models using Unsloth")
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
        default="What is machine learning?",
        help="Test prompt for model testing"
    )
    parser.add_argument(
        "--memory-report",
        action="store_true",
        help="Show detailed memory usage report"
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
    
    # Set memory optimization environment variables
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    
    # Initialize fine-tuner
    finetuner = OptimizedUnslothFineTuner(config_path=args.config)
    
    # Override HF settings from command line arguments
    if args.upload_to_hf:
        finetuner.config.setdefault('huggingface', {})
        finetuner.config['huggingface']['upload'] = True
        if args.hf_repo_user and args.hf_repo_model_name:
            # find model name from config, extract the model size, from the last part of the model name
            model_size = finetuner.config['model']['name'].split('-')[-1]
            model_tag = f"{args.hf_repo_model_name}-{model_size}"
            time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
            finetuner.config['huggingface']['repo_name'] = f"{args.hf_repo_user}/{model_tag}_{time_tag}"
            finetuner.config['huggingface']['create_model_card'] = args.hf_create_model_card
    
    if args.memory_report and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"Available GPU memory: {finetuner._get_available_gpu_memory():.2f} GB")
    
    if args.test_only:
        # Only test the model
        finetuner.setup_model_and_tokenizer()
        finetuner.test_model(prompt=args.prompt)
    else:
        # Run full pipeline
        finetuner.run_full_pipeline()


if __name__ == "__main__":
    main() 
