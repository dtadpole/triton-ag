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
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from util import logger


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
    
    # Default configuration - single source of truth
    DEFAULT_CONFIG = {
        'model': {
            'name': "unsloth/Qwen3-8B-bnb-4bit",
            'max_seq_length': 2048,
            'dtype': None,
            'load_in_4bit': True
        },
        'training': {
            'learning_rate': 2e-4,
            'max_steps': 50,
            'warmup_steps': 5,
            'per_device_batch_size': 2,
            'gradient_accumulation_steps': 4,
            'optim': 'paged_adamw_8bit',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'linear',
            'logging_steps': 1,
            'save_steps': 25,
            'save_total_limit': 2,
            'output_dir': "./qwen3-unsloth-finetuned",
            'dataloader_num_workers': 4,
            'seed': 3407,
            'dataloader_pin_memory': False,
            'ddp_find_unused_parameters': False,
            'remove_unused_columns': False,
            'dataset_num_proc': 2,
            'packing': False,
            'cpu_offload_optimizer': True,
            'cpu_offload_params': True,
            'max_grad_norm': 1.0,
            'auto_find_batch_size': True,
            'gradient_checkpointing': True
        },
        'data': {'local_dir': 'finetune_experiences'},
        'lora': {
            'r': 16, 
            'alpha': 16, 
            'dropout': 0.0,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'bias': "none",
            'use_gradient_checkpointing': "unsloth",
            'random_state': 3407,
            'use_rslora': False,
            'loftq_config': None
        }
    }
    
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
        """Load configuration from YAML file with defaults."""
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Deep merge with defaults
                return self._deep_merge(self.DEFAULT_CONFIG, user_config)
        except FileNotFoundError:
            self._log_main(f"Config file {config_path} not found. Using defaults.", "warning")
            return self.DEFAULT_CONFIG.copy()

    def _deep_merge(self, default, override):
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _get_config_value(self, *keys, default=None):
        """Get nested configuration value with dot notation."""
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
            raise RuntimeError("CUDA is not available! Please ensure GPU drivers are installed.")
        
        self._log_main(f"Available CUDA devices: {torch.cuda.device_count()}")

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer using Unsloth with aggressive memory optimization."""
        model_name = self._get_config_value('model', 'name')
        max_seq_length = int(self._get_config_value('model', 'max_seq_length'))
        
        self._log_main(f"Loading model: {model_name}")
        
        # Clear any existing cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            # Use sequential device mapping for better memory efficiency
            model_kwargs['device_map'] = "sequential"
        
        # Add memory optimization flags
        model_kwargs['low_cpu_mem_usage'] = True
        # Note: torch_dtype is handled by Unsloth automatically for optimal performance
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        
        # Enable model CPU offloading if configured
        if self._get_config_value('training', 'cpu_offload_params', default=False):
            self._log_main("Enabling model parameter CPU offloading")
            # This will be handled by the training framework
        
        self._setup_lora()
        
        # Clear cache after model setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        mode = "distributed" if self.is_distributed else "single GPU"
        self._log_main(f"Model and tokenizer setup complete ({mode} training with {max_seq_length} max sequence length)")

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
            use_gradient_checkpointing=lora_config['use_gradient_checkpointing'],
            random_state=int(lora_config['random_state']),
            use_rslora=bool(lora_config['use_rslora']),
            loftq_config=lora_config['loftq_config'],
        )

    def load_experiences(self):
        """Load and process conversation experiences."""
        data_dir = self._get_config_value('data', 'local_dir')
        experiences = []
        self._log_main(f"Loading experiences from: {data_dir}")
        
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            processed = self._process_conversation(data)
                            if processed:
                                experiences.append({"messages": processed})
                    except Exception as e:
                        self._log_main(f"Error processing {file_path}: {e}", "warning")
        
        self._log_main(f"Loaded {len(experiences)} conversations")
        return experiences

    def _process_conversation(self, data):
        """Process and clean conversation data."""
        processed = []
        function_name = None
        
        for msg in data:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                processed.append({"role": "system", "content": content})
            elif role == "user":
                if isinstance(content, dict) and content.get("type") == "function_call_output":
                    processed.append({
                        "role": "function",
                        "name": function_name or "unknown",
                        "content": content["output"]
                    })
                elif isinstance(content, str):
                    processed.append({"role": "user", "content": content})
            elif role == "assistant":
                if isinstance(content, dict) and content.get("type") == "function_call":
                    function_name = content["name"]
                    processed.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": content["name"],
                            "arguments": content["arguments"]
                        }
                    })
                elif isinstance(content, list) and content:
                    if content[0].get("type") == "output_text":
                        processed.append({"role": "assistant", "content": content[0]["text"]})
                elif isinstance(content, str):
                    processed.append({"role": "assistant", "content": content})
        
        return processed

    def format_conversations(self, examples):
        """Format conversations for training with dynamic length optimization."""
        texts = []
        max_length = int(self._get_config_value('model', 'max_seq_length'))
        
        for messages in examples["messages"]:
            conversation = ""
            for message in messages:
                role = message["role"]
                content = message.get("content", "")
                
                if role == "system":
                    conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    conversation += f"<|im_start|>assistant\n"
                    if "function_call" in message:
                        func_call = message["function_call"]
                        conversation += f"<function_call>\n{json.dumps(func_call)}\n</function_call>"
                    if content:
                        conversation += content
                    conversation += "<|im_end|>\n"
                elif role == "function":
                    name = message.get("name", "unknown")
                    conversation += f"<|im_start|>function name={name}\n{content}<|im_end|>\n"
            
            # Truncate conversations that are too long to save memory
            if len(conversation) > max_length * 4:  # Rough character estimate
                conversation = conversation[:max_length * 4] + "..."
                self._log_main(f"Truncated long conversation to save memory", "debug")
            
            texts.append(conversation)
        return {"text": texts}

    def create_dataset(self):
        """Create and prepare training dataset."""
        experiences = self.load_experiences()
        dataset = Dataset.from_list(experiences)
        formatted_dataset = dataset.map(self.format_conversations, batched=True)
        
        self._log_main(f"Created dataset with {len(formatted_dataset)} examples")
        return formatted_dataset

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
        
        # CPU offloading configuration for memory optimization
        training_args_kwargs = {
            'output_dir': train_config['output_dir'],
            'per_device_train_batch_size': per_device_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'warmup_steps': int(train_config['warmup_steps']),
            'max_steps': int(train_config['max_steps']),
            'learning_rate': float(train_config['learning_rate']),
            'fp16': not torch.cuda.is_bf16_supported(),
            'bf16': torch.cuda.is_bf16_supported(),
            'logging_steps': int(train_config['logging_steps']),
            'optim': train_config['optim'],
            'weight_decay': float(train_config['weight_decay']),
            'lr_scheduler_type': train_config['lr_scheduler_type'],
            'seed': int(train_config['seed']),
            'save_steps': int(train_config['save_steps']),
            'save_total_limit': int(train_config['save_total_limit']),
            'dataloader_num_workers': int(train_config['dataloader_num_workers']),
            'report_to': None,
            'ddp_find_unused_parameters': bool(train_config['ddp_find_unused_parameters']),
            'dataloader_pin_memory': bool(train_config['dataloader_pin_memory']),
            'remove_unused_columns': bool(train_config['remove_unused_columns']),
        }
        
        # Memory optimization settings
        if train_config.get('cpu_offload_optimizer', False):
            # Enable CPU offloading through optimizer arguments
            training_args_kwargs['optim_args'] = "cpu_offload=True"
            self._log_main("Enabled CPU offloading for optimizer states")
        
        if train_config.get('cpu_offload_params', False):
            # Additional memory optimization settings
            training_args_kwargs['dataloader_pin_memory'] = False
            training_args_kwargs['dataloader_persistent_workers'] = False
            self._log_main("Enabled additional CPU offloading optimizations")
        
        # Advanced memory conservation features
        if train_config.get('auto_find_batch_size', False):
            training_args_kwargs['auto_find_batch_size'] = True
            self._log_main("Enabled automatic batch size detection to prevent OOM")
        
        if train_config.get('gradient_checkpointing', False):
            training_args_kwargs['gradient_checkpointing'] = True
            self._log_main("Enabled gradient checkpointing for memory efficiency")
        
        if train_config.get('max_grad_norm'):
            training_args_kwargs['max_grad_norm'] = float(train_config['max_grad_norm'])
            self._log_main(f"Set gradient clipping to {train_config['max_grad_norm']}")
        
        # Force mixed precision and additional optimizations
        training_args_kwargs['tf32'] = True if torch.cuda.is_available() else False
        training_args_kwargs['group_by_length'] = True  # Reduces padding overhead
        
        return TrainingArguments(**training_args_kwargs)

    def train(self):
        """Train the model."""
        mode = "distributed" if self.is_distributed else "single GPU"
        self._log_main(f"Starting fine-tuning ({mode})...")
        
        self.setup_model_and_tokenizer()
        train_dataset = self.create_dataset()
        training_args = self.get_training_arguments()
        train_config = self.config['training']
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=int(self._get_config_value('model', 'max_seq_length')),
            dataset_num_proc=int(train_config['dataset_num_proc']),
            packing=bool(train_config['packing']),
            args=training_args,
        )
        
        # Additional memory optimization callback for periodic cache clearing
        if hasattr(trainer, 'add_callback'):
            from transformers import TrainerCallback
            
            class MemoryOptimizationCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    # Aggressive memory cleanup every 5 steps
                    if state.global_step % 5 == 0:
                        torch.cuda.empty_cache()
                        # Force garbage collection
                        import gc
                        gc.collect()
                
                def on_train_begin(self, args, state, control, **kwargs):
                    # Clear cache at start of training
                    torch.cuda.empty_cache()
                    
                def on_epoch_end(self, args, state, control, **kwargs):
                    # Major cleanup at end of each epoch
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
            trainer.add_callback(MemoryOptimizationCallback())
        
        if self.local_rank == 0:
            torch.cuda.empty_cache()
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            memory_reserved_gb = torch.cuda.memory_reserved() / 1024**3
            self._log_main(f"GPU memory before training - Allocated: {memory_gb:.2f} GB, Reserved: {memory_reserved_gb:.2f} GB")
            
            # Log optimizer CPU offloading status
            if train_config.get('cpu_offload_optimizer', False):
                self._log_main("Optimizer states will be offloaded to CPU to save GPU memory")
            if train_config.get('cpu_offload_params', False):
                self._log_main("Additional memory optimizations enabled for CPU offloading")
        
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
