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
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from util import logger
import numpy as np


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


class CustomDataCollatorWithMasking(DataCollatorForLanguageModeling):
    """Custom data collator that masks system, user, and function tokens."""
    
    def __init__(self, tokenizer, mlm=False, ignore_index=-100):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.ignore_index = ignore_index
        
    def torch_call(self, examples):
        # Handle different input formats
        if isinstance(examples[0], dict):
            # Check if examples are already tokenized (have input_ids)
            if "input_ids" in examples[0]:
                # Examples are already tokenized tensors - use them directly
                batch = self._collate_tokenized_examples(examples)
            elif "text" in examples[0]:
                # Examples have text field - extract and tokenize
                texts = [example["text"] for example in examples]
                batch = self._tokenize_and_prepare(texts)
            else:
                # Unknown dict format - try to use as is
                batch = super().torch_call(examples)
        elif isinstance(examples[0], str):
            # Examples are raw text strings
            batch = self._tokenize_and_prepare(examples)
        else:
            # Fall back to parent class
            batch = super().torch_call(examples)
        
        # Apply masking to labels
        if "labels" in batch:
            input_ids = batch["input_ids"]
            labels = batch["labels"].clone()
            
            # Create masks for different roles
            for i, input_seq in enumerate(input_ids):
                labels[i] = self._mask_non_assistant_tokens(input_seq, labels[i])
            
            batch["labels"] = labels
        
        return batch
    
    def _tokenize_and_prepare(self, texts):
        """Tokenize texts and prepare batch."""
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=getattr(self.tokenizer, 'model_max_length', 2048),
            return_tensors="pt"
        )
        # Create labels from input_ids
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    def _collate_tokenized_examples(self, examples):
        """Collate examples that are already tokenized."""
        # Extract data from examples and convert to tensors if needed
        input_ids = []
        attention_mask = []
        labels = []
        
        for ex in examples:
            # Convert to tensor if it's a list
            if isinstance(ex["input_ids"], list):
                seq_input = torch.tensor(ex["input_ids"], dtype=torch.long)
            else:
                seq_input = ex["input_ids"].clone()
            
            if isinstance(ex["attention_mask"], list):
                seq_attention = torch.tensor(ex["attention_mask"], dtype=torch.long)
            else:
                seq_attention = ex["attention_mask"].clone()
            
            # Handle labels - use input_ids if labels not present
            if "labels" in ex:
                if isinstance(ex["labels"], list):
                    seq_labels = torch.tensor(ex["labels"], dtype=torch.long)
                else:
                    seq_labels = ex["labels"].clone()
            else:
                seq_labels = seq_input.clone()
            
            input_ids.append(seq_input)
            attention_mask.append(seq_attention)
            labels.append(seq_labels)
        
        # Pad sequences to same length
        max_length = max(len(seq.squeeze()) for seq in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(input_ids)):
            seq_input = input_ids[i].squeeze()
            seq_attention = attention_mask[i].squeeze()
            seq_labels = labels[i].squeeze()
            
            # Pad sequences
            pad_length = max_length - len(seq_input)
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                seq_input = torch.cat([seq_input, torch.full((pad_length,), pad_token_id, dtype=seq_input.dtype)])
                seq_attention = torch.cat([seq_attention, torch.zeros(pad_length, dtype=seq_attention.dtype)])
                seq_labels = torch.cat([seq_labels, torch.full((pad_length,), self.ignore_index, dtype=seq_labels.dtype)])
            
            padded_input_ids.append(seq_input)
            padded_attention_mask.append(seq_attention)
            padded_labels.append(seq_labels)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels)
        }
    
    def _mask_non_assistant_tokens(self, input_ids, labels):
        """Mask tokens that are not assistant responses."""
        # Convert to list for easier processing and ensure 1D
        input_ids_list = input_ids.squeeze().tolist() if input_ids.dim() > 1 else input_ids.tolist()
        labels_list = labels.squeeze().tolist() if labels.dim() > 1 else labels.tolist()
        original_loss_tokens = sum(1 for l in labels_list if l != self.ignore_index)
        
        # Special tokens for Qwen format - ensure we catch all variations
        system_start = self.tokenizer.encode("<|im_start|>system", add_special_tokens=False)
        user_start = self.tokenizer.encode("<|im_start|>user", add_special_tokens=False)
        assistant_start = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        function_start = self.tokenizer.encode("<|im_start|>function", add_special_tokens=False)
        im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        function_call_start = self.tokenizer.encode("<function_call>", add_special_tokens=False)
        function_call_end = self.tokenizer.encode("</function_call>", add_special_tokens=False)
        
        # Debug output removed - masking working correctly
        
        # Track current state and what we're masking
        current_role = "unknown"  # Track current role for debugging
        in_assistant_content = False  # Only true when in actual assistant response content
        in_function_call = False
        i = 0
        
        # Initially mask everything until we know the role
        while i < len(input_ids_list):
            # Check for role transitions
            if self._token_sequence_match(input_ids_list, i, system_start):
                current_role = "system"
                in_assistant_content = False
                in_function_call = False
                # Mask the entire system role marker and advance
                for j in range(len(system_start)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                i += len(system_start)
                
            elif self._token_sequence_match(input_ids_list, i, user_start):
                current_role = "user"
                in_assistant_content = False
                in_function_call = False
                # Mask the entire user role marker and advance
                for j in range(len(user_start)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                i += len(user_start)
                
            elif self._token_sequence_match(input_ids_list, i, assistant_start):
                current_role = "assistant"
                in_assistant_content = True
                in_function_call = False
                # Mask the assistant start tokens themselves (role marker should not be trained on)
                for j in range(len(assistant_start)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                i += len(assistant_start)
                
            elif self._token_sequence_match(input_ids_list, i, function_start):
                current_role = "function"
                in_assistant_content = False
                in_function_call = False
                # Mask the entire function role marker and advance
                for j in range(len(function_start)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                i += len(function_start)
                
            elif self._token_sequence_match(input_ids_list, i, function_call_start):
                # Function calls within assistant responses should be TRAINED ON (not masked)
                # Only set the flag to track we're in a function call, but don't mask
                in_function_call = True
                i += len(function_call_start)
                
            elif self._token_sequence_match(input_ids_list, i, function_call_end):
                in_function_call = False
                i += len(function_call_end)
                
            elif self._token_sequence_match(input_ids_list, i, im_end):
                # End of any role - mask the end marker and reset state
                for j in range(len(im_end)):
                    if i + j < len(labels_list):
                        labels_list[i + j] = self.ignore_index
                current_role = "unknown"
                in_assistant_content = False
                in_function_call = False
                i += len(im_end)
                
            else:
                # Apply masking based on current state
                should_mask = True
                
                if current_role == "assistant" and in_assistant_content:
                    # Train on ALL assistant content including function calls
                    should_mask = False
                    
                # Always mask: system instructions, user prompts, function outputs
                # Note: function calls within assistant responses are now trained on
                if current_role in ["system", "user", "function"]:
                    should_mask = True
                
                if should_mask:
                    labels_list[i] = self.ignore_index
                    
                i += 1
        
        # Debug: Count how many tokens will be used for loss calculation
        masked_loss_tokens = sum(1 for l in labels_list if l != self.ignore_index)
        
        return torch.tensor(labels_list, dtype=labels.dtype)
    
    def _token_sequence_match(self, input_ids, start_idx, target_sequence):
        """Check if token sequence matches at given position."""
        if start_idx + len(target_sequence) > len(input_ids):
            return False
        return input_ids[start_idx:start_idx + len(target_sequence)] == target_sequence


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
            'gradient_checkpointing': True,
            'use_custom_loss_masking': True
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
            'num_train_epochs': int(train_config.get('num_train_epochs', 3)),
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
        
        # Create trainer with optional custom data collator
        trainer_kwargs = {
            'model': self.model,
            'train_dataset': train_dataset,
            'dataset_text_field': "text",
            'max_seq_length': int(self._get_config_value('model', 'max_seq_length')),
            'dataset_num_proc': int(train_config['dataset_num_proc']),
            'packing': bool(train_config['packing']),
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
            self._log_main("Using custom loss function with masking for system, user, and function outputs (but training on assistant function calls)")
            
            # Test masking with a sample
            self._test_masking_sample(data_collator)
            
            # Verify masking on actual training data  
            self._verify_dataset_masking(train_dataset, data_collator)
        else:
            self._log_main("Using standard loss function without custom masking")
        
        trainer = SFTTrainer(**trainer_kwargs)
        
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

    def _test_masking_sample(self, data_collator):
        """Test the masking function with comprehensive sample conversations."""
        if self.local_rank == 0:  # Only test on main process
            # Test case 1: Basic conversation
            sample_text1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\nAI stands for Artificial Intelligence.<|im_end|>"
            
            # Test case 2: Conversation with function call
            sample_text2 = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n<function_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</function_call>\nThe weather is sunny.<|im_end|>\n<|im_start|>function name=get_weather\nSunny, 75F<|im_end|>\n<|im_start|>assistant\nBased on the weather data, it's a beautiful sunny day at 75°F!<|im_end|>"
            
            self._log_main("=== Testing Masking Function ===")
            
            for i, (name, sample_text) in enumerate([("Basic", sample_text1), ("Function Call", sample_text2)], 1):
                self._log_main(f"\n--- Test Case {i}: {name} ---")
                
                # Tokenize sample
                tokenized = self.tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)
                
                # Create a mock batch
                example = {
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'labels': tokenized['input_ids'].clone()
                }
                
                # Apply masking
                masked_batch = data_collator.torch_call([example])
                
                # Detailed analysis
                input_ids = tokenized['input_ids'].squeeze()
                labels = masked_batch['labels'].squeeze()
                
                # Count tokens by category
                total_tokens = len(input_ids)
                training_tokens = (labels != -100).sum().item()
                masked_tokens = total_tokens - training_tokens
                
                self._log_main(f"Total tokens: {total_tokens}, Training tokens: {training_tokens}, Masked tokens: {masked_tokens}")
                self._log_main(f"Masking ratio: {masked_tokens/total_tokens*100:.1f}%")
                
                # Show what's being trained on
                if training_tokens > 0:
                    training_token_ids = labels[labels != -100]
                    training_text = self.tokenizer.decode(training_token_ids, skip_special_tokens=False)
                    self._log_main(f"Training content: '{training_text}'")
                
                # Verify masking by showing token-by-token breakdown
                self._log_main("Token breakdown (first 10 tokens):")
                for j in range(min(10, len(input_ids))):
                    token = self.tokenizer.decode([input_ids[j]], skip_special_tokens=False)
                    masked = "MASKED" if labels[j].item() == -100 else "TRAIN"
                    self._log_main(f"  {j:2d}: '{token}' -> {masked}")
                
                # Verify specific requirements
                full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                
                # Check that system instructions are masked
                if "<|im_start|>system" in full_text:
                    self._log_main("✓ System instructions detected - should be masked")
                
                # Check that user prompts are masked  
                if "<|im_start|>user" in full_text:
                    self._log_main("✓ User prompts detected - should be masked")
                
                # Check that function outputs are masked
                if "<|im_start|>function" in full_text:
                    self._log_main("✓ Function outputs detected - should be masked")
                
                # Check that function calls are NOT masked (should be trained on)
                if "<function_call>" in full_text:
                    self._log_main("✓ Function calls detected - should be TRAINED ON (not masked)")
                
            self._log_main("=== Masking Test Complete ===\n")

    def _verify_dataset_masking(self, dataset, data_collator):
        """Verify masking is working correctly on actual training data."""
        if self.local_rank == 0 and len(dataset) > 0:
            self._log_main("=== Verifying Dataset Masking ===")
            
            # Check a few random samples from the dataset
            import random
            sample_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
            
            total_masking_stats = {"total_tokens": 0, "training_tokens": 0, "masked_tokens": 0}
            
            for i, idx in enumerate(sample_indices):
                self._log_main(f"\n--- Dataset Sample {i+1} (index {idx}) ---")
                
                sample = dataset[idx]
                text = sample["text"]
                
                # Tokenize
                tokenized = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                
                # Create example
                example = {
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'labels': tokenized['input_ids'].clone()
                }
                
                # Apply masking
                masked_batch = data_collator.torch_call([example])
                
                # Analyze
                input_ids = tokenized['input_ids'].squeeze()
                labels = masked_batch['labels'].squeeze()
                
                total_tokens = len(input_ids)
                training_tokens = (labels != -100).sum().item()
                masked_tokens = total_tokens - training_tokens
                
                # Accumulate stats
                total_masking_stats["total_tokens"] += total_tokens
                total_masking_stats["training_tokens"] += training_tokens
                total_masking_stats["masked_tokens"] += masked_tokens
                
                self._log_main(f"Sample length: {len(text)} chars, {total_tokens} tokens")
                self._log_main(f"Training on: {training_tokens} tokens ({training_tokens/total_tokens*100:.1f}%)")
                self._log_main(f"Masked: {masked_tokens} tokens ({masked_tokens/total_tokens*100:.1f}%)")
                
                # Show what content is being trained on (first 100 chars)
                if training_tokens > 0:
                    training_token_ids = labels[labels != -100]
                    training_text = self.tokenizer.decode(training_token_ids, skip_special_tokens=False)
                    self._log_main(f"Training content preview: '{training_text[:100]}{'...' if len(training_text) > 100 else ''}'")
                
                # Verify key requirements
                checks = []
                if "<|im_start|>system" in text:
                    checks.append("System instructions present (will be masked)")
                if "<|im_start|>user" in text:
                    checks.append("User prompts present (will be masked)")
                if "<|im_start|>function" in text:
                    checks.append("Function outputs present (will be masked)")
                if "<function_call>" in text:
                    checks.append("Function calls present (will be trained on)")
                
                if checks:
                    self._log_main(f"Content analysis: {', '.join(checks)}")
            
            # Overall statistics
            if total_masking_stats["total_tokens"] > 0:
                overall_training_ratio = total_masking_stats["training_tokens"] / total_masking_stats["total_tokens"]
                overall_masking_ratio = total_masking_stats["masked_tokens"] / total_masking_stats["total_tokens"]
                
                self._log_main(f"\n--- Overall Dataset Masking Statistics ---")
                self._log_main(f"Total tokens across samples: {total_masking_stats['total_tokens']}")
                self._log_main(f"Training tokens: {total_masking_stats['training_tokens']} ({overall_training_ratio*100:.1f}%)")
                self._log_main(f"Masked tokens: {total_masking_stats['masked_tokens']} ({overall_masking_ratio*100:.1f}%)")
                
                # Validate the masking is working as expected
                if overall_masking_ratio > 0.5:
                    self._log_main("✓ Good masking ratio - most tokens are properly masked")
                elif overall_masking_ratio > 0.3:
                    self._log_main("⚠ Moderate masking ratio - verify system/user/function content is being masked")
                else:
                    self._log_main("⚠ Low masking ratio - check if masking logic is working correctly")
            
            self._log_main("=== Dataset Masking Verification Complete ===\n")

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
