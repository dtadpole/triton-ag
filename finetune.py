#!/usr/bin/env python3
"""
Qwen Fine-tuning with Unsloth
A clean and efficient script for fine-tuning Qwen models using Unsloth for optimized training.
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
    else:
        return 0


def is_distributed():
    """Check if we're running in distributed mode."""
    return "RANK" in os.environ


class QwenUnslothTrainer:
    """Fine-tune Qwen models using Unsloth for optimized training with distributed support."""
    
    def __init__(self, config_path="finetune.yaml"):
        """Initialize trainer with configuration."""
        self.local_rank = setup_distributed()
        self.is_distributed = is_distributed()
        self.config = self._load_config(config_path)
        self._setup_configuration()
        self._validate_cuda()
        self.tokenizer = None
        self.model = None
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            if self.local_rank == 0:
                logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration."""
        return {
            'model': {
                'name': "unsloth/Qwen3-8B-bnb-4bit",
                'max_seq_length': 2048,
                'dtype': None,
                'load_in_4bit': True
            },
            'training': {
                'num_epochs': 2,
                'learning_rate': 2e-4,
                'max_length': 2048,
                'per_device_batch_size': 2,
                'gradient_accumulation_steps': 4,
                'gpu_count': 1,
                'strategy': 'unsloth',
                'output_dir': "./qwen3-unsloth-finetuned",
                'max_steps': 50,
                'warmup_steps': 5
            },
            'data': {'local_dir': 'finetune_experiences'},
            'lora': {'r': 16, 'alpha': 16, 'dropout': 0.0}
        }
    
    def _setup_configuration(self):
        """Setup configuration parameters."""
        # Model settings
        model = self.config.get('model', {})
        self.model_name = model.get('name', "unsloth/Qwen3-8B-bnb-4bit")
        self.max_seq_length = model.get('max_seq_length', 2048)
        self.dtype = model.get('dtype', None)
        self.load_in_4bit = model.get('load_in_4bit', True)
        
        # Training settings
        training = self.config.get('training', {})
        self.num_epochs = training.get('num_epochs', 2)
        self.learning_rate = training.get('learning_rate', 2e-4)
        self.max_length = training.get('max_length', 2048)
        self.warmup_steps = training.get('warmup_steps', 5)
        self.max_steps = training.get('max_steps', 50)
        self.output_dir = training.get('output_dir', "./qwen3-unsloth-finetuned")
        
        # Batch settings
        self.per_device_batch_size = training.get('per_device_batch_size', 2)
        self.gradient_accumulation_steps = training.get('gradient_accumulation_steps', 4)
        
        # Unsloth specific settings
        self.optim = training.get('optim', 'adamw_8bit')
        self.weight_decay = training.get('weight_decay', 0.01)
        self.lr_scheduler_type = training.get('lr_scheduler_type', 'linear')
        self.logging_steps = training.get('logging_steps', 1)
        self.save_steps = training.get('save_steps', 25)
        self.save_total_limit = training.get('save_total_limit', 2)
        
        # Multi-GPU settings (simplified for unsloth)
        self.gpu_count = training.get('gpu_count', 1)
        self.gpu_strategy = training.get('strategy', 'unsloth')
        self.dataloader_num_workers = training.get('dataloader_num_workers', 2)
        
        # Data settings
        self.experience_dir = self.config.get('data', {}).get('local_dir', 'finetune_experiences')
        
        if self.local_rank == 0:
            strategy_desc = "distributed" if self.is_distributed else self.gpu_strategy
            logger.info(f"Configuration loaded: {self.gpu_count} GPU(s) with {strategy_desc} strategy")
    
    def _validate_cuda(self):
        """Validate CUDA availability and adjust GPU count."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please ensure GPU drivers are installed.")
        
        available_gpus = torch.cuda.device_count()
        logger.info(f"Available CUDA devices: {available_gpus}")
        
        if self.gpu_count > available_gpus:
            logger.warning(f"Requested {self.gpu_count} GPUs but only {available_gpus} available.")
            self.gpu_count = available_gpus
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer using Unsloth with distributed support."""
        if self.local_rank == 0:
            logger.info(f"Loading model with Unsloth: {self.model_name}")
        
        # Load model and tokenizer using Unsloth FastLanguageModel
        model_kwargs = {
            'model_name': self.model_name,
            'max_seq_length': self.max_seq_length,
            'dtype': self.dtype,
            'load_in_4bit': self.load_in_4bit,
        }
        
        # Add device mapping for distributed training
        if self.is_distributed:
            model_kwargs['device_map'] = {"": self.local_rank}
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        
        # Setup LoRA using Unsloth
        self._setup_unsloth_lora()
        
        if self.local_rank == 0:
            logger.info("Model and tokenizer setup complete with Unsloth")
    
    def _setup_unsloth_lora(self):
        """Configure and apply LoRA using Unsloth."""
        lora_config = self.config.get('lora', {})
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config.get('r', 16),
            target_modules=lora_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_alpha=lora_config.get('alpha', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    
    def load_experiences(self):
        """Load and process conversation experiences."""
        experiences = []
        if self.local_rank == 0:
            logger.info(f"Loading experiences from: {self.experience_dir}")
        
        for root, _, files in os.walk(self.experience_dir):
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
                        if self.local_rank == 0:
                            logger.warning(f"Error processing {file_path}: {e}")
        
        if self.local_rank == 0:
            logger.info(f"Loaded {len(experiences)} conversations")
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
    
    def format_conversation(self, example):
        """Convert conversation to training format using Qwen's chat template."""
        messages = example["messages"]
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
        
        return {"text": conversation}
    
    def formatting_prompts_func(self, examples):
        """Format data for SFTTrainer - batch processing."""
        texts = []
        messages_list = examples["messages"]
        for messages in messages_list:
            text = self.format_conversation({"messages": messages})["text"]
            texts.append(text)
        return {"text": texts}
    
    def create_dataset(self):
        """Create and prepare training dataset."""
        experiences = self.load_experiences()
        dataset = Dataset.from_list(experiences)
        
        # Format the dataset for SFTTrainer
        formatted_dataset = dataset.map(
            self.formatting_prompts_func,
            batched=True,
        )
        
        if self.local_rank == 0:
            logger.info(f"Created dataset with {len(formatted_dataset)} examples")
        return formatted_dataset
    
    def get_training_arguments(self):
        """Get training arguments for Unsloth with distributed support."""
        if self.is_distributed:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            effective_batch_size = self.per_device_batch_size * world_size * self.gradient_accumulation_steps
        else:
            effective_batch_size = self.per_device_batch_size * self.gpu_count * self.gradient_accumulation_steps
        
        if self.local_rank == 0:
            logger.info(f"Effective batch size: {effective_batch_size}")
        
        args = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "logging_steps": self.logging_steps,
            "optim": self.optim,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": 3407,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "dataloader_num_workers": self.dataloader_num_workers,
            "report_to": None,
            # Distributed training settings
            "ddp_find_unused_parameters": False,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
        }
        
        return TrainingArguments(**args)
    
    def train(self):
        """Train the model with Unsloth with distributed support."""
        if self.local_rank == 0:
            mode = "distributed" if self.is_distributed else "single GPU"
            logger.info(f"Starting fine-tuning process with Unsloth ({mode})...")
        
        # Setup
        self.setup_model_and_tokenizer()
        train_dataset = self.create_dataset()
        training_args = self.get_training_arguments()
        
        # Initialize SFTTrainer with Unsloth
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,  # Reduced for distributed stability
            packing=False,  # Disable packing for distributed training stability
            args=training_args,
        )
        
        if self.local_rank == 0:
            torch.cuda.empty_cache()
            logger.info(f"GPU {self.local_rank} memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Start training
        trainer.train()
        
        # Save model only on rank 0 for distributed training
        if not self.is_distributed or self.local_rank == 0:
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Training completed! Model saved to {self.output_dir}")
    
    def test_model(self, prompt):
        """Test the fine-tuned model using Unsloth."""
        if self.model is None or self.tokenizer is None:
            if self.local_rank == 0:
                logger.error("Model not loaded! Run train() first.")
            return None
        
        # Only test on rank 0 for distributed training
        if self.is_distributed and self.local_rank != 0:
            return "Inference skipped on non-zero rank for distributed training"
        
        # Enable fast inference with Unsloth
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
            logger.error(f"Error during inference: {e}")
            return f"Inference failed: {e}"


def main():
    """Main function to run the fine-tuning process."""
    # Check if we're running in distributed mode
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = is_distributed()
    
    if local_rank == 0:
        mode = "distributed" if is_dist else "single GPU"
        logger.info(f"Starting Qwen fine-tuning with Unsloth ({mode})...")
    
    trainer = QwenUnslothTrainer()
    
    # Train the model
    trainer.train()
    
    # Test model only on rank 0
    if not is_dist or local_rank == 0:
        test_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n"
        logger.info("Testing the fine-tuned model...")
        response = trainer.test_model(test_prompt)
        logger.info(f"Model response: {response}")


if __name__ == "__main__":
    main()
