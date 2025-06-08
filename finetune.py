#!/usr/bin/env python3
"""
Qwen Fine-tuning with LoRA and FSDP
A clean and efficient script for fine-tuning Qwen models using LoRA with distributed training support.
"""

import os
import json
import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from util import logger


class QwenLoRATrainer:
    """Fine-tune Qwen models using LoRA with distributed training support."""
    
    def __init__(self, config_path="finetune.yaml"):
        """Initialize trainer with configuration."""
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
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration."""
        return {
            'model': {'name': "Qwen/Qwen3-0.6B"},
            'training': {
                'num_epochs': 2,
                'learning_rate': 5e-5,
                'max_length': 4096,
                'per_device_batch_size': 1,
                'gradient_accumulation_steps': 8,
                'gpu_count': 1,
                'strategy': 'data_parallel',
                'output_dir': "./qwen-lora-finetuned"
            },
            'data': {'local_dir': 'finetune_experiences'},
            'lora': {'r': 16, 'alpha': 32, 'dropout': 0.05}
        }
    
    def _setup_configuration(self):
        """Setup configuration parameters."""
        # Model settings
        self.model_name = self.config.get('model', {}).get('name', "Qwen/Qwen2.5-0.5B-Instruct")
        
        # Training settings
        training = self.config.get('training', {})
        self.num_epochs = training.get('num_epochs', 2)
        self.learning_rate = training.get('learning_rate', 5e-5)
        self.max_length = training.get('max_length', 4096)
        self.warmup_ratio = training.get('warmup_ratio', 0.1)
        self.output_dir = training.get('output_dir', "./qwen-lora-finetuned")
        
        # Batch settings
        self.per_device_batch_size = training.get('per_device_batch_size', 1)
        self.gradient_accumulation_steps = training.get('gradient_accumulation_steps', 4)
        
        # Multi-GPU settings
        self.gpu_count = training.get('gpu_count', 1)
        self.gpu_strategy = training.get('strategy', 'data_parallel')
        self.dataloader_num_workers = training.get('dataloader_num_workers', 4)
        
        # Data settings
        self.experience_dir = self.config.get('data', {}).get('local_dir', 'finetune_experiences')
        
        logger.info(f"Configuration loaded: {self.gpu_count} GPU(s) with {self.gpu_strategy} strategy")
    
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
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup model with appropriate device mapping
        device_map = self._get_device_map()
        model_kwargs = {
            'torch_dtype': torch.float16,
            'trust_remote_code': True
        }
        
        if device_map is not None:
            model_kwargs['device_map'] = device_map
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        
        # Configure gradient checkpointing
        if self.gpu_strategy != "fsdp":
            self.model.gradient_checkpointing_enable()
        
        # Resize embeddings and setup LoRA
        self.model.resize_token_embeddings(len(self.tokenizer))
        self._setup_lora()
        
        logger.info("Model and tokenizer setup complete")
    
    def _get_device_map(self):
        """Get appropriate device mapping based on strategy."""
        if self.gpu_strategy == "fsdp":
            return None  # FSDP handles device mapping
        elif self.gpu_strategy == "model_parallel":
            return "auto"  # Automatic model parallelism
        else:
            return "cuda:0"  # Data parallel or single GPU
    
    def _setup_lora(self):
        """Configure and apply LoRA to the model."""
        lora_config = self.config.get('lora', {})
        
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ])
        )
        
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        self.model.enable_input_require_grads()
    
    def load_experiences(self):
        """Load and process conversation experiences."""
        experiences = []
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
                        logger.warning(f"Error processing {file_path}: {e}")
        
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
        """Convert conversation to training format."""
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
    
    def tokenize_data(self, example):
        """Tokenize conversation data for training."""
        text = self.format_conversation(example)["text"]
        
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def create_dataset(self):
        """Create and prepare training dataset."""
        experiences = self.load_experiences()
        dataset = Dataset.from_list(experiences)
        
        tokenized_dataset = dataset.map(
            self.tokenize_data,
            batched=False,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"Created dataset with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def _get_decoder_layer_class(self):
        """Get the appropriate decoder layer class based on model architecture."""
        if "qwen3" in self.model_name.lower():
            return "Qwen3DecoderLayer"
        elif "qwen2" in self.model_name.lower():
            return "Qwen2DecoderLayer"
        else:
            # Default fallback - try to detect from model config
            logger.warning(f"Could not detect model architecture from {self.model_name}, using Qwen3DecoderLayer")
            return "Qwen3DecoderLayer"

    def get_training_arguments(self):
        """Get training arguments based on strategy."""
        effective_batch_size = self.per_device_batch_size * self.gpu_count * self.gradient_accumulation_steps
        logger.info(f"Effective batch size: {effective_batch_size}")
        
        args = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": 1,
            "save_steps": 10,
            "save_total_limit": 2,
            "prediction_loss_only": True,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "dataloader_num_workers": self.dataloader_num_workers,
            "gradient_checkpointing": False,
            "ddp_find_unused_parameters": False,
            "report_to": None
        }
        
        # Configure precision and FSDP
        if self.gpu_strategy == "fsdp":
            decoder_layer_class = self._get_decoder_layer_class()
            args.update({
                "bf16": True,
                "fsdp": "full_shard auto_wrap",
                "fsdp_config": {
                    "min_num_params": 0,
                    "xla": False,
                    "xla_fsdp_v2": False,
                    "xla_fsdp_grad_ckpt": False,
                },
                "fsdp_transformer_layer_cls_to_wrap": decoder_layer_class,
            })
        else:
            args["fp16"] = True
        
        return TrainingArguments(**args)
    
    def train(self):
        """Train the model with LoRA."""
        logger.info("Starting fine-tuning process...")
        
        # Setup
        self.setup_model_and_tokenizer()
        train_dataset = self.create_dataset()
        training_args = self.get_training_arguments()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize and run trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Training completed! Model saved to {self.output_dir}")
    
    def test_model(self, prompt):
        """Test the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded! Run train() first.")
            return None
        
        # For FSDP models in distributed mode, only test on rank 0
        if self.gpu_strategy == "fsdp" and "RANK" in os.environ:
            rank = int(os.environ.get("RANK", "0"))
            if rank != 0:
                return "Inference skipped on non-zero rank for FSDP"
        
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as model
        if torch.cuda.is_available():
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                dtype = torch.bfloat16 if self.gpu_strategy == "fsdp" else torch.float16
                with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return f"Inference failed: {e}"


def main():
    """Main function to run the fine-tuning process."""
    logger.info("Starting Qwen LoRA fine-tuning...")
    
    trainer = QwenLoRATrainer()
    
    # Check for distributed training requirement
    if trainer.gpu_strategy == "fsdp" and "RANK" not in os.environ:
        logger.error("FSDP requires distributed training. Please run with:")
        logger.error("torchrun --nproc_per_node=4 finetune.py")
        return
    
    # Train the model
    trainer.train()
    
    # Test model (only on rank 0 for distributed training)
    if trainer.gpu_strategy != "fsdp" or os.environ.get("RANK", "0") == "0":
        test_prompt = "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n"
        logger.info("Testing the fine-tuned model...")
        response = trainer.test_model(test_prompt)
        logger.info(f"Model response: {response}")


if __name__ == "__main__":
    main()
