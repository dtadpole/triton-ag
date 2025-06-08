#!/usr/bin/env python3
"""
Qwen Fine-tuning with LoRA
A clean and efficient script for fine-tuning Qwen models using LoRA (Low-Rank Adaptation)
on function calling and conversation data.
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
    """Fine-tune Qwen models using LoRA for memory-efficient training."""
    
    def __init__(self, config=None):
        # Load configuration
        if config is None:
            config = self.load_config()
        
        self.config = config
        self.model_name = config.get('model', {}).get('name', "Qwen/Qwen2.5-0.5B-Instruct")
        self.output_dir = config.get('training', {}).get('output_dir', "./qwen-lora-finetuned")
        self.max_length = config.get('training', {}).get('max_length', 4096)
        self.tokenizer = None
        self.model = None
        
        # Multi-GPU configuration
        self.gpu_count = config.get('gpu', {}).get('count', 1)
        self.gpu_strategy = config.get('gpu', {}).get('strategy', 'data_parallel')
        self.per_device_batch_size = config.get('gpu', {}).get('per_device_batch_size', 1)
        
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please ensure GPU drivers are installed.")
        
        available_gpus = torch.cuda.device_count()
        logger.info(f"Available CUDA devices: {available_gpus}")
        
        # Adjust GPU count if more requested than available
        if self.gpu_count > available_gpus:
            logger.warning(f"Requested {self.gpu_count} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
            self.gpu_count = available_gpus
        
        logger.info(f"Using {self.gpu_count} GPU(s) with {self.gpu_strategy} strategy")
    
    def load_config(self, config_path="finetune.yaml"):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return {}
    
    def load_experiences(self, experience_dir="./finetune_experiences"):
        """Load conversation experiences from JSON files."""
        experiences = []
        
        logger.info(f"Loading experiences from: {experience_dir}")
        for root, _, files in os.walk(experience_dir):
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    logger.info(f"Processing: {file_path}")
                    
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            # Process conversation data
                            processed_data = self._process_conversation(data)
                            if processed_data:
                                experiences.append({"messages": processed_data})
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
                if isinstance(content, dict):
                    if content["type"] == "function_call_output":
                        # Convert to function response
                        processed.append({
                            "role": "function",
                            "name": function_name or "unknown",
                            "content": content["output"]
                        })
                elif isinstance(content, str):
                    processed.append({"role": "user", "content": content})
            
            elif role == "assistant":
                if isinstance(content, dict):
                    if content["type"] == "function_call":
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
                    if content[0]["type"] == "output_text":
                        processed.append({"role": "assistant", "content": content[0]["text"]})
                elif isinstance(content, str):
                    processed.append({"role": "assistant", "content": content})
        
        return processed
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with multi-GPU optimization."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure device mapping for multi-GPU
        if self.gpu_count > 1 and self.gpu_strategy == "fsdp":
            # FSDP: Don't use device_map, let FSDP handle distribution
            device_map = None
        elif self.gpu_count > 1 and self.gpu_strategy == "model_parallel":
            # Model parallel: split model across GPUs
            device_map = "auto"
        elif self.gpu_count > 1 and self.gpu_strategy == "data_parallel":
            # Data parallel: model on all GPUs, data split
            device_map = "cuda:0"  # Primary GPU for model loading
        else:
            # Single GPU
            device_map = "cuda:0"
        
        # Load model with multi-GPU optimization
        if device_map is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            )
        else:
            # For FSDP, don't specify device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Enable gradient checkpointing for memory efficiency with long sequences
        # Only enable if not using FSDP, as FSDP handles this through training arguments
        if self.gpu_strategy != "fsdp":
            self.model.gradient_checkpointing_enable()
        
        # Resize embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup LoRA
        self._setup_lora()
        
        # Setup data parallel if using multiple GPUs
        if self.gpu_count > 1 and self.gpu_strategy == "data_parallel":
            logger.info(f"Setting up DataParallel across {self.gpu_count} GPUs")
            # DataParallel will be handled by the Trainer with proper TrainingArguments
        
        logger.info("Model and tokenizer setup complete")
    
    def _setup_lora(self):
        """Configure and apply LoRA to the model."""
        lora_config_params = self.config.get('lora', {})
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config_params.get('r', 16),
            lora_alpha=lora_config_params.get('alpha', 32),
            lora_dropout=lora_config_params.get('dropout', 0.1),
            target_modules=lora_config_params.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj"       # MLP layers
            ])
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Enable gradient computation for input embeddings (required for PEFT)
        self.model.enable_input_require_grads()
    
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
                
                # Handle function calls
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
        
        # Set labels for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def create_dataset(self, experience_dir="./finetune_experiences", max_examples=None):
        """Create and prepare training dataset."""
        # Load experiences
        experiences = self.load_experiences(experience_dir)
        
        # Use all experiences unless max_examples is specified
        if max_examples is not None and len(experiences) > max_examples:
            logger.info(f"Limiting dataset from {len(experiences)} to {max_examples} examples")
            experiences = experiences[:max_examples]
        else:
            logger.info(f"Using all {len(experiences)} experiences for comprehensive training")
        
        # Create dataset
        dataset = Dataset.from_list(experiences)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_data,
            batched=False,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"Created dataset with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(self, experience_dir="./finetune_experiences"):
        """Train the model with LoRA using configuration from YAML."""
        logger.info("Starting fine-tuning process...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Create dataset
        train_dataset = self.create_dataset(experience_dir)
        
        # Get training parameters from config
        training_config = self.config.get('training', {})
        gpu_config = self.config.get('gpu', {})
        
        epochs = training_config.get('num_epochs', 1)
        learning_rate = training_config.get('learning_rate', 5e-5)
        per_device_batch_size = gpu_config.get('per_device_batch_size', 1)
        gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 32)
        dataloader_num_workers = gpu_config.get('dataloader_num_workers', 4)
        
        # Calculate effective batch size
        effective_batch_size = per_device_batch_size * self.gpu_count * gradient_accumulation_steps
        logger.info(f"Effective batch size: {effective_batch_size} (per_device: {per_device_batch_size}, "
                   f"gpus: {self.gpu_count}, grad_accum: {gradient_accumulation_steps})")
        
        # Training arguments with multi-GPU support
        training_args_dict = {
            "output_dir": self.output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_ratio": training_config.get('warmup_ratio', 0.1),
            "logging_steps": 1,
            "save_steps": 10,  # Less frequent saving for long sequences
            "save_total_limit": 2,
            "prediction_loss_only": True,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "dataloader_num_workers": dataloader_num_workers,
            "gradient_checkpointing": True,  # Enable gradient checkpointing
            "ddp_find_unused_parameters": False,  # Optimize for multi-GPU training
            "report_to": None
        }
        
        # Configure precision and FSDP based on strategy
        if self.gpu_strategy == "fsdp":
            # FSDP configuration from test4.py learnings
            training_args_dict.update({
                "bf16": True,  # Use bf16 instead of fp16 for better FSDP compatibility
                "fsdp": "full_shard auto_wrap",  # Enable full sharding with auto wrapping
                "fsdp_config": {
                    "min_num_params": 0,  # Minimum number of parameters for a layer to be wrapped
                    "xla": False,
                    "xla_fsdp_v2": False,
                    "xla_fsdp_grad_ckpt": False,
                },
                "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",  # Wrap each transformer layer
            })
        else:
            # Use fp16 for non-FSDP strategies
            training_args_dict["fp16"] = True
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal language modeling
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Clear GPU memory and start training
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
        
        # For FSDP models, we need to ensure the model is in the right state for inference
        self.model.eval()  # Set to evaluation mode
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to CUDA if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use torch.autocast for mixed precision inference
            if self.gpu_strategy == "fsdp":
                # Use bf16 autocast for FSDP compatibility
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            else:
                # Use standard fp16 autocast for other strategies
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()


def main():
    """Main function to run the fine-tuning process."""
    logger.info("Starting Qwen LoRA fine-tuning with multi-GPU support...")
    
    # Initialize trainer with YAML configuration
    trainer = QwenLoRATrainer()
    
    # Check if FSDP is being used and if we're in distributed mode
    if trainer.gpu_strategy == "fsdp":
        import os
        if "RANK" not in os.environ:
            logger.error("FSDP requires distributed training. Please run with:")
            logger.error("torchrun --nproc_per_node=4 finetune.py")
            return
    
    # Get experience directory from config
    experience_dir = trainer.config.get('data', {}).get('local_dir', './finetune_experiences')
    
    # Train the model
    trainer.train(experience_dir=experience_dir)
    
    # Test the model (only on rank 0 for distributed training)
    if trainer.gpu_strategy != "fsdp" or os.environ.get("RANK", "0") == "0":
        test_prompt = "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n"
        logger.info("Testing the fine-tuned model...")
        response = trainer.test_model(test_prompt)
        logger.info(f"Model response: {response}")


if __name__ == "__main__":
    main()
