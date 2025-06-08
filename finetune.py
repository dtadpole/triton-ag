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
                'learning_rate': 2e-4,
                'max_steps': 50,
                'warmup_steps': 5,
                'per_device_batch_size': 2,
                'gradient_accumulation_steps': 4,
                'optim': 'adamw_8bit',
                'weight_decay': 0.01,
                'lr_scheduler_type': 'linear',
                'logging_steps': 1,
                'save_steps': 25,
                'save_total_limit': 2,
                'output_dir': "./qwen3-unsloth-finetuned",
                'dataloader_num_workers': 4
            },
            'data': {'local_dir': 'finetune_experiences'},
            'lora': {
                'r': 16, 
                'alpha': 16, 
                'dropout': 0.0,
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            }
        }

    def _setup_configuration(self):
        """Setup configuration parameters from loaded config."""
        # Model settings
        model_cfg = self.config.get('model', {})
        self.model_name = model_cfg.get('name', "unsloth/Qwen3-8B-bnb-4bit")
        self.max_seq_length = int(model_cfg.get('max_seq_length', 2048))
        self.dtype = model_cfg.get('dtype', None)
        self.load_in_4bit = bool(model_cfg.get('load_in_4bit', True))
        
        # Training settings
        train_cfg = self.config.get('training', {})
        self.learning_rate = float(train_cfg.get('learning_rate', 2e-4))
        self.max_steps = int(train_cfg.get('max_steps', 50))
        self.warmup_steps = int(train_cfg.get('warmup_steps', 5))
        self.per_device_batch_size = int(train_cfg.get('per_device_batch_size', 2))
        self.gradient_accumulation_steps = int(train_cfg.get('gradient_accumulation_steps', 4))
        self.output_dir = train_cfg.get('output_dir', "./qwen3-unsloth-finetuned")
        
        # Optimization settings
        self.optim = train_cfg.get('optim', 'adamw_8bit')
        self.weight_decay = float(train_cfg.get('weight_decay', 0.01))
        self.lr_scheduler_type = train_cfg.get('lr_scheduler_type', 'linear')
        
        # Logging and saving
        self.logging_steps = int(train_cfg.get('logging_steps', 1))
        self.save_steps = int(train_cfg.get('save_steps', 25))
        self.save_total_limit = int(train_cfg.get('save_total_limit', 2))
        self.dataloader_num_workers = int(train_cfg.get('dataloader_num_workers', 4))
        
        # Data and LoRA settings
        self.data_dir = self.config.get('data', {}).get('local_dir', 'finetune_experiences')
        self.lora_config = self.config.get('lora', {})
        
        if self.local_rank == 0:
            mode = "distributed" if self.is_distributed else "single GPU"
            logger.info(f"Configuration loaded: {mode} training with {self.max_seq_length} max sequence length")

    def _validate_cuda(self):
        """Validate CUDA availability."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please ensure GPU drivers are installed.")
        
        if self.local_rank == 0:
            logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer using Unsloth."""
        if self.local_rank == 0:
            logger.info(f"Loading model: {self.model_name}")
        
        model_kwargs = {
            'model_name': self.model_name,
            'max_seq_length': self.max_seq_length,
            'dtype': self.dtype,
            'load_in_4bit': self.load_in_4bit,
        }
        
        if self.is_distributed:
            model_kwargs['device_map'] = {"": self.local_rank}
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        self._setup_lora()
        
        if self.local_rank == 0:
            logger.info("Model and tokenizer setup complete")

    def _setup_lora(self):
        """Configure and apply LoRA using Unsloth."""
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.get('r', 16),
            target_modules=self.lora_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]),
            lora_alpha=self.lora_config.get('alpha', 16),
            lora_dropout=self.lora_config.get('dropout', 0.0),
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
            logger.info(f"Loading experiences from: {self.data_dir}")
        
        for root, _, files in os.walk(self.data_dir):
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

    def format_conversations(self, examples):
        """Format conversations for training."""
        texts = []
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
            
            texts.append(conversation)
        return {"text": texts}

    def create_dataset(self):
        """Create and prepare training dataset."""
        experiences = self.load_experiences()
        dataset = Dataset.from_list(experiences)
        formatted_dataset = dataset.map(self.format_conversations, batched=True)
        
        if self.local_rank == 0:
            logger.info(f"Created dataset with {len(formatted_dataset)} examples")
        return formatted_dataset

    def get_training_arguments(self):
        """Get training arguments."""
        if self.is_distributed:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            effective_batch_size = self.per_device_batch_size * world_size * self.gradient_accumulation_steps
        else:
            effective_batch_size = self.per_device_batch_size * self.gradient_accumulation_steps
        
        if self.local_rank == 0:
            logger.info(f"Effective batch size: {effective_batch_size}")
        
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.logging_steps,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            seed=3407,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            dataloader_num_workers=self.dataloader_num_workers,
            report_to=None,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

    def train(self):
        """Train the model."""
        if self.local_rank == 0:
            mode = "distributed" if self.is_distributed else "single GPU"
            logger.info(f"Starting fine-tuning ({mode})...")
        
        self.setup_model_and_tokenizer()
        train_dataset = self.create_dataset()
        training_args = self.get_training_arguments()
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        
        if self.local_rank == 0:
            torch.cuda.empty_cache()
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory before training: {memory_gb:.2f} GB")
        
        trainer.train()
        
        # Save model only on rank 0 for distributed training
        if not self.is_distributed or self.local_rank == 0:
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Training completed! Model saved to {self.output_dir}")

    def test_model(self, prompt):
        """Test the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            if self.local_rank == 0:
                logger.error("Model not loaded! Run train() first.")
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
            logger.error(f"Error during inference: {e}")
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
