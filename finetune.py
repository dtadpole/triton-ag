#!/usr/bin/env python3
"""
Qwen Fine-tuning with LoRA
A clean and efficient script for fine-tuning Qwen models using LoRA (Low-Rank Adaptation)
on function calling and conversation data.
"""

import os
import json
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
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", output_dir="./qwen-lora-finetuned"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = 4096  # Maximum sequence length for comprehensive conversations
        self.tokenizer = None
        self.model = None
        
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please ensure GPU drivers are installed.")
        
        logger.info(f"Using CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")
    
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
        """Initialize model and tokenizer with CUDA optimization."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with CUDA optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True
        )
        
        # Resize embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup LoRA
        self._setup_lora()
        
        logger.info("Model and tokenizer setup complete")
    
    def _setup_lora(self):
        """Configure and apply LoRA to the model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj"       # MLP layers
            ]
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
    
    def train(self, epochs=1, learning_rate=5e-4, batch_size=1, experience_dir="./finetune_experiences"):
        """Train the model with LoRA."""
        logger.info("Starting fine-tuning process...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Create dataset
        train_dataset = self.create_dataset(experience_dir)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=32,  # Increased for larger dataset
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            logging_steps=1,
            save_steps=5,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True,
            report_to=None
        )
        
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
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
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
    # Configuration
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "output_dir": "./qwen-lora-finetuned",
        "experience_dir": "./finetune_experiences",
        "epochs": 1,
        "learning_rate": 5e-4,
        "batch_size": 1
    }
    
    # Initialize trainer
    trainer = QwenLoRATrainer(
        model_name=config["model_name"],
        output_dir=config["output_dir"]
    )
    
    # Train the model
    trainer.train(
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        experience_dir=config["experience_dir"]
    )
    
    # Test the model
    test_prompt = "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n"
    logger.info("Testing the fine-tuned model...")
    response = trainer.test_model(test_prompt)
    logger.info(f"Model response: {response}")


if __name__ == "__main__":
    main()
