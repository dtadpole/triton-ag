import os
import torch
import json
import yaml
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

def load_config(config_path="finetune.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_experiences(local_dir):
    """Load and process experiences from all local directories recursively."""
    experiences = []
    
    # Walk through all subdirectories recursively
    for root, _, files in os.walk(local_dir):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                print(f"Loading experiences from: {file_path}")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    experiences.extend(data)
    
    print(f"Total number of experiences loaded: {len(experiences)}")
    return experiences

def prepare_dataset(experiences):
    """Convert experiences to HuggingFace dataset format."""
    texts = []
    for exp in experiences:
        # Adjust this based on your experience format
        text = f"Instruction: {exp.get('instruction', '')}\n"
        text += f"Input: {exp.get('input', '')}\n"
        text += f"Output: {exp.get('output', '')}\n"
        texts.append(text)
    
    return Dataset.from_dict({"text": texts})

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune Qwen model with local experiences')
    parser.add_argument('-e', '--experiences_dir', type=str, default='finetune_experiences', help='Directory containing experience JSON files')
    parser.add_argument('-c', '--config', type=str, default='finetune.yaml',
                      help='Path to configuration file (default: finetune.yaml)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load and prepare dataset
    print(f"Loading experiences from directory: {args.experiences_dir}")
    experiences = load_experiences(args.experiences_dir)
    dataset = prepare_dataset(experiences)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config['training']['max_length']
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=config['training']['warmup_ratio'],
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config['training']['output_dir'])

if __name__ == "__main__":
    main()
