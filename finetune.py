import os
import torch
import boto3
import json
import yaml
from urllib.parse import urlparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
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

def parse_s3_url(s3_url):
    """Parse S3 URL into bucket and prefix."""
    parsed = urlparse(s3_url)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    return bucket, prefix

def download_from_s3(bucket_name, prefix, local_dir):
    """Download files from S3 bucket to local directory, maintaining folder structure."""
    s3_client = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)
    
    # List all objects under the prefix, including those in subfolders
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            
            # Skip if it's the prefix itself
            if key == prefix:
                continue
                
            # Get the relative path from the prefix
            relative_path = key[len(prefix):].lstrip('/')
            if not relative_path:
                continue
                
            # Create the full local path, maintaining the S3 folder structure
            local_path = os.path.join(local_dir, relative_path)
            
            # Create all necessary subdirectories
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            s3_client.download_file(bucket_name, key, local_path)
            print(f"Downloaded {key} to {local_path}")

def download_all_targets(config):
    """Download all targets specified in the configuration, maintaining S3 folder structure."""
    local_base_dir = config['data']['local_dir']
    
    for s3_url in config['data']['s3_folders']:
        bucket, prefix = parse_s3_url(s3_url)
        
        # Create a subdirectory structure that mirrors the S3 path
        # For example, s3://agent-xyz/kernel_coder/anthropic becomes
        # finetune_experiences/kernel_coder/anthropic
        path_parts = prefix.split('/')
        if len(path_parts) > 1:
            # Use the full path structure after the bucket name
            local_dir = os.path.join(local_base_dir, *path_parts)
        else:
            # If it's just a single folder, use it directly
            local_dir = os.path.join(local_base_dir, prefix)
        
        print(f"Downloading from {s3_url} to {local_dir}")
        download_from_s3(bucket, prefix, local_dir)

def load_experiences(config):
    """Load and process experiences from all local directories."""
    experiences = []
    base_dir = config['data']['local_dir']
    
    # Walk through all subdirectories recursively
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(root, filename), 'r') as f:
                    data = json.load(f)
                    experiences.extend(data)
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
    # Load configuration
    config = load_config()
    
    # Download experiences from S3
    print("Downloading experiences from S3...")
    download_all_targets(config)
    
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    experiences = load_experiences(config)
    dataset = prepare_dataset(experiences)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        device_map="auto",
        trust_remote_code=True,
        quantization_config={"load_in_4bit": config['model']['quantization']}
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
