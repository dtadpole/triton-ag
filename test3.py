import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np

def main():
    print("Starting script...")
    
    # Load model and tokenizer
    print("Loading tokenizer...")
    model_name = "Qwen/Qwen-1_8B"  # Using smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set up tokenizer for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
    print(f"Tokenizer pad token: {tokenizer.pad_token}, pad token id: {tokenizer.pad_token_id}")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        pad_token_id=tokenizer.pad_token_id  # Explicitly set pad_token_id in model config
    )
    print("Model loaded successfully")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
    print("Dataset loaded successfully")
    
    def preprocess_function(examples):
        # Format the text for training
        texts = [f"Instruction: {instruction}\nInput: {input}\nOutput: {output}"
                for instruction, input, output in zip(examples['instruction'], 
                                                    examples['input'], 
                                                    examples['output'])]
        
        # Tokenize the texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None
        )
        
        # Add labels for language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    print("Dataset preprocessing completed")
    
    # Set up training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./qwen-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )
    print("Training arguments set up")
    
    # Initialize trainer with default data collator
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Helps with performance
        ),
    )
    print("Trainer initialized")
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed")
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./qwen-finetuned")
    print("Model saved successfully")

if __name__ == "__main__":
    main()
