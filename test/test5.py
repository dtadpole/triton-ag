#!/usr/bin/env python3
"""
Simple Qwen model fine-tuning with LoRA
Requirements: pip install transformers peft datasets torch accelerate bitsandbytes
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def main():
    # 1. Load model and tokenizer
    model_name = "Qwen/Qwen2-0.5B-Instruct"  # Small model for quick testing
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 2. Setup LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,                    # Rank
        lora_alpha=16,          # LoRA scaling parameter
        lora_dropout=0.1,       # Dropout probability
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Target attention layers
    )
    
    # 3. Apply LoRA to model
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Prepare sample dataset (replace with your own data)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Python is a versatile programming language.",
        "Artificial intelligence will shape the future.",
        "Fine-tuning models with LoRA is efficient."
    ]
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False,  # Don't pad here, let the data collator handle it
            max_length=128,
        )
    
    # Create dataset
    dataset = Dataset.from_dict({"text": sample_texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 5. Setup training arguments
    training_args = TrainingArguments(
        output_dir="./qwen-lora-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=5e-4,
        fp16=True,
        remove_unused_columns=False,
    )
    
    # 6. Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # 7. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 8. Start training
    print("Starting training...")
    trainer.train()
    
    # 9. Save the fine-tuned model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./qwen-lora-finetuned")
    
    print("Training completed! Model saved to ./qwen-lora-finetuned")
    
    # 10. Test the fine-tuned model
    print("\nTesting the fine-tuned model:")
    model.eval()
    test_input = "The future of AI is"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_input}")
    print(f"Output: {generated_text}")

if __name__ == "__main__":
    main()