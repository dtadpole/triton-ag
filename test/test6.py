#!/usr/bin/env python3
"""
Simple Unsloth Qwen3-8B Finetuning Script
Install unsloth first: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
"""

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# Configuration
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",  # The specific model you requested
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters for finetuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None, # And LoftQ
)

# Sample training data - replace with your own dataset
training_data = [
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Explain what machine learning is.",
        "input": "",
        "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
    },
    {
        "instruction": "Write a Python function to calculate fibonacci numbers.",
        "input": "",
        "output": "Here's a Python function to calculate fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
    }
]

# Format data for training
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}"
        if input_text:
            text += f"\n{input_text}"
        text += f"<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return {"text": texts}

# Create dataset
dataset = Dataset.from_list(training_data)
dataset = dataset.map(formatting_prompts_func, batched=True)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=50,  # Set to -1 for full training, or set a specific number
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    save_steps=25,
    save_total_limit=2,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=training_args,
)

# Start training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving model...")
model.save_pretrained("qwen3_finetuned")  # Local saving
tokenizer.save_pretrained("qwen3_finetuned")

# Optional: Save to Hugging Face Hub
# model.push_to_hub("your_username/qwen3_finetuned", token="your_hf_token")
# tokenizer.push_to_hub("your_username/qwen3_finetuned", token="your_hf_token")

print("Training completed!")

# Test the fine-tuned model
print("\nTesting the model...")
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is artificial intelligence?<|im_end|>\n<|im_start|>assistant\n"
    ], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.batch_decode(outputs))
