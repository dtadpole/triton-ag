"""
Simple Qwen3 Model Fine-tuning with Hugging Face Transformers and FSDP
Minimal setup for fine-tuning Qwen3 models using Fully Sharded Data Parallel
"""

import os
# Disable GPU 0 - only use GPUs 1-5
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

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
import json

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Qwen3 model and tokenizer
model_name = "Qwen/Qwen3-0.6B"  # Using Qwen3 0.6B model
# Other Qwen3 options: "Qwen/Qwen3-1.8B", "Qwen/Qwen3-8B", etc.

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    # Remove device_map for FSDP compatibility
    trust_remote_code=True,
    # Remove load_in_8bit for FSDP compatibility - FSDP handles memory efficiency
)

# Apply LoRA configuration
print("Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,                    # Rank - controls the number of trainable parameters
    lora_alpha=16,          # LoRA scaling parameter
    lora_dropout=0.1,       # Dropout probability for LoRA layers
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Target modules for Qwen3
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print("LoRA applied successfully!")
model.print_trainable_parameters()

# 2. Prepare sample training data for Qwen3
# Replace this with your actual dataset
sample_data = [
    {"text": "<|im_start|>user\nWhat is Python?<|im_end|>\n<|im_start|>assistant\nPython is a high-level programming language known for its simplicity and versatility.<|im_end|>"},
    {"text": "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\nMachine learning is a subset of artificial intelligence that enables computers to learn and improve from data without explicit programming.<|im_end|>"},
    {"text": "<|im_start|>user\nWhat is fine-tuning?<|im_end|>\n<|im_start|>assistant\nFine-tuning is the process of adapting a pre-trained model to a specific task by training it on task-specific data.<|im_end|>"},
    {"text": "<|im_start|>user\nExplain Qwen3 models<|im_end|>\n<|im_start|>assistant\nQwen3 is a series of large language models developed by Alibaba, featuring improved capabilities and efficiency compared to previous versions.<|im_end|>"},
    # Add more training examples here
]

# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # Don't pad here, let the data collator handle it
        max_length=512,
    )

# Convert to HuggingFace dataset
dataset = Dataset.from_list(sample_data)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Set up training arguments with FSDP configuration
training_args = TrainingArguments(
    output_dir="./qwen3_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small batch size for memory efficiency
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    learning_rate=5e-4,  # Higher learning rate often works better with LoRA
    bf16=True,  # Use bf16 instead of fp16 for better FSDP compatibility
    logging_dir="./logs",
    report_to=None,  # Disable wandb logging
    save_strategy="steps",
    eval_strategy="no",  # No evaluation for simplicity
    
    # FSDP Configuration
    fsdp="full_shard auto_wrap",  # Enable full sharding with auto wrapping
    fsdp_config={
        "min_num_params": 0,  # Minimum number of parameters for a layer to be wrapped
        "xla": False,  # Set to True if using TPUs
        "xla_fsdp_v2": False,
        "xla_fsdp_grad_ckpt": False,
    },
    # FSDP transformer wrapping policy
    fsdp_transformer_layer_cls_to_wrap="Qwen3DecoderLayer",  # Wrap each transformer layer for Qwen3
    
    # Additional FSDP settings
    dataloader_pin_memory=False,  # Disable pin memory for FSDP
    remove_unused_columns=False,  # Keep all columns for FSDP
)

# 5. Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

# 6. Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 7. Start fine-tuning
print("Starting fine-tuning with FSDP...")
try:
    trainer.train()
    print("Fine-tuning completed!")
    
    # Save the fine-tuned LoRA model
    trainer.save_model("./qwen3_lora_finetuned_final")
    tokenizer.save_pretrained("./qwen3_lora_finetuned_final")
    print("LoRA model saved to ./qwen3_lora_finetuned_final")
    
except Exception as e:
    print(f"Error during training: {e}")

# 8. Test the fine-tuned model
def test_model(prompt):
    # For FSDP models, we need to ensure the model is in the right state for inference
    model.eval()  # Set to evaluation mode
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model and ensure consistent dtype
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Ensure model parameters are in consistent dtype for inference
    # Convert model to float16 for inference to avoid dtype mismatch
    model_dtype = next(model.parameters()).dtype
    if model_dtype != torch.float16:
        # Convert inputs to match model dtype
        if hasattr(inputs, 'input_ids'):
            # input_ids should remain as long/int, only convert embeddings if needed
            pass
    
    with torch.no_grad():
        # Use torch.autocast to handle mixed precision inference
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

# Example usage
if __name__ == "__main__":
    # Test the model after training
    test_prompt = "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n"
    print(f"\nTest prompt: {test_prompt}")
    print(f"Response: {test_model(test_prompt)}")