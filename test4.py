"""
Simple Qwen Model Fine-tuning with Hugging Face Transformers
Minimal setup for fine-tuning Qwen models
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
import json

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Qwen model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Using smallest model for demo
# Other options: "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", etc.

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency with LoRA
    device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else None,
    trust_remote_code=True,
    load_in_8bit=True  # Use 8-bit quantization for memory efficiency
)

# Apply LoRA configuration
print("Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,                    # Rank - controls the number of trainable parameters
    lora_alpha=16,          # LoRA scaling parameter
    lora_dropout=0.1,       # Dropout probability for LoRA layers
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Target modules for Qwen2.5
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print("LoRA applied successfully!")
model.print_trainable_parameters()

# 2. Prepare sample training data
# Replace this with your actual dataset
sample_data = [
    {"text": "Question: What is Python? Answer: Python is a programming language."},
    {"text": "Question: What is machine learning? Answer: Machine learning is a subset of AI."},
    {"text": "Question: What is fine-tuning? Answer: Fine-tuning adapts pre-trained models."},
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

# 4. Set up training arguments
training_args = TrainingArguments(
    output_dir="./qwen_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small batch size for memory efficiency
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    learning_rate=5e-4,  # Higher learning rate often works better with LoRA
    fp16=True,  # Enable mixed precision for memory efficiency with LoRA
    logging_dir="./logs",
    report_to=None,  # Disable wandb logging
    save_strategy="steps",
    eval_strategy="no",  # No evaluation for simplicity
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
print("Starting fine-tuning...")
try:
    trainer.train()
    print("Fine-tuning completed!")
    
    # Save the fine-tuned LoRA model
    trainer.save_model("./qwen_lora_finetuned_final")
    tokenizer.save_pretrained("./qwen_lora_finetuned_final")
    print("LoRA model saved to ./qwen_lora_finetuned_final")
    
except Exception as e:
    print(f"Error during training: {e}")

# 8. Test the fine-tuned model
def test_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    with torch.no_grad():
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
    test_prompt = "Question: What is AI?"
    print(f"\nTest prompt: {test_prompt}")
    print(f"Response: {test_model(test_prompt)}")