from unsloth import FastLanguageModel
import torch

# Load the model - Two options:

# Option 1: Regular Qwen3-32B with 4-bit quantization (currently working)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-32B",  # Using regular Qwen3-32B
    max_seq_length=2048,
    dtype=None,  # Auto detection
    load_in_4bit=True,  # Add 4-bit loading for regular model
)

# Option 2: For AWQ models, use this instead (commented out):
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="Qwen/Qwen3-32B-AWQ",  # AWQ model
#     max_seq_length=2048,
#     dtype=None,
#     # Don't use load_in_4bit=True with AWQ models
# )

# Prepare model for finetuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Simple training data
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# Dummy training data
dataset = [
    {"instruction": "What is 2+2?", "output": "2+2 equals 4."},
    {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
]

# Format dataset
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# Simple training
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Convert to dataset
train_dataset = Dataset.from_list(dataset)
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

# Training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10,  # Very short training for demo
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to=None,  # Disable wandb
    ),
)

# Start training
print("Starting finetuning...")
trainer.train()

# Save model
model.save_pretrained("qwen_finetuned")
tokenizer.save_pretrained("qwen_finetuned")

print("Finetuning completed! Model saved to 'qwen_finetuned' directory.")
