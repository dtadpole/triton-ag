from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)

# Load 4-bit quantized model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
    quantization_config=quantization_config,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load dataset (assuming it's prepared locally)
dataset = load_dataset("json", data_files="your_data.jsonl")["train"]

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not FastLanguageModel.is_bfloat16_supported(),
    bf16=FastLanguageModel.is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Adjust field name as needed
    max_seq_length=2048,
    dataset_num_proc=2,
    args=training_args,
)

# Start training
trainer.train()

# Save model
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")