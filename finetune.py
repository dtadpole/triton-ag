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
                    for d in data:
                        if d["role"] == "system":
                            pass
                        elif d["role"] == "user":
                            if isinstance(d["content"], dict):
                                if d["content"]["type"] == "function_call_output":
                                    d["role"] = "function"
                                    d["content"] = d["content"]["output"]
                                else:
                                    raise ValueError(f"Unknown content type: {d['content']}")
                            elif isinstance(d["content"], str):
                                pass
                            else:
                                raise ValueError(f"Unknown content type: {d['content']}")
                        elif d["role"] == "assistant":
                            if isinstance(d["content"], dict):
                                if d["content"]["type"] == "function_call":
                                    d["function_call"] = {
                                        "name": d["content"]["name"],
                                        "arguments": d["content"]["arguments"]
                                    }
                                    d["content"] = None
                                else:
                                    raise ValueError(f"Unknown content type: {d['content']}")
                            elif isinstance(d["content"], list) and len(d["content"]) > 0:
                                if d["content"][0]["type"] == "output_text":
                                    d["content"] = d["content"][0]["text"]
                                else:
                                    raise ValueError(f"Unknown content type: {d['content'][0]['type']}")
                            elif isinstance(d["content"], str):
                                pass
                            else:
                                raise ValueError(f"Unknown content type: {d['content']}")
                        else:
                            raise ValueError(f"Unknown role: {d['role']}")


                    conversation = { "messages": data }
                    experiences.append(conversation)
    
    print(f"Total number of experiences loaded: {len(experiences)}")
    return experiences

def prepare_dataset(experiences):
    # prepare the dataset in the format of { "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}] }
    dataset = Dataset.from_list(experiences[:1])
    dataset = dataset.map(lambda x: { "messages": [{"role": "user", "content": x["messages"][0]["content"]}, {"role": "assistant", "content": x["messages"][1]["content"]}] })
    return dataset

def format_function_call_data(example):
    """Convert function call examples to training format"""
    messages = example["messages"]
    formatted_text = ""
    
    for msg in messages:
        if msg["role"] == "system":
            formatted_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            formatted_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            formatted_text += f"<|im_start|>assistant\n"
            if msg.get("function_call"):
                func_call = msg["function_call"]
                formatted_text += f"<function_call>\n{func_call['name']}\n{func_call['arguments']}\n</function_call>"
            else:
                formatted_text += msg["content"]
            formatted_text += "<|im_end|>\n"
        elif msg["role"] == "function":
            formatted_text += f"<function_response>\n{msg['content']}\n</function_response>\n"
    
    return {"text": formatted_text}


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
    # load the tokenizer and the model
    model_name = "Qwen/Qwen3-32B-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype="auto",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare model for training
    # model = prepare_model_for_kbit_training(model)
    
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
