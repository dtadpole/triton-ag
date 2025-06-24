#!/usr/bin/env python3
"""
Script to show actual examples of masked vs non-masked tokens with real words.
"""

import yaml
from unsloth import FastLanguageModel
from data_processor import create_dataset, CustomDataCollatorWithMasking
from util import logger
import torch

def show_masking_examples():
    # Load config
    with open('finetune.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['name']
    max_seq_length = config['model']['max_seq_length']
    data_dir = config['data']['local_dir']
    
    logger.info(f"Loading tokenizer for: {model_name}")
    
    # Load tokenizer
    try:
        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
            device_map={"": 0}
        )
        
        # Create dataset
        train_dataset = create_dataset(data_dir, max_seq_length, 0, tokenizer)
        
        # Create data collator
        data_collator = CustomDataCollatorWithMasking(
            tokenizer=tokenizer,
            mlm=False,
            ignore_index=-100
        )
        
        # Show examples for first few samples
        for i in range(min(3, len(train_dataset))):
            print(f"\n{'='*80}")
            print(f"SAMPLE #{i+1}:")
            print(f"{'='*80}")
            
            # Get sample
            sample = train_dataset[i]
            text = sample['text']
            
            # Tokenize
            tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length)
            
            # Apply masking using data collator
            batch = data_collator([{
                'input_ids': tokenized['input_ids'].squeeze(),
                'attention_mask': tokenized['attention_mask'].squeeze()
            }])
            
            input_ids = batch['input_ids'][0]
            labels = batch['labels'][0]
            
            # Convert back to tokens for display
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # Separate masked and non-masked tokens
            masked_tokens = []
            unmasked_tokens = []
            
            current_masked_sequence = []
            current_unmasked_sequence = []
            
            for j, (token, label) in enumerate(zip(tokens, labels)):
                if label == -100:  # Masked token
                    if current_unmasked_sequence:
                        unmasked_tokens.append(' '.join(current_unmasked_sequence))
                        current_unmasked_sequence = []
                    current_masked_sequence.append(token)
                else:  # Unmasked token
                    if current_masked_sequence:
                        masked_tokens.append(' '.join(current_masked_sequence))
                        current_masked_sequence = []
                    current_unmasked_sequence.append(token)
            
            # Add any remaining sequences
            if current_masked_sequence:
                masked_tokens.append(' '.join(current_masked_sequence))
            if current_unmasked_sequence:
                unmasked_tokens.append(' '.join(current_unmasked_sequence))
            
            # Clean up tokens for display (remove special tokenizer artifacts)
            def clean_tokens(token_sequences):
                cleaned = []
                for seq in token_sequences:
                    # Replace tokenizer artifacts
                    cleaned_seq = seq.replace('Ġ', ' ').replace('▁', ' ').replace('<0x0A>', '\n')
                    # Remove extra spaces
                    cleaned_seq = ' '.join(cleaned_seq.split())
                    if cleaned_seq.strip():
                        cleaned.append(cleaned_seq)
                return cleaned
            
            masked_tokens = clean_tokens(masked_tokens)
            unmasked_tokens = clean_tokens(unmasked_tokens)
            
            # Display results
            print("\nMASKED (not trained on):")
            print("-" * 40)
            for idx, masked_seq in enumerate(masked_tokens[:5]):  # Show first 5 sequences
                print(f"{idx+1}. {masked_seq}")
            if len(masked_tokens) > 5:
                print(f"... and {len(masked_tokens) - 5} more masked sequences")
            
            print("\nNOT-MASKED (trained on):")
            print("-" * 40)
            for idx, unmasked_seq in enumerate(unmasked_tokens[:5]):  # Show first 5 sequences
                print(f"{idx+1}. {unmasked_seq}")
            if len(unmasked_tokens) > 5:
                print(f"... and {len(unmasked_tokens) - 5} more unmasked sequences")
            
            # Show statistics
            total_tokens = len(tokens)
            masked_count = (labels == -100).sum().item()
            unmasked_count = total_tokens - masked_count
            
            print(f"\nSTATISTICS:")
            print(f"Total tokens: {total_tokens}")
            print(f"Masked tokens: {masked_count} ({masked_count/total_tokens*100:.1f}%)")
            print(f"Unmasked tokens: {unmasked_count} ({unmasked_count/total_tokens*100:.1f}%)")
            
            # Show a snippet of the original conversation for context
            print(f"\nORIGINAL CONVERSATION (first 500 chars):")
            print("-" * 40)
            print(text[:500] + "..." if len(text) > 500 else text)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_masking_examples() 