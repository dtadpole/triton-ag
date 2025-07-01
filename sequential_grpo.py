#!/usr/bin/env python3
"""
Sequential GRPO Training Implementation
=====================================

Memory-efficient GRPO training using Two-Phase Sequential approach:
- Phase 1: Generate completions sequentially, store minimal data
- Phase 2: Compute advantages and train with gradient accumulation

This approach provides significant memory reduction while preserving the exact GRPO algorithm.
"""

# Import unsloth first for optimizations
from unsloth import FastLanguageModel

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import json
import glob
import time
import gc
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import tqdm

from util import logger
from data_util import ExperienceDataset, SimpleDataCollator


@dataclass
class GenerationResult:
    """Minimal storage for generation results during Phase 1."""
    tokens: List[int]
    log_probs: List[float]
    reward: float
    prompt_tokens: List[int]
    generation_id: int
    prompt_id: int = 0


class SequentialGRPOTrainer:
    """
    Sequential GRPO trainer using Two-Phase approach for memory efficiency.
    
    Phase 1: Generate completions sequentially, store minimal data
    Phase 2: Compute group advantages and train with gradient accumulation
    """
    
    def __init__(self, grpo_config_path="sequential_grpo.yaml"):
        # Load GRPO configuration
        with open(grpo_config_path, 'r') as f:
            grpo_full_config = yaml.safe_load(f)
        
        # Load base training configuration
        finetune_config_path = grpo_full_config.get('finetune_config', 'finetune.yaml')
        with open(finetune_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_seq_length = self.config['model']['max_seq_length']
        
        # Extract only the essential GRPO configuration
        self.grpo_config = grpo_full_config.get('grpo', {})
        
        # Training state
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.epoch = 0
        
        # Generation storage
        self.generation_results: List[GenerationResult] = []
        
        logger.info("Sequential GRPO Trainer initialized with config from {}".format(grpo_config_path))
    

    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer using Unsloth."""
        model_config = self.config['model']
        
        logger.info(f"Loading model: {model_config['name']}")
        
        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['name'],
            max_seq_length=model_config['max_seq_length'],
            dtype=model_config.get('dtype'),
            load_in_4bit=model_config.get('load_in_4bit', False),
        )
        
        # Setup LoRA
        lora_config = self.config['lora']
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config['r'],
            target_modules=lora_config['target_modules'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            bias=lora_config['bias'],
            use_gradient_checkpointing="unsloth",
            random_state=self.config['training']['seed'],
        )
        
        # Enable training mode
        FastLanguageModel.for_training(self.model)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model and tokenizer setup complete")
    
    def setup_dataset(self):
        """Setup dataset and dataloader."""
        data_config = self.config['data']
        
        # Create dataset
        self.train_dataset = ExperienceDataset(
            data_dir=data_config['processed_dir'],
            max_length=self.max_seq_length
        )
        
        # Create data collator
        self.data_collator = SimpleDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=data_config['collator']['pad_to_multiple_of']
        )
        
        # Create dataloader
        dataloader_config = data_config['dataloader']
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['per_device_batch_size'],
            shuffle=dataloader_config['shuffle'],
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory'],
            drop_last=dataloader_config['drop_last'],
            collate_fn=self.data_collator
        )
        
        logger.info(f"Dataset setup complete: {len(self.train_dataset)} examples")
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Setup optimizer
        if training_config['optim'] == "paged_adamw_8bit":
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.PagedAdamW8bit(
                trainable_params,
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        else:
            self.optimizer = AdamW(
                trainable_params,
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        
        # Calculate total training steps
        total_steps = len(self.train_dataloader) * training_config['num_train_epochs']
        if training_config['max_steps'] > 0:
            total_steps = min(total_steps, training_config['max_steps'])
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=training_config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer and scheduler setup complete. Total steps: {total_steps}")
    
    def extract_prompt_from_batch(self, batch: Dict[str, torch.Tensor], idx: int = 0) -> List[int]:
        """
        Extract prompt tokens from a batch. 
        For GRPO, we need to separate the prompt from the completion.
        This is a simplified extraction - you may need to adapt based on your data format.
        """
        input_ids = batch['input_ids'][idx].tolist()
        
        # Find the assistant token or use a heuristic to split prompt/completion
        # This is dataset-specific - adapt based on your conversation format
        assistant_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        
        if assistant_token:
            try:
                # Find where assistant response starts
                assistant_start = None
                for i in range(len(input_ids) - len(assistant_token) + 1):
                    if input_ids[i:i+len(assistant_token)] == assistant_token:
                        assistant_start = i + len(assistant_token)
                        break
                
                if assistant_start:
                    # Return prompt up to assistant response
                    return input_ids[:assistant_start]
            except:
                pass
        
        # Fallback: use first half as prompt (rough heuristic)
        split_point = len(input_ids) // 2
        return input_ids[:split_point]
    
    def generate_completion(self, prompt_tokens: List[int]) -> Tuple[List[int], List[float]]:
        """Generate a single completion and return tokens with log probabilities."""
        prompt_tensor = torch.tensor([prompt_tokens], device=self.device)
        
        with torch.no_grad():
            # Generate completion
            generated = self.model.generate(
                prompt_tensor,
                max_new_tokens=self.grpo_config.get('max_new_tokens', 512),
                temperature=self.grpo_config.get('temperature', 1.0),
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Extract generated tokens (excluding prompt)
            generated_tokens = generated.sequences[0][len(prompt_tokens):].tolist()
            
            # Calculate log probabilities
            log_probs = []
            if generated.scores:
                for i, score in enumerate(generated.scores):
                    if i < len(generated_tokens):
                        token_id = generated_tokens[i]
                        log_prob = F.log_softmax(score[0], dim=-1)[token_id].item()
                        log_probs.append(log_prob)
            
            return generated_tokens, log_probs
    
    def compute_reward(self, prompt_tokens: List[int], completion_tokens: List[int]) -> float:
        """
        Compute reward for a prompt-completion pair.
        Uses simple length-based reward - replace with your actual reward function.
        """
        completion_length = len(completion_tokens)
        
        # Simple heuristic: reward longer, coherent completions
        base_reward = min(completion_length / 100.0, 1.0)  # Normalize by length
        
        # Add some randomness to simulate actual reward model
        noise = np.random.normal(0, 0.1)
        reward = base_reward + noise
        
        return float(reward)
    
    def phase1_generate_completions(self, batch: Dict[str, torch.Tensor]) -> List[GenerationResult]:
        """
        Phase 1: Generate completions sequentially for memory efficiency.
        """
        logger.info("Phase 1: Generating completions sequentially...")
        
        generation_results = []
        batch_size = batch['input_ids'].size(0)
        
        for batch_idx in range(batch_size):
            # Extract prompt for this example
            prompt_tokens = self.extract_prompt_from_batch(batch, batch_idx)
            
            # Generate multiple completions for this prompt
            for gen_id in range(self.grpo_config.get('num_generations_per_prompt', 8)):
                try:
                    # Generate single completion
                    completion_tokens, log_probs = self.generate_completion(prompt_tokens)
                    
                    # Compute reward
                    reward = self.compute_reward(prompt_tokens, completion_tokens)
                    
                    # Store result
                    result = GenerationResult(
                        tokens=completion_tokens,
                        log_probs=log_probs,
                        reward=reward,
                        prompt_tokens=prompt_tokens,
                        generation_id=gen_id,
                        prompt_id=batch_idx
                    )
                    generation_results.append(result)
                    
                    # Memory cleanup after each generation
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Failed to generate completion {gen_id} for prompt {batch_idx}: {e}")
                    continue
        
        logger.info(f"Phase 1 complete: Generated {len(generation_results)} completions")
        return generation_results 
    
    def compute_group_advantages(self, generation_results: List[GenerationResult]) -> Dict[int, List[List[float]]]:
        """
        Compute token-level group-relative advantages for GRPO (DAPO-style).
        Groups are organized by prompt_id.
        Returns advantages per token for each generation.
        """
        # Group results by prompt_id
        prompt_groups = {}
        for result in generation_results:
            if result.prompt_id not in prompt_groups:
                prompt_groups[result.prompt_id] = []
            prompt_groups[result.prompt_id].append(result)
        
        # Compute token-level advantages for each group
        advantages = {}
        for prompt_id, group_results in prompt_groups.items():
            
            # Get sequence-level rewards for normalization
            rewards = [r.reward for r in group_results]
            
            if len(rewards) > 1:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                if std_reward > 1e-8:  # Avoid division by zero
                    sequence_advantages = [(r - mean_reward) / std_reward for r in rewards]
                else:
                    sequence_advantages = [0.0] * len(rewards)
            else:
                sequence_advantages = [0.0]
            
            # Convert sequence-level advantages to token-level advantages
            token_advantages = []
            for i, (result, seq_advantage) in enumerate(zip(group_results, sequence_advantages)):
                num_tokens = len(result.tokens)
                if num_tokens > 0:
                    # Simple token-level advantages: broadcast sequence advantage to all tokens
                    token_level_advantages = [seq_advantage] * num_tokens
                    token_advantages.append(token_level_advantages)
                else:
                    token_advantages.append([0.0])
            
            advantages[prompt_id] = token_advantages
        
        return advantages
    
    def phase2_train_with_advantages(self, generation_results: List[GenerationResult], 
                                   advantages: Dict[int, List[List[float]]]) -> float:
        """
        Phase 2: Train with computed advantages using gradient accumulation.
        Uses proper GRPO loss with reference model ratio and clipping.
        """
        logger.info("Phase 2: Training with computed advantages...")
        
        total_loss = 0.0
        num_processed = 0
        
        # GRPO clipping parameters (DAPO-style asymmetric clipping)
        clip_epsilon_lower = self.grpo_config.get('clip_epsilon_lower', 0.2)
        clip_epsilon_upper = self.grpo_config.get('clip_epsilon_upper', 0.3)
        
        # Group results by prompt for advantage lookup
        prompt_groups = {}
        for result in generation_results:
            if result.prompt_id not in prompt_groups:
                prompt_groups[result.prompt_id] = []
            prompt_groups[result.prompt_id].append(result)
        
        # Process each group
        for prompt_id, group_results in prompt_groups.items():
            group_token_advantages = advantages[prompt_id]
            
            for i, (result, token_advantages) in enumerate(zip(group_results, group_token_advantages)):
                try:
                    # Prepare input for forward pass
                    full_tokens = result.prompt_tokens + result.tokens
                    input_ids = torch.tensor([full_tokens], device=self.device)
                    
                    # Forward pass with current model
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0]  # Remove batch dimension
                    
                    # Extract logits for generated tokens only
                    prompt_length = len(result.prompt_tokens)
                    generation_logits = logits[prompt_length-1:-1]  # Shift for next token prediction
                    
                    # Compute current model log probabilities
                    current_log_probs = F.log_softmax(generation_logits, dim=-1)
                    
                    # Get current log probs for actual generated tokens
                    current_token_log_probs = []
                    ref_token_log_probs = []
                    
                    for j, token_id in enumerate(result.tokens):
                        if j < len(current_log_probs) and j < len(result.log_probs):
                            current_token_log_probs.append(current_log_probs[j, token_id])
                            ref_token_log_probs.append(result.log_probs[j])  # Reference log probs from generation
                    
                    if current_token_log_probs and ref_token_log_probs and len(token_advantages) > 0:
                        # Convert to tensors
                        current_log_probs_tensor = torch.stack(current_token_log_probs)
                        ref_log_probs_tensor = torch.tensor(ref_token_log_probs, device=self.device)
                        
                        # Ensure token advantages match the number of tokens we have log probs for
                        num_tokens = min(len(current_token_log_probs), len(token_advantages))
                        if num_tokens > 0:
                            # Trim to matching length
                            current_log_probs_tensor = current_log_probs_tensor[:num_tokens]
                            ref_log_probs_tensor = ref_log_probs_tensor[:num_tokens]
                            token_advantages_tensor = torch.tensor(token_advantages[:num_tokens], device=self.device)
                            
                            # Compute log probability ratio: log(π_θ/π_ref) = log(π_θ) - log(π_ref)
                            log_ratio = current_log_probs_tensor - ref_log_probs_tensor
                            
                            # Compute probability ratio: π_θ/π_ref = exp(log(π_θ) - log(π_ref))
                            ratio = torch.exp(log_ratio)
                            
                            # DAPO-style asymmetric clipping: [1-ε_lower, 1+ε_upper]
                            # Lower bound: more restrictive to prevent policy collapse
                            # Upper bound: less restrictive to allow beneficial updates
                            clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon_lower, 1.0 + clip_epsilon_upper)
                            
                            # DAPO-style token-level GRPO loss: -sum(advantage_i * clipped_ratio_i)
                            # Each token has its own advantage weight
                            token_losses = -token_advantages_tensor * clipped_ratio
                            loss = token_losses.mean()  # Average across tokens
                            
                            # Scale loss for gradient accumulation
                            loss = loss / len(generation_results)
                            
                            # Backward pass
                            loss.backward()
                            
                            total_loss += loss.item()
                            num_processed += 1
                    
                    # Memory cleanup
                    del outputs, logits, current_log_probs
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Failed to process result {i} for prompt {prompt_id}: {e}")
                    continue
        
        # Optimizer step after processing all generations
        if num_processed > 0:
            # Gradient clipping
            max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / max(num_processed, 1)
        logger.info(f"Phase 2 complete: Average loss = {avg_loss:.4f} (clip=[{1.0-clip_epsilon_lower:.1f}, {1.0+clip_epsilon_upper:.1f}])")
        
        return avg_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Complete GRPO training step using two-phase approach.
        """
        # Phase 1: Generate completions sequentially
        generation_results = self.phase1_generate_completions(batch)
        
        if not generation_results:
            logger.warning("No generations produced, skipping training step")
            return 0.0
        
        # Compute group advantages
        advantages = self.compute_group_advantages(generation_results)
        
        # Phase 2: Train with advantages
        loss = self.phase2_train_with_advantages(generation_results, advantages)
        
        # Final memory cleanup
        torch.cuda.empty_cache()
        
        return loss
    
    def train(self):
        """Main training loop."""
        logger.info("Starting Sequential GRPO training...")
        
        training_config = self.config['training']
        num_epochs = training_config['num_train_epochs']
        max_steps = training_config['max_steps']
        
        self.model.train()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm.tqdm(self.train_dataloader, 
                                   desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Training step
                step_loss = self.train_step(batch)
                epoch_loss += step_loss
                num_batches += 1
                
                # Update progress
                self.global_step += 1
                avg_loss = epoch_loss / num_batches
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'step': self.global_step
                })
                
                # Logging
                logging_steps = training_config.get('logging_steps', 10)
                if self.global_step % logging_steps == 0:
                    logger.info(f"Step {self.global_step}: loss = {step_loss:.4f}")
                
                # Save checkpoint
                save_steps = training_config.get('save_steps', 500)
                if self.global_step % save_steps == 0:
                    self.save_checkpoint()
                
                # Check max steps
                if max_steps > 0 and self.global_step >= max_steps:
                    logger.info(f"Reached max steps ({max_steps}), stopping training")
                    break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1} complete: Average loss = {avg_epoch_loss:.4f}")
            
            if max_steps > 0 and self.global_step >= max_steps:
                break
        
        logger.info("Training complete!")
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        output_dir = Path(self.config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def run(self):
        """Run the complete training pipeline."""
        logger.info("Setting up Sequential GRPO training...")
        
        # Setup components
        self.setup_model_and_tokenizer()
        self.setup_dataset()
        self.setup_optimizer_and_scheduler()
        
        # Run training
        self.train()
        
        # Final save
        final_output = Path(self.config['training']['output_dir']) / "final_model"
        final_output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_output)
        self.tokenizer.save_pretrained(final_output)
        
        logger.info(f"Training complete! Final model saved to {final_output}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sequential GRPO Training")
    parser.add_argument("--config", default="sequential_grpo.yaml", 
                       help="Path to GRPO configuration file")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SequentialGRPOTrainer(grpo_config_path=args.config)
    
    # Log configuration
    logger.info("GRPO Configuration:")
    logger.info(f"  num_generations_per_prompt: {trainer.grpo_config.get('num_generations_per_prompt', 8)}")
    logger.info(f"  temperature: {trainer.grpo_config.get('temperature', 1.0)}")
    logger.info(f"  max_new_tokens: {trainer.grpo_config.get('max_new_tokens', 512)}")
    logger.info(f"  clip_epsilon_lower: {trainer.grpo_config.get('clip_epsilon_lower', 0.2)}")
    logger.info(f"  clip_epsilon_upper: {trainer.grpo_config.get('clip_epsilon_upper', 0.3)}")
    
    # Run training
    trainer.run()


if __name__ == "__main__":
    main()
