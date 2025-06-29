#!/usr/bin/env python3
"""
Example Usage of Sequential GRPO Training
=========================================

This script demonstrates how to use the Sequential GRPO trainer with different configurations.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sequential_grpo import SequentialGRPOTrainer
from util import logger


def example_basic_usage():
    """Basic usage example with default configuration."""
    logger.info("Example 1: Basic Usage")
    logger.info("-" * 30)
    
    # Create trainer with default config
    trainer = SequentialGRPOTrainer("finetune.yaml")
    
    # Run training
    trainer.run()


def example_custom_grpo_config():
    """Example with custom GRPO configuration."""
    logger.info("Example 2: Custom GRPO Configuration")
    logger.info("-" * 40)
    
    # Create trainer
    trainer = SequentialGRPOTrainer("finetune.yaml")
    
    # Customize GRPO settings
    trainer.grpo_config.update({
        'num_generations_per_prompt': 4,  # Fewer generations for faster training
        'reward_weight': 0.5,             # Lower reward weight
        'temperature': 0.8,               # Higher temperature for more diverse generations
        'max_new_tokens': 256,            # Shorter completions
        'clip_epsilon_lower': 0.1,        # More restrictive lower bound
        'clip_epsilon_upper': 0.5,        # Less restrictive upper bound (DAPO-style)
    })
    
    logger.info(f"GRPO Config: {trainer.grpo_config}")
    
    # Run training
    trainer.run()


def example_memory_efficient_config():
    """Example optimized for memory efficiency."""
    logger.info("Example 3: Memory-Efficient Configuration")
    logger.info("-" * 45)
    
    # Create trainer
    trainer = SequentialGRPOTrainer("finetune.yaml")
    
    # Very memory-efficient settings
    trainer.grpo_config.update({
        'num_generations_per_prompt': 2,  # Minimal generations
        'max_new_tokens': 128,            # Short completions
        'generation_batch_size': 1,       # One at a time
    })
    
    # You could also modify the model config for smaller models
    # trainer.config['model']['name'] = "Qwen/Qwen3-1.5B"  # Smaller model
    
    logger.info(f"Memory-efficient GRPO Config: {trainer.grpo_config}")
    
    # Run training
    trainer.run()


def example_with_custom_reward_function():
    """Example showing how to implement a custom reward function."""
    logger.info("Example 4: Custom Reward Function")
    logger.info("-" * 35)
    
    class CustomGRPOTrainer(SequentialGRPOTrainer):
        """Custom trainer with specialized reward function."""
        
        def compute_reward(self, prompt_tokens, completion_tokens):
            """
            Custom reward function that rewards specific patterns.
            Replace this with your actual reward model.
            """
            # Convert tokens back to text for analysis
            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            completion_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
            
            # Example: Reward completions that contain certain keywords
            reward = 0.0
            
            # Reward technical terms (since this is for Triton development)
            technical_terms = ['kernel', 'triton', 'cuda', 'gpu', 'optimization', 'performance']
            for term in technical_terms:
                if term.lower() in completion_text.lower():
                    reward += 0.1
            
            # Reward longer, more detailed responses
            reward += min(len(completion_tokens) / 200.0, 0.5)
            
            # Penalize very short responses
            if len(completion_tokens) < 10:
                reward -= 0.2
            
            # Add some noise to simulate real reward model uncertainty
            import numpy as np
            noise = np.random.normal(0, 0.05)
            reward += noise
            
            return float(reward)
    
    # Create custom trainer
    trainer = CustomGRPOTrainer("finetune.yaml")
    
    # Customize for this example
    trainer.grpo_config['num_generations_per_prompt'] = 6
    
    logger.info("Using custom reward function that rewards technical content")
    
    # Run training
    trainer.run()


def example_command_line_usage():
    """Show how to use command line arguments."""
    logger.info("Example 5: Command Line Usage")
    logger.info("-" * 30)
    
    example_commands = [
        # Basic usage
        "python sequential_grpo.py",
        
        # Custom config file
        "python sequential_grpo.py --config my_config.yaml",
        
        # Custom number of generations
        "python sequential_grpo.py --num_generations 4",
        
        # DAPO-style asymmetric clipping with token-level advantages
        "python sequential_grpo.py --clip_epsilon_lower 0.1 --clip_epsilon_upper 0.5",
        
        # Custom reward weight
        "python sequential_grpo.py --reward_weight 0.5",
        
        # Combined options with token-level advantages
        "python sequential_grpo.py --config my_config.yaml --num_generations 6 --reward_weight 0.8 --clip_epsilon_lower 0.15 --clip_epsilon_upper 0.45",
    ]
    
    logger.info("Command line examples:")
    for cmd in example_commands:
        logger.info(f"  {cmd}")


def main():
    """Main function to run examples."""
    logger.info("Sequential GRPO Training Examples")
    logger.info("=" * 50)
    
    # Check if data directory exists
    data_dir = "finetune_processed_experiences"
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found!")
        logger.error("Please ensure you have the processed experience files.")
        return 1
    
    # Show command line usage (doesn't require running)
    example_command_line_usage()
    
    logger.info("\n" + "=" * 50)
    logger.info("Note: To run the actual training examples, uncomment the desired example below:")
    logger.info("⚠️  Training examples are commented out to avoid accidental long runs")
    
    # Uncomment ONE of these to run actual training:
    
    # example_basic_usage()
    # example_custom_grpo_config()
    # example_memory_efficient_config()
    # example_with_custom_reward_function()
    
    logger.info("\n🚀 To start training, run:")
    logger.info("python sequential_grpo.py")
    
    return 0


if __name__ == "__main__":
    exit(main()) 