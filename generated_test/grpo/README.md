# Sequential GRPO Training with Simplified Configuration

This directory contains the Sequential GRPO training implementation with a **simplified** YAML configuration system.

## Overview

The Sequential GRPO trainer has been simplified to focus on **only the essential GRPO parameters**. Complex configurations like reward functions, generation parameters, and memory management are now handled with sensible defaults in the code, making it much easier to use and maintain.

## Key Files

- `sequential_grpo.yaml` - **Simplified** GRPO configuration file with only essential parameters
- `sequential_grpo.py` - Updated trainer implementation  
- `test_sequential_grpo.py` - Test script demonstrating the simplified configuration system

## Simplified Configuration Structure

The `sequential_grpo.yaml` file now contains **only the essential GRPO parameters**:

```yaml
# Base training configuration file
finetune_config: "finetune.yaml"

# Core GRPO parameters
grpo:
  # Number of generations per prompt
  num_generations_per_prompt: 8
  
  # Generation temperature
  temperature: 1.0
  
  # Maximum new tokens per generation
  max_new_tokens: 512
  
  # DAPO-style asymmetric clipping parameters
  clip_epsilon_lower: 0.2  # Lower bound: more restrictive to prevent policy collapse
  clip_epsilon_upper: 0.3  # Upper bound: less restrictive to allow beneficial updates
```

## What Was Simplified

### ✅ Kept (Essential GRPO Parameters)
- `num_generations_per_prompt` - Core to GRPO algorithm
- `temperature` - Important for generation diversity  
- `max_new_tokens` - Generation length control
- `clip_epsilon_lower/upper` - DAPO-style asymmetric clipping

### ❌ Removed (Now Hardcoded Defaults)
- **Reward configuration** - Use the `compute_reward()` method in code
- **Generation parameters** - Sensible defaults (top_k=50, top_p=0.95, etc.)
- **Memory management** - Automatic CUDA cache clearing
- **Advantage computation** - Standard group-relative advantages with standardization
- **Debug settings** - Minimal essential logging only

## Usage Examples

### 1. Basic Usage
```bash
# Using default configuration
python sequential_grpo.py

# Using custom config file
python sequential_grpo.py --config my_grpo_config.yaml
```

### 2. Programmatic Usage
```python
from sequential_grpo import SequentialGRPOTrainer

# Load trainer with config
trainer = SequentialGRPOTrainer('sequential_grpo.yaml')

# Modify specific values if needed
trainer.grpo_config['temperature'] = 0.8
trainer.grpo_config['num_generations_per_prompt'] = 16

# Run training
trainer.run()
```

### 3. Custom Reward Function
To implement your own reward function, simply modify the `compute_reward()` method:

```python
def compute_reward(self, prompt_tokens: List[int], completion_tokens: List[int]) -> float:
    # Your custom reward logic here
    # Examples:
    # - Call your reward model
    # - Use semantic similarity
    # - Apply task-specific metrics
    
    # Simple example: prefer completions with specific tokens
    reward = 0.5  # base reward
    if any(token_id in completion_tokens for token_id in [123, 456]):  # special tokens
        reward += 0.5
    return reward
```

### 4. Minimal Custom Configuration
```yaml
# my_simple_config.yaml
finetune_config: "my_finetune.yaml"

grpo:
  num_generations_per_prompt: 4  # Fewer generations for faster training
  temperature: 0.7               # More focused generation
  max_new_tokens: 256           # Shorter completions
  clip_epsilon_lower: 0.1       # Tighter clipping
  clip_epsilon_upper: 0.4       # Looser upper bound
```

## Key Benefits of Simplification

### 🚀 **Easier to Use**
- Only 5 essential parameters to configure
- No complex nested configuration sections
- Clear, focused YAML structure

### 🛠️ **Easier to Maintain**
- Reward logic in code where it belongs
- Sensible defaults eliminate most configuration needs
- Less chance for configuration errors

### 📚 **Easier to Understand**
- Focus on what matters for GRPO
- Clear separation of concerns
- Minimal cognitive overhead

### ⚡ **Faster to Get Started**
- Copy the 5-line GRPO config section
- Implement `compute_reward()` for your task
- Run training immediately

## Default Behavior

When parameters are not specified in the config, these defaults are used:

### Generation Parameters
- `do_sample: true`
- `top_k: 50`
- `top_p: 0.95` 
- `repetition_penalty: 1.0`

### Memory Management
- Automatic CUDA cache clearing after each generation and batch
- Memory optimization handled transparently

### Advantage Computation
- Group-relative advantages (standard GRPO)
- Standardization with mean-centering and std normalization
- Automatic handling of edge cases (single generation, zero std, etc.)

### Reward Function
- Simple length-based reward with noise (placeholder)
- **Replace with your task-specific reward function**

## Migration Guide

### From Previous Complex Config
**Old (Complex)**:
```yaml
reward:
  type: "length_based"
  length_based:
    target_length: 100
    noise_std: 0.1
generation:
  do_sample: true
  top_k: 50
memory:
  clear_cache_after_generation: true
```

**New (Simplified)**:
```yaml
grpo:
  num_generations_per_prompt: 8
  temperature: 1.0
  max_new_tokens: 512
  clip_epsilon_lower: 0.2
  clip_epsilon_upper: 0.3
```

### Command Line Changes
**Old**: Many CLI options for overriding config values  
**New**: Only `--config` to specify the YAML file

## Testing

Run the simplified test suite:
```bash
cd generated_test/grpo
python test_sequential_grpo.py
```

The tests verify:
- ✅ Configuration loading
- ✅ Custom config files
- ✅ Reward function computation

## Example Workflow

1. **Start with defaults**: Use `sequential_grpo.yaml` as-is
2. **Implement reward**: Modify `compute_reward()` method for your task
3. **Tune GRPO params**: Adjust the 5 parameters based on your needs
4. **Run training**: `python sequential_grpo.py --config your_config.yaml`

This simplified approach gets you to productive GRPO training much faster! 