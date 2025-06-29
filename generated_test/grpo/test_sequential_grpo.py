#!/usr/bin/env python3
"""
Test script for Sequential GRPO with simplified YAML configuration
=================================================================

This script demonstrates how to use the simplified YAML configuration system
for Sequential GRPO training.
"""

import sys
import os
import yaml
from pathlib import Path

# Add parent directory to path to import sequential_grpo
sys.path.append(str(Path(__file__).parent.parent.parent))

from sequential_grpo import SequentialGRPOTrainer
from util import logger


def test_config_loading():
    """Test loading configuration from YAML file."""
    print("Testing YAML configuration loading...")
    
    try:
        # Test with default config
        trainer = SequentialGRPOTrainer()
        
        print("✓ Successfully loaded default configuration")
        print(f"  GRPO config: {trainer.grpo_config}")
        
        # Verify essential GRPO parameters are present
        assert 'num_generations_per_prompt' in trainer.grpo_config
        assert 'temperature' in trainer.grpo_config
        assert 'max_new_tokens' in trainer.grpo_config
        assert 'clip_epsilon_lower' in trainer.grpo_config
        assert 'clip_epsilon_upper' in trainer.grpo_config
        
        print("✓ All essential GRPO parameters loaded")
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False
    
    return True


def test_custom_config_file():
    """Test using a custom configuration file."""
    print("\nTesting custom configuration file...")
    
    # Create a simplified test config file
    test_config = {
        'finetune_config': 'finetune.yaml',
        'grpo': {
            'num_generations_per_prompt': 4,
            'temperature': 0.5,
            'max_new_tokens': 256,
            'clip_epsilon_lower': 0.1,
            'clip_epsilon_upper': 0.4
        }
    }
    
    # Write test config
    test_config_path = 'test_grpo_config.yaml'
    try:
        with open(test_config_path, 'w') as f:
            yaml.dump(test_config, f, default_flow_style=False)
        
        # Test loading the custom config
        trainer = SequentialGRPOTrainer(grpo_config_path=test_config_path)
        
        # Verify custom values
        assert trainer.grpo_config['num_generations_per_prompt'] == 4
        assert trainer.grpo_config['temperature'] == 0.5
        assert trainer.grpo_config['max_new_tokens'] == 256
        assert trainer.grpo_config['clip_epsilon_lower'] == 0.1
        assert trainer.grpo_config['clip_epsilon_upper'] == 0.4
        
        print("✓ Successfully loaded custom configuration file")
        
        # Cleanup
        os.remove(test_config_path)
        
    except Exception as e:
        print(f"✗ Failed to use custom configuration file: {e}")
        # Cleanup on failure
        if os.path.exists(test_config_path):
            os.remove(test_config_path)
        return False
    
    return True


def test_reward_function():
    """Test the simplified reward function."""
    print("\nTesting simplified reward function...")
    
    try:
        trainer = SequentialGRPOTrainer()
        
        # Test reward computation
        prompt_tokens = [1, 2, 3, 4, 5]
        completion_tokens = [6, 7, 8, 9, 10] * 20  # 100 tokens
        
        reward = trainer.compute_reward(prompt_tokens, completion_tokens)
        
        # Should be close to 1.0 for 100 tokens + some noise
        assert 0.5 < reward < 1.5, f"Reward {reward} seems out of expected range"
        
        print(f"✓ Reward function working: {reward:.3f} for {len(completion_tokens)} tokens")
        
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        return False
    
    return True


def print_example_usage():
    """Print example usage instructions."""
    print("\n" + "="*60)
    print("SIMPLIFIED CONFIGURATION USAGE")
    print("="*60)
    
    print("\n1. Using default configuration:")
    print("   python sequential_grpo.py")
    
    print("\n2. Using custom GRPO config file:")
    print("   python sequential_grpo.py --config my_grpo_config.yaml")
    
    print("\n3. Minimal sequential_grpo.yaml structure:")
    print("   ```yaml")
    print("   finetune_config: 'finetune.yaml'")
    print("   grpo:")
    print("     num_generations_per_prompt: 8")
    print("     temperature: 1.0")
    print("     max_new_tokens: 512")
    print("     clip_epsilon_lower: 0.2")
    print("     clip_epsilon_upper: 0.3")
    print("   ```")
    
    print("\n4. Programmatic usage:")
    print("   ```python")
    print("   trainer = SequentialGRPOTrainer('my_config.yaml')")
    print("   # Modify specific values if needed:")
    print("   trainer.grpo_config['temperature'] = 0.8")
    print("   trainer.run()")
    print("   ```")
    
    print("\n5. Key simplifications:")
    print("   - Reward function is now in code (compute_reward method)")
    print("   - Generation parameters use hardcoded defaults")
    print("   - Memory management is automatic")
    print("   - Only essential GRPO parameters in config file")


def main():
    """Run all tests."""
    print("Sequential GRPO Simplified Configuration Tests")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_custom_config_file,
        test_reward_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        print_example_usage()
    else:
        print("❌ Some tests failed. Please check the configuration setup.")
        sys.exit(1)


if __name__ == "__main__":
    main() 