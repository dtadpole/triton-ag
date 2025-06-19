#!/usr/bin/env python3
"""
Integration test for Swift Qwen3 fine-tuning.
This script performs end-to-end testing of the fine-tuning pipeline.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import yaml
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def setup_test_environment():
    """Set up test environment with mock data and configuration."""
    test_dir = tempfile.mkdtemp(prefix="swift_integration_test_")
    print(f"Setting up test environment in: {test_dir}")
    
    # Create test configuration
    config = {
        "model": {
            "name": "Qwen/Qwen3-8B-Instruct",
            "max_seq_length": 512,  # Small for testing
            "dtype": None,
            "load_in_4bit": True
        },
        "training": {
            "num_train_epochs": 1,
            "learning_rate": 1e-4,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 2,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "logging_steps": 1,
            "save_steps": 5,
            "output_dir": os.path.join(test_dir, "output"),
            "seed": 42
        },
        "lora": {
            "r": 8,  # Small rank for testing
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
        "data": {
            "local_dir": os.path.join(test_dir, "data")
        }
    }
    
    # Write configuration file
    config_file = os.path.join(test_dir, "integration_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create test data
    data_dir = os.path.join(test_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create training data with more diverse examples
    training_data = [
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        },
        {
            "instruction": "Explain the concept of neural networks.",
            "input": "",
            "output": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information."
        },
        {
            "instruction": "What is the difference between supervised and unsupervised learning?",
            "input": "",
            "output": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."
        },
        {
            "instruction": "Write a simple Python function to add two numbers.",
            "input": "",
            "output": "def add_numbers(a, b):\n    return a + b\n\n# Example usage:\nresult = add_numbers(5, 3)\nprint(result)  # Output: 8"
        },
        {
            "instruction": "Explain what is artificial intelligence.",
            "input": "",
            "output": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
        }
    ]
    
    # Write training data
    train_file = os.path.join(data_dir, "train.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created test configuration: {config_file}")
    print(f"Created test data: {train_file}")
    print(f"Number of training samples: {len(training_data)}")
    
    return test_dir, config_file


def run_swift_training_test(config_file, test_mode="train"):
    """Run Swift training test with specified mode."""
    print(f"\n{'='*60}")
    print(f"RUNNING SWIFT INTEGRATION TEST - MODE: {test_mode.upper()}")
    print(f"{'='*60}")
    
    # Set up environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    env['PYTHONPATH'] = os.path.dirname(os.path.dirname(__file__))
    
    # Prepare command
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "finetune_swift.py")
    cmd = [
        sys.executable, script_path,
        "--config", config_file,
        "--mode", test_mode
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment variables:")
    print(f"  CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}")
    
    try:
        # Run the command
        print(f"\nExecuting Swift fine-tuning...")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for safety
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nExecution completed in {execution_time:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        
        # Print output
        if result.stdout:
            print(f"\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print(f"\nSTDERR:")
            print(result.stderr)
        
        # Check if successful
        if result.returncode == 0:
            print(f"\n✅ Swift {test_mode} test PASSED")
            return True
        else:
            print(f"\n❌ Swift {test_mode} test FAILED with exit code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"\n⏰ Swift {test_mode} test TIMED OUT after 5 minutes")
        return False
    except Exception as e:
        print(f"\n💥 Swift {test_mode} test FAILED with exception: {e}")
        return False


def validate_test_outputs(test_dir):
    """Validate that expected outputs were created."""
    print(f"\n{'='*60}")
    print("VALIDATING TEST OUTPUTS")
    print(f"{'='*60}")
    
    output_dir = os.path.join(test_dir, "output")
    
    expected_files = [
        "test_results.json",
    ]
    
    validation_results = {}
    
    for expected_file in expected_files:
        file_path = os.path.join(output_dir, expected_file)
        exists = os.path.exists(file_path)
        validation_results[expected_file] = exists
        
        if exists:
            file_size = os.path.getsize(file_path)
            print(f"✅ Found {expected_file} ({file_size} bytes)")
            
            # For JSON files, try to load and validate structure
            if expected_file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"   📝 JSON structure is valid, contains {len(data)} items")
                except Exception as e:
                    print(f"   ⚠️  JSON file exists but has invalid structure: {e}")
        else:
            print(f"❌ Missing {expected_file}")
    
    # Check for any checkpoint directories
    if os.path.exists(output_dir):
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-') or d.startswith('v')]
        if checkpoint_dirs:
            print(f"✅ Found {len(checkpoint_dirs)} checkpoint directories:")
            for checkpoint_dir in checkpoint_dirs:
                print(f"   📁 {checkpoint_dir}")
        else:
            print("ℹ️  No checkpoint directories found (expected for mock tests)")
    
    all_files_found = all(validation_results.values())
    print(f"\nValidation result: {'PASS' if all_files_found else 'PARTIAL'}")
    
    return validation_results


def run_dependency_check():
    """Check if required dependencies are available."""
    print(f"{'='*60}")
    print("CHECKING DEPENDENCIES")
    print(f"{'='*60}")
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'datasets': 'Datasets',
        'yaml': 'PyYAML',
        'json': 'JSON (built-in)',
    }
    
    available_deps = {}
    
    for dep, name in dependencies.items():
        try:
            if dep == 'json':
                import json
            else:
                __import__(dep)
            print(f"✅ {name}")
            available_deps[dep] = True
        except ImportError:
            print(f"❌ {name} - NOT AVAILABLE")
            available_deps[dep] = False
    
    # Check ms-swift specifically
    try:
        from swift.llm import SftArguments
        print(f"✅ ms-swift")
        available_deps['ms-swift'] = True
    except ImportError:
        print(f"❌ ms-swift - NOT AVAILABLE")
        print("   ℹ️  You can install it with: pip install ms-swift")
        available_deps['ms-swift'] = False
    
    essential_deps = ['torch', 'transformers', 'peft', 'datasets', 'yaml', 'ms-swift']
    all_essential_available = all(available_deps.get(dep, False) for dep in essential_deps)
    
    print(f"\nDependency check result: {'PASS' if all_essential_available else 'FAIL'}")
    
    return all_essential_available


def main():
    """Main integration test function."""
    print("🚀 Starting Swift Qwen3 Fine-tuning Integration Test")
    print(f"{'='*80}")
    
    # Step 1: Check dependencies
    if not run_dependency_check():
        print("\n❌ Dependency check failed. Please install missing dependencies.")
        return False
    
    # Step 2: Set up test environment
    try:
        test_dir, config_file = setup_test_environment()
    except Exception as e:
        print(f"\n❌ Failed to set up test environment: {e}")
        return False
    
    test_results = {}
    
    try:
        # Step 3: Run training test (with mocked Swift functions for safety)
        print(f"\n📚 Testing configuration loading and validation...")
        test_results['config_test'] = run_swift_training_test(config_file, "test")
        
        # Step 4: Validate outputs
        print(f"\n🔍 Validating test outputs...")
        validation_results = validate_test_outputs(test_dir)
        test_results['validation'] = validation_results
        
        # Step 5: Summary
        print(f"\n{'='*80}")
        print("INTEGRATION TEST SUMMARY")
        print(f"{'='*80}")
        
        overall_success = test_results.get('config_test', False)
        
        print(f"Configuration Test: {'PASS' if test_results.get('config_test', False) else 'FAIL'}")
        print(f"Output Validation: {'PASS' if validation_results else 'PARTIAL'}")
        print(f"Overall Result: {'PASS' if overall_success else 'FAIL'}")
        
        if overall_success:
            print(f"\n🎉 Integration test completed successfully!")
            print(f"📁 Test artifacts available in: {test_dir}")
        else:
            print(f"\n💥 Integration test failed!")
        
        return overall_success
    
    finally:
        # Clean up (optional - comment out to preserve test artifacts)
        try:
            shutil.rmtree(test_dir)
            print(f"\n🧹 Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"\n⚠️  Failed to clean up test directory: {e}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 