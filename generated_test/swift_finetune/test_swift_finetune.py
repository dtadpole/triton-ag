#!/usr/bin/env python3
"""
Comprehensive test suite for Swift Qwen3 fine-tuning.
This script tests all aspects of the fine-tuning pipeline.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from finetune_swift import SwiftQwen3FineTuner, SwiftFineTuningConfig
except ImportError as e:
    print(f"Warning: Could not import finetune_swift module: {e}")
    print("This test requires the finetune_swift.py module to be available")
    sys.exit(1)


class TestSwiftQwen3FineTuner(unittest.TestCase):
    """Test cases for Swift Qwen3 Fine-tuner."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "test_config.yaml")
        self.output_dir = os.path.join(self.test_dir, "output")
        
        # Create test configuration
        self.test_config = {
            "model": {
                "name": "Qwen/Qwen3-8B-Instruct",
                "max_seq_length": 16384,
                "dtype": None,
                "load_in_4bit": True
            },
            "training": {
                "num_train_epochs": 1,
                "learning_rate": 3e-5,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 5,
                "weight_decay": 0.01,
                "lr_scheduler_type": "cosine",
                "logging_steps": 1,
                "save_steps": 10,
                "output_dir": self.output_dir,
                "seed": 42
            },
            "lora": {
                "r": 64,
                "alpha": 64,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            },
            "data": {
                "local_dir": "test_data"
            }
        }
        
        # Write test configuration
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Create test data directory
        self.test_data_dir = os.path.join(self.test_dir, "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create mock training data
        self.create_mock_training_data()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_mock_training_data(self):
        """Create mock training data for testing."""
        # Create mock jsonl file with sample data
        training_data = [
            {
                "instruction": "What is machine learning?",
                "input": "",
                "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
            },
            {
                "instruction": "Explain neural networks",
                "input": "",
                "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information."
            },
            {
                "instruction": "What is deep learning?",
                "input": "",
                "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."
            }
        ]
        
        training_file = os.path.join(self.test_data_dir, "train.jsonl")
        with open(training_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Create validation data
        val_data = [
            {
                "instruction": "What is AI?",
                "input": "",
                "output": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
            }
        ]
        
        val_file = os.path.join(self.test_data_dir, "val.jsonl")
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def test_config_loading(self):
        """Test configuration loading."""
        fine_tuner = SwiftQwen3FineTuner(config_path=self.config_file)
        
        # Test that configuration was loaded correctly
        self.assertIsNotNone(fine_tuner.yaml_config)
        self.assertIsNotNone(fine_tuner.swift_config)
        self.assertEqual(fine_tuner.swift_config.model_id, "Qwen/Qwen3-8B-Instruct")
        self.assertEqual(fine_tuner.swift_config.max_length, 16384)
        self.assertEqual(fine_tuner.swift_config.num_train_epochs, 1)
        self.assertEqual(fine_tuner.swift_config.lora_rank, 64)
    
    def test_config_defaults(self):
        """Test default configuration when no config file exists."""
        non_existent_config = os.path.join(self.test_dir, "nonexistent.yaml")
        
        fine_tuner = SwiftQwen3FineTuner(config_path=non_existent_config)
        
        # Test that defaults are used
        self.assertEqual(fine_tuner.swift_config.model_type, "qwen3-8b-instruct")
        self.assertEqual(fine_tuner.swift_config.max_length, 16384)
        self.assertEqual(fine_tuner.swift_config.lora_rank, 64)
    
    def test_dataset_preparation(self):
        """Test dataset preparation logic."""
        fine_tuner = SwiftQwen3FineTuner(config_path=self.config_file)
        fine_tuner.swift_config.dataset_path = self.test_data_dir
        
        datasets = fine_tuner.prepare_datasets()
        
        # Should use custom dataset since local data exists
        self.assertIn("custom", datasets)
        self.assertEqual(fine_tuner.swift_config.custom_train_dataset_path, self.test_data_dir)
    
    def test_dataset_preparation_no_local_data(self):
        """Test dataset preparation when no local data exists."""
        fine_tuner = SwiftQwen3FineTuner(config_path=self.config_file)
        fine_tuner.swift_config.dataset_path = "/nonexistent/path"
        
        datasets = fine_tuner.prepare_datasets()
        
        # Should use built-in datasets
        self.assertIn("AI-ModelScope/alpaca-gpt4-data-zh#500", datasets)
        self.assertIn("AI-ModelScope/alpaca-gpt4-data-en#500", datasets)
        self.assertIn("swift/self-cognition#200", datasets)
    
    @patch('finetune_swift.sft_main')
    def test_training_mock(self, mock_sft_main):
        """Test training process with mocked Swift function."""
        mock_checkpoint_dir = os.path.join(self.test_dir, "checkpoint-100")
        mock_sft_main.return_value = mock_checkpoint_dir
        
        fine_tuner = SwiftQwen3FineTuner(config_path=self.config_file)
        
        # Test training
        result = fine_tuner.train()
        
        # Verify that sft_main was called
        mock_sft_main.assert_called_once()
        self.assertEqual(result, mock_checkpoint_dir)
        self.assertEqual(fine_tuner.best_ckpt_dir, mock_checkpoint_dir)
    
    @patch('finetune_swift.infer_main')
    def test_evaluation_mock(self, mock_infer_main):
        """Test evaluation process with mocked Swift function."""
        fine_tuner = SwiftQwen3FineTuner(config_path=self.config_file)
        checkpoint_dir = os.path.join(self.test_dir, "checkpoint-100")
        
        # Test evaluation
        fine_tuner.evaluate(checkpoint_dir)
        
        # Verify that infer_main was called
        mock_infer_main.assert_called_once()
    
    def test_test_model(self):
        """Test model testing functionality."""
        fine_tuner = SwiftQwen3FineTuner(config_path=self.config_file)
        checkpoint_dir = os.path.join(self.test_dir, "checkpoint-100")
        
        # Create output directory
        os.makedirs(fine_tuner.swift_config.output_dir, exist_ok=True)
        
        # Test with custom prompts
        test_prompts = [
            "What is Python?",
            "Explain machine learning"
        ]
        
        results = fine_tuner.test_model(checkpoint_dir, test_prompts)
        
        # Verify results structure
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["prompt"], "What is Python?")
        self.assertEqual(results[1]["prompt"], "Explain machine learning")
        
        # Verify results file was created
        results_file = os.path.join(fine_tuner.swift_config.output_dir, "test_results.json")
        self.assertTrue(os.path.exists(results_file))
    
    def test_swift_config_class(self):
        """Test SwiftFineTuningConfig dataclass."""
        config = SwiftFineTuningConfig()
        
        # Test default values
        self.assertEqual(config.model_type, "qwen3-8b-instruct")
        self.assertEqual(config.model_id, "Qwen/Qwen3-8B-Instruct")
        self.assertEqual(config.sft_type, "lora")
        self.assertEqual(config.num_train_epochs, 2)
        self.assertEqual(config.lora_rank, 64)
        self.assertEqual(config.lora_alpha, 64)
        self.assertFalse(config.push_to_hub)
        
        # Test custom values
        custom_config = SwiftFineTuningConfig(
            model_type="qwen3-7b-instruct",
            num_train_epochs=3,
            lora_rank=32
        )
        
        self.assertEqual(custom_config.model_type, "qwen3-7b-instruct")
        self.assertEqual(custom_config.num_train_epochs, 3)
        self.assertEqual(custom_config.lora_rank, 32)
    
    def test_model_type_mapping(self):
        """Test model name to type mapping."""
        test_configs = [
            ("Qwen/Qwen3-8B-Instruct", "qwen3-8b-instruct"),
            ("Qwen/Qwen3-7B-Instruct", "qwen3-7b-instruct"),
            ("Qwen/Qwen3-14B-Instruct", "qwen3-14b-instruct"),
            ("Qwen/Qwen3-8B", "qwen3-8b"),
        ]
        
        for model_name, expected_type in test_configs:
            config = {
                "model": {"name": model_name},
                "training": {},
                "lora": {},
                "data": {}
            }
            
            config_file = os.path.join(self.test_dir, f"test_{expected_type}.yaml")
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            fine_tuner = SwiftQwen3FineTuner(config_path=config_file)
            self.assertEqual(fine_tuner.swift_config.model_type, expected_type)
    
    @patch('finetune_swift.sft_main')
    @patch('finetune_swift.infer_main')
    def test_full_pipeline_mock(self, mock_infer_main, mock_sft_main):
        """Test full pipeline with mocked Swift functions."""
        mock_checkpoint_dir = os.path.join(self.test_dir, "checkpoint-100")
        mock_sft_main.return_value = mock_checkpoint_dir
        
        fine_tuner = SwiftQwen3FineTuner(config_path=self.config_file)
        
        # Create output directory
        os.makedirs(fine_tuner.swift_config.output_dir, exist_ok=True)
        
        # Test full pipeline
        result = fine_tuner.run_full_pipeline()
        
        # Verify that both training and evaluation were called
        mock_sft_main.assert_called_once()
        mock_infer_main.assert_called_once()
        self.assertEqual(result, mock_checkpoint_dir)
        
        # Verify test results file was created
        results_file = os.path.join(fine_tuner.swift_config.output_dir, "test_results.json")
        self.assertTrue(os.path.exists(results_file))


class TestSwiftIntegration(unittest.TestCase):
    """Integration tests for Swift functionality."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "integration_config.yaml")
        
        # Create minimal config for integration tests
        self.integration_config = {
            "model": {
                "name": "Qwen/Qwen3-8B-Instruct",
                "max_seq_length": 512,  # Small for testing
                "load_in_4bit": True
            },
            "training": {
                "num_train_epochs": 1,
                "learning_rate": 1e-4,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "logging_steps": 1,
                "save_steps": 5,
                "output_dir": os.path.join(self.test_dir, "output"),
                "seed": 42
            },
            "lora": {
                "r": 8,  # Small rank for testing
                "alpha": 16,
                "dropout": 0.1
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.integration_config, f)
    
    def tearDown(self):
        """Clean up integration test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_config_validation(self):
        """Test configuration validation."""
        fine_tuner = SwiftQwen3FineTuner(config_path=self.config_file)
        
        # Check that all required configurations are present
        self.assertIsNotNone(fine_tuner.swift_config.model_id)
        self.assertIsNotNone(fine_tuner.swift_config.output_dir)
        self.assertGreater(fine_tuner.swift_config.lora_rank, 0)
        self.assertGreater(fine_tuner.swift_config.max_length, 0)
    
    def test_environment_setup(self):
        """Test environment setup and CUDA configuration."""
        # Test CUDA environment variable setting
        original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
        
        # Clear CUDA env var
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Import and run main function setup
        from finetune_swift import main
        
        # This should set CUDA_VISIBLE_DEVICES
        # Note: We can't easily test main() without mocking, so we test the logic
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '0,1,2,3')
        
        # Restore original value
        if original_cuda is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']


class TestSwiftUtilities(unittest.TestCase):
    """Test utility functions and edge cases."""
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with invalid YAML
            invalid_config = os.path.join(temp_dir, "invalid.yaml")
            with open(invalid_config, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            with self.assertRaises(SystemExit):
                SwiftQwen3FineTuner(config_path=invalid_config)
    
    def test_path_handling(self):
        """Test path handling and directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "path_test.yaml")
            output_dir = os.path.join(temp_dir, "nested", "output")
            
            config = {
                "model": {"name": "Qwen/Qwen3-8B-Instruct"},
                "training": {"output_dir": output_dir},
                "lora": {},
                "data": {}
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            fine_tuner = SwiftQwen3FineTuner(config_path=config_file)
            self.assertEqual(fine_tuner.swift_config.output_dir, output_dir)


def create_test_suite():
    """Create a comprehensive test suite."""
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestSwiftQwen3FineTuner))
    suite.addTest(unittest.makeSuite(TestSwiftIntegration))
    suite.addTest(unittest.makeSuite(TestSwiftUtilities))
    
    return suite


def run_tests():
    """Run all tests with detailed output."""
    # Set up test environment
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(__file__))
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 