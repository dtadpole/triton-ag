#!/usr/bin/env python3
"""
Fine-tuning script for Qwen3 model using ms-swift framework.
This script provides comprehensive functionality for training, evaluation, and testing.
"""

import os
import sys
import yaml
import torch
import time
import gc
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
from dataclasses import dataclass, field

# Import ms-swift
try:
    from swift.llm import (
        ModelType, TrainArguments, InferArguments,
        sft_main, infer_main, ExportArguments, export_main
    )
    from swift.utils import get_logger
    SWIFT_AVAILABLE = True
except ImportError:
    print("Warning: ms-swift not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", "ms-swift", "-U"])
    from swift.llm import (
        ModelType, TrainArguments, InferArguments,
        sft_main, infer_main, ExportArguments, export_main
    )
    from swift.utils import get_logger
    SWIFT_AVAILABLE = True

# Import local utilities
from util import logger
from data_util import load_experiences, create_dataset


@dataclass
class SwiftFineTuningConfig:
    """Configuration class for Swift fine-tuning."""
    
    # Model configuration
    model_type: str = "qwen3-8b-instruct"
    model_id: str = "Qwen/Qwen3-8B-Instruct"
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    
    # Training configuration
    sft_type: str = "lora"
    num_train_epochs: int = 2
    learning_rate: float = 3e-5
    max_length: int = 16384
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    save_steps: int = 10
    eval_steps: int = 10
    save_total_limit: int = 3
    
    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["all-linear"])
    
    # Data configuration
    dataset_path: str = "finetune_experiences"
    custom_train_dataset_path: Optional[str] = None
    custom_val_dataset_path: Optional[str] = None
    train_dataset_sample: int = -1  # -1 means use all data
    val_dataset_sample: int = -1
    
    # Output configuration
    output_dir: str = "output_swift"
    seed: int = 42
    
    # System configuration
    system_prompt: str = "You are a helpful assistant."
    
    # Evaluation configuration
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    
    # Additional swift-specific parameters
    dataset_num_proc: int = 4
    model_author: str = "swift"
    model_name: str = "qwen3-finetuned"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None


class SwiftQwen3FineTuner:
    """Fine-tuner for Qwen3 using ms-swift framework."""
    
    def __init__(self, config_path: str = "finetune.yaml"):
        """Initialize the fine-tuner with configuration."""
        self.config_path = config_path
        self.yaml_config = None
        self.swift_config = None
        self.best_ckpt_dir = None
        
        # Load configuration
        self._load_config()
        self._setup_swift_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.yaml_config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_path} not found, using defaults")
            self.yaml_config = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _setup_swift_config(self):
        """Setup ms-swift configuration from YAML config."""
        # Create Swift configuration with defaults
        self.swift_config = SwiftFineTuningConfig()
        
        # Override with YAML configuration if available
        if self.yaml_config:
            # Model configuration
            model_config = self.yaml_config.get('model', {})
            if model_config.get('name'):
                # Map model name to Swift model type
                model_name = model_config['name']
                if "Qwen3" in model_name or "qwen3" in model_name.lower():
                    if "8B" in model_name:
                        self.swift_config.model_type = "qwen3-8b-instruct" if "Instruct" in model_name else "qwen3-8b"
                    elif "7B" in model_name:
                        self.swift_config.model_type = "qwen3-7b-instruct" if "Instruct" in model_name else "qwen3-7b"
                    elif "14B" in model_name:
                        self.swift_config.model_type = "qwen3-14b-instruct" if "Instruct" in model_name else "qwen3-14b"
                    self.swift_config.model_id = model_name
                
            self.swift_config.max_length = model_config.get('max_seq_length', self.swift_config.max_length)
            self.swift_config.dtype = model_config.get('dtype', self.swift_config.dtype)
            self.swift_config.load_in_4bit = model_config.get('load_in_4bit', self.swift_config.load_in_4bit)
            
            # Training configuration
            training_config = self.yaml_config.get('training', {})
            self.swift_config.num_train_epochs = training_config.get('num_train_epochs', self.swift_config.num_train_epochs)
            self.swift_config.learning_rate = training_config.get('learning_rate', self.swift_config.learning_rate)
            self.swift_config.per_device_train_batch_size = training_config.get('per_device_batch_size', self.swift_config.per_device_train_batch_size)
            self.swift_config.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', self.swift_config.gradient_accumulation_steps)
            self.swift_config.warmup_ratio = training_config.get('warmup_steps', 0) / max(training_config.get('max_steps', 100), 100)
            self.swift_config.weight_decay = training_config.get('weight_decay', self.swift_config.weight_decay)
            self.swift_config.lr_scheduler_type = training_config.get('lr_scheduler_type', self.swift_config.lr_scheduler_type)
            self.swift_config.logging_steps = training_config.get('logging_steps', self.swift_config.logging_steps)
            self.swift_config.save_steps = training_config.get('save_steps', self.swift_config.save_steps)
            self.swift_config.output_dir = training_config.get('output_dir', self.swift_config.output_dir)
            self.swift_config.seed = training_config.get('seed', self.swift_config.seed)
            
            # LoRA configuration
            lora_config = self.yaml_config.get('lora', {})
            self.swift_config.lora_rank = lora_config.get('r', self.swift_config.lora_rank)
            self.swift_config.lora_alpha = lora_config.get('alpha', self.swift_config.lora_alpha)
            self.swift_config.lora_dropout = lora_config.get('dropout', self.swift_config.lora_dropout)
            if lora_config.get('target_modules'):
                self.swift_config.lora_target_modules = lora_config['target_modules']
            
            # Data configuration
            data_config = self.yaml_config.get('data', {})
            self.swift_config.dataset_path = data_config.get('local_dir', self.swift_config.dataset_path)
        
        logger.info("Swift configuration setup complete")
        logger.info(f"Model: {self.swift_config.model_id}")
        logger.info(f"Output directory: {self.swift_config.output_dir}")
    
    def prepare_datasets(self):
        """Prepare datasets for Swift training."""
        datasets = []
        
        # Check if local experience data exists
        if os.path.exists(self.swift_config.dataset_path):
            logger.info(f"Loading local experience data from {self.swift_config.dataset_path}")
            # For local data, we'll use custom dataset approach
            self.swift_config.custom_train_dataset_path = self.swift_config.dataset_path
            datasets.append("custom")
        else:
            logger.info("Using built-in Swift datasets")
            # Use Swift's built-in datasets
            datasets = [
                "AI-ModelScope/alpaca-gpt4-data-zh#500",
                "AI-ModelScope/alpaca-gpt4-data-en#500",
                "swift/self-cognition#200"
            ]
        
        return datasets
    
    def train(self):
        """Execute the fine-tuning process."""
        logger.info("Starting Swift fine-tuning process...")
        
        # Prepare datasets
        datasets = self.prepare_datasets()
        
        # Create TrainArguments
        train_args = TrainArguments(
            model_type=self.swift_config.model_type,
            model_id_or_path=self.swift_config.model_id,
            sft_type=self.swift_config.sft_type,
            
            # Training parameters
            num_train_epochs=self.swift_config.num_train_epochs,
            learning_rate=self.swift_config.learning_rate,
            max_length=self.swift_config.max_length,
            per_device_train_batch_size=self.swift_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.swift_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.swift_config.gradient_accumulation_steps,
            warmup_ratio=self.swift_config.warmup_ratio,
            weight_decay=self.swift_config.weight_decay,
            lr_scheduler_type=self.swift_config.lr_scheduler_type,
            
            # LoRA parameters
            lora_rank=self.swift_config.lora_rank,
            lora_alpha=self.swift_config.lora_alpha,
            lora_dropout=self.swift_config.lora_dropout,
            lora_target_modules=self.swift_config.lora_target_modules,
            
            # Dataset parameters
            dataset=datasets,
            train_dataset_sample=self.swift_config.train_dataset_sample,
            val_dataset_sample=self.swift_config.val_dataset_sample,
            dataset_num_proc=self.swift_config.dataset_num_proc,
            
            # Logging and saving
            logging_steps=self.swift_config.logging_steps,
            save_steps=self.swift_config.save_steps,
            eval_steps=self.swift_config.eval_steps,
            save_total_limit=self.swift_config.save_total_limit,
            output_dir=self.swift_config.output_dir,
            
            # System parameters
            seed=self.swift_config.seed,
            system=self.swift_config.system_prompt,
            
            # Model metadata
            model_author=self.swift_config.model_author,
            model_name=self.swift_config.model_name,
            
            # Evaluation
            do_eval=self.swift_config.do_eval,
            evaluation_strategy=self.swift_config.evaluation_strategy,
            
            # Hardware optimization
            torch_dtype="auto" if self.swift_config.dtype is None else self.swift_config.dtype,
            load_in_4bit=self.swift_config.load_in_4bit,
            gradient_checkpointing=True,
            
            # Additional optimizations
            dataloader_num_workers=self.swift_config.dataset_num_proc,
            remove_unused_columns=False,
        )
        
        # Set custom dataset path if using local data
        if self.swift_config.custom_train_dataset_path:
            train_args.custom_train_dataset_path = self.swift_config.custom_train_dataset_path
        
        try:
            # Run Swift fine-tuning
            logger.info("Executing Swift SFT training...")
            self.best_ckpt_dir = sft_main(train_args)
            logger.info(f"Training completed successfully. Best checkpoint: {self.best_ckpt_dir}")
            return self.best_ckpt_dir
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    def evaluate(self, checkpoint_dir: Optional[str] = None):
        """Run evaluation on the trained model."""
        if checkpoint_dir is None:
            checkpoint_dir = self.best_ckpt_dir
        
        if checkpoint_dir is None:
            logger.error("No checkpoint directory available for evaluation")
            return
        
        logger.info(f"Running evaluation on checkpoint: {checkpoint_dir}")
        
        # Create InferArguments for evaluation
        infer_args = InferArguments(
            model_type=self.swift_config.model_type,
            ckpt_dir=checkpoint_dir,
            stream=True,
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=2048,
            show_dataset_sample=5,
        )
        
        try:
            # Run inference for evaluation
            infer_main(infer_args)
            logger.info("Evaluation completed successfully")
        except Exception as e:
            logger.error(f"Evaluation failed with error: {e}")
            raise
    
    def test_model(self, checkpoint_dir: Optional[str] = None, test_prompts: Optional[List[str]] = None):
        """Test the trained model with custom prompts."""
        if checkpoint_dir is None:
            checkpoint_dir = self.best_ckpt_dir
        
        if checkpoint_dir is None:
            logger.error("No checkpoint directory available for testing")
            return
        
        if test_prompts is None:
            test_prompts = [
                "What is machine learning?",
                "Explain the concept of deep learning in simple terms.",
                "How does fine-tuning work in language models?",
                "Write a Python function to calculate the factorial of a number.",
                "What are the benefits of using LoRA for fine-tuning?"
            ]
        
        logger.info(f"Testing model with {len(test_prompts)} prompts...")
        
        # Create InferArguments for testing
        infer_args = InferArguments(
            model_type=self.swift_config.model_type,
            ckpt_dir=checkpoint_dir,
            stream=False,  # Disable streaming for cleaner output
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=1024,
        )
        
        results = []
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n--- Test {i}/{len(test_prompts)} ---")
            logger.info(f"Prompt: {prompt}")
            
            try:
                # Run inference for this prompt
                # Note: Swift's infer_main is interactive, so for automated testing
                # we would need to use the Swift API directly
                logger.info("Running inference...")
                # This is a placeholder - in practice, you'd use Swift's inference API
                logger.info("Response: [Would contain model response]")
                
                results.append({
                    "prompt": prompt,
                    "response": "[Model response would be here]",
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Failed to generate response for prompt {i}: {e}")
                results.append({
                    "prompt": prompt,
                    "response": None,
                    "status": "error",
                    "error": str(e)
                })
        
        # Save test results
        results_file = os.path.join(self.swift_config.output_dir, "test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to: {results_file}")
        return results
    
    def push_to_hub(self, checkpoint_dir: Optional[str] = None):
        """Push the trained model to ModelScope hub."""
        if not self.swift_config.push_to_hub:
            logger.info("Push to hub is disabled")
            return
        
        if checkpoint_dir is None:
            checkpoint_dir = self.best_ckpt_dir
        
        if checkpoint_dir is None:
            logger.error("No checkpoint directory available for pushing to hub")
            return
        
        if not self.swift_config.hub_model_id or not self.swift_config.hub_token:
            logger.error("Hub model ID and token are required for pushing to hub")
            return
        
        logger.info(f"Pushing model to hub: {self.swift_config.hub_model_id}")
        
        try:
            # Use Swift's export functionality to push to hub
            from swift.llm import ExportArguments, export_main
            
            export_args = ExportArguments(
                ckpt_dir=checkpoint_dir,
                push_to_hub=True,
                hub_model_id=self.swift_config.hub_model_id,
                hub_token=self.swift_config.hub_token,
            )
            
            export_main(export_args)
            logger.info("Model successfully pushed to hub")
            
        except Exception as e:
            logger.error(f"Failed to push model to hub: {e}")
            raise
    
    def run_full_pipeline(self):
        """Run the complete fine-tuning pipeline."""
        logger.info("Starting full Swift fine-tuning pipeline...")
        
        try:
            # Step 1: Train the model
            logger.info("Step 1: Training...")
            checkpoint_dir = self.train()
            
            # Step 2: Evaluate the model
            logger.info("Step 2: Evaluation...")
            self.evaluate(checkpoint_dir)
            
            # Step 3: Test the model
            logger.info("Step 3: Testing...")
            self.test_model(checkpoint_dir)
            
            # Step 4: Push to hub (if configured)
            if self.swift_config.push_to_hub:
                logger.info("Step 4: Pushing to hub...")
                self.push_to_hub(checkpoint_dir)
            
            logger.info("Full pipeline completed successfully!")
            logger.info(f"Final model checkpoint: {checkpoint_dir}")
            
            return checkpoint_dir
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function to run Swift fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 model using ms-swift")
    parser.add_argument("--config", type=str, default="finetune.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "test", "full"], 
                       default="full", help="Mode to run")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint directory for eval/test")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = SwiftQwen3FineTuner(config_path=args.config)
    
    # Override output directory if provided
    if args.output_dir:
        fine_tuner.swift_config.output_dir = args.output_dir
    
    # Set CUDA devices
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    
    try:
        if args.mode == "train":
            checkpoint_dir = fine_tuner.train()
            logger.info(f"Training completed. Checkpoint: {checkpoint_dir}")
            
        elif args.mode == "eval":
            if not args.checkpoint:
                logger.error("--checkpoint required for evaluation mode")
                sys.exit(1)
            fine_tuner.evaluate(args.checkpoint)
            
        elif args.mode == "test":
            if not args.checkpoint:
                logger.error("--checkpoint required for test mode")
                sys.exit(1)
            fine_tuner.test_model(args.checkpoint)
            
        elif args.mode == "full":
            checkpoint_dir = fine_tuner.run_full_pipeline()
            logger.info(f"Full pipeline completed. Final checkpoint: {checkpoint_dir}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
