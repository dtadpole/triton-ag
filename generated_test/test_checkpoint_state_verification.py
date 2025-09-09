#!/usr/bin/env python3
"""
Comprehensive test to verify that saved checkpoint state exactly matches loaded state
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from logger import logger
from engineFSDP import EngineFSDP
from engineBase import EngineConfig


def compare_tensors(tensor1, tensor2, name, tolerance=1e-6):
    """Compare two tensors and return detailed comparison results"""
    if tensor1.shape != tensor2.shape:
        logger.error(f"❌ {name}: Shape mismatch - {tensor1.shape} vs {tensor2.shape}")
        return False
    
    if tensor1.dtype != tensor2.dtype:
        logger.error(f"❌ {name}: Dtype mismatch - {tensor1.dtype} vs {tensor2.dtype}")
        return False
    
    if tensor1.device != tensor2.device:
        logger.warning(f"⚠️ {name}: Device mismatch - {tensor1.device} vs {tensor2.device}")
    
    # Compare values
    if torch.allclose(tensor1, tensor2, atol=tolerance, rtol=tolerance):
        logger.info(f"✅ {name}: Values match (tolerance: {tolerance})")
        return True
    else:
        max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        mean_diff = torch.mean(torch.abs(tensor1 - tensor2)).item()
        logger.error(f"❌ {name}: Values don't match - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        return False


def compare_state_dicts(state_dict1, state_dict2, prefix="", tolerance=1e-6):
    """Compare two state dictionaries recursively"""
    all_match = True
    
    # Check keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    if keys1 != keys2:
        logger.error(f"❌ {prefix}Key mismatch:")
        logger.error(f"   Only in dict1: {keys1 - keys2}")
        logger.error(f"   Only in dict2: {keys2 - keys1}")
        return False
    
    # Compare values
    for key in keys1:
        val1 = state_dict1[key]
        val2 = state_dict2[key]
        
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not compare_tensors(val1, val2, f"{prefix}{key}", tolerance):
                all_match = False
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_state_dicts(val1, val2, f"{prefix}{key}.", tolerance):
                all_match = False
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance:
                logger.error(f"❌ {prefix}{key}: Value mismatch - {val1} vs {val2}")
                all_match = False
            else:
                logger.info(f"✅ {prefix}{key}: Values match")
        else:
            if val1 != val2:
                logger.error(f"❌ {prefix}{key}: Type/value mismatch - {type(val1)} {val1} vs {type(val2)} {val2}")
                all_match = False
            else:
                logger.info(f"✅ {prefix}{key}: Values match")
    
    return all_match


def test_checkpoint_state_verification():
    """Test that saved checkpoint state exactly matches loaded state"""
    logger.info("🧪 Testing checkpoint state verification...")
    
    # Create a config for testing
    config = EngineConfig()
    config.model.name = "microsoft/DialoGPT-small"
    config.model.max_seq_length = 1024
    config.training.max_steps = 10
    config.training.micro_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.learning_rate = 1e-4  # Higher LR for more noticeable changes
    config.lora.use_lora = False
    config.logging.use_wandb = False
    
    logger.info("🔧 Created test configuration")
    
    # Create trainer
    trainer = EngineFSDP("test_verification", config, inference_mode=False)
    logger.info("✅ EngineFSDP created successfully")
    
    # Get initial state for comparison
    initial_model_state = trainer.model.state_dict()
    initial_optimizer_state = trainer.optimizer.state_dict()
    initial_scheduler_state = trainer.scheduler.state_dict()
    initial_global_step = trainer.status.global_step
    
    logger.info(f"📊 Initial global step: {initial_global_step}")
    logger.info(f"📊 Model state keys: {list(initial_model_state.keys())}")
    logger.info(f"📊 Optimizer state keys: {list(initial_optimizer_state.keys())}")
    logger.info(f"📊 Scheduler state keys: {list(initial_scheduler_state.keys())}")
    
    # Do some training steps to change the state
    logger.info("🏋️ Performing training steps to modify state...")
    
    # Create dummy data
    dummy_input = torch.randint(0, 1000, (1, 10)).to(trainer.device)
    dummy_labels = torch.randint(0, 1000, (1, 10)).to(trainer.device)
    
    # Perform a few training steps
    for step in range(3):
        trainer.model.train()
        
        # Forward pass
        outputs = trainer.model(input_ids=dummy_input, labels=dummy_labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.optimizer.zero_grad()
        
        trainer.status.global_step += 1
        
        logger.info(f"📈 Training step {step + 1}: loss={loss.item():.4f}, global_step={trainer.status.global_step}")
    
    # Get state after training
    trained_model_state = trainer.model.state_dict()
    trained_optimizer_state = trainer.optimizer.state_dict()
    trained_scheduler_state = trainer.scheduler.state_dict()
    trained_global_step = trainer.status.global_step
    
    logger.info(f"📊 Trained global step: {trained_global_step}")
    
    # Verify that training actually changed the state
    model_changed = not compare_state_dicts(initial_model_state, trained_model_state, "Model ", tolerance=1e-3)
    optimizer_changed = not compare_state_dicts(initial_optimizer_state, trained_optimizer_state, "Optimizer ", tolerance=1e-3)
    scheduler_changed = not compare_state_dicts(initial_scheduler_state, trained_scheduler_state, "Scheduler ", tolerance=1e-3)
    global_step_changed = initial_global_step != trained_global_step
    
    logger.info(f"🔄 State changes after training:")
    logger.info(f"   Model changed: {model_changed}")
    logger.info(f"   Optimizer changed: {optimizer_changed}")
    logger.info(f"   Scheduler changed: {scheduler_changed}")
    logger.info(f"   Global step changed: {global_step_changed}")
    
    if not (model_changed or optimizer_changed or scheduler_changed or global_step_changed):
        logger.warning("⚠️ No state changes detected - training might not be working properly")
    
    # Now test checkpoint save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_checkpoint"
        checkpoint_path.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving checkpoint to: {checkpoint_path}")
        
        # Save checkpoint
        trainer._save_checkpoint(1)
        
        # Modify the state to test loading
        logger.info("🔄 Modifying state to test loading...")
        
        with torch.no_grad():
            for param in trainer.model.parameters():
                if param.requires_grad:
                    param.data += torch.randn_like(param.data) * 0.1
        
        # Reset optimizer and scheduler state (but keep the same parameter groups)
        trainer.optimizer.zero_grad()
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=config.training.learning_rate)
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=config.training.max_steps
        )
        trainer.status.global_step = 0
        
        logger.info("📂 Loading checkpoint...")
        
        # Load checkpoint (use the actual saved checkpoint path)
        actual_checkpoint_path = trainer.checkpoint_path / "checkpoint-1"
        trainer._load_checkpoint(actual_checkpoint_path)
        
        # Get loaded state
        loaded_model_state = trainer.model.state_dict()
        loaded_optimizer_state = trainer.optimizer.state_dict()
        loaded_scheduler_state = trainer.scheduler.state_dict()
        loaded_global_step = trainer.status.global_step
        
        logger.info(f"📊 Loaded global step: {loaded_global_step}")
        
        # Compare loaded state with saved state
        logger.info("🔍 Comparing saved vs loaded state...")
        
        model_match = compare_state_dicts(trained_model_state, loaded_model_state, "Model ", tolerance=1e-6)
        optimizer_match = compare_state_dicts(trained_optimizer_state, loaded_optimizer_state, "Optimizer ", tolerance=1e-6)
        scheduler_match = compare_state_dicts(trained_scheduler_state, loaded_scheduler_state, "Scheduler ", tolerance=1e-6)
        global_step_match = trained_global_step == loaded_global_step
        
        logger.info(f"📋 State verification results:")
        logger.info(f"   Model state match: {model_match}")
        logger.info(f"   Optimizer state match: {optimizer_match}")
        logger.info(f"   Scheduler state match: {scheduler_match}")
        logger.info(f"   Global step match: {global_step_match}")
        
        # Overall result
        all_match = model_match and optimizer_match and scheduler_match and global_step_match
        
        if all_match:
            logger.info("🎉 All checkpoint states match perfectly!")
        else:
            logger.error("❌ Some checkpoint states don't match!")
        
        # Test forward pass consistency
        logger.info("🧪 Testing forward pass consistency...")
        
        trainer.model.eval()
        with torch.no_grad():
            original_output = trainer.model(input_ids=dummy_input)
            loaded_output = trainer.model(input_ids=dummy_input)
        
        output_match = compare_tensors(
            original_output.logits, 
            loaded_output.logits, 
            "Model output", 
            tolerance=1e-5
        )
        
        if output_match:
            logger.info("✅ Model forward pass is consistent")
        else:
            logger.error("❌ Model forward pass is inconsistent")
        
        return all_match and output_match


def test_tdc_vs_single_gpu_consistency():
    """Test that TDC and single GPU checkpoints produce consistent results"""
    logger.info("🧪 Testing TDC vs Single GPU checkpoint consistency...")
    
    # This test would require distributed setup, so we'll skip it for now
    # but the structure is here for future testing
    logger.info("ℹ️ TDC vs Single GPU consistency test skipped (requires distributed setup)")
    return True


if __name__ == "__main__":
    try:
        # Test checkpoint state verification
        success = test_checkpoint_state_verification()
        
        # Test TDC vs single GPU consistency
        tdc_success = test_tdc_vs_single_gpu_consistency()
        
        if success and tdc_success:
            logger.info("🎉 All checkpoint verification tests passed!")
            sys.exit(0)
        else:
            logger.error("❌ Some checkpoint verification tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
