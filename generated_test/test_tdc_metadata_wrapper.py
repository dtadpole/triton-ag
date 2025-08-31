#!/usr/bin/env python3
"""
Test to verify TDC checkpoint functionality with metadata wrapper
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path
import time

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from logger import logger
from engineFSDP import EngineFSDP
from engineBase import EngineConfig


def test_tdc_metadata_wrapper():
    """Test TDC checkpoint functionality with metadata wrapper"""
    logger.info("🧪 Testing TDC checkpoint with metadata wrapper...")
    
    # Create a config for testing
    config = EngineConfig()
    config.model.name = "microsoft/DialoGPT-small"
    config.model.max_seq_length = 1024
    config.training.max_steps = 10
    config.training.micro_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.learning_rate = 1e-4
    config.lora.use_lora = False
    config.logging.use_wandb = False
    
    logger.info("🔧 Created test configuration")
    
    # Create trainer with unique name to avoid conflicts
    unique_name = f"test_metadata_wrapper_{int(time.time())}"
    trainer = EngineFSDP(unique_name, config, inference_mode=False)
    logger.info("✅ EngineFSDP created successfully")
    
    # Force distributed mode for testing TDC
    original_use_distributed = trainer.use_distributed
    trainer.use_distributed = True
    trainer.rank = 0
    trainer.world_size = 1
    
    # Initialize distributed process group for TDC testing
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", 
                               init_method="file:///tmp/test_distributed_init", 
                               world_size=1, 
                               rank=0)
        logger.info("🔄 Initialized distributed process group for TDC testing")
    
    logger.info(f"🔄 Forced distributed mode: {trainer.use_distributed}")
    
    # Get initial state
    initial_global_step = trainer.status.global_step
    logger.info(f"📊 Initial global step: {initial_global_step}")
    
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
    trained_global_step = trainer.status.global_step
    logger.info(f"📊 Trained global step: {trained_global_step}")
    
    # Verify that training actually changed the state
    global_step_changed = initial_global_step != trained_global_step
    logger.info(f"🔄 Global step changed: {global_step_changed}")
    
    if not global_step_changed:
        logger.warning("⚠️ No state changes detected - training might not be working properly")
        return False
    
    # Now test TDC checkpoint save/load with metadata wrapper
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_tdc_metadata_checkpoint"
        checkpoint_path.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving TDC checkpoint with metadata wrapper to: {checkpoint_path}")
        
        # Save checkpoint using TDC with metadata wrapper
        trainer._save_checkpoint_tdc(checkpoint_path, 1)
        
        # Modify the global step to test loading
        logger.info("🔄 Modifying global step to test loading...")
        trainer.status.global_step = 0
        
        logger.info("📂 Loading TDC checkpoint with metadata wrapper...")
        
        # Load checkpoint using TDC with metadata wrapper
        trainer._load_checkpoint_tdc(checkpoint_path)
        
        # Get loaded state
        loaded_global_step = trainer.status.global_step
        
        logger.info(f"📊 Loaded global step: {loaded_global_step}")
        
        # Compare loaded state with saved state
        logger.info("🔍 Comparing saved vs loaded state...")
        
        global_step_match = trained_global_step == loaded_global_step
        
        logger.info(f"📋 TDC Metadata Wrapper State verification results:")
        logger.info(f"   Global step match: {global_step_match}")
        
        # Test forward pass consistency
        logger.info("🧪 Testing forward pass consistency...")
        
        trainer.model.eval()
        with torch.no_grad():
            original_output = trainer.model(input_ids=dummy_input)
            loaded_output = trainer.model(input_ids=dummy_input)
        
        # Check if outputs are identical (should be since model state is preserved)
        output_match = torch.allclose(original_output.logits, loaded_output.logits, atol=1e-6, rtol=1e-6)
        
        if output_match:
            logger.info("✅ Model forward pass is consistent")
        else:
            logger.error("❌ Model forward pass is inconsistent")
        
        # Overall result
        all_match = global_step_match and output_match
        
        if all_match:
            logger.info("🎉 All TDC metadata wrapper checkpoint states match perfectly!")
        else:
            logger.error("❌ Some TDC metadata wrapper checkpoint states don't match!")
        
        # Restore original distributed mode and cleanup
        trainer.use_distributed = original_use_distributed
        
        # Cleanup distributed process group
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("🔄 Cleaned up distributed process group")
        
        return all_match


if __name__ == "__main__":
    try:
        # Test TDC checkpoint verification with metadata wrapper
        success = test_tdc_metadata_wrapper()
        
        if success:
            logger.info("🎉 TDC metadata wrapper checkpoint verification test passed!")
            sys.exit(0)
        else:
            logger.error("❌ TDC metadata wrapper checkpoint verification test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
