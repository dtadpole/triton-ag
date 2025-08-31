#!/usr/bin/env python3
"""
Test script to verify TDC checkpoint saving and loading functionality
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engineBase import EngineConfig
from engineFSDP import EngineFSDP
from logger import logger

def test_tdc_checkpoint():
    """Test TDC checkpoint saving and loading"""
    
    # Set up environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use 2 GPUs for testing
    
    # Initialize distributed training
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logger.info(f"🔍 [Test] Starting TDC checkpoint test on rank {rank}/{world_size}")
    
    # Create a simple config using the YAML structure
    config = EngineConfig.from_yaml("engineBase.yaml")
    
    # Override with test-specific settings
    config.model.name = "Qwen/Qwen3-4B"  # Small model for testing
    config.model.engine = "fsdp"
    config.model.max_seq_length = 128
    config.model.load_in_4bit = False
    config.model.load_in_8bit = False
    
    config.training.micro_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.max_steps = 10
    config.training.learning_rate = 1e-5
    config.training.save_steps = 5
    
    config.lora.use_lora = True
    config.lora.rank = 8
    config.lora.alpha = 16
    
    try:
        # Initialize trainer
        trainer = EngineFSDP(f"test_rank_{rank}", config)
        logger.info(f"✅ [Test] Trainer initialized on rank {rank}")
        
        # Create a temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🔍 [Test] Testing checkpoint save to: {checkpoint_path}")
            
            # Test checkpoint saving
            trainer._save_checkpoint(5)  # Save at step 5
            logger.info(f"✅ [Test] Checkpoint saved successfully on rank {rank}")
            
            # Synchronize all ranks
            dist.barrier()
            
            # Test checkpoint loading
            if rank == 0:
                logger.info(f"🔍 [Test] Testing checkpoint load from: {checkpoint_path}")
            
            # Create a new trainer instance to test loading
            trainer2 = EngineFSDP(f"test_rank_{rank}_loaded", config)
            
            # Load the checkpoint
            trainer2._load_checkpoint(str(checkpoint_path))
            logger.info(f"✅ [Test] Checkpoint loaded successfully on rank {rank}")
            
            # Verify the global step was loaded correctly
            if trainer2.status.global_step == 5:
                logger.info(f"✅ [Test] Global step loaded correctly: {trainer2.status.global_step}")
            else:
                logger.error(f"❌ [Test] Global step mismatch: expected 5, got {trainer2.status.global_step}")
                return False
            
            # Synchronize all ranks
            dist.barrier()
            
            if rank == 0:
                logger.info("🎉 [Test] TDC checkpoint test completed successfully!")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ [Test] TDC checkpoint test failed on rank {rank}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Clean up distributed training
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    success = test_tdc_checkpoint()
    sys.exit(0 if success else 1)
