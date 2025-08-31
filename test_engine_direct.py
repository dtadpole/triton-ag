#!/usr/bin/env python3
"""
Test script to reproduce the issue when running engineFSDP.py directly with torchrun
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from logger import logger
from engineFSDP import EngineFSDP
from engineBase import EngineConfig


def setup_distributed():
    """Initialize distributed process group"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    
    logger.info(f"🔍 Rank {rank}/{world_size} initialized on device {device}")
    return rank, world_size, device


def test_engine_direct():
    """Test running engineFSDP directly"""
    try:
        # Setup distributed
        rank, world_size, device = setup_distributed()
        
        # Create a simple config
        config = EngineConfig()
        config.model.name = "microsoft/DialoGPT-small"
        config.model.max_seq_length = 1024
        config.training.max_steps = 2
        config.training.save_steps = 1
        config.training.micro_batch_size = 1
        config.training.gradient_accumulation_steps = 1
        
        # Create temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config.training.checkpoint_path = temp_dir
            
            logger.info(f"🔍 [{rank}] Creating EngineFSDP...")
            trainer = EngineFSDP("test_direct", config)
            
            logger.info(f"🔍 [{rank}] EngineFSDP created successfully")
            logger.info(f"🔍 [{rank}] use_distributed: {trainer.use_distributed}")
            logger.info(f"🔍 [{rank}] rank: {trainer.rank}")
            logger.info(f"🔍 [{rank}] world_size: {trainer.world_size}")
            
            # Test checkpoint save
            logger.info(f"🔍 [{rank}] Testing checkpoint save...")
            trainer._save_checkpoint(1)
            logger.info(f"🔍 [{rank}] Checkpoint save completed successfully!")
            
            # Synchronize all ranks
            if trainer.use_distributed:
                dist.barrier()
                logger.info(f"🔍 [{rank}] All ranks synchronized")
            
            logger.info(f"✅ [{rank}] Test completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ [{rank}] Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    test_engine_direct()

