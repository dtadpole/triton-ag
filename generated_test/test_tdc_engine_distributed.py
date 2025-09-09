#!/usr/bin/env python3
"""
Test TDC functionality using EngineFSDP in a distributed environment
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
    
    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_tdc_engine_distributed():
    """Test TDC functionality using EngineFSDP"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🚀 Rank {rank}/{world_size}: Starting EngineFSDP TDC distributed test")
    
    # Create a config for testing with a tiny model
    config = EngineConfig()
    config.model.name = "microsoft/DialoGPT-small"  # Small model (117M params)
    config.model.max_seq_length = 1024  # Use the model's default sequence length
    config.training.max_steps = 5
    config.training.micro_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.learning_rate = 1e-4
    config.lora.use_lora = False
    config.logging.use_wandb = False
    config.model.load_in_4bit = True  # Use 4-bit quantization to reduce memory
    config.model.use_gradient_checkpointing = False  # Disable for faster testing
    
    # Create trainer with unique name to avoid conflicts
    unique_name = f"test_engine_distributed_{int(time.time())}"
    trainer = EngineFSDP(unique_name, config, inference_mode=False)
    
    # Force distributed mode
    trainer.use_distributed = True
    trainer.rank = rank
    trainer.world_size = world_size
    
    logger.info(f"🔄 Rank {rank}: EngineFSDP created successfully")
    
    # Get initial state
    initial_global_step = trainer.status.global_step
    logger.info(f"📊 Rank {rank}: Initial global step: {initial_global_step}")
    
    # Create checkpoint path
    checkpoint_path = Path("/tmp/test_engine_tdc_distributed")
    checkpoint_path.mkdir(exist_ok=True)
    
    # Save checkpoint using TDC (all ranks participate)
    logger.info(f"💾 Rank {rank}: Saving TDC checkpoint...")
    trainer._save_checkpoint_tdc(checkpoint_path, 1)
    logger.info(f"💾 Rank {rank}: TDC checkpoint saved")
    
    # Modify the global step to test loading
    logger.info(f"🔄 Rank {rank}: Modifying global step to test loading...")
    trainer.status.global_step = 0
    
    # Load checkpoint using TDC
    logger.info(f"📂 Rank {rank}: Loading TDC checkpoint...")
    trainer._load_checkpoint_tdc(checkpoint_path)
    
    # Get loaded state
    loaded_global_step = trainer.status.global_step
    logger.info(f"📊 Rank {rank}: Loaded global step: {loaded_global_step}")
    
    # Test forward pass consistency
    logger.info(f"🧪 Rank {rank}: Testing forward pass consistency...")
    
    # Create dummy data
    dummy_input = torch.randint(0, 1000, (1, 10)).to(trainer.device)
    
    trainer.model.eval()
    with torch.no_grad():
        output = trainer.model(input_ids=dummy_input)
    
    logger.info(f"📊 Rank {rank}: Model forward pass completed, output shape: {output.logits.shape}")
    
    # Check global step consistency
    global_step_match = loaded_global_step == initial_global_step
    if global_step_match:
        logger.info(f"✅ Rank {rank}: Global step match: {initial_global_step} == {loaded_global_step}")
    else:
        logger.error(f"❌ Rank {rank}: Global step mismatch: {initial_global_step} != {loaded_global_step}")
    
    # Collect results from all ranks
    results = torch.tensor([
        int(global_step_match),
        loaded_global_step
    ], dtype=torch.int32, device=device)
    
    gathered_results = [torch.zeros_like(results) for _ in range(world_size)]
    dist.all_gather(gathered_results, results)
    
    # Analyze results on rank 0
    if rank == 0:
        logger.info("📋 EngineFSDP TDC Test Results:")
        
        all_success = True
        for i, result in enumerate(gathered_results):
            rank_global_step_match = bool(result[0].item())
            rank_global_step = result[1].item()
            rank_success = rank_global_step_match
            
            logger.info(f"📊 Rank {i}:")
            logger.info(f"   Global step match: {rank_global_step_match}")
            logger.info(f"   Global step: {rank_global_step}")
            logger.info(f"   Success: {rank_success}")
            
            if not rank_success:
                all_success = False
        
        # Check consistency across ranks
        if world_size >= 2:
            rank0_global_step = gathered_results[0][1].item()
            rank1_global_step = gathered_results[1][1].item()
            
            if rank0_global_step == rank1_global_step:
                logger.info("✅ Global step consistency across ranks: PASSED")
            else:
                logger.error(f"❌ Global step consistency across ranks: FAILED ({rank0_global_step} vs {rank1_global_step})")
                all_success = False
        
        if all_success:
            logger.info("🎉 All EngineFSDP TDC tests passed!")
        else:
            logger.error("❌ Some EngineFSDP TDC tests failed!")
        
        return all_success
    
    return True


def main():
    """Main function"""
    try:
        success = test_tdc_engine_distributed()
        cleanup_distributed()
        
        if success:
            logger.info("🎉 EngineFSDP TDC test completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ EngineFSDP TDC test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cleanup_distributed()
        sys.exit(1)


if __name__ == "__main__":
    main()
