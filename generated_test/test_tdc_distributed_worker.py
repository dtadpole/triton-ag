#!/usr/bin/env python3
"""
Distributed worker script for TDC testing - to be run with torchrun
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


def main():
    """Main function for distributed worker"""
    # Initialize distributed
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logger.info(f"🚀 Rank {rank}/{world_size}: Starting distributed TDC test")
    
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
    
    # Create trainer with unique name to avoid conflicts
    unique_name = f"test_distributed_{int(time.time())}"
    trainer = EngineFSDP(unique_name, config, inference_mode=False)
    
    # Force distributed mode
    trainer.use_distributed = True
    trainer.rank = rank
    trainer.world_size = world_size
    
    logger.info(f"🔄 Rank {rank}: EngineFSDP created successfully")
    
    # Get initial state
    initial_global_step = trainer.status.global_step
    logger.info(f"📊 Rank {rank}: Initial global step: {initial_global_step}")
    
    # Do some training steps to change the state
    logger.info(f"🏋️ Rank {rank}: Performing training steps...")
    
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
        
        logger.info(f"📈 Rank {rank}: Training step {step + 1}: loss={loss.item():.4f}, global_step={trainer.status.global_step}")
    
    # Get state after training
    trained_global_step = trainer.status.global_step
    logger.info(f"📊 Rank {rank}: Trained global step: {trained_global_step}")
    
    # Use a shared checkpoint path
    checkpoint_path = Path("/tmp/test_distributed_checkpoint")
    checkpoint_path.mkdir(exist_ok=True)
    
    # Save checkpoint using TDC (only rank 0 should save)
    if rank == 0:
        logger.info(f"💾 Rank {rank}: Saving TDC checkpoint...")
        trainer._save_checkpoint_tdc(checkpoint_path, 1)
        logger.info(f"💾 Rank {rank}: TDC checkpoint saved")
    
    # Synchronize all ranks
    dist.barrier()
    
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
    
    trainer.model.eval()
    with torch.no_grad():
        original_output = trainer.model(input_ids=dummy_input)
        loaded_output = trainer.model(input_ids=dummy_input)
    
    # Check if outputs are identical
    output_match = torch.allclose(original_output.logits, loaded_output.logits, atol=1e-6, rtol=1e-6)
    
    if output_match:
        logger.info(f"✅ Rank {rank}: Model forward pass is consistent")
    else:
        logger.error(f"❌ Rank {rank}: Model forward pass is inconsistent")
    
    # Check global step consistency
    global_step_match = trained_global_step == loaded_global_step
    if global_step_match:
        logger.info(f"✅ Rank {rank}: Global step match: {trained_global_step} == {loaded_global_step}")
    else:
        logger.error(f"❌ Rank {rank}: Global step mismatch: {trained_global_step} != {loaded_global_step}")
    
    # Collect results from all ranks
    results = torch.tensor([trained_global_step, loaded_global_step, int(global_step_match), int(output_match)], dtype=torch.int32)
    gathered_results = [torch.zeros_like(results) for _ in range(world_size)]
    dist.all_gather(gathered_results, results)
    
    # Analyze results on rank 0
    if rank == 0:
        logger.info("📋 Distributed TDC Test Results:")
        all_success = True
        
        for i, result in enumerate(gathered_results):
            rank_trained_step = result[0].item()
            rank_loaded_step = result[1].item()
            rank_global_match = bool(result[2].item())
            rank_output_match = bool(result[3].item())
            rank_success = rank_global_match and rank_output_match
            
            logger.info(f"📊 Rank {i}:")
            logger.info(f"   Trained global step: {rank_trained_step}")
            logger.info(f"   Loaded global step: {rank_loaded_step}")
            logger.info(f"   Global step match: {rank_global_match}")
            logger.info(f"   Output match: {rank_output_match}")
            logger.info(f"   Success: {rank_success}")
            
            if not rank_success:
                all_success = False
        
        # Check consistency across ranks
        if world_size >= 2:
            rank0_loaded_step = gathered_results[0][1].item()
            rank1_loaded_step = gathered_results[1][1].item()
            
            if rank0_loaded_step == rank1_loaded_step:
                logger.info("✅ Global step consistency across ranks: PASSED")
            else:
                logger.error(f"❌ Global step consistency across ranks: FAILED ({rank0_loaded_step} vs {rank1_loaded_step})")
                all_success = False
        
        if all_success:
            logger.info("🎉 All distributed TDC checkpoint tests passed!")
        else:
            logger.error("❌ Some distributed TDC checkpoint tests failed!")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
