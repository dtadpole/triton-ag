#!/usr/bin/env python3
"""
Comprehensive test to verify TDC state consistency across ranks:
1. Model parameters are identical before and after save/load
2. Optimizer state is identical before and after save/load  
3. Metadata (global_step, etc.) is loaded consistently on all ranks
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path
import time
import hashlib

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


def get_model_state_hash(model):
    """Get a hash of all model parameters for comparison"""
    param_hashes = []
    for name, param in model.named_parameters():
        # Convert to CPU and flatten for consistent hashing
        param_data = param.detach().cpu().flatten()
        # Convert to float32 to handle BFloat16 and other types
        if param_data.dtype != torch.float32:
            param_data = param_data.float()
        param_numpy = param_data.numpy()
        param_hash = hashlib.md5(param_numpy.tobytes()).hexdigest()
        param_hashes.append(f"{name}:{param_hash}")
    
    # Sort to ensure consistent ordering across ranks
    param_hashes.sort()
    combined_hash = hashlib.md5("|".join(param_hashes).encode()).hexdigest()
    return combined_hash


def get_optimizer_state_hash(optimizer):
    """Get a hash of optimizer state for comparison"""
    state_hashes = []
    
    for group_idx, group in enumerate(optimizer.param_groups):
        for param_idx, param in enumerate(group['params']):
            if param in optimizer.state:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        # Convert to CPU and flatten for consistent hashing
                        value_data = value.detach().cpu().flatten()
                        # Convert to float32 to handle BFloat16 and other types
                        if value_data.dtype != torch.float32:
                            value_data = value_data.float()
                        value_numpy = value_data.numpy()
                        value_hash = hashlib.md5(value_numpy.tobytes()).hexdigest()
                        state_hashes.append(f"g{group_idx}_p{param_idx}_{key}:{value_hash}")
    
    # Sort to ensure consistent ordering
    state_hashes.sort()
    combined_hash = hashlib.md5("|".join(state_hashes).encode()).hexdigest()
    return combined_hash


def test_tdc_state_consistency():
    """Test TDC state consistency across ranks"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🚀 Rank {rank}/{world_size}: Starting TDC state consistency test")
    
    # Create a config for testing
    config = EngineConfig()
    config.model.name = "microsoft/DialoGPT-small"
    config.model.max_seq_length = 1024
    config.training.max_steps = 5
    config.training.micro_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.learning_rate = 1e-4
    config.lora.use_lora = False
    config.logging.use_wandb = False
    config.model.load_in_4bit = True
    config.model.use_gradient_checkpointing = False
    
    # Create trainer
    unique_name = f"test_state_consistency_{int(time.time())}"
    trainer = EngineFSDP(unique_name, config, inference_mode=False)
    
    # Force distributed mode
    trainer.use_distributed = True
    trainer.rank = rank
    trainer.world_size = world_size
    
    logger.info(f"🔄 Rank {rank}: EngineFSDP created successfully")
    
    # Perform some training to change the state
    logger.info(f"🏋️ Rank {rank}: Performing training to modify state...")
    
    # Create dummy data
    dummy_input = torch.randint(0, 1000, (1, 10)).to(trainer.device)
    dummy_labels = torch.randint(0, 1000, (1, 10)).to(trainer.device)
    
    # Perform training steps to modify model and optimizer state
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
    
    # Get state BEFORE checkpointing
    pre_save_global_step = trainer.status.global_step
    pre_save_model_hash = get_model_state_hash(trainer.model)
    pre_save_optimizer_hash = get_optimizer_state_hash(trainer.optimizer)
    
    logger.info(f"📊 Rank {rank}: PRE-SAVE State:")
    logger.info(f"   Global step: {pre_save_global_step}")
    logger.info(f"   Model hash: {pre_save_model_hash[:16]}...")
    logger.info(f"   Optimizer hash: {pre_save_optimizer_hash[:16]}...")
    
    # Create checkpoint path
    checkpoint_path = Path("/tmp/test_tdc_state_consistency")
    checkpoint_path.mkdir(exist_ok=True)
    
    # Save checkpoint using TDC (all ranks participate)
    logger.info(f"💾 Rank {rank}: Saving TDC checkpoint...")
    trainer._save_checkpoint_tdc(checkpoint_path, 1)
    logger.info(f"💾 Rank {rank}: TDC checkpoint saved")
    
    # Modify the state to test loading
    logger.info(f"🔄 Rank {rank}: Modifying state to test loading...")
    
    # Reset global step
    trainer.status.global_step = 0
    
    # Reset model parameters (add some noise)
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # Reset optimizer state
    trainer.optimizer.zero_grad()
    for group in trainer.optimizer.param_groups:
        for p in group['params']:
            if p in trainer.optimizer.state:
                trainer.optimizer.state[p].clear()
    
    # Get state AFTER modification (before loading)
    post_modify_global_step = trainer.status.global_step
    post_modify_model_hash = get_model_state_hash(trainer.model)
    post_modify_optimizer_hash = get_optimizer_state_hash(trainer.optimizer)
    
    logger.info(f"📊 Rank {rank}: POST-MODIFY State:")
    logger.info(f"   Global step: {post_modify_global_step}")
    logger.info(f"   Model hash: {post_modify_model_hash[:16]}...")
    logger.info(f"   Optimizer hash: {post_modify_optimizer_hash[:16]}...")
    
    # Load checkpoint using TDC
    logger.info(f"📂 Rank {rank}: Loading TDC checkpoint...")
    trainer._load_checkpoint_tdc(checkpoint_path)
    logger.info(f"📂 Rank {rank}: TDC checkpoint loaded")
    
    # Get state AFTER loading
    post_load_global_step = trainer.status.global_step
    post_load_model_hash = get_model_state_hash(trainer.model)
    post_load_optimizer_hash = get_optimizer_state_hash(trainer.optimizer)
    
    logger.info(f"📊 Rank {rank}: POST-LOAD State:")
    logger.info(f"   Global step: {post_load_global_step}")
    logger.info(f"   Model hash: {post_load_model_hash[:16]}...")
    logger.info(f"   Optimizer hash: {post_load_optimizer_hash[:16]}...")
    
    # Test consistency
    global_step_consistency = (pre_save_global_step == post_load_global_step)
    model_consistency = (pre_save_model_hash == post_load_model_hash)
    optimizer_consistency = (pre_save_optimizer_hash == post_load_optimizer_hash)
    
    logger.info(f"🧪 Rank {rank}: Consistency Tests:")
    logger.info(f"   Global step consistency: {global_step_consistency} ({pre_save_global_step} == {post_load_global_step})")
    logger.info(f"   Model consistency: {model_consistency}")
    logger.info(f"   Optimizer consistency: {optimizer_consistency}")
    
    # Collect results from all ranks
    results = torch.tensor([
        int(global_step_consistency),
        int(model_consistency),
        int(optimizer_consistency),
        post_load_global_step
    ], dtype=torch.int32, device=device)
    
    gathered_results = [torch.zeros_like(results) for _ in range(world_size)]
    dist.all_gather(gathered_results, results)
    
    # Analyze results on rank 0
    if rank == 0:
        logger.info("📋 TDC State Consistency Test Results:")
        
        all_success = True
        for i, result in enumerate(gathered_results):
            rank_global_consistency = bool(result[0].item())
            rank_model_consistency = bool(result[1].item())
            rank_optimizer_consistency = bool(result[2].item())
            rank_global_step = result[3].item()
            rank_success = (rank_global_consistency and rank_model_consistency and rank_optimizer_consistency)
            
            logger.info(f"📊 Rank {i}:")
            logger.info(f"   Global step consistency: {rank_global_consistency}")
            logger.info(f"   Model consistency: {rank_model_consistency}")
            logger.info(f"   Optimizer consistency: {rank_optimizer_consistency}")
            logger.info(f"   Global step: {rank_global_step}")
            logger.info(f"   Overall success: {rank_success}")
            
            if not rank_success:
                all_success = False
        
        # Check cross-rank consistency
        if world_size >= 2:
            rank0_global_step = gathered_results[0][3].item()
            rank1_global_step = gathered_results[1][3].item()
            
            if rank0_global_step == rank1_global_step:
                logger.info("✅ Global step consistency across ranks: PASSED")
            else:
                logger.error(f"❌ Global step consistency across ranks: FAILED ({rank0_global_step} vs {rank1_global_step})")
                all_success = False
        
        if all_success:
            logger.info("🎉 All TDC state consistency tests passed!")
        else:
            logger.error("❌ Some TDC state consistency tests failed!")
        
        return all_success
    
    return True


def main():
    """Main function"""
    try:
        success = test_tdc_state_consistency()
        cleanup_distributed()
        
        if success:
            logger.info("🎉 TDC state consistency test completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ TDC state consistency test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cleanup_distributed()
        sys.exit(1)


if __name__ == "__main__":
    main()
