#!/usr/bin/env python3
"""
Comprehensive test for TDC implementation to verify all states are correct across ranks
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType
from pathlib import Path
import time
import hashlib
import json

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

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
    """Get hash of model parameters for consistency checking in FSDP"""
    hasher = hashlib.md5()
    rank = dist.get_rank()
    
    # For FSDP models, we need to use the state dict approach to get consistent hashing
    # across all ranks. Each rank will hash its own sharded parameters, but we need
    # to ensure consistent ordering and handling.
    
    # Get the model state dict - this gives us the sharded parameters for this rank
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model.state_dict()
        
        # Count parameters for logging - handle both regular tensors and ShardedTensors
        num_params = 0
        for p in model_state_dict.values():
            # Check for ShardedTensor first (it's a subclass of torch.Tensor)
            if hasattr(p, 'local_shards'):  # This is a ShardedTensor
                num_params += p.local_shards()[0].tensor.numel()
            elif isinstance(p, torch.Tensor):  # This is a regular tensor
                num_params += p.numel()
        logger.info(f"🔍 Rank {rank}: Model has [{num_params:,}] parameters in state dict")
        
        # Sort by parameter names for consistent ordering across ranks
        # Each rank will hash its own sharded parameters
        for param_name in sorted(model_state_dict.keys()):
            param_tensor = model_state_dict[param_name]
            
            # Skip non-tensor entries (like metadata)
            if not isinstance(param_tensor, torch.Tensor) and not hasattr(param_tensor, 'size'):
                continue
            
            # For ShardedTensor, we need to get the local shard
            if hasattr(param_tensor, 'local_shards') and param_tensor.local_shards():
                # This is a ShardedTensor - get the local shard data
                local_shard = param_tensor.local_shards()[0].tensor
                tensor_to_hash = local_shard
            elif isinstance(param_tensor, torch.Tensor):
                # This is a regular tensor
                tensor_to_hash = param_tensor
            else:
                continue
                
            # Convert to float32 if needed for consistent hashing
            if tensor_to_hash.dtype == torch.bfloat16:
                tensor_to_hash = tensor_to_hash.float()
            elif tensor_to_hash.dtype == torch.float16:
                tensor_to_hash = tensor_to_hash.float()
                
            # Move to CPU and hash the tensor data
            hasher.update(tensor_to_hash.cpu().numpy().tobytes())
    
    return hasher.hexdigest()


def get_optimizer_state_hash(optimizer):
    """Get hash of optimizer state for consistency checking"""
    hasher = hashlib.md5()
    
    # Get optimizer state dict
    optim_state = optimizer.state_dict()
    
    # Convert to JSON string for hashing (handles nested structures)
    state_str = json.dumps(optim_state, sort_keys=True, default=str)
    hasher.update(state_str.encode())
    
    return hasher.hexdigest()


def get_scheduler_state_hash(scheduler):
    """Get hash of scheduler state for consistency checking"""
    if scheduler is None:
        return "no_scheduler"
    
    hasher = hashlib.md5()
    
    # Get scheduler state dict
    scheduler_state = scheduler.state_dict()
    
    # Convert to JSON string for hashing
    state_str = json.dumps(scheduler_state, sort_keys=True, default=str)
    hasher.update(state_str.encode())
    
    return hasher.hexdigest()


def test_tdc_comprehensive():
    """Comprehensive test for TDC implementation"""
    # rank, world_size, device = setup_distributed()
    
    # Create a config for testing
    config = EngineConfig.from_yaml("engineBase.yaml")
    
    # Create trainer
    unique_name = f"test_tdc_comprehensive_{int(time.time())}"
    trainer = EngineFSDP(unique_name, config, inference_mode=False)

    # get rank, world_size, device from trainer
    rank, world_size, device = trainer.rank, trainer.world_size, trainer.device
    logger.info(f"🚀 Rank {rank}/{world_size}: Starting comprehensive TDC test")
    
    
    logger.info(f"🔄 Rank {rank}: EngineFSDP created successfully")
    
    # Perform some training to create optimizer state
    logger.info(f"🏋️ Rank {rank}: Performing training to create state...")
    
    # Create dummy data
    # get vocabulary size of the given model
    vocabulary_size = trainer.model.config.vocab_size
    
    # Perform training steps to create optimizer state
    for step in range(20):
        dummy_input = torch.randint(0, vocabulary_size, (1, 2048)).to(trainer.device)
        dummy_labels = torch.randint(0, vocabulary_size, (1, 2048)).to(trainer.device)

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
    logger.info(f"🔍 Rank {rank}: Calculating PRE-SAVE hashes...")
    start_time = time.time()
    pre_save_global_step = trainer.status.global_step
    pre_save_model_hash = get_model_state_hash(trainer.model)
    pre_save_optimizer_hash = get_optimizer_state_hash(trainer.optimizer)
    pre_save_scheduler_hash = get_scheduler_state_hash(trainer.scheduler)
    
    logger.info(f"📊 Rank {rank}: PRE-SAVE State in {time.time() - start_time:.2f} seconds:")
    logger.info(f"   Global step: {pre_save_global_step}")
    logger.info(f"   Model hash: {pre_save_model_hash[:16]}...")
    logger.info(f"   Optimizer hash: {pre_save_optimizer_hash[:16]}...")
    logger.info(f"   Scheduler hash: {pre_save_scheduler_hash[:16]}...")
    
    # Create checkpoint path
    checkpoint_path = Path("/tmp/test_tdc_comprehensive")
    checkpoint_path.mkdir(exist_ok=True)
    
    # Save checkpoint using TDC
    logger.info(f"💾 Rank {rank}: Saving TDC checkpoint...")
    start_time = time.time()
    trainer._save_checkpoint(1)
    logger.info(f"💾 Rank {rank}: TDC checkpoint saved in {time.time() - start_time:.2f} seconds")
    
    # Modify the state to test loading
    logger.info(f"🔄 Rank {rank}: Modifying state to test loading...")
    
    # Reset global step
    trainer.status.global_step = 0
    
    # Modify model parameters slightly
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.add_(torch.randn_like(param) * 0.01)

    # Modify optimizer state slightly
    for param in trainer.optimizer.state_dict()["state"].values():
        param["exp_avg"] = torch.randn_like(param["exp_avg"]) * 0.01
        param["exp_avg_sq"] = torch.randn_like(param["exp_avg_sq"]) * 0.01

    # Modify scheduler state slightly
    # For LambdaLR scheduler (created by get_scheduler with cosine), we can modify the last_epoch
    if hasattr(trainer.scheduler, 'last_epoch'):
        trainer.scheduler.last_epoch = trainer.scheduler.last_epoch + 1
    # For other scheduler types, we can modify the step count
    if hasattr(trainer.scheduler, '_step_count'):
        trainer.scheduler._step_count = trainer.scheduler._step_count + 1
    
    # Get state AFTER modification (before loading)
    post_modify_global_step = trainer.status.global_step
    post_modify_model_hash = get_model_state_hash(trainer.model)
    post_modify_optimizer_hash = get_optimizer_state_hash(trainer.optimizer)
    post_modify_scheduler_hash = get_scheduler_state_hash(trainer.scheduler)
    
    logger.info(f"📊 Rank {rank}: POST-MODIFY State:")
    logger.info(f"   Global step: {post_modify_global_step}")
    logger.info(f"   Model hash: {post_modify_model_hash[:16]}...")
    logger.info(f"   Optimizer hash: {post_modify_optimizer_hash[:16]}...")
    logger.info(f"   Scheduler hash: {post_modify_scheduler_hash[:16]}...")

    # assert global step is different
    assert post_modify_global_step != pre_save_global_step
    logger.info(f"✅ Rank {rank}: Global step is different: {post_modify_global_step} != {pre_save_global_step}")
    # ensure that post_modify hash is different from pre_save hash
    assert post_modify_model_hash != pre_save_model_hash
    logger.info(f"✅ Rank {rank}: Model hash is different: {post_modify_model_hash} != {pre_save_model_hash}")
    assert post_modify_optimizer_hash != pre_save_optimizer_hash
    logger.info(f"✅ Rank {rank}: Optimizer hash is different: {post_modify_optimizer_hash} != {pre_save_optimizer_hash}")
    assert post_modify_scheduler_hash != pre_save_scheduler_hash
    logger.info(f"✅ Rank {rank}: Scheduler hash is different: {post_modify_scheduler_hash} != {pre_save_scheduler_hash}")
    
    # Load checkpoint using TDC
    checkpoint_location = trainer.checkpoint_path / 'checkpoint-1'
    logger.info(f"📂 Rank {rank}: Loading TDC checkpoint from [{checkpoint_location}]...")
    start_time = time.time()
    trainer._load_checkpoint(checkpoint_location)
    logger.info(f"📂 Rank {rank}: TDC checkpoint loaded in {time.time() - start_time:.2f} seconds")
    
    # Get state AFTER loading
    logger.info(f"🔍 Rank {rank}: Calculating POST-LOAD hashes...")
    start_time = time.time()
    post_load_global_step = trainer.status.global_step
    post_load_model_hash = get_model_state_hash(trainer.model)
    post_load_optimizer_hash = get_optimizer_state_hash(trainer.optimizer)
    post_load_scheduler_hash = get_scheduler_state_hash(trainer.scheduler)
    
    logger.info(f"📊 Rank {rank}: POST-LOAD State in {time.time() - start_time:.2f} seconds:")
    logger.info(f"   Global step: {post_load_global_step}")
    logger.info(f"   Model hash: {post_load_model_hash[:16]}...")
    logger.info(f"   Optimizer hash: {post_load_optimizer_hash[:16]}...")
    logger.info(f"   Scheduler hash: {post_load_scheduler_hash[:16]}...")
    
    # Test consistency
    global_step_consistency = (pre_save_global_step == post_load_global_step)
    model_consistency = (pre_save_model_hash == post_load_model_hash)
    optimizer_consistency = (pre_save_optimizer_hash == post_load_optimizer_hash)
    scheduler_consistency = (pre_save_scheduler_hash == post_load_scheduler_hash)
    
    logger.info(f"🧪 Rank {rank}: Consistency Tests:")
    logger.info(f"   Global step consistency: {global_step_consistency} ({pre_save_global_step} == {post_load_global_step})")
    logger.info(f"   Model consistency: {model_consistency}")
    logger.info(f"   Optimizer consistency: {optimizer_consistency}")
    logger.info(f"   Scheduler consistency: {scheduler_consistency}")
    
    # Collect results from all ranks
    results = torch.tensor([
        int(global_step_consistency),
        int(model_consistency),
        int(optimizer_consistency),
        int(scheduler_consistency),
        post_load_global_step
    ], dtype=torch.int32, device=device)
    
    gathered_results = [torch.zeros_like(results) for _ in range(world_size)]
    dist.all_gather(gathered_results, results)
    
    # Analyze results on rank 0
    if rank == 0:
        logger.info("📋 Comprehensive TDC Test Results:")
        
        all_success = True
        for i, result in enumerate(gathered_results):
            rank_global_consistency = bool(result[0].item())
            rank_model_consistency = bool(result[1].item())
            rank_optimizer_consistency = bool(result[2].item())
            rank_scheduler_consistency = bool(result[3].item())
            rank_global_step = result[4].item()
            
            rank_success = (rank_global_consistency and rank_model_consistency and 
                          rank_optimizer_consistency and rank_scheduler_consistency)
            
            logger.info(f"📊 Rank {i}:")
            logger.info(f"   Global step consistency: {rank_global_consistency}")
            logger.info(f"   Model consistency: {rank_model_consistency}")
            logger.info(f"   Optimizer consistency: {rank_optimizer_consistency}")
            logger.info(f"   Scheduler consistency: {rank_scheduler_consistency}")
            logger.info(f"   Global step: {rank_global_step}")
            logger.info(f"   Overall success: {rank_success}")
            
            if not rank_success:
                all_success = False
        
        # Check cross-rank consistency
        if world_size >= 2:
            rank0_global_step = gathered_results[0][4].item()
            rank1_global_step = gathered_results[1][4].item()
            
            if rank0_global_step == rank1_global_step:
                logger.info("✅ Global step consistency across ranks: PASSED")
            else:
                logger.error(f"❌ Global step consistency across ranks: FAILED ({rank0_global_step} vs {rank1_global_step})")
                all_success = False
        
        if all_success:
            logger.info("🎉 All comprehensive TDC tests passed!")
        else:
            logger.error("❌ Some comprehensive TDC tests failed!")
        
        return all_success
    
    return True


def main():
    """Main function"""
    try:
        success = test_tdc_comprehensive()
        cleanup_distributed()
        
        if success:
            logger.info("🎉 Comprehensive TDC test completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Comprehensive TDC test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cleanup_distributed()
        sys.exit(1)


if __name__ == "__main__":
    main()
