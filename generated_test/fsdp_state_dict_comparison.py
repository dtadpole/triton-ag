#!/usr/bin/env python3
"""
Comparison of LOCAL_STATE_DICT vs SHARDED_STATE_DICT in FSDP

This script demonstrates the differences between the two state dict types:
- LOCAL_STATE_DICT: Contains only local shard data as regular tensors
- SHARDED_STATE_DICT: Contains ShardedTensor objects with metadata
"""

import torch
import torch.distributed as dist
from pathlib import Path
import sys

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from logger import logger
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType


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


def compare_state_dict_types(model):
    """Compare LOCAL_STATE_DICT vs SHARDED_STATE_DICT"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🔍 Rank {rank}: Comparing state dict types")
    
    # Test LOCAL_STATE_DICT
    logger.info(f"\n{'='*60}")
    logger.info(f"Rank {rank}: LOCAL_STATE_DICT")
    logger.info(f"{'='*60}")
    
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        local_state_dict = model.state_dict()
        
        logger.info(f"📊 Rank {rank}: LOCAL_STATE_DICT contains {len(local_state_dict)} parameters")
        
        # Analyze the first few parameters
        for i, (name, param) in enumerate(local_state_dict.items()):
            if i >= 3:  # Only show first 3 parameters
                break
                
            logger.info(f"  Parameter {i+1}: {name}")
            logger.info(f"    Type: {type(param)}")
            logger.info(f"    Shape: {param.shape}")
            logger.info(f"    Dtype: {param.dtype}")
            logger.info(f"    Device: {param.device}")
            logger.info(f"    Numel: {param.numel()}")
            logger.info(f"    Is ShardedTensor: {hasattr(param, 'local_shards')}")
    
    # Test SHARDED_STATE_DICT
    logger.info(f"\n{'='*60}")
    logger.info(f"Rank {rank}: SHARDED_STATE_DICT")
    logger.info(f"{'='*60}")
    
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        sharded_state_dict = model.state_dict()
        
        logger.info(f"📊 Rank {rank}: SHARDED_STATE_DICT contains {len(sharded_state_dict)} parameters")
        
        # Analyze the first few parameters
        for i, (name, param) in enumerate(sharded_state_dict.items()):
            if i >= 3:  # Only show first 3 parameters
                break
                
            logger.info(f"  Parameter {i+1}: {name}")
            logger.info(f"    Type: {type(param)}")
            
            if hasattr(param, 'local_shards') and param.local_shards():
                # This is a ShardedTensor
                logger.info(f"    Is ShardedTensor: True")
                logger.info(f"    Full tensor shape: {param.size()}")
                logger.info(f"    Full tensor numel: {param.size().numel()}")
                logger.info(f"    Local shards count: {len(param.local_shards())}")
                
                local_shard = param.local_shards()[0].tensor
                logger.info(f"    Local shard shape: {local_shard.shape}")
                logger.info(f"    Local shard numel: {local_shard.numel()}")
                logger.info(f"    Local shard dtype: {local_shard.dtype}")
                logger.info(f"    Local shard device: {local_shard.device}")
            else:
                # This is a regular tensor
                logger.info(f"    Is ShardedTensor: False")
                logger.info(f"    Shape: {param.shape}")
                logger.info(f"    Dtype: {param.dtype}")
                logger.info(f"    Device: {param.device}")
                logger.info(f"    Numel: {param.numel()}")
    
    # Compare parameter counts
    logger.info(f"\n{'='*60}")
    logger.info(f"Rank {rank}: Summary Comparison")
    logger.info(f"{'='*60}")
    
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        local_state_dict = model.state_dict()
        local_total_params = sum(p.numel() for p in local_state_dict.values() if isinstance(p, torch.Tensor))
    
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        sharded_state_dict = model.state_dict()
        sharded_total_params = 0
        for p in sharded_state_dict.values():
            if hasattr(p, 'local_shards') and p.local_shards():
                sharded_total_params += p.local_shards()[0].tensor.numel()
            elif isinstance(p, torch.Tensor):
                sharded_total_params += p.numel()
    
    logger.info(f"📊 Rank {rank}: LOCAL_STATE_DICT total parameters: {local_total_params:,}")
    logger.info(f"📊 Rank {rank}: SHARDED_STATE_DICT total parameters: {sharded_total_params:,}")
    logger.info(f"📊 Rank {rank}: Parameter counts match: {local_total_params == sharded_total_params}")
    
    cleanup_distributed()


def demonstrate_hashing_differences(model):
    """Demonstrate how hashing works with different state dict types"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Rank {rank}: Hashing Comparison")
    logger.info(f"{'='*60}")
    
    import hashlib
    
    # Hash with LOCAL_STATE_DICT
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        local_state_dict = model.state_dict()
        local_hasher = hashlib.md5()
        
        for param_name in sorted(local_state_dict.keys()):
            param = local_state_dict[param_name]
            if isinstance(param, torch.Tensor):
                local_hasher.update(param.cpu().numpy().tobytes())
        
        local_hash = local_hasher.hexdigest()
    
    # Hash with SHARDED_STATE_DICT
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        sharded_state_dict = model.state_dict()
        sharded_hasher = hashlib.md5()
        
        for param_name in sorted(sharded_state_dict.keys()):
            param = sharded_state_dict[param_name]
            
            if hasattr(param, 'local_shards') and param.local_shards():
                # Extract local shard
                local_shard = param.local_shards()[0].tensor
                sharded_hasher.update(local_shard.cpu().numpy().tobytes())
            elif isinstance(param, torch.Tensor):
                sharded_hasher.update(param.cpu().numpy().tobytes())
        
        sharded_hash = sharded_hasher.hexdigest()
    
    logger.info(f"🔍 Rank {rank}: LOCAL_STATE_DICT hash: {local_hash[:16]}...")
    logger.info(f"🔍 Rank {rank}: SHARDED_STATE_DICT hash: {sharded_hash[:16]}...")
    logger.info(f"🔍 Rank {rank}: Hashes match: {local_hash == sharded_hash}")
    
    cleanup_distributed()


def main():
    """Main function"""
    try:
        rank, world_size, device = setup_distributed()
        
        logger.info(f"🚀 Rank {rank}/{world_size}: Starting FSDP state dict comparison")
        
        # Create a simple model for demonstration
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 100)
        ).to(device)
        
        # Wrap with FSDP
        model = FSDP(model)
        
        logger.info(f"📊 Rank {rank}: Model wrapped with FSDP")
        
        # Compare state dict types
        compare_state_dict_types(model)
        
        # Demonstrate hashing differences
        demonstrate_hashing_differences(model)
        
        logger.info(f"🎉 Rank {rank}: FSDP state dict comparison completed!")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cleanup_distributed()


if __name__ == "__main__":
    main()
