#!/usr/bin/env python3
"""
Example demonstrating proper model parameter hashing in FSDP

This script shows the difference between:
1. Incorrect approach: Direct parameter iteration (won't work in FSDP)
2. Correct approach: Using state_dict() for sharded parameters
3. Full model approach: Gathering all parameters to rank 0 (expensive)
"""

import torch
import torch.distributed as dist
import hashlib
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


def incorrect_model_hash(model):
    """INCORRECT: This won't work properly in FSDP"""
    hasher = hashlib.md5()
    rank = dist.get_rank()
    
    logger.info(f"❌ Rank {rank}: Using INCORRECT approach (direct parameter iteration)")
    
    # This is WRONG for FSDP because:
    # 1. Each rank only sees its sharded parameters
    # 2. Different ranks will have different hashes
    # 3. You can't sort parameters directly
    try:
        model_params = model.parameters()
        num_params = sum(p.numel() for p in model_params)
        logger.info(f"❌ Rank {rank}: Model has [{num_params:,}] parameters (sharded)")
        
        # This will fail or give inconsistent results
        for param in sorted(model_params):  # This sorting won't work properly
            tensor = param.data
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            hasher.update(tensor.cpu().numpy().tobytes())
        
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"❌ Rank {rank}: Error in incorrect approach: {e}")
        return "ERROR"


def correct_model_hash(model):
    """CORRECT: Using state_dict() for FSDP sharded parameters"""
    hasher = hashlib.md5()
    rank = dist.get_rank()
    
    logger.info(f"✅ Rank {rank}: Using CORRECT approach (state_dict)")
    
    # Get the model state dict - this gives us the sharded parameters for this rank
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model.state_dict()
        
        # Count parameters for logging - handle both regular tensors and ShardedTensors
        num_params = 0
        for p in model_state_dict.values():
            # Check for ShardedTensor first (it's a subclass of torch.Tensor)
            if hasattr(p, 'local_shards'):  # This is a ShardedTensor
                num_params += p.size().numel()
            elif isinstance(p, torch.Tensor):  # This is a regular tensor
                num_params += p.numel()
        logger.info(f"✅ Rank {rank}: Model has [{num_params:,}] parameters in state dict")
        
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


def full_model_hash(model):
    """EXPENSIVE: Get hash of the FULL model parameters (gathers all shards to rank 0)"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        logger.info("🔍 Rank 0: Computing full model hash (expensive operation)")
    
    # Get the full state dict on rank 0 only
    if rank == 0:
        # Use FSDP's full state dict functionality
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            full_state_dict = model.state_dict()
        
        hasher = hashlib.md5()
        num_params = 0
        
        # Sort by parameter names for consistent ordering
        for param_name in sorted(full_state_dict.keys()):
            param_tensor = full_state_dict[param_name]
            
            # Skip non-tensor entries
            if not isinstance(param_tensor, torch.Tensor):
                continue
                
            num_params += param_tensor.numel()
            
            # Convert to float32 if needed for consistent hashing
            if param_tensor.dtype == torch.bfloat16:
                param_tensor = param_tensor.float()
            elif param_tensor.dtype == torch.float16:
                param_tensor = param_tensor.float()
                
            # Hash the tensor data
            hasher.update(param_tensor.cpu().numpy().tobytes())
        
        logger.info(f"🔍 Rank 0: Full model has [{num_params:,}] parameters")
        full_hash = hasher.hexdigest()
        
        # Broadcast the hash to all ranks
        hash_bytes = full_hash.encode()
        for other_rank in range(1, world_size):
            dist.send(torch.tensor([ord(c) for c in hash_bytes], dtype=torch.uint8), dst=other_rank)
        
        return full_hash
    else:
        # Receive the hash from rank 0
        hash_tensor = torch.zeros(32, dtype=torch.uint8)  # MD5 hash is 32 hex chars
        dist.recv(hash_tensor, src=0)
        return ''.join(chr(c.item()) for c in hash_tensor)


def demonstrate_fsdp_hashing():
    """Demonstrate the different approaches to FSDP model hashing"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🚀 Rank {rank}/{world_size}: Starting FSDP hashing demonstration")
    
    # Create a simple model for demonstration
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    ).to(device)
    
    # Wrap with FSDP
    model = FSDP(model)
    
    logger.info(f"📊 Rank {rank}: Model wrapped with FSDP")
    
    # Test the different approaches
    logger.info(f"\n{'='*60}")
    logger.info(f"Rank {rank}: Testing different hashing approaches")
    logger.info(f"{'='*60}")
    
    # 1. Incorrect approach
    incorrect_hash = incorrect_model_hash(model)
    logger.info(f"❌ Rank {rank}: Incorrect hash: {incorrect_hash[:16]}...")
    
    # 2. Correct approach (sharded)
    correct_hash = correct_model_hash(model)
    logger.info(f"✅ Rank {rank}: Correct hash (sharded): {correct_hash[:16]}...")
    
    # 3. Full model approach
    full_hash = full_model_hash(model)
    logger.info(f"🔍 Rank {rank}: Full model hash: {full_hash[:16]}...")
    
    # Gather results from all ranks
    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info("Summary of hashing approaches:")
        logger.info(f"{'='*60}")
        logger.info("❌ INCORRECT: Direct parameter iteration")
        logger.info("   - Each rank sees different parameters (sharded)")
        logger.info("   - Hashes will be different across ranks")
        logger.info("   - Cannot sort parameters directly")
        logger.info("")
        logger.info("✅ CORRECT: Using model.state_dict()")
        logger.info("   - Each rank hashes its own sharded parameters")
        logger.info("   - Consistent ordering by parameter names")
        logger.info("   - Efficient and works well for distributed training")
        logger.info("")
        logger.info("🔍 FULL MODEL: Gathering all parameters to rank 0")
        logger.info("   - Expensive operation (can cause OOM)")
        logger.info("   - All ranks get the same hash")
        logger.info("   - Only use for debugging/verification")
    
    cleanup_distributed()


if __name__ == "__main__":
    try:
        demonstrate_fsdp_hashing()
        logger.info("🎉 FSDP hashing demonstration completed!")
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cleanup_distributed()
