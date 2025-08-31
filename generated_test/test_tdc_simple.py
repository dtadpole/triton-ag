#!/usr/bin/env python3
"""
Simple test script to verify TDC checkpoint saving and loading functionality
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import logger

def test_tdc_basic():
    """Test basic TDC functionality without complex models"""
    
    # Set up environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use 2 GPUs for testing
    
    # Initialize distributed training
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logger.info(f"🔍 [Test] Starting basic TDC test on rank {rank}/{world_size}")
    
    try:
        # Test basic TDC imports and functionality
        from torch.distributed.checkpoint import save as tdc_save, load as tdc_load
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        logger.info(f"✅ [Test] TDC imports successful on rank {rank}")
        
        # Create a simple model for testing
        model = torch.nn.Linear(10, 5).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create some dummy data
        x = torch.randn(2, 10).cuda()
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        logger.info(f"✅ [Test] Simple model created and trained on rank {rank}")
        
        # Test TDC state dict collection
        try:
            model_state_dict, optim_state_dict = get_state_dict(
                model=model,
                optimizers=optimizer,
            )
            logger.info(f"✅ [Test] TDC get_state_dict successful on rank {rank}")
        except Exception as e:
            logger.error(f"❌ [Test] TDC get_state_dict failed on rank {rank}: {e}")
            return False
        
        # Create a temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🔍 [Test] Testing TDC save to: {checkpoint_path}")
            
            # Test TDC save
            try:
                state_dict = {
                    "model": model_state_dict,
                    "optimizer": optim_state_dict,
                }
                
                tdc_save(
                    state_dict=state_dict,
                    checkpoint_id=checkpoint_path,
                )
                logger.info(f"✅ [Test] TDC save successful on rank {rank}")
            except Exception as e:
                logger.error(f"❌ [Test] TDC save failed on rank {rank}: {e}")
                return False
            
            # Synchronize all ranks
            dist.barrier()
            
            # Test TDC load
            try:
                # Create new model and optimizer
                model2 = torch.nn.Linear(10, 5).cuda()
                optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
                
                # Get state dict for loading
                model_state_dict2, optim_state_dict2 = get_state_dict(
                    model=model2,
                    optimizers=optimizer2,
                )
                
                state_dict2 = {
                    "model": model_state_dict2,
                    "optimizer": optim_state_dict2,
                }
                
                tdc_load(
                    state_dict=state_dict2,
                    checkpoint_id=checkpoint_path,
                )
                
                # Set the loaded state
                set_state_dict(
                    model=model2,
                    optimizers=optimizer2,
                    model_state_dict=state_dict2["model"],
                    optim_state_dict=state_dict2["optimizer"],
                )
                
                logger.info(f"✅ [Test] TDC load successful on rank {rank}")
            except Exception as e:
                logger.error(f"❌ [Test] TDC load failed on rank {rank}: {e}")
                return False
            
            # Synchronize all ranks
            dist.barrier()
            
            if rank == 0:
                logger.info("🎉 [Test] Basic TDC test completed successfully!")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ [Test] Basic TDC test failed on rank {rank}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Clean up distributed training
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    success = test_tdc_basic()
    sys.exit(0 if success else 1)
