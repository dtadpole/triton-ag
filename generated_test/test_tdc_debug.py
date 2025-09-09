#!/usr/bin/env python3
"""
Debug version of TDC test to identify hanging issues
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path
import time
import hashlib
import json
import signal
import threading

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


def timeout_handler(signum, frame):
    """Handle timeout signals"""
    logger.error("⏰ TIMEOUT: Operation took too long!")
    import traceback
    logger.error(traceback.format_stack(frame))
    cleanup_distributed()
    sys.exit(1)


def safe_operation(operation_name, operation_func, timeout=300):
    """Safely execute an operation with timeout"""
    rank = dist.get_rank()
    logger.info(f"🔄 Rank {rank}: Starting {operation_name}...")
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        start_time = time.time()
        result = operation_func()
        elapsed = time.time() - start_time
        logger.info(f"✅ Rank {rank}: {operation_name} completed in {elapsed:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"❌ Rank {rank}: {operation_name} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)  # Cancel timeout


def get_model_state_hash(model):
    """Get hash of model parameters for consistency checking in FSDP"""
    hasher = hashlib.md5()
    rank = dist.get_rank()
    
    # Use LOCAL_STATE_DICT for simplicity
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import StateDictType
    
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        model_state_dict = model.state_dict()
        
        # Count parameters for logging
        num_params = 0
        for p in model_state_dict.values():
            if isinstance(p, torch.Tensor):
                num_params += p.numel()
        logger.info(f"🔍 Rank {rank}: Model has [{num_params:,}] parameters in state dict")
        
        # Sort by parameter names for consistent ordering across ranks
        for param_name in sorted(model_state_dict.keys()):
            param_tensor = model_state_dict[param_name]
            
            if not isinstance(param_tensor, torch.Tensor):
                continue
                
            # Convert to float32 if needed for consistent hashing
            if param_tensor.dtype == torch.bfloat16:
                param_tensor = param_tensor.float()
            elif param_tensor.dtype == torch.float16:
                param_tensor = param_tensor.float()
                
            # Move to CPU and hash the tensor data
            hasher.update(param_tensor.cpu().numpy().tobytes())
    
    return hasher.hexdigest()


def test_tdc_debug():
    """Debug version of TDC test with timeout protection"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🚀 Rank {rank}/{world_size}: Starting debug TDC test")
    
    try:
        # Step 1: Create config and trainer
        def create_trainer():
            config = EngineConfig.from_yaml("engineBase.yaml")
            unique_name = f"test_tdc_debug_{int(time.time())}"
            return EngineFSDP(unique_name, config, inference_mode=False)
        
        trainer = safe_operation("Trainer Creation", create_trainer, timeout=60)
        logger.info(f"🔄 Rank {rank}: EngineFSDP created successfully")
        
        # Step 2: Perform minimal training
        def perform_training():
            vocabulary_size = trainer.model.config.vocab_size
            
            # Only do 5 steps instead of 20
            for step in range(5):
                dummy_input = torch.randint(0, vocabulary_size, (1, 512)).to(trainer.device)  # Smaller sequence
                dummy_labels = torch.randint(0, vocabulary_size, (1, 512)).to(trainer.device)

                trainer.model.train()
                
                outputs = trainer.model(input_ids=dummy_input, labels=dummy_labels)
                loss = outputs.loss
                
                loss.backward()
                trainer.optimizer.step()
                trainer.scheduler.step()
                trainer.optimizer.zero_grad()
                
                trainer.status.global_step += 1
                
                logger.info(f"📈 Rank {rank}: Training step {step + 1}: loss={loss.item():.4f}")
        
        safe_operation("Training", perform_training, timeout=120)
        
        # Step 3: Calculate hashes
        def calculate_hashes():
            pre_save_global_step = trainer.status.global_step
            pre_save_model_hash = get_model_state_hash(trainer.model)
            return pre_save_global_step, pre_save_model_hash
        
        pre_save_global_step, pre_save_model_hash = safe_operation("Hash Calculation", calculate_hashes, timeout=60)
        
        logger.info(f"📊 Rank {rank}: PRE-SAVE State:")
        logger.info(f"   Global step: {pre_save_global_step}")
        logger.info(f"   Model hash: {pre_save_model_hash[:16]}...")
        
        # Step 4: Save checkpoint
        def save_checkpoint():
            trainer._save_checkpoint(1)
        
        safe_operation("Checkpoint Save", save_checkpoint, timeout=180)
        logger.info(f"💾 Rank {rank}: Checkpoint saved successfully")
        
        # Step 5: Modify state (simplified)
        def modify_state():
            # Reset global step
            trainer.status.global_step = 0
            
            # Modify model parameters (safer approach)
            with torch.no_grad():
                # Use state dict instead of direct parameter iteration
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.api import StateDictType
                
                with FSDP.state_dict_type(trainer.model, StateDictType.LOCAL_STATE_DICT):
                    state_dict = trainer.model.state_dict()
                    for name, param in state_dict.items():
                        if isinstance(param, torch.Tensor):
                            param.add_(torch.randn_like(param) * 0.01)
        
        safe_operation("State Modification", modify_state, timeout=60)
        
        # Step 6: Calculate modified hashes
        def calculate_modified_hashes():
            post_modify_global_step = trainer.status.global_step
            post_modify_model_hash = get_model_state_hash(trainer.model)
            return post_modify_global_step, post_modify_model_hash
        
        post_modify_global_step, post_modify_model_hash = safe_operation("Modified Hash Calculation", calculate_modified_hashes, timeout=60)
        
        logger.info(f"📊 Rank {rank}: POST-MODIFY State:")
        logger.info(f"   Global step: {post_modify_global_step}")
        logger.info(f"   Model hash: {post_modify_model_hash[:16]}...")
        
        # Step 7: Load checkpoint
        def load_checkpoint():
            checkpoint_location = trainer.checkpoint_path / 'checkpoint-1'
            trainer._load_checkpoint(checkpoint_location)
        
        safe_operation("Checkpoint Load", load_checkpoint, timeout=180)
        logger.info(f"📂 Rank {rank}: Checkpoint loaded successfully")
        
        # Step 8: Verify consistency
        def verify_consistency():
            post_load_global_step = trainer.status.global_step
            post_load_model_hash = get_model_state_hash(trainer.model)
            
            global_step_consistency = (pre_save_global_step == post_load_global_step)
            model_consistency = (pre_save_model_hash == post_load_model_hash)
            
            logger.info(f"🧪 Rank {rank}: Consistency Tests:")
            logger.info(f"   Global step consistency: {global_step_consistency}")
            logger.info(f"   Model consistency: {model_consistency}")
            
            return global_step_consistency and model_consistency
        
        success = safe_operation("Consistency Verification", verify_consistency, timeout=60)
        
        if success:
            logger.info(f"🎉 Rank {rank}: Debug test completed successfully!")
        else:
            logger.error(f"❌ Rank {rank}: Debug test failed!")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Rank {rank}: Debug test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        cleanup_distributed()


def main():
    """Main function"""
    try:
        success = test_tdc_debug()
        
        if success:
            logger.info("🎉 Debug TDC test completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Debug TDC test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
