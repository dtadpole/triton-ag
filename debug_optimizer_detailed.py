#!/usr/bin/env python3
"""
Detailed debug script to investigate optimizer state issues in TDC
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


def debug_optimizer_detailed():
    """Detailed debug of optimizer state handling in TDC"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🚀 Rank {rank}/{world_size}: Starting detailed optimizer state debug")
    
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
    unique_name = f"debug_optimizer_detailed_{int(time.time())}"
    trainer = EngineFSDP(unique_name, config, inference_mode=False)
    
    # Force distributed mode
    trainer.use_distributed = True
    trainer.rank = rank
    trainer.world_size = world_size
    
    logger.info(f"🔄 Rank {rank}: EngineFSDP created successfully")
    
    # Perform some training to create optimizer state
    logger.info(f"🏋️ Rank {rank}: Performing training to create optimizer state...")
    
    # Create dummy data
    dummy_input = torch.randint(0, 1000, (1, 10)).to(trainer.device)
    dummy_labels = torch.randint(0, 1000, (1, 10)).to(trainer.device)
    
    # Perform training steps to create optimizer state
    for step in range(2):
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
    
    # Debug optimizer state BEFORE saving
    logger.info(f"🔍 Rank {rank}: Optimizer state BEFORE saving:")
    optimizer_state_count = 0
    for group_idx, group in enumerate(trainer.optimizer.param_groups):
        for param_idx, param in enumerate(group['params']):
            if param in trainer.optimizer.state:
                state = trainer.optimizer.state[param]
                optimizer_state_count += len(state)
                logger.info(f"   Group {group_idx}, Param {param_idx}: {list(state.keys())}")
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"     {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
    
    logger.info(f"🔍 Rank {rank}: Total optimizer state entries: {optimizer_state_count}")
    
    # Test TDC save/load directly
    logger.info(f"🧪 Rank {rank}: Testing TDC save/load directly...")
    
    # Create checkpoint path
    checkpoint_path = Path("/tmp/debug_optimizer_detailed")
    checkpoint_path.mkdir(exist_ok=True)
    
    # Import TDC functions
    from torch.distributed.checkpoint import save as tdc_save, load as tdc_load
    from engineFSDP import MetadataWrapper
    
    # Create metadata wrapper
    metadata_wrapper = MetadataWrapper(
        global_step=trainer.status.global_step,
        config=trainer.config.model_dump()
    )
    
    # Prepare state dict
    state_dict = {
        "model": trainer.model,
        "optimizer": trainer.optimizer,
        "scheduler": trainer.scheduler,
        "metadata": metadata_wrapper,
    }
    
    # Save using TDC
    logger.info(f"💾 Rank {rank}: Saving with TDC directly...")
    tdc_save(
        state_dict=state_dict,
        checkpoint_id=checkpoint_path,
    )
    logger.info(f"💾 Rank {rank}: TDC save completed")
    
    # Clear optimizer state
    logger.info(f"🔄 Rank {rank}: Clearing optimizer state...")
    trainer.optimizer.zero_grad()
    for group in trainer.optimizer.param_groups:
        for p in group['params']:
            if p in trainer.optimizer.state:
                trainer.optimizer.state[p].clear()
    
    # Debug optimizer state AFTER clearing
    logger.info(f"🔍 Rank {rank}: Optimizer state AFTER clearing:")
    optimizer_state_count_after_clear = 0
    for group_idx, group in enumerate(trainer.optimizer.param_groups):
        for param_idx, param in enumerate(group['params']):
            if param in trainer.optimizer.state:
                state = trainer.optimizer.state[param]
                optimizer_state_count_after_clear += len(state)
                logger.info(f"   Group {group_idx}, Param {param_idx}: {list(state.keys())}")
    
    logger.info(f"🔍 Rank {rank}: Total optimizer state entries after clear: {optimizer_state_count_after_clear}")
    
    # Create new metadata wrapper for loading
    loaded_metadata = MetadataWrapper(global_step=0, config={})
    
    # Prepare state dict for loading
    state_dict_load = {
        "model": trainer.model,
        "optimizer": trainer.optimizer,
        "scheduler": trainer.scheduler,
        "metadata": loaded_metadata,
    }
    
    # Load using TDC
    logger.info(f"📂 Rank {rank}: Loading with TDC directly...")
    tdc_load(
        state_dict=state_dict_load,
        checkpoint_id=checkpoint_path,
    )
    logger.info(f"📂 Rank {rank}: TDC load completed")
    
    # Debug optimizer state AFTER loading
    logger.info(f"🔍 Rank {rank}: Optimizer state AFTER loading:")
    optimizer_state_count_after_load = 0
    for group_idx, group in enumerate(trainer.optimizer.param_groups):
        for param_idx, param in enumerate(group['params']):
            if param in trainer.optimizer.state:
                state = trainer.optimizer.state[param]
                optimizer_state_count_after_load += len(state)
                logger.info(f"   Group {group_idx}, Param {param_idx}: {list(state.keys())}")
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"     {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
    
    logger.info(f"🔍 Rank {rank}: Total optimizer state entries after load: {optimizer_state_count_after_load}")
    
    # Summary
    logger.info(f"📊 Rank {rank}: Optimizer State Summary:")
    logger.info(f"   Before save: {optimizer_state_count} entries")
    logger.info(f"   After clear: {optimizer_state_count_after_clear} entries")
    logger.info(f"   After load: {optimizer_state_count_after_load} entries")
    logger.info(f"   Restored: {optimizer_state_count_after_load == optimizer_state_count}")
    
    cleanup_distributed()


def main():
    """Main function"""
    try:
        debug_optimizer_detailed()
        logger.info("🎉 Detailed optimizer state debug completed!")
        sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Debug failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cleanup_distributed()
        sys.exit(1)


if __name__ == "__main__":
    main()
