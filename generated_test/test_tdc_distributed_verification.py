#!/usr/bin/env python3
"""
Multi-process distributed test to verify TDC checkpoint functionality across ranks
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path
import time
import subprocess
import multiprocessing as mp

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from logger import logger
from engineFSDP import EngineFSDP
from engineBase import EngineConfig


def setup_distributed(rank, world_size, backend="nccl"):
    """Initialize distributed process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_worker(rank, world_size, checkpoint_path, results_queue):
    """Worker function for each rank"""
    try:
        # Setup distributed
        setup_distributed(rank, world_size)
        
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
        
        # Collect results
        results = {
            'rank': rank,
            'trained_global_step': trained_global_step,
            'loaded_global_step': loaded_global_step,
            'global_step_match': trained_global_step == loaded_global_step,
            'output_match': output_match,
            'success': (trained_global_step == loaded_global_step) and output_match
        }
        
        results_queue.put(results)
        
        # Cleanup
        cleanup_distributed()
        
    except Exception as e:
        logger.error(f"❌ Rank {rank}: Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results_queue.put({'rank': rank, 'error': str(e), 'success': False})
        cleanup_distributed()


def test_tdc_distributed_verification():
    """Test TDC checkpoint functionality across multiple ranks"""
    logger.info("🧪 Testing TDC checkpoint verification across multiple ranks...")
    
    # Use 2 ranks for testing
    world_size = 2
    
    # Create temporary checkpoint directory
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_distributed_checkpoint"
        checkpoint_path.mkdir(exist_ok=True)
        
        logger.info(f"💾 Checkpoint path: {checkpoint_path}")
        
        # Create a queue to collect results from all ranks
        results_queue = mp.Queue()
        
        # Create and start processes for each rank
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=run_worker, args=(rank, world_size, checkpoint_path, results_queue))
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results from all ranks
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Sort results by rank
        results.sort(key=lambda x: x['rank'])
        
        # Analyze results
        logger.info("📋 Distributed TDC Test Results:")
        
        all_success = True
        for result in results:
            rank = result['rank']
            if 'error' in result:
                logger.error(f"❌ Rank {rank}: Failed with error: {result['error']}")
                all_success = False
            else:
                logger.info(f"📊 Rank {rank}:")
                logger.info(f"   Trained global step: {result['trained_global_step']}")
                logger.info(f"   Loaded global step: {result['loaded_global_step']}")
                logger.info(f"   Global step match: {result['global_step_match']}")
                logger.info(f"   Output match: {result['output_match']}")
                logger.info(f"   Success: {result['success']}")
                
                if not result['success']:
                    all_success = False
        
        # Check consistency across ranks
        if len(results) >= 2:
            rank0_loaded_step = results[0].get('loaded_global_step', -1)
            rank1_loaded_step = results[1].get('loaded_global_step', -1)
            
            if rank0_loaded_step == rank1_loaded_step:
                logger.info("✅ Global step consistency across ranks: PASSED")
            else:
                logger.error(f"❌ Global step consistency across ranks: FAILED ({rank0_loaded_step} vs {rank1_loaded_step})")
                all_success = False
        
        if all_success:
            logger.info("🎉 All distributed TDC checkpoint tests passed!")
        else:
            logger.error("❌ Some distributed TDC checkpoint tests failed!")
        
        return all_success


if __name__ == "__main__":
    try:
        # Test distributed TDC checkpoint verification
        success = test_tdc_distributed_verification()
        
        if success:
            logger.info("🎉 Distributed TDC checkpoint verification test passed!")
            sys.exit(0)
        else:
            logger.error("❌ Distributed TDC checkpoint verification test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
