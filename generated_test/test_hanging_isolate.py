#!/usr/bin/env python3
"""
Isolate hanging issues by testing individual components
"""

import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
import time
import signal

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
    cleanup_distributed()
    sys.exit(1)


def test_barrier_sync():
    """Test if distributed barriers are working"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🔄 Rank {rank}: Testing barrier synchronization...")
    
    try:
        # Test basic barrier
        logger.info(f"🔄 Rank {rank}: Calling dist.barrier()...")
        dist.barrier()
        logger.info(f"✅ Rank {rank}: Barrier passed")
        
        # Test barrier with different delays
        time.sleep(rank * 0.1)  # Different delays per rank
        logger.info(f"🔄 Rank {rank}: Calling dist.barrier() with delays...")
        dist.barrier()
        logger.info(f"✅ Rank {rank}: Delayed barrier passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rank {rank}: Barrier test failed: {e}")
        return False
    finally:
        cleanup_distributed()


def test_trainer_creation():
    """Test if trainer creation hangs"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🔄 Rank {rank}: Testing trainer creation...")
    
    try:
        config = EngineConfig.from_yaml("engineBase.yaml")
        unique_name = f"test_hanging_{int(time.time())}"
        
        logger.info(f"🔄 Rank {rank}: Creating EngineFSDP...")
        trainer = EngineFSDP(unique_name, config, inference_mode=False)
        logger.info(f"✅ Rank {rank}: EngineFSDP created successfully")
        
        # Test basic operations
        logger.info(f"🔄 Rank {rank}: Testing model forward pass...")
        vocabulary_size = trainer.model.config.vocab_size
        dummy_input = torch.randint(0, vocabulary_size, (1, 128)).to(trainer.device)
        
        with torch.no_grad():
            outputs = trainer.model(input_ids=dummy_input)
        
        logger.info(f"✅ Rank {rank}: Model forward pass successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rank {rank}: Trainer creation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        cleanup_distributed()


def test_checkpoint_save():
    """Test if checkpoint saving hangs"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🔄 Rank {rank}: Testing checkpoint save...")
    
    try:
        config = EngineConfig.from_yaml("engineBase.yaml")
        unique_name = f"test_checkpoint_{int(time.time())}"
        trainer = EngineFSDP(unique_name, config, inference_mode=False)
        
        logger.info(f"🔄 Rank {rank}: Performing minimal training...")
        vocabulary_size = trainer.model.config.vocab_size
        dummy_input = torch.randint(0, vocabulary_size, (1, 128)).to(trainer.device)
        dummy_labels = torch.randint(0, vocabulary_size, (1, 128)).to(trainer.device)

        trainer.model.train()
        outputs = trainer.model(input_ids=dummy_input, labels=dummy_labels)
        loss = outputs.loss
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.status.global_step = 1
        
        logger.info(f"🔄 Rank {rank}: Saving checkpoint...")
        trainer._save_checkpoint(1)
        logger.info(f"✅ Rank {rank}: Checkpoint saved successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rank {rank}: Checkpoint save test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        cleanup_distributed()


def test_checkpoint_load():
    """Test if checkpoint loading hangs"""
    rank, world_size, device = setup_distributed()
    
    logger.info(f"🔄 Rank {rank}: Testing checkpoint load...")
    
    try:
        config = EngineConfig.from_yaml("engineBase.yaml")
        unique_name = f"test_checkpoint_{int(time.time())}"
        trainer = EngineFSDP(unique_name, config, inference_mode=False)
        
        # First save a checkpoint
        logger.info(f"🔄 Rank {rank}: Creating checkpoint to load...")
        vocabulary_size = trainer.model.config.vocab_size
        dummy_input = torch.randint(0, vocabulary_size, (1, 128)).to(trainer.device)
        dummy_labels = torch.randint(0, vocabulary_size, (1, 128)).to(trainer.device)

        trainer.model.train()
        outputs = trainer.model(input_ids=dummy_input, labels=dummy_labels)
        loss = outputs.loss
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.status.global_step = 1
        
        trainer._save_checkpoint(1)
        logger.info(f"✅ Rank {rank}: Checkpoint created")
        
        # Now test loading
        logger.info(f"🔄 Rank {rank}: Loading checkpoint...")
        checkpoint_location = trainer.checkpoint_path / 'checkpoint-1'
        trainer._load_checkpoint(checkpoint_location)
        logger.info(f"✅ Rank {rank}: Checkpoint loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rank {rank}: Checkpoint load test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        cleanup_distributed()


def main():
    """Main function to run isolation tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Isolate hanging issues")
    parser.add_argument("--test", choices=["barrier", "trainer", "save", "load", "all"], 
                       default="all", help="Which test to run")
    args = parser.parse_args()
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minute timeout
    
    try:
        if args.test == "barrier" or args.test == "all":
            logger.info("🧪 Testing barrier synchronization...")
            success = test_barrier_sync()
            if not success:
                logger.error("❌ Barrier test failed!")
                return
        
        if args.test == "trainer" or args.test == "all":
            logger.info("🧪 Testing trainer creation...")
            success = test_trainer_creation()
            if not success:
                logger.error("❌ Trainer creation test failed!")
                return
        
        if args.test == "save" or args.test == "all":
            logger.info("🧪 Testing checkpoint save...")
            success = test_checkpoint_save()
            if not success:
                logger.error("❌ Checkpoint save test failed!")
                return
        
        if args.test == "load" or args.test == "all":
            logger.info("🧪 Testing checkpoint load...")
            success = test_checkpoint_load()
            if not success:
                logger.error("❌ Checkpoint load test failed!")
                return
        
        logger.info("🎉 All isolation tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Isolation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        signal.alarm(0)  # Cancel timeout


if __name__ == "__main__":
    main()
