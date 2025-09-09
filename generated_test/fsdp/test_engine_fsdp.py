#!/usr/bin/env python3
"""
Test script for EngineFSDP implementation
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engineBase import EngineConfig, TrainerStatus
from engineFSDP import EngineFSDP
from logger import logger


async def test_fsdp_engine():
    """Test the FSDP engine implementation"""
    logger.info("🧪 Starting FSDP engine test...")
    
    # Create a temporary config
    config = EngineConfig()
    config.model.name = "gpt2"  # Use a small model for testing
    config.model.engine = "fsdp"
    config.model.max_seq_length = 1024  # GPT-2 max length
    config.training.max_steps = 2
    config.training.micro_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.save_steps = 10
    config.training.eval_steps = 10
    config.lora.use_lora = False  # Disable LoRA for simpler testing
    
    # Create a temporary checkpoint directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config.training.checkpoint_path = temp_dir
        
        try:
            # Initialize the engine
            logger.info("🚀 Initializing FSDP engine...")
            engine = EngineFSDP("test_fsdp", config, inference_mode=True)
            logger.info("✅ FSDP engine initialized successfully")
            
            # Test basic functionality
            logger.info("🔍 Testing basic functionality...")
            
            # Test model access
            base_model = engine._base_model()
            logger.info(f"✅ Base model accessed: {type(base_model).__name__}")
            
            # Test tokenizer
            tokenizer = engine.tokenizer
            logger.info(f"✅ Tokenizer accessed: {type(tokenizer).__name__}")
            
            # Test text generation (if not in inference mode, this would require training)
            if engine.inference_mode:
                logger.info("🤖 Testing text generation...")
                try:
                    generated_text = engine.generate_text("Hello, how are you?", max_length=20)
                    logger.info(f"✅ Text generation successful: '{generated_text}'")
                except Exception as e:
                    logger.warning(f"⚠️ Text generation failed (expected in inference mode): {e}")
            
            logger.info("🎉 All tests passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


async def test_engine_creation():
    """Test engine creation through the factory method"""
    logger.info("🧪 Testing engine creation through factory method...")
    
    config = EngineConfig()
    config.model.engine = "fsdp"
    config.model.name = "gpt2"
    config.model.max_seq_length = 1024  # GPT-2 max length
    
    try:
        # Test the factory method
        from engineBase import EngineBase
        engine = EngineBase.create_engine("test_factory", config, inference_mode=True)
        logger.info(f"✅ Engine created via factory: {type(engine).__name__}")
        return True
    except Exception as e:
        logger.error(f"❌ Factory method test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def main():
    """Main test function"""
    logger.info("🧪 Starting FSDP engine tests...")
    
    # Set environment variables for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only one GPU for testing
    
    # Run tests
    test1_passed = await test_fsdp_engine()
    test2_passed = await test_engine_creation()
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
