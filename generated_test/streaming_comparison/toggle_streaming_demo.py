#!/usr/bin/env python3
"""
Demonstration of Streaming vs Non-Streaming Toggle
==================================================

This demo shows how to use the global STREAMING flag to switch between:
- STREAMING = True  -> Real-time token streaming
- STREAMING = False -> Single complete response

Usage: python toggle_streaming_demo.py
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Add parent directory to path to import sequential_inference
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import and modify the STREAMING flag
import sequential_inference

# Test configuration
TEST_CODE = """
import triton

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""

async def test_mode(mode_name: str, streaming: bool):
    """Test a specific mode (streaming or non-streaming)"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {mode_name}")
    print(f"{'='*60}")
    
    # Set the global streaming flag
    sequential_inference.STREAMING = streaming
    
    # Create client
    client = sequential_inference.VLLMClient(
        client_type="vllm", 
        config_file="sequential_inference.yaml"
    )
    
    # Test generation
    print(f"🔍 Generating response using {mode_name}...")
    start_time = time.time()
    
    try:
        result = await client.generate(
            source_code=TEST_CODE,
            max_tokens=200,  # Keep it short for demo
            temperature=0.7
        )
        
        elapsed = time.time() - start_time
        generated_text = result.get('text', '')
        tokens = result.get('tokens', [])
        
        print(f"✅ {mode_name} completed!")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📊 Tokens: {len(tokens)}")
        print(f"📝 Characters: {len(generated_text)}")
        print(f"📖 Preview: {generated_text[:100]}...")
        
        return {
            'mode': mode_name,
            'streaming': streaming,
            'time': elapsed,
            'tokens': len(tokens),
            'characters': len(generated_text),
            'text': generated_text
        }
        
    except Exception as e:
        print(f"❌ {mode_name} failed: {e}")
        return None

async def main():
    """Compare streaming vs non-streaming modes"""
    print("🚀 vLLM Streaming vs Non-Streaming Comparison")
    print("Make sure your vLLM server is running!")
    print()
    
    # Test both modes
    results = []
    
    # Test streaming mode
    streaming_result = await test_mode("STREAMING MODE", True)
    if streaming_result:
        results.append(streaming_result)
    
    # Test non-streaming mode
    non_streaming_result = await test_mode("NON-STREAMING MODE", False)
    if non_streaming_result:
        results.append(non_streaming_result)
    
    # Compare results
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("📊 COMPARISON RESULTS")
        print(f"{'='*60}")
        
        streaming_res = results[0]
        non_streaming_res = results[1]
        
        print(f"Mode              | Time    | Tokens | Characters")
        print(f"------------------|---------|--------|----------")
        print(f"Streaming         | {streaming_res['time']:6.2f}s | {streaming_res['tokens']:6d} | {streaming_res['characters']:10d}")
        print(f"Non-streaming     | {non_streaming_res['time']:6.2f}s | {non_streaming_res['tokens']:6d} | {non_streaming_res['characters']:10d}")
        
        # Check consistency
        if streaming_res['text'] == non_streaming_res['text']:
            print("\n🎉 Both modes produced IDENTICAL results!")
        else:
            print("\n⚠️  Results differ between modes - this might be expected due to randomness")
            
        # Performance comparison
        time_diff = abs(streaming_res['time'] - non_streaming_res['time'])
        if streaming_res['time'] < non_streaming_res['time']:
            print(f"⚡ Streaming was {time_diff:.2f}s faster")
        else:
            print(f"📄 Non-streaming was {time_diff:.2f}s faster")
    
    print(f"\n{'='*60}")
    print("💡 How to Toggle Modes:")
    print("In sequential_inference.py, change the global flag:")
    print("  STREAMING = True   # For real-time streaming")
    print("  STREAMING = False  # For single response")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main()) 