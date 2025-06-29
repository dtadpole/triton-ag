#!/usr/bin/env python3
"""
OpenAI Client Streaming Demo for vLLM
=====================================

This demo shows how the new OpenAI client implementation works for both:
- STREAMING = True  -> Real-time token streaming with OpenAI client
- STREAMING = False -> Single complete response with OpenAI client

Usage: python openai_streaming_demo.py
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
import triton.language as tl

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
    
def add(x, y):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    N = output.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, N, BLOCK_SIZE=1024)
    return output
"""

async def test_openai_streaming(streaming_mode: bool):
    """Test OpenAI client with streaming or non-streaming"""
    mode_name = "STREAMING" if streaming_mode else "NON-STREAMING"
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing OpenAI Client - {mode_name} Mode")
    print(f"{'='*60}")
    
    # Set the global streaming flag
    sequential_inference.STREAMING = streaming_mode
    
    # Create client (this will use OpenAI client internally)
    client = sequential_inference.VLLMClient(
        client_type="vllm", 
        config_file="sequential_inference.yaml"
    )
    
    # Test generation
    print(f"🔍 Generating response using OpenAI client ({mode_name})...")
    start_time = time.time()
    
    try:
        result = await client.generate(
            source_code=TEST_CODE,
            max_tokens=300,
            temperature=0.7
        )
        
        elapsed = time.time() - start_time
        generated_text = result.get('text', '')
        tokens = result.get('tokens', [])
        
        print(f"✅ OpenAI {mode_name} completed!")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📊 Tokens: {len(tokens)}")
        print(f"📝 Characters: {len(generated_text)}")
        print(f"📖 Preview: {generated_text[:150]}...")
        
        return {
            'mode': mode_name,
            'streaming': streaming_mode,
            'time': elapsed,
            'tokens': len(tokens),
            'characters': len(generated_text),
            'text': generated_text,
            'client': 'OpenAI'
        }
        
    except Exception as e:
        print(f"❌ OpenAI {mode_name} failed: {e}")
        return None

async def main():
    """Test both OpenAI streaming and non-streaming modes"""
    print("🚀 vLLM with OpenAI Client - Streaming vs Non-Streaming")
    print("Make sure your vLLM server is running on localhost:8000!")
    print()
    
    # Test both modes
    results = []
    
    # Test streaming mode
    print("🌊 Testing STREAMING mode with OpenAI client...")
    streaming_result = await test_openai_streaming(True)
    if streaming_result:
        results.append(streaming_result)
    
    # Test non-streaming mode
    print("📄 Testing NON-STREAMING mode with OpenAI client...")
    non_streaming_result = await test_openai_streaming(False)
    if non_streaming_result:
        results.append(non_streaming_result)
    
    # Compare results
    if len(results) >= 1:
        print(f"\n{'='*60}")
        print("📊 OPENAI CLIENT RESULTS SUMMARY")
        print(f"{'='*60}")
        
        for result in results:
            print(f"{result['mode']:15} | {result['time']:6.2f}s | {result['tokens']:6d} tokens | {result['characters']:6d} chars")
        
        # Compare if we have both results
        if len(results) == 2:
            streaming_res = results[0]
            non_streaming_res = results[1]
            
            # Check consistency
            if streaming_res['text'] == non_streaming_res['text']:
                print("\n🎉 Both OpenAI modes produced IDENTICAL results!")
            else:
                print("\n⚠️  Results differ between modes (expected due to randomness)")
                
            # Performance comparison
            time_diff = abs(streaming_res['time'] - non_streaming_res['time'])
            if streaming_res['time'] < non_streaming_res['time']:
                print(f"⚡ OpenAI streaming was {time_diff:.2f}s faster")
            else:
                print(f"📄 OpenAI non-streaming was {time_diff:.2f}s faster")
    
    print(f"\n{'='*60}")
    print("🔧 NEW IMPLEMENTATION BENEFITS:")
    print("✅ Uses official OpenAI Python client")
    print("✅ Automatic streaming handling (no manual SSE parsing)")
    print("✅ Built-in error handling and retries")
    print("✅ Cleaner, more maintainable code")
    print("✅ Better compatibility with OpenAI ecosystem")
    print()
    print("💡 Toggle modes by changing in sequential_inference.py:")
    print("  STREAMING = True   # OpenAI streaming mode")
    print("  STREAMING = False  # OpenAI non-streaming mode")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main()) 