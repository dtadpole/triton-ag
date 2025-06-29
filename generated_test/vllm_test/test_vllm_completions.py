#!/usr/bin/env python3
"""
Test script for vLLM completions API with logprobs
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sequential_inference import VLLMClient

async def test_vllm_completions():
    """Test vLLM completions API functionality."""
    print("Testing vLLM completions API...")
    
    # Create client
    client = VLLMClient()
    
    # Test simple generation
    test_prompt = "def factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n\n# Optimize this code for better performance"
    
    try:
        print("Testing basic generation...")
        result = await client.generate(
            prompt=test_prompt,
            max_tokens=100,
            temperature=0.1
        )
        
        print(f"Generated text: {result['text'][:200]}...")
        print(f"Input logprobs count: {len(result.get('input_logprobs', []))}")
        print(f"Output logprobs count: {len(result.get('output_logprobs', []))}")
        
        # Print first few logprobs for debugging
        if result.get('output_logprobs'):
            print("\nFirst few output logprobs:")
            for i, logprob in enumerate(result['output_logprobs'][:3]):
                print(f"  {i}: {logprob}")
        
        if result.get('input_logprobs'):
            print("\nFirst few input logprobs:")
            for i, logprob in enumerate(result['input_logprobs'][:3]):
                print(f"  {i}: {logprob}")
                
        print("\n✓ vLLM completions API test successful!")
        return True
        
    except Exception as e:
        print(f"✗ vLLM completions API test failed: {e}")
        return False

async def test_health_check():
    """Test vLLM health check."""
    print("\nTesting vLLM health check...")
    
    client = VLLMClient()
    
    try:
        is_healthy = await client.health_check()
        if is_healthy:
            print("✓ vLLM server is healthy")
        else:
            print("✗ vLLM server health check failed")
        return is_healthy
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_models():
    """Test getting available models."""
    print("\nTesting model discovery...")
    
    client = VLLMClient()
    
    try:
        models = client.get_models()
        print(f"Available models: {models}")
        print("✓ Model discovery successful")
        return True
    except Exception as e:
        print(f"✗ Model discovery failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("=== vLLM Completions API Test Suite ===\n")
    
    # Test 1: Health check
    health_ok = await test_health_check()
    
    # Test 2: Model discovery  
    models_ok = test_models()
    
    # Test 3: Completions API
    completions_ok = await test_vllm_completions()
    
    print("\n=== Test Results ===")
    print(f"Health check: {'✓' if health_ok else '✗'}")
    print(f"Model discovery: {'✓' if models_ok else '✗'}")
    print(f"Completions API: {'✓' if completions_ok else '✗'}")
    
    if all([health_ok, models_ok, completions_ok]):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 