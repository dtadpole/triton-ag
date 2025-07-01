#!/usr/bin/env python3
"""
Demonstration of Different Streaming Approaches for vLLM
========================================================

This file shows 4 different ways to implement streaming with vLLM:
1. aiter_lines() - Simple line-by-line (current approach)
2. aiter_text() - Manual buffering 
3. httpx-sse - Specialized SSE library
4. requests - Synchronous streaming

Usage: python streaming_methods_demo.py
"""

import asyncio
import json
import time
import httpx
# pip install httpx-sse
try:
    from httpx_sse import aconnect_sse
    HAS_SSE = True
except ImportError:
    HAS_SSE = False
import requests


# Test configuration
BASE_URL = "http://localhost:8000/v1"
MODEL = "your-model-name"
PROMPT = "Hello, write a short story about a robot."


async def method_1_aiter_lines():
    """Method 1: aiter_lines() - Current approach (simple and reliable)"""
    print("🔄 Method 1: aiter_lines()")
    
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": True,
        "max_tokens": 100,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    start_time = time.time()
    generated_text = ""
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{BASE_URL}/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data)
                            choices = chunk.get('choices', [])
                            if choices:
                                text = choices[0].get('text', '')
                                if text:
                                    generated_text += text
                        except json.JSONDecodeError:
                            continue
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"✅ Generated {len(generated_text)} characters in {elapsed:.2f}s")
    return generated_text


async def method_2_aiter_text():
    """Method 2: aiter_text() with manual buffering (potentially faster)"""
    print("🔄 Method 2: aiter_text() with buffering")
    
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": True,
        "max_tokens": 100,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    start_time = time.time()
    generated_text = ""
    buffer = ""
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{BASE_URL}/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                
                async for chunk in response.aiter_text():
                    buffer += chunk
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                choices = chunk_data.get('choices', [])
                                if choices:
                                    text = choices[0].get('text', '')
                                    if text:
                                        generated_text += text
                            except json.JSONDecodeError:
                                continue
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"✅ Generated {len(generated_text)} characters in {elapsed:.2f}s")
    return generated_text


async def method_3_httpx_sse():
    """Method 3: httpx-sse library (most robust for SSE)"""
    if not HAS_SSE:
        print("❌ Method 3: httpx-sse not available (pip install httpx-sse)")
        return None
    
    print("🔄 Method 3: httpx-sse library")
    
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": True,
        "max_tokens": 100,
    }
    
    headers = {
        "Content-Type": "application/json",
    }
    
    start_time = time.time()
    generated_text = ""
    
    try:
        async with httpx.AsyncClient() as client:
            async with aconnect_sse(client, "POST", f"{BASE_URL}/completions", json=payload, headers=headers) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.data == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(sse.data)
                        choices = chunk.get('choices', [])
                        if choices:
                            text = choices[0].get('text', '')
                            if text:
                                generated_text += text
                    except json.JSONDecodeError:
                        continue
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"✅ Generated {len(generated_text)} characters in {elapsed:.2f}s")
    return generated_text


def method_4_requests_sync():
    """Method 4: Synchronous requests library (simple but blocking)"""
    print("🔄 Method 4: requests (synchronous)")
    
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": True,
        "max_tokens": 100,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    start_time = time.time()
    generated_text = ""
    
    try:
        response = requests.post(f"{BASE_URL}/completions", json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                
                try:
                    chunk = json.loads(data)
                    choices = chunk.get('choices', [])
                    if choices:
                        text = choices[0].get('text', '')
                        if text:
                            generated_text += text
                except json.JSONDecodeError:
                    continue
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"✅ Generated {len(generated_text)} characters in {elapsed:.2f}s")
    return generated_text


async def main():
    """Compare all streaming methods"""
    print("📊 Comparing Streaming Methods for vLLM")
    print("="*50)
    
    # Test all async methods
    methods = [
        ("aiter_lines()", method_1_aiter_lines()),
        ("aiter_text()", method_2_aiter_text()),
    ]
    
    if HAS_SSE:
        methods.append(("httpx-sse", method_3_httpx_sse()))
    
    results = {}
    
    # Test async methods
    for name, coro in methods:
        print(f"\n🧪 Testing {name}...")
        result = await coro
        results[name] = result
        
    # Test sync method
    print(f"\n🧪 Testing requests (sync)...")
    results["requests"] = method_4_requests_sync()
    
    # Compare results
    print("\n📈 Results Summary:")
    print("="*50)
    for method, result in results.items():
        if result:
            print(f"✅ {method:15} - {len(result):4d} chars")
        else:
            print(f"❌ {method:15} - Failed")
    
    # Check if all results are identical
    successful_results = [r for r in results.values() if r is not None]
    if len(set(successful_results)) <= 1:
        print("\n🎉 All methods produced identical results!")
    else:
        print("\n⚠️  Methods produced different results - investigate!")


if __name__ == "__main__":
    print("🚀 vLLM Streaming Methods Comparison")
    print("Make sure your vLLM server is running on localhost:8000")
    print("Update MODEL variable with your actual model name")
    print()
    
    asyncio.run(main()) 