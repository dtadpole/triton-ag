#!/usr/bin/env python3
"""
Debug script to see exact vLLM completions response format
"""

import asyncio
import httpx
import json

async def debug_vllm_response():
    """Debug vLLM completions API response format."""
    base_url = "http://10.12.0.204:8091/v1"
    completions_url = f"{base_url}/completions"
    
    payload = {
        "model": "Qwen/Qwen3-14B",
        "prompt": "Hello world",
        "temperature": 0.1,
        "max_tokens": 10,
        "stream": False,
        "echo": False,
        # Enable both types of logprobs
        "logprobs": 1,
        "prompt_logprobs": 1
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy_key"
    }
    
    try:
        print("Sending request to vLLM...")
        print(f"URL: {completions_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                completions_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
        print("\n" + "="*80)
        print("FULL RESPONSE:")
        print("="*80)
        print(json.dumps(data, indent=2))
        
        print("\n" + "="*80)
        print("LOGPROBS ANALYSIS:")
        print("="*80)
        
        choices = data.get('choices', [])
        if choices:
            choice = choices[0]
            logprobs_data = choice.get('logprobs')
            
            if logprobs_data:
                print("Logprobs data found!")
                print(f"Keys in logprobs: {list(logprobs_data.keys())}")
                
                # Output logprobs
                if 'tokens' in logprobs_data:
                    print(f"\nOutput tokens: {logprobs_data['tokens']}")
                if 'token_logprobs' in logprobs_data:
                    print(f"Output token_logprobs: {logprobs_data['token_logprobs']}")
                if 'top_logprobs' in logprobs_data:
                    print(f"Output top_logprobs: {logprobs_data['top_logprobs']}")
                
                # Prompt logprobs
                if 'prompt_logprobs' in logprobs_data:
                    prompt_logprobs = logprobs_data['prompt_logprobs']
                    print(f"\nPrompt logprobs type: {type(prompt_logprobs)}")
                    print(f"Prompt logprobs: {prompt_logprobs}")
                    
                    if isinstance(prompt_logprobs, list) and prompt_logprobs:
                        print(f"First prompt logprob entry type: {type(prompt_logprobs[0])}")
                        print(f"First prompt logprob entry: {prompt_logprobs[0]}")
                else:
                    print("\nNo prompt_logprobs found in response")
            else:
                print("No logprobs data found in response")
        else:
            print("No choices found in response")
            
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(debug_vllm_response()) 