#!/usr/bin/env python3
"""
Example: Using GEPA with a vLLM-hosted model.

GEPA uses litellm under the hood, which supports OpenAI-compatible APIs.
vLLM exposes an OpenAI-compatible endpoint, so you can easily use it.

This example shows 3 ways to use your own hosted LLM:
1. Using litellm with custom base_url
2. Using a custom callable function
3. Using environment variables
"""

import os
import sys

# Append paths to sys.path at the END (after site-packages)
# - /workspace: for custom modules in workspace root
# - /workspace/gepa: for importing adapters directly (without gepa. prefix)
# This ensures installed packages (like gepa) take precedence.
if '/workspace' not in sys.path:
    sys.path.append('/workspace')
if '/workspace/gepa' not in sys.path:
    sys.path.append('/workspace/gepa')


# =============================================================================
# METHOD 1: Using litellm with custom base_url (Recommended)
# =============================================================================
def method1_litellm_base_url():
    """
    Use litellm with a custom base_url pointing to your vLLM server.

    This is the cleanest approach - just set environment variables
    and GEPA will work automatically.
    """
    import gepa

    # Set the base URL for your vLLM server
    # vLLM typically runs on port 8000 with /v1 endpoint
    os.environ["OPENAI_API_BASE"] = "http://your-vllm-server:8000/v1"
    os.environ["OPENAI_API_KEY"] = "dummy"  # vLLM doesn't need a real key

    # Your model name as registered in vLLM
    # Use "openai/" prefix to tell litellm to use OpenAI-compatible API
    VLLM_MODEL = "openai/Qwen/Qwen2.5-32B-Instruct"

    # Example dataset
    trainset = [
        {"input": "What is 2+2?", "additional_context": {}, "answer": "4"},
        {"input": "What is 3+3?", "additional_context": {}, "answer": "6"},
    ]

    seed_candidate = {"system_prompt": "You are a helpful math assistant. Be concise."}

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=trainset,  # Use same for demo
        task_lm=VLLM_MODEL,  # Your vLLM model for task execution
        reflection_lm=VLLM_MODEL,  # Can use same or different model
        max_metric_calls=50,
    )

    print("Best prompt:", result.best_candidate)
    return result


# =============================================================================
# METHOD 2: Using a custom callable (Most flexible)
# =============================================================================
def method2_custom_callable():
    """
    Use a custom function to call your LLM.

    This gives you full control over how the LLM is called.
    Useful when you have special authentication, headers, or processing.
    """
    import gepa
    import requests
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    VLLM_URL = "http://your-vllm-server:8000/v1/chat/completions"
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

    def call_vllm(messages):
        """Custom function to call vLLM."""
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # Create adapter with custom callable
    adapter = DefaultAdapter(model=call_vllm)

    trainset = [
        {"input": "What is 2+2?", "additional_context": {}, "answer": "4"},
        {"input": "What is 3+3?", "additional_context": {}, "answer": "6"},
    ]

    seed_candidate = {"system_prompt": "You are a helpful math assistant."}

    # For reflection_lm, you can also use a custom callable
    def reflection_lm(prompt: str) -> str:
        """Custom reflection LM."""
        return call_vllm([{"role": "user", "content": prompt}])

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=trainset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=50,
    )

    print("Best prompt:", result.best_candidate)
    return result


# =============================================================================
# METHOD 3: Using litellm with explicit api_base (Per-call configuration)
# =============================================================================
def method3_litellm_explicit():
    """
    Configure litellm with explicit api_base per call.

    Useful when you want to use different servers for different models.
    """
    import gepa
    import litellm
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    VLLM_BASE_URL = "http://your-vllm-server:8000/v1"
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

    def call_vllm_litellm(messages):
        """Call vLLM using litellm with explicit api_base."""
        response = litellm.completion(
            model=f"openai/{MODEL_NAME}",
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            api_base=VLLM_BASE_URL,
            api_key="dummy",  # vLLM doesn't need real key
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content

    adapter = DefaultAdapter(model=call_vllm_litellm)

    def reflection_lm(prompt: str) -> str:
        response = litellm.completion(
            model=f"openai/{MODEL_NAME}",
            messages=[{"role": "user", "content": prompt}],
            api_base=VLLM_BASE_URL,
            api_key="dummy",
            max_tokens=4096,
            temperature=1.0,  # Higher temp for diversity in proposals
        )
        return response.choices[0].message.content

    trainset = [
        {"input": "What is 2+2?", "additional_context": {}, "answer": "4"},
        {"input": "What is 3+3?", "additional_context": {}, "answer": "6"},
    ]

    seed_candidate = {"system_prompt": "You are a helpful math assistant."}

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=trainset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=50,
    )

    print("Best prompt:", result.best_candidate)
    return result


# =============================================================================
# QUICK REFERENCE: Environment Variables for litellm
# =============================================================================
"""
For vLLM (OpenAI-compatible):
    export OPENAI_API_BASE=http://your-vllm-server:8000/v1
    export OPENAI_API_KEY=dummy

For Ollama:
    export OLLAMA_API_BASE=http://localhost:11434

For Azure OpenAI:
    export AZURE_API_KEY=your_key
    export AZURE_API_BASE=https://your-resource.openai.azure.com/
    export AZURE_API_VERSION=2023-05-15

For Anthropic:
    export ANTHROPIC_API_KEY=your_key

See: https://docs.litellm.ai/docs/providers
"""


if __name__ == "__main__":
    print("=" * 60)
    print("GEPA + vLLM Integration Examples")
    print("=" * 60)
    print(
        """
Choose a method:
1. method1_litellm_base_url() - Recommended, uses env vars
2. method2_custom_callable()  - Most flexible, full control
3. method3_litellm_explicit() - Per-call api_base config

Before running, update the VLLM_URL and MODEL_NAME variables
to match your vLLM server configuration.

Example vLLM server command:
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-32B-Instruct \\
        --host 0.0.0.0 \\
        --port 8000
"""
    )
