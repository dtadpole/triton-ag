#!/usr/bin/env python3
"""
vLLM Throughput Benchmark Script
Measures Tokens Per Second (TPS) for a given model and dataset.
"""

import argparse
import time

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM TPS")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model name or path (e.g., "mistralai/Mistral-7B-Instruct-v0.2")',
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="File containing one prompt per line",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate per prompt",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    args = parser.parse_args()

    # 1. Read the prompts from the file
    print(f"[+] Loading prompts from {args.prompt_file}...")
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"[+] Loaded {len(prompts)} prompts.")

    # 2. Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for benchmarking
        top_p=1.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )

    # 3. Load the model into vLLM's engine
    # This is where you can configure tensor parallelism, dtype, etc.
    print(f"[+] Loading model '{args.model}' onto {args.num_gpus} GPUs...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,  # Only set to True if you trust the model's source
        # gpu_memory_utilization=0.9, # Optional: control GPU memory usage
        # dtype="half", # Optional: use "float16" or "bfloat16" for faster inference
    )

    # 4. Warm-up run (optional but recommended)
    # The first run compiles kernels and loads weights, which is slower.
    print("[+] Performing warm-up run...")
    _ = llm.generate("Hello, warm up!", sampling_params)

    # 5. The actual benchmark run
    print(f"[+] Starting benchmark for {len(prompts)} prompts...")
    start_time = time.perf_counter()

    # This is the core generation call. vLLM's engine will automatically
    # batch these prompts together using PagedAttention for efficiency.
    outputs = llm.generate(prompts, sampling_params)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # 6. Calculate statistics
    total_tokens_generated = 0
    for output in outputs:
        # Count the tokens in the generated text for each request
        total_tokens_generated += len(output.outputs[0].token_ids)

    # Calculate throughput
    tokens_per_second = total_tokens_generated / total_time

    # 7. Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Number of Prompts: {len(prompts)}")
    print(f"Max New Tokens: {args.max_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Total Generated Tokens: {total_tokens_generated}")
    print(f"Throughput: {tokens_per_second:.2f} tokens/second")
    print(f"Throughput: {tokens_per_second * 60:.2f} tokens/minute")
    print("=" * 50)


if __name__ == "__main__":
    main()
