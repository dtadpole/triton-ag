import asyncio

from agents import (
    Agent,
    function_tool,
    OpenAIResponsesModel,
    Runner,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

set_tracing_disabled(True)

import os

# TPS Benchmarking Functions
import time
from typing import Any, Dict

from openai import OpenAI

# Configuration options - easily modify these settings
BENCHMARK_CONFIG = {
    "concurrent_queries": 100,  # Number of concurrent queries
    "sequential_queries": 10,  # Number of sequential queries
    "reasoning_efforts": ["low", "medium", "high"],  # Reasoning efforts to test
    "run_sequential": False,  # Whether to run sequential benchmark (usually not needed for TPS)
    "run_effort_comparison": True,  # Whether to compare different reasoning efforts
    "max_output_tokens": 16000,  # Maximum output tokens (16K context length)
}


def load_test_prompts(filename: str = "test_prompts.txt") -> list[str]:
    """Load test prompts from external file, filtering out comments and empty lines."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    prompts = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    prompts.append(line)

        if not prompts:
            print(f"Warning: No prompts found in {filename}, using fallback prompts")
            return [
                "Write a Python function to implement binary search on a sorted array.",
                "Explain the concept of quantum computing in simple terms.",
                "Create a SQL query to find the top 10 customers by total order value.",
            ]

        print(f"Loaded {len(prompts)} test prompts from {filename}")
        return prompts

    except FileNotFoundError:
        print(f"Warning: {filename} not found, using fallback prompts")
        return [
            "Write a Python function to implement binary search on a sorted array.",
            "Explain the concept of quantum computing in simple terms.",
            "Create a SQL query to find the top 10 customers by total order value.",
        ]


# Load test prompts from external file
test_prompts = load_test_prompts()

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple approximation (4 chars per token on average)."""
    return max(1, len(text) // 4)


async def send_query(
    prompt: str, query_id: int, reasoning_effort: str = "high"
) -> Dict[str, Any]:
    """Send a single query and return response with metadata."""
    start_time = time.time()

    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model="hub/models--openai--gpt-oss-120b/snapshots/bc75b44b8a2a116a0e4c6659bcd1b7969885f423",
            reasoning={"effort": reasoning_effort},
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=BENCHMARK_CONFIG["max_output_tokens"],
        )

        end_time = time.time()

        # Estimate tokens in prompt and response
        prompt_tokens = estimate_tokens(prompt)
        response_tokens = estimate_tokens(response.output_text)
        total_tokens = prompt_tokens + response_tokens

        return {
            "query_id": query_id,
            "status": response.status,
            "prompt": prompt,
            "response": response.output_text,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "time_taken": end_time - start_time,
            "tokens_per_second": (
                response_tokens / (end_time - start_time)
                if end_time > start_time
                else 0
            ),
        }
    except Exception as e:
        end_time = time.time()
        return {
            "query_id": query_id,
            "status": "error",
            "prompt": prompt,
            "response": f"Error: {str(e)}",
            "prompt_tokens": estimate_tokens(prompt),
            "response_tokens": 0,
            "total_tokens": estimate_tokens(prompt),
            "time_taken": end_time - start_time,
            "tokens_per_second": 0,
            "error": str(e),
        }


async def benchmark_tps(
    num_queries: int = 20, concurrent: bool = True, reasoning_effort: str = "high"
) -> Dict[str, Any]:
    """Benchmark tokens per second for the model."""
    print(f"Starting TPS benchmark with {num_queries} queries...")
    print(f"Concurrent mode: {concurrent}")
    print(f"Reasoning effort: {reasoning_effort}")
    print("-" * 60)

    # Create prompts by cycling through test prompts
    prompts = [test_prompts[i % len(test_prompts)] for i in range(num_queries)]

    start_time = time.time()

    if concurrent:
        # Run queries concurrently
        tasks = [
            send_query(prompt, i, reasoning_effort) for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
    else:
        # Run queries sequentially
        results = []
        for i, prompt in enumerate(prompts):
            result = await send_query(prompt, i, reasoning_effort)
            results.append(result)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    successful_results = [r for r in results if r["status"] != "error"]
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_response_tokens = sum(r["response_tokens"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)

    successful_queries = len(successful_results)
    failed_queries = len(results) - successful_queries

    # Calculate TPS metrics
    overall_tps = total_response_tokens / total_time if total_time > 0 else 0
    avg_individual_tps = (
        sum(r["tokens_per_second"] for r in successful_results) / successful_queries
        if successful_queries > 0
        else 0
    )

    # Print detailed results
    print("\n" + "=" * 60)
    print("TOKENS PER SECOND BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total queries: {num_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Failed queries: {failed_queries}")
    print(f"Total time: {total_time:.2f} seconds")
    print("-" * 60)
    print("TOKEN STATISTICS (estimated):")
    print(f"  Total prompt tokens: {total_prompt_tokens:,}")
    print(f"  Total response tokens: {total_response_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens per query: {total_tokens/num_queries:.1f}")
    print("-" * 60)
    print("PERFORMANCE METRICS:")
    print(f"  Overall TPS (response tokens): {overall_tps:.2f} tokens/sec")
    print(f"  Average individual TPS: {avg_individual_tps:.2f} tokens/sec")
    print(f"  Queries per second: {successful_queries/total_time:.2f} queries/sec")
    print(f"  Average time per query: {total_time/num_queries:.2f} seconds")

    if successful_results:
        min_tps = min(r["tokens_per_second"] for r in successful_results)
        max_tps = max(r["tokens_per_second"] for r in successful_results)
        print(f"  Min TPS: {min_tps:.2f} tokens/sec")
        print(f"  Max TPS: {max_tps:.2f} tokens/sec")

    print("=" * 60)

    # Print individual query details if requested
    if num_queries <= 10:
        print("\nINDIVIDUAL QUERY DETAILS:")
        print("-" * 60)
        for result in results:
            status_indicator = "✓" if result["status"] != "error" else "✗"
            print(
                f"{status_indicator} Query {result['query_id']}: {result['tokens_per_second']:.2f} TPS "
                f"({result['response_tokens']} tokens in {result['time_taken']:.2f}s)"
            )

    return {
        "total_queries": num_queries,
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "total_time": total_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_response_tokens": total_response_tokens,
        "total_tokens": total_tokens,
        "overall_tps": overall_tps,
        "avg_individual_tps": avg_individual_tps,
        "queries_per_second": successful_queries / total_time if total_time > 0 else 0,
        "results": results,
    }


async def run_tps_benchmark():
    """Main function to run TPS benchmarking with configurable options."""
    print("Token Per Second (TPS) Benchmarking Tool")
    print("Model: GPT-OSS-120B on localhost:8000")
    print("Note: Token counts are estimated (4 chars ≈ 1 token)")
    print(f"Available test prompts: {len(test_prompts)}")
    print()

    if BENCHMARK_CONFIG["run_effort_comparison"]:
        # Test different reasoning efforts
        for effort in BENCHMARK_CONFIG["reasoning_efforts"]:
            print(f"\n{'='*80}")
            print(f"TESTING REASONING EFFORT: {effort.upper()}")
            print(f"{'='*80}")

            await benchmark_tps(
                num_queries=BENCHMARK_CONFIG["concurrent_queries"],
                concurrent=True,
                reasoning_effort=effort,
            )
    else:
        # Run with default "high" reasoning effort
        await benchmark_tps(
            num_queries=BENCHMARK_CONFIG["concurrent_queries"],
            concurrent=True,
            reasoning_effort="high",
        )

    if BENCHMARK_CONFIG["run_sequential"]:
        print(f"\n{'='*80}")
        print("RUNNING SEQUENTIAL BENCHMARK FOR COMPARISON")
        print(f"{'='*80}")

        await benchmark_tps(
            num_queries=BENCHMARK_CONFIG["sequential_queries"],
            concurrent=False,
            reasoning_effort="high",
        )


async def main():
    """Main function - choose what to run."""
    # Option 1: Run TPS benchmarking (default)
    await run_tps_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
