import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from vllm import LLM, SamplingParams


def benchmark_reasoning_effort(prompt_file_path, num_gpus=4):
    """
    Benchmark TPS for different reasoning effort levels using the same prompt set.
    """

    # 1. Read all prompts from the file
    print(f"[+] Loading prompts from {prompt_file_path}...")
    with open(prompt_file_path, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    print(f"[+] Loaded {len(prompts)} prompts for benchmarking.")
    if len(prompts) < 10:
        print("[!] Warning: For accurate TPS measurement, use at least 20-30 prompts.")

    # 2. Initialize the LLM (ONCE for all tests)
    print(f"[+] Initializing model on {num_gpus} GPUs...")
    llm = LLM(
        model="hub/models--openai--gpt-oss-120b/snapshots/bc75b44b8a2a116a0e4c6659bcd1b7969885f423",
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )

    # 3. Define reasoning effort configurations with appropriate sampling parameters
    effort_configs = {
        "low": {
            "sampling_params": SamplingParams(
                temperature=0.1,  # Very deterministic
                top_p=0.9,
                max_tokens=256,  # Short responses
                stop=['"}\n', "</s>", "<|im_end|>"],
            ),
            "color": "green",
        },
        "medium": {
            "sampling_params": SamplingParams(
                temperature=0.5,  # Balanced
                top_p=0.95,
                max_tokens=1024,  # Medium responses
                stop=['"}\n', "</s>", "<|im_end|>"],
            ),
            "color": "blue",
        },
        "high": {
            "sampling_params": SamplingParams(
                temperature=0.8,  # Creative/exploratory
                top_p=0.98,
                max_tokens=2048,  # Long, detailed responses
                frequency_penalty=0.3,
                stop=['"}\n', "</s>", "<|im_end|>"],
            ),
            "color": "red",
        },
    }

    # 4. Harmony system prompt template
    harmony_system_prompt = """You are a helpful AI assistant. You always respond in a precise JSON format:

{
  "reasoning": "Your internal reasoning process goes here.",
  "final_answer": "Your concise answer here."
}

Ensure your response is parseable by a JSON parser."""

    # 5. Benchmark each effort level
    results = {}

    for effort_level, config in effort_configs.items():
        print(f"\n[+] Benchmarking '{effort_level}' effort level...")

        # Build prompts for this effort level
        full_prompts = []
        for user_input in prompts:
            full_prompt = f"""<|im_start|>system
{harmony_system_prompt}<|im_end|>
<|im_start|>user
reasoning_effort: {effort_level}
{user_input}<|im_end|>
<|im_start|>assistant
{{
  "reasoning": "
"""
            full_prompts.append(full_prompt)

        # Warm-up run (optional but recommended)
        if effort_level == "low":  # Only need to warm up once
            print("    Performing warm-up run...")
            _ = llm.generate(full_prompts[:1], config["sampling_params"])

        # Actual benchmark run
        start_time = time.perf_counter()

        outputs = llm.generate(full_prompts, config["sampling_params"])

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate statistics
        total_tokens = 0
        successful_responses = 0

        for output in outputs:
            total_tokens += len(output.outputs[0].token_ids)
            # Check if response is valid JSON
            try:
                raw_output = output.outputs[0].text
                complete_json_str = '{ "reasoning": "' + raw_output + '"}\n'
                json.loads(complete_json_str)
                successful_responses += 1
            except:
                pass

        # Calculate TPS
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        # Store results
        results[effort_level] = {
            "total_time": total_time,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "avg_time_per_prompt": total_time / len(prompts),
            "success_rate": (successful_responses / len(prompts)) * 100,
            "avg_tokens_per_response": total_tokens / len(prompts),
        }

        print(
            f"    Time: {total_time:.2f}s, Tokens: {total_tokens}, TPS: {tokens_per_second:.2f}"
        )
        print(f"    Success rate: {results[effort_level]['success_rate']:.1f}%")

    return results, effort_configs


def plot_results(results, effort_configs, output_file=None):
    """Plot the benchmarking results."""
    effort_levels = list(results.keys())
    tps_values = [results[level]["tokens_per_second"] for level in effort_levels]
    colors = [effort_configs[level]["color"] for level in effort_levels]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # TPS bar chart
    bars = ax1.bar(effort_levels, tps_values, color=colors, alpha=0.7)
    ax1.set_ylabel("Tokens Per Second (TPS)")
    ax1.set_title("Throughput by Reasoning Effort Level")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, tps_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    # Additional metrics
    metrics = ["avg_tokens_per_response", "success_rate"]
    metric_names = ["Avg Tokens/Response", "Success Rate (%)"]

    x = np.arange(len(effort_levels))
    width = 0.35

    for i, metric in enumerate(metrics):
        values = [results[level][metric] for level in effort_levels]
        offset = width * i
        ax2.bar(x + offset, values, width, label=metric_names[i], alpha=0.7)

    ax2.set_xlabel("Reasoning Effort Level")
    ax2.set_ylabel("Metrics")
    ax2.set_title("Response Characteristics")
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(effort_levels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"[+] Plot saved to {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TPS for different reasoning effort levels"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="File containing one prompt per line",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--output-plot", type=str, help="Optional filename to save results plot"
    )

    args = parser.parse_args()

    # Run the benchmark
    results, effort_configs = benchmark_reasoning_effort(
        prompt_file_path=args.prompt_file, num_gpus=args.num_gpus
    )

    # Print detailed results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    for effort_level, data in results.items():
        print(f"\n{effort_level.upper()} EFFORT:")
        print(f"  Throughput: {data['tokens_per_second']:.2f} TPS")
        print(f"  Total Time: {data['total_time']:.2f} seconds")
        print(f"  Total Tokens: {data['total_tokens']}")
        print(f"  Avg Time per Prompt: {data['avg_time_per_prompt']:.2f}s")
        print(f"  Avg Tokens per Response: {data['avg_tokens_per_response']:.1f}")
        print(f"  Success Rate: {data['success_rate']:.1f}%")

    # Plot results
    plot_results(results, effort_configs, args.output_plot)


if __name__ == "__main__":
    main()
