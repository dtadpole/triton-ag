#!/usr/bin/env python3
"""
CUDA Kernel Prompt Optimization with GEPA.

This script optimizes the system prompt for CUDA kernel generation
using GEPA's evolutionary prompt optimization.

Usage:
    python optimize_cuda_kernel.py --config configs/cuda_kernel.yaml
    python optimize_cuda_kernel.py --provider h100_8_5_a --model qwen3-32b
"""

import argparse
import sys
from pathlib import Path

# Append paths to sys.path at the END (after site-packages)
# - /workspace: for custom modules in workspace root
# - /workspace/gepa: for importing adapters directly (without gepa. prefix)
# This ensures installed packages (like gepa) take precedence.
if "/workspace" not in sys.path:
    sys.path.append("/workspace")
if "/workspace/gepa" not in sys.path:
    sys.path.append("/workspace/gepa")

import gepa
import yaml
from adapters.cuda_kernel_adapter import CudaKernelAdapter, CudaKernelDataInst


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_cuda_examples(examples_path: str) -> dict:
    """Load CUDA examples from YAML file."""
    with open(examples_path, "r") as f:
        return yaml.safe_load(f)


def create_sample_dataset() -> list:
    """
    Create a sample dataset for CUDA kernel optimization.

    In practice, you would load this from your actual dataset.
    """
    # Sample reference codes for different CUDA operations
    samples = [
        {
            "task_id": "elementwise_add",
            "reference_code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    return [x, y]

def get_init_inputs():
    return []
""",
        },
        {
            "task_id": "elementwise_mul",
            "reference_code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a * b

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    return [x, y]

def get_init_inputs():
    return []
""",
        },
        {
            "task_id": "relu_activation",
            "reference_code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
""",
        },
        {
            "task_id": "softmax",
            "reference_code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)

batch_size = 16
dim = 1024

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
""",
        },
        {
            "task_id": "layer_norm",
            "reference_code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)

batch_size = 16
dim = 1024

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [dim]
""",
        },
    ]

    return [
        CudaKernelDataInst(
            task_id=s["task_id"],
            reference_code=s["reference_code"],
        )
        for s in samples
    ]


def create_reflection_lm(
    model_name: str,
    provider_name: str,
    config_file: str,
):
    """
    Create a reflection LM callable using InferenceClient.

    This is used by GEPA to propose improved prompts.
    """
    import asyncio

    from inferenceClient import InferenceClient

    client = InferenceClient(
        model_name=model_name,
        config_file=config_file,
    )

    def reflection_lm(prompt: str) -> str:
        """Call the reflection LM synchronously."""
        messages = [{"role": "user", "content": prompt}]

        async def _call():
            result = await client.chat_completion(
                provider=provider_name,
                messages=messages,
                max_tokens=4096,
            )
            if result is None:
                return ""
            return result.get("content", "") or result.get("reasoning_content", "")

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_call())
        finally:
            loop.close()

    return reflection_lm


def main():
    parser = argparse.ArgumentParser(
        description="Optimize CUDA kernel generation prompts with GEPA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="gepa/configs/cuda_kernel.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Provider name (overrides config)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Max metric calls (overrides config)",
    )
    parser.add_argument(
        "--inference-config",
        type=str,
        default="inferenceClient.yaml",
        help="Path to inference client config",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--eval-provider",
        type=str,
        default="h8_4",
        help="Provider for KbEvalClient (e.g., h8_4, h8_2)",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="kbEval.yaml",
        help="Path to kbEval.yaml config",
    )
    parser.add_argument(
        "--no-kb-eval",
        action="store_true",
        help="Disable KbEvalClient (use simple heuristic evaluation)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {}

    # Get settings from config or args
    defaults = config.get("defaults", {})
    optimization = config.get("optimization", {})
    logging_config = config.get("logging", {})

    model_name = args.model or defaults.get("task_model", "qwen3-32b")
    provider_name = args.provider or defaults.get("task_provider", "h100_8_5_a")
    reflection_model = defaults.get("reflection_model", model_name)
    reflection_provider = defaults.get("reflection_provider", provider_name)

    max_metric_calls = args.max_metric_calls or optimization.get("max_metric_calls", 50)
    run_dir = args.run_dir or logging_config.get("run_dir")

    # Get seed candidate from config or use default
    seed_candidate = config.get("seed_candidate", {})
    if "system_prompt" not in seed_candidate:
        seed_candidate[
            "system_prompt"
        ] = """You are an experienced CUDA developer specializing in GPU performance optimization and parallel programming.

You are provided with PyTorch reference code and shall examine the code to understand how it works, then generate high-performance CUDA code that matches the reference code for correctness.

When generating code, always follow these instructions:
- Replace pytorch operators in the reference code with raw CUDA kernels, optimizing for performance on NVIDIA architecture.
- Use torch.utils.cpp_extension.load_inline and name your optimized output module ModelNew.
- You're NOT allowed to use torch.nn (except for Parameter, containers, and init).
- The input and output must be on CUDA device. Your answer must be the complete new module (no testing code, no other code)."""

    print("=" * 60)
    print("GEPA CUDA Kernel Prompt Optimization")
    print("=" * 60)
    print(f"Task Model: {model_name}")
    print(f"Task Provider: {provider_name}")
    print(f"Reflection Model: {reflection_model}")
    print(f"Reflection Provider: {reflection_provider}")
    print(f"Max Metric Calls: {max_metric_calls}")
    print(f"Run Directory: {run_dir}")
    print("=" * 60)

    # Create dataset
    print("\nCreating dataset...")
    dataset = create_sample_dataset()
    print(f"Dataset size: {len(dataset)} samples")

    # Split into train/val
    trainset = dataset[:3]
    valset = dataset[3:]
    print(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Create adapter with KbEvalClient
    print("\nCreating adapter...")
    use_kb_eval = not args.no_kb_eval
    print(f"KbEvalClient enabled: {use_kb_eval}")
    if use_kb_eval:
        print(f"Eval Provider: {args.eval_provider}")
        print(f"Eval Config: {args.eval_config}")

    adapter = CudaKernelAdapter(
        model_name=model_name,
        provider_name=provider_name,
        config_file=args.inference_config,
        eval_provider_name=args.eval_provider,
        eval_config_file=args.eval_config,
        use_kb_eval=use_kb_eval,
        run_tag=f"gepa_{model_name}",
        model_tag=model_name,
        run_dir=run_dir,
        enable_tracing=True,
    )

    # Create reflection LM
    print("Creating reflection LM...")
    reflection_lm = create_reflection_lm(
        model_name=reflection_model,
        provider_name=reflection_provider,
        config_file=args.inference_config,
    )

    # Run GEPA optimization
    print("\nStarting GEPA optimization...")
    print("-" * 60)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        candidate_selection_strategy=optimization.get(
            "candidate_selection_strategy", "pareto"
        ),
        use_merge=optimization.get("use_merge", True),
        skip_perfect_score=optimization.get("skip_perfect_score", True),
        display_progress_bar=logging_config.get("display_progress_bar", True),
        run_dir=run_dir,
        seed=optimization.get("seed", 42),
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nBest Score: {best_score:.4f}")
    print(f"Best Candidate Index: {result.best_idx}")
    print(f"Total Candidates Evaluated: {result.num_candidates}")
    print("\nBest System Prompt:")
    print("-" * 60)
    print(result.best_candidate.get("system_prompt", ""))
    print("-" * 60)

    # Save results
    if run_dir:
        output_path = Path(run_dir) / "best_prompt.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(result.best_candidate.get("system_prompt", ""))
        print(f"\nBest prompt saved to: {output_path}")

    return result


if __name__ == "__main__":
    main()
