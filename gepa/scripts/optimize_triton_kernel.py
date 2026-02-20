#!/usr/bin/env python3
"""
Triton Kernel Prompt Optimization with GEPA.

This script optimizes the system prompt for Triton kernel generation
using GEPA's evolutionary prompt optimization.

Usage:
    python optimize_triton_kernel.py --config configs/triton_kernel.yaml
    python optimize_triton_kernel.py --provider h100_8_5_a --model qwen3-32b
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

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
from adapters.triton_kernel_adapter import TritonKernelAdapter, TritonKernelDataInst


# Directory for caching reference runtimes
REFERENCE_CACHE_DIR = Path("shared/.kbeval/reference")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts_from_config(config: dict) -> Dict[str, Any]:
    """
    Load and generate prompts from the modular config.

    This function reads the prompts section from the config, builds the full
    system_prompt from modular components (role_description, task_description,
    task_instruction, examples), and creates a seed_candidate containing only
    the optimizable components.

    Args:
        config: The loaded configuration dictionary

    Returns:
        Dict with:
            - 'system_prompt': The full assembled system prompt
            - 'user_prompt_template': The user prompt template (with {reference_code} placeholder)
            - 'seed_candidate': Dict of optimizable components for GEPA to evolve
            - 'fixed_components': Dict of non-optimizable components (kept constant)
            - 'component_order': List of component names in assembly order
            - 'optimizable_flags': Dict mapping component names to their optimizable status
            - 'component_metadata': Dict mapping component names to their label/description

    Config Structure:
        prompts:
            role_description:
                optimizable: true
                label: "ROLE DESCRIPTION"
                description: "This component defines..."
                content: |
                    ...
            task_description:
                optimizable: true
                label: "TASK DESCRIPTION"
                description: "This component describes..."
                content: |
                    ...
            task_instruction:
                optimizable: true
                label: "TASK INSTRUCTIONS"
                description: "This component provides..."
                content: |
                    ...
            examples:
                optimizable: false
                label: "EXAMPLES"
                description: "This component provides..."
                content: |
                    ... {example_reference_code} ... {example_generated_code} ...
                reference_code: |
                    ...
                generated_code: |
                    ...
            user_prompt_template: |
                ... {reference_code} ...

    Example:
        >>> config = load_config("configs/triton_kernel.yaml")
        >>> result = load_prompts_from_config(config)
        >>> print(result["system_prompt"][:100])
        'You are an experienced Triton developer...'
        >>> print(result["seed_candidate"].keys())
        dict_keys(['role_description', 'task_description', 'task_instruction'])
    """
    prompts_config = config.get("prompts", {})

    # Define the component order for assembling system_prompt
    component_order = ["role_description", "task_description", "task_instruction", "examples"]

    # Get example code for substitution in examples component
    examples_config = prompts_config.get("examples", {})
    example_reference_code = examples_config.get("reference_code", "")
    example_generated_code = examples_config.get("generated_code", "")

    # Build components
    seed_candidate = {}  # Optimizable components (GEPA will evolve these)
    fixed_components = {}  # Non-optimizable components (kept constant)
    optimizable_flags = {}  # Track which components are optimizable
    component_metadata = {}  # Store label/description for each component
    system_prompt_parts = []

    for component_name in component_order:
        component_config = prompts_config.get(component_name, {})

        # Handle both dict format (with optimizable flag) and direct string format
        if isinstance(component_config, dict):
            optimizable = component_config.get("optimizable", True)
            content = component_config.get("content", "")
            label = component_config.get("label", component_name.upper().replace("_", " "))
            description = component_config.get("description", "")
        else:
            # Legacy format: direct string content, assume optimizable
            optimizable = True
            content = component_config
            label = component_name.upper().replace("_", " ")
            description = ""

        # Handle examples component specially - substitute example code
        if component_name == "examples" and content:
            content = content.format(
                example_reference_code=example_reference_code.strip(),
                example_generated_code=example_generated_code.strip(),
            )

        optimizable_flags[component_name] = optimizable
        component_metadata[component_name] = {
            "label": label,
            "description": description,
        }

        if content:
            if optimizable:
                seed_candidate[component_name] = content
            else:
                fixed_components[component_name] = content
            system_prompt_parts.append(content)

    # Assemble the full system_prompt
    system_prompt = "\n".join(system_prompt_parts)

    # Get user prompt template
    user_prompt_template = prompts_config.get("user_prompt_template", "")

    # Print summary
    print("Loaded prompts from modular config:")
    print(f"  System prompt length: {len(system_prompt)} chars")
    print(f"  User prompt template length: {len(user_prompt_template)} chars")
    print(f"  Components:")
    for component_name in component_order:
        meta = component_metadata.get(component_name, {})
        label = meta.get("label", "N/A")
        if component_name in seed_candidate:
            print(f"    - {component_name} [{label}]: optimizable=True, length={len(seed_candidate[component_name])} chars")
        elif component_name in fixed_components:
            print(f"    - {component_name} [{label}]: optimizable=False, length={len(fixed_components[component_name])} chars")
    print(f"  Seed candidate keys: {list(seed_candidate.keys())}")

    return {
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "seed_candidate": seed_candidate,
        "fixed_components": fixed_components,
        "component_order": component_order,
        "optimizable_flags": optimizable_flags,
        "component_metadata": component_metadata,
    }


def get_reference_cache_path(hostname: str, task_id: str) -> Path:
    """Get the cache file path for a reference runtime result.

    Structure: shared/.kbeval/reference/{task_id}/{hostname}.json
    """
    safe_task_id = task_id.replace("/", "_").replace("\\", "_")
    safe_hostname = hostname.replace("/", "_").replace("\\", "_").replace(":", "_")
    return REFERENCE_CACHE_DIR / safe_task_id / f"{safe_hostname}.json"


def load_cached_reference_runtime(hostname: str, task_id: str) -> Optional[Dict[str, Any]]:
    """Load cached reference runtime if available."""
    cache_path = get_reference_cache_path(hostname, task_id)
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load cache for {task_id}: {e}")
    return None


def save_reference_runtime(hostname: str, task_id: str, result: Dict[str, Any]) -> None:
    """Save reference runtime result to cache.

    Structure: shared/.kbeval/reference/{task_id}/{hostname}.json
    """
    cache_path = get_reference_cache_path(hostname, task_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Cached reference runtime to: {cache_path}")


async def generate_reference_runtime_async(
    eval_client,
    provider: str,
    hostname: str,
    data: TritonKernelDataInst,
    run_tag: str = "reference_runtime",
    model_tag: str = "reference",
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Generate reference runtime for a single task using kb_eval_ref.

    Args:
        eval_client: KbEvalClient instance
        provider: Provider name for evaluation
        hostname: Hostname of the kbEval server (for caching)
        data: TritonKernelDataInst containing reference code
        run_tag: Tag for the evaluation run
        model_tag: Tag for the model
        force_refresh: If True, regenerate even if cached

    Returns:
        Reference runtime result dict or None if failed
    """
    # Check cache first
    if not force_refresh:
        cached = load_cached_reference_runtime(hostname, data.task_id)
        if cached is not None:
            print(f"  [{data.task_id}] Using cached reference runtime")
            return cached

    # Generate reference runtime
    print(f"  [{data.task_id}] Generating reference runtime...")
    result = await eval_client.kb_eval_ref(
        provider=provider,
        reference_code=data.reference_code,
        run_tag=run_tag,
        model_tag=model_tag,
        task_tag=data.task_id,
        code_type="triton",  # KEY DIFFERENCE: Use "triton" instead of "cuda"
    )

    if result is not None:
        # Save to cache
        save_reference_runtime(hostname, data.task_id, result)
        return result
    else:
        print(f"  [{data.task_id}] Failed to generate reference runtime")
        return None


def generate_reference_runtimes(
    dataset: List[TritonKernelDataInst],
    eval_config_file: str,
    eval_provider: str,
    run_tag: str = "reference_runtime",
    force_refresh: bool = False,
    max_concurrent: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate reference runtimes for all tasks in a dataset (in parallel).

    Args:
        dataset: List of TritonKernelDataInst
        eval_config_file: Path to kbEval.yaml config
        eval_provider: Provider name for evaluation
        run_tag: Tag for the evaluation run
        force_refresh: If True, regenerate even if cached
        max_concurrent: Maximum number of concurrent tasks

    Returns:
        Dict mapping task_id to reference runtime result
    """
    from kbEvalClient import KbEvalClient

    # Create eval client
    eval_client = KbEvalClient(config_file=eval_config_file)

    # Get hostname from provider config
    provider_config = eval_client._provider_config_from_yaml(eval_provider)
    hostname = provider_config.get("hostname", "unknown")

    print(f"\nGenerating reference runtimes for {len(dataset)} tasks (parallel, max_concurrent={max_concurrent})...")
    print(f"  Provider: {eval_provider}")
    print(f"  Hostname: {hostname}")
    print(f"  Cache dir: {REFERENCE_CACHE_DIR}")
    print(f"  Force refresh: {force_refresh}")

    results = {}

    async def _generate_single(data: TritonKernelDataInst) -> tuple:
        """Generate reference runtime for a single task and return (task_id, result)."""
        result = await generate_reference_runtime_async(
            eval_client=eval_client,
            provider=eval_provider,
            hostname=hostname,
            data=data,
            run_tag=run_tag,
            force_refresh=force_refresh,
        )
        return (data.task_id, result)

    async def _generate_all_parallel():
        """Generate all reference runtimes in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited_generate(data: TritonKernelDataInst) -> tuple:
            async with semaphore:
                return await _generate_single(data)

        # Run all tasks in parallel
        tasks = [_limited_generate(data) for data in dataset]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for task_result in task_results:
            if isinstance(task_result, Exception):
                print(f"  Error: {task_result}")
                continue
            task_id, result = task_result
            if result is not None:
                results[task_id] = result

    # Run async
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_generate_all_parallel())
    finally:
        loop.close()

    # Print summary
    print(f"\nReference runtime summary:")
    for task_id, result in results.items():
        runtime = result.get("runtime", -1)
        compiled = result.get("compiled", False)
        correctness = result.get("correctness", False)
        status = "✅" if compiled and correctness else "❌"
        print(f"  {status} [{task_id}] runtime={runtime:.2f}μs, compiled={compiled}, correctness={correctness}")

    return results


def load_all_reference_runtimes(
    dataset: List[TritonKernelDataInst],
    hostname: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all cached reference runtimes for a dataset.

    Args:
        dataset: List of TritonKernelDataInst
        hostname: Hostname of the kbEval server

    Returns:
        Dict mapping task_id to reference runtime result (only cached ones)
    """
    results = {}
    for data in dataset:
        cached = load_cached_reference_runtime(hostname, data.task_id)
        if cached is not None:
            results[data.task_id] = cached
    return results


def rename_output_folders_to_task_names(
    run_dir: str,
    valset: List[TritonKernelDataInst],
    folder_name: str = "generated_best_outputs_valset",
) -> None:
    """
    Rename GEPA output folders from task_0, task_1, etc. to actual task names.

    GEPA creates folders like:
        generated_best_outputs_valset/task_0/
        generated_best_outputs_valset/task_1/

    This function renames them to:
        generated_best_outputs_valset/elementwise_add/
        generated_best_outputs_valset/softmax/

    Args:
        run_dir: GEPA run directory
        valset: Validation set with task_id for each task
        folder_name: Name of the output folder (default: generated_best_outputs_valset)
    """
    import shutil

    output_dir = Path(run_dir) / folder_name
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    print(f"\nRenaming output folders to task names...")
    for task_idx, data in enumerate(valset):
        old_folder = output_dir / f"task_{task_idx}"
        new_folder = output_dir / data.task_id

        if old_folder.exists():
            if new_folder.exists():
                # If target exists, merge contents
                for item in old_folder.iterdir():
                    shutil.move(str(item), str(new_folder / item.name))
                old_folder.rmdir()
                print(f"  Merged task_{task_idx} -> {data.task_id}")
            else:
                old_folder.rename(new_folder)
                print(f"  Renamed task_{task_idx} -> {data.task_id}")
        elif new_folder.exists():
            print(f"  Already renamed: {data.task_id}")
        else:
            print(f"  Folder not found: task_{task_idx}")


def load_kernel_bench_dataset(
    kernel_bench_path: str = "kernel_bench",
    levels: List[int] = None,
    total_tasks: Optional[int] = None,
    train_size: int = 10,
    shuffle: bool = True,
    dataset_seed: int = 42,
    include_tasks: List[str] = None,
    exclude_tasks: List[str] = None,
) -> tuple:
    """
    Load dataset from kernel_bench/ directory.

    Args:
        kernel_bench_path: Path to kernel_bench directory (relative or absolute)
        levels: List of levels to load [1, 2, 3, 4]. Default: [1, 2]
        total_tasks: Total number of tasks to sample. If None or 0, use all tasks.
        train_size: Number of samples for training set
        shuffle: Whether to shuffle the dataset before splitting
        dataset_seed: Random seed for task sampling and shuffling
        include_tasks: Optional list of specific task IDs to include
        exclude_tasks: Optional list of task IDs to exclude

    Returns:
        Tuple of (trainset, valset) where each is a list of TritonKernelDataInst

    Note:
        - If train_size >= total_tasks (or all available tasks), val set = train set
        - dataset_seed controls both task sampling and shuffle order

    Example:
        >>> # Sample 10 tasks, use 3 for train, 7 for val
        >>> trainset, valset = load_kernel_bench_dataset(
        ...     levels=[1, 2],
        ...     total_tasks=10,
        ...     train_size=3,
        ...     dataset_seed=42,
        ... )

        >>> # Use all tasks, 5 for train
        >>> trainset, valset = load_kernel_bench_dataset(
        ...     levels=[1],
        ...     total_tasks=None,  # Use all
        ...     train_size=5,
        ... )
    """
    import random

    if levels is None:
        levels = [1, 2]

    include_tasks = include_tasks or []
    exclude_tasks = exclude_tasks or []

    kernel_bench_dir = Path(kernel_bench_path)
    if not kernel_bench_dir.exists():
        raise FileNotFoundError(f"kernel_bench directory not found: {kernel_bench_dir}")

    all_tasks = []

    for level in sorted(levels):
        level_dir = kernel_bench_dir / f"level{level}"
        if not level_dir.exists():
            print(f"Warning: Level {level} directory not found: {level_dir}")
            continue

        # Get all .py files in the level directory
        py_files = sorted(level_dir.glob("*.py"))
        print(f"Found {len(py_files)} tasks in level{level}")

        for py_file in py_files:
            # Generate task_id: level{N}_{filename_without_extension}
            filename = py_file.stem  # e.g., "94_MSELoss"
            task_id = f"level{level}_{filename}"

            # Apply include/exclude filters
            if include_tasks and task_id not in include_tasks:
                continue
            if task_id in exclude_tasks:
                continue

            # Read the reference code
            try:
                reference_code = py_file.read_text()
            except Exception as e:
                print(f"Warning: Failed to read {py_file}: {e}")
                continue

            all_tasks.append(
                TritonKernelDataInst(
                    task_id=task_id,
                    reference_code=reference_code,
                    additional_context={"level": level, "source_file": str(py_file)},
                )
            )

    print(f"Total tasks available: {len(all_tasks)}")

    if len(all_tasks) == 0:
        raise ValueError(f"No tasks found in kernel_bench with levels={levels}")

    # Set random seed for reproducibility
    rng = random.Random(dataset_seed)

    # Sample total_tasks if specified
    if total_tasks is not None and total_tasks > 0 and total_tasks < len(all_tasks):
        print(f"Sampling {total_tasks} tasks from {len(all_tasks)} available (seed={dataset_seed})")
        all_tasks = rng.sample(all_tasks, total_tasks)
    else:
        # Use all tasks, but shuffle if requested
        if shuffle:
            rng.shuffle(all_tasks)

    # If we sampled, we may still want to shuffle for train/val split
    if total_tasks is not None and total_tasks > 0 and shuffle:
        rng.shuffle(all_tasks)

    print(f"Tasks after sampling: {len(all_tasks)}")

    # Split into train/val
    actual_total = len(all_tasks)
    if train_size >= actual_total:
        # If train_size >= total, use all for train and val = train (same data)
        trainset = all_tasks
        valset = all_tasks.copy()
        print(f"Train size ({train_size}) >= total tasks ({actual_total})")
        print(f"Using same data for train and val: {len(trainset)} tasks each")
    else:
        trainset = all_tasks[:train_size]
        valset = all_tasks[train_size:]
        print(f"Dataset split: train={len(trainset)}, val={len(valset)}")

    # Print task IDs for reference
    print(f"Train tasks: {[t.task_id for t in trainset]}")
    print(f"Val tasks: {[t.task_id for t in valset]}")

    return trainset, valset


def create_sample_dataset() -> list:
    """
    Create a sample dataset for Triton kernel optimization.

    This is a simple hardcoded dataset for testing.
    For production, use load_kernel_bench_dataset() instead.
    """
    # Sample reference codes for different Triton operations
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
        TritonKernelDataInst(
            task_id=s["task_id"],
            reference_code=s["reference_code"],
        )
        for s in samples
    ]


def create_reflection_lm(
    model_name: str,
    provider_name: str,
    config_file: str,
    max_tokens: int = 4096,
    trace_logger: Optional[Any] = None,
):
    """
    Create a reflection LM callable using InferenceClient.

    This is used by GEPA to propose improved prompts.
    The wrapper captures reflection inputs/outputs for logging.

    Args:
        model_name: Model name for reflection
        provider_name: Provider name for inference
        config_file: Path to inferenceClient.yaml
        max_tokens: Maximum tokens for generation
        trace_logger: Optional TraceLogger instance for logging reflection data
    """
    import asyncio

    from inferenceClient import InferenceClient

    client = InferenceClient(
        model_name=model_name,
        config_file=config_file,
    )

    # Track iteration for reflection logging
    reflection_call_count = [0]

    def reflection_lm(prompt: str) -> str:
        """Call the reflection LM synchronously with logging."""
        messages = [{"role": "user", "content": prompt}]

        async def _call():
            result = await client.chat_completion(
                provider=provider_name,
                messages=messages,
                max_tokens=max_tokens,
            )
            if result is None:
                return ""
            return result.get("content", "") or result.get("reasoning_content", "")

        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(_call())
        finally:
            loop.close()

        # Log reflection input/output if trace_logger is provided
        if trace_logger is not None:
            reflection_call_count[0] += 1
            iteration = reflection_call_count[0]

            # Log the reflection input (the prompt from GEPA)
            trace_logger.log_reflection_input(
                iteration=iteration,
                candidate={},  # We don't have the candidate here
                feedback_data={"reflection_prompt": prompt},
            )

            # Try to extract the new prompt from the response
            # GEPA typically wraps the new prompt in specific tags
            new_prompt = _extract_new_prompt_from_response(response)

            # Log the reflection response
            trace_logger.log_reflection_response(
                iteration=iteration,
                raw_response=response,
                new_prompt=new_prompt,
            )

        return response

    return reflection_lm


def _extract_new_prompt_from_response(response: str) -> str:
    """
    Extract the new prompt from GEPA reflection response.

    GEPA typically wraps the new prompt in specific tags like:
    - [NEW PROMPT] ... [/NEW PROMPT]
    - ```prompt ... ```
    - Or the response itself is the new prompt

    This function tries to extract the actual new prompt.
    """
    import re

    # Try to find [NEW PROMPT] tags
    match = re.search(r'\[NEW PROMPT\](.*?)\[/NEW PROMPT\]', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to find ```prompt blocks
    match = re.search(r'```prompt\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find generic code blocks that might contain the prompt
    match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no specific format found, return the whole response
    # (it might be the prompt itself)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Triton kernel generation prompts with GEPA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="gepa/configs/triton_kernel.yaml",
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
        help="Max metric calls / evaluation budget (overrides config). "
             "Use --max-iterations for a more intuitive option.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Number of GEPA optimization iterations (overrides config). "
             "More intuitive than --max-metric-calls. "
             "Calculated as: max_metric_calls = max_iterations × (trainset_size + valset_size)",
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
        default="shared/gepa_runs/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--eval-provider",
        type=str,
        default=None,
        help="Provider for KbEvalClient (e.g., h8_4, h8_2). If not specified, uses config file value.",
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
    parser.add_argument(
        "--compile-score",
        type=float,
        default=None,
        help="Score for code that compiles but fails correctness (default: 0.15)",
    )
    parser.add_argument(
        "--correct-score",
        type=float,
        default=None,
        help="Base score for correct code (default: 0.3)",
    )
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        default=None,
        help="Minimum speedup required to earn bonus (default: 1.0)",
    )
    parser.add_argument(
        "--speedup-score",
        type=float,
        default=None,
        help="Bonus score when speedup >= threshold (default: 0.3)",
    )
    parser.add_argument(
        "--generate-reference-runtimes",
        action="store_true",
        help="Generate reference runtimes for trainset and valset before optimization",
    )
    parser.add_argument(
        "--force-refresh-reference",
        action="store_true",
        help="Force regenerate reference runtimes even if cached",
    )
    parser.add_argument(
        "--reference-runtimes-only",
        action="store_true",
        help="Only generate reference runtimes, skip optimization",
    )
    # Dataset arguments
    parser.add_argument(
        "--kernel-bench-path",
        type=str,
        default=None,
        help="Path to kernel_bench directory (overrides config)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of levels to load, e.g., '1,2' (overrides config)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of samples for training set (overrides config)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable dataset shuffling",
    )
    parser.add_argument(
        "--use-sample-dataset",
        action="store_true",
        help="Use the hardcoded sample dataset instead of kernel_bench",
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=None,
        help="Total number of tasks to sample from kernel_bench (overrides config)",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Random seed for dataset sampling and shuffling (overrides config)",
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
    evaluation_config = config.get("evaluation", {})

    model_name = args.model or defaults.get("task_model", "qwen3-32b")
    provider_name = args.provider or defaults.get("task_provider", "h100_8_5_a")
    reflection_model = defaults.get("reflection_model", model_name)
    reflection_provider = defaults.get("reflection_provider", provider_name)

    # max_metric_calls will be calculated later after dataset is loaded
    # if max_iterations is specified
    run_dir = args.run_dir or logging_config.get("run_dir")

    # Load prompts from modular config
    # The prompts section contains modular components:
    # - role_description: optimizable flag + content
    # - task_description: optimizable flag + content
    # - task_instruction: optimizable flag + content
    # - examples: optimizable flag + content + reference_code + generated_code
    # - user_prompt_template: template with {reference_code} placeholder
    print("\nLoading prompts from config...")
    prompts_data = load_prompts_from_config(config)

    # seed_candidate contains only optimizable components (GEPA will evolve these)
    # The adapter will assemble the full system_prompt from optimizable + fixed components
    seed_candidate = prompts_data["seed_candidate"]
    fixed_components = prompts_data["fixed_components"]
    component_order = prompts_data["component_order"]
    component_metadata = prompts_data.get("component_metadata", {})

    # user_prompt_template is passed to adapter separately (not optimized by GEPA)
    user_prompt_template = prompts_data.get("user_prompt_template")

    print("=" * 60)
    print("GEPA Triton Kernel Prompt Optimization")
    print("=" * 60)
    print(f"Task Model: {model_name}")
    print(f"Task Provider: {provider_name}")
    print(f"Reflection Model: {reflection_model}")
    print(f"Reflection Provider: {reflection_provider}")
    print(f"Run Directory: {run_dir}")
    print("=" * 60)

    # Create dataset
    print("\nCreating dataset...")

    # Get dataset config from config file
    dataset_config = config.get("dataset", {})

    if args.use_sample_dataset:
        # Use the hardcoded sample dataset
        print("Using hardcoded sample dataset")
        dataset = create_sample_dataset()
        trainset = dataset[:3]
        valset = dataset[3:]
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Train: {len(trainset)}, Val: {len(valset)}")
    else:
        # Use kernel_bench dataset
        kernel_bench_path = (
            args.kernel_bench_path
            or dataset_config.get("kernel_bench_path", "kernel_bench")
        )

        # Parse levels from args or config
        if args.levels:
            levels = [int(x.strip()) for x in args.levels.split(",")]
        else:
            levels = dataset_config.get("levels", [1, 2])

        train_size = (
            args.train_size
            if args.train_size is not None
            else dataset_config.get("train_size", 10)
        )

        total_tasks = (
            args.total_tasks
            if args.total_tasks is not None
            else dataset_config.get("total_tasks", None)
        )

        shuffle = not args.no_shuffle and dataset_config.get("shuffle", True)

        # Get dataset_seed from args, config, or fall back to optimization seed
        dataset_seed = (
            args.dataset_seed
            if args.dataset_seed is not None
            else dataset_config.get("dataset_seed", optimization.get("seed", 42))
        )

        include_tasks = dataset_config.get("include_tasks", [])
        exclude_tasks = dataset_config.get("exclude_tasks", [])

        print(f"Loading dataset from: {kernel_bench_path}")
        print(f"Levels: {levels}")
        print(f"Total tasks to sample: {total_tasks if total_tasks else 'all'}")
        print(f"Train size: {train_size}")
        print(f"Shuffle: {shuffle}")
        print(f"Dataset seed: {dataset_seed}")

        trainset, valset = load_kernel_bench_dataset(
            kernel_bench_path=kernel_bench_path,
            levels=levels,
            total_tasks=total_tasks,
            train_size=train_size,
            shuffle=shuffle,
            dataset_seed=dataset_seed,
            include_tasks=include_tasks,
            exclude_tasks=exclude_tasks,
        )

    # Get eval_provider from args or config
    eval_provider = (
        args.eval_provider
        if args.eval_provider is not None
        else evaluation_config.get("eval_provider", "h8_4")
    )
    print(f"Eval Provider: {eval_provider}")

    # Generate reference runtimes if requested
    if args.generate_reference_runtimes or args.reference_runtimes_only:
        print("\n" + "=" * 60)
        print("GENERATING REFERENCE RUNTIMES")
        print("=" * 60)

        # Generate for trainset
        print("\n--- Trainset Reference Runtimes ---")
        train_ref_runtimes = generate_reference_runtimes(
            dataset=trainset,
            eval_config_file=args.eval_config,
            eval_provider=eval_provider,
            run_tag="reference_runtime_train",
            force_refresh=args.force_refresh_reference,
        )

        # Generate for valset
        print("\n--- Valset Reference Runtimes ---")
        val_ref_runtimes = generate_reference_runtimes(
            dataset=valset,
            eval_config_file=args.eval_config,
            eval_provider=eval_provider,
            run_tag="reference_runtime_val",
            force_refresh=args.force_refresh_reference,
        )

        print(f"\nTotal reference runtimes generated:")
        print(f"  Trainset: {len(train_ref_runtimes)}/{len(trainset)}")
        print(f"  Valset: {len(val_ref_runtimes)}/{len(valset)}")

        if args.reference_runtimes_only:
            print("\n--reference-runtimes-only flag set, skipping optimization.")
            return None

    # Load reference runtimes for the adapter (from cache)
    # Get hostname from eval provider config
    from kbEvalClient import KbEvalClient
    eval_client_for_config = KbEvalClient(config_file=args.eval_config)
    provider_config = eval_client_for_config._provider_config_from_yaml(eval_provider)
    hostname = provider_config.get("hostname", "unknown")

    # Load cached reference runtimes for all tasks
    all_dataset = trainset + valset
    reference_runtimes = load_all_reference_runtimes(all_dataset, hostname)
    print(f"\nLoaded {len(reference_runtimes)} cached reference runtimes for hostname: {hostname}")
    for task_id, ref_data in reference_runtimes.items():
        ref_runtime = ref_data.get("runtime", -1)
        print(f"  [{task_id}] reference_runtime={ref_runtime:.2f}μs")

    # Create adapter with KbEvalClient
    print("\nCreating adapter...")
    use_kb_eval = not args.no_kb_eval
    print(f"KbEvalClient enabled: {use_kb_eval}")
    if use_kb_eval:
        print(f"Eval Config: {args.eval_config}")

    # Get reward function parameters from args or config
    compile_score = (
        args.compile_score
        if args.compile_score is not None
        else evaluation_config.get("compile_score", 0.15)
    )
    correct_score = (
        args.correct_score
        if args.correct_score is not None
        else evaluation_config.get("correct_score", 0.3)
    )
    speedup_threshold = (
        args.speedup_threshold
        if args.speedup_threshold is not None
        else evaluation_config.get("speedup_threshold", 1.0)
    )
    speedup_score = (
        args.speedup_score
        if args.speedup_score is not None
        else evaluation_config.get("speedup_score", 0.3)
    )

    print(f"Reward parameters: compile_score={compile_score}, correct_score={correct_score}, "
          f"speedup_threshold={speedup_threshold}, speedup_score={speedup_score}")
    print(f"Reference runtimes loaded: {len(reference_runtimes)}")

    # Get concurrency settings from args or config
    concurrency_config = config.get("concurrency", {})
    max_concurrent_llm = concurrency_config.get("max_concurrent_llm", 8)
    max_concurrent_eval = concurrency_config.get("max_concurrent_eval", 4)

    print(f"Concurrency: max_concurrent_llm={max_concurrent_llm}, max_concurrent_eval={max_concurrent_eval}")

    # Get reflection config from config file
    reflection_config = config.get("reflection", {})
    if reflection_config:
        print(f"Reflection config: custom template loaded")

    adapter = TritonKernelAdapter(
        model_name=model_name,
        provider_name=provider_name,
        config_file=args.inference_config,
        eval_provider_name=eval_provider,
        eval_config_file=args.eval_config,
        use_kb_eval=use_kb_eval,
        compile_score=compile_score,
        correct_score=correct_score,
        speedup_threshold=speedup_threshold,
        speedup_score=speedup_score,
        reference_runtimes=reference_runtimes,
        run_tag=f"gepa_{model_name}",
        model_tag=model_name,
        run_dir=run_dir,
        enable_tracing=True,
        user_prompt_template=user_prompt_template,  # Passed separately, not part of seed_candidate
        max_concurrent_llm=max_concurrent_llm,
        max_concurrent_eval=max_concurrent_eval,
        # Modular prompt settings - fixed components and assembly order
        fixed_components=fixed_components,
        component_order=component_order,
        component_metadata=component_metadata,  # For custom reflection prompts
        reflection_config=reflection_config,  # Configurable reflection prompt template
    )

    # Create reflection LM with trace logger for logging reflection inputs/outputs
    print("Creating reflection LM...")
    reflection_lm = create_reflection_lm(
        model_name=reflection_model,
        provider_name=reflection_provider,
        config_file=args.inference_config,
        trace_logger=adapter.trace_logger,  # Pass trace logger for reflection logging
    )

    # Set the reflection LM on the adapter for custom propose_new_texts
    adapter.set_reflection_lm(reflection_lm)

    # Calculate max_metric_calls from max_iterations or use direct value
    # Priority: CLI args > config max_iterations > config max_metric_calls > default
    trainset_size = len(trainset)
    valset_size = len(valset)

    if args.max_iterations is not None:
        # CLI --max-iterations takes highest priority
        max_iterations = args.max_iterations
        max_metric_calls = max_iterations * (trainset_size + valset_size)
        print(f"\nUsing --max-iterations={max_iterations} from CLI")
        print(f"  Calculated max_metric_calls = {max_iterations} × ({trainset_size} + {valset_size}) = {max_metric_calls}")
    elif args.max_metric_calls is not None:
        # CLI --max-metric-calls
        max_metric_calls = args.max_metric_calls
        estimated_iterations = max_metric_calls / (trainset_size + valset_size) if (trainset_size + valset_size) > 0 else 0
        print(f"\nUsing --max-metric-calls={max_metric_calls} from CLI")
        print(f"  Estimated iterations: ~{estimated_iterations:.1f}")
    elif optimization.get("max_iterations") is not None:
        # Config max_iterations
        max_iterations = optimization.get("max_iterations")
        max_metric_calls = max_iterations * (trainset_size + valset_size)
        print(f"\nUsing max_iterations={max_iterations} from config")
        print(f"  Calculated max_metric_calls = {max_iterations} × ({trainset_size} + {valset_size}) = {max_metric_calls}")
    elif optimization.get("max_metric_calls") is not None:
        # Config max_metric_calls
        max_metric_calls = optimization.get("max_metric_calls")
        estimated_iterations = max_metric_calls / (trainset_size + valset_size) if (trainset_size + valset_size) > 0 else 0
        print(f"\nUsing max_metric_calls={max_metric_calls} from config")
        print(f"  Estimated iterations: ~{estimated_iterations:.1f}")
    else:
        # Default fallback
        max_iterations = 3
        max_metric_calls = max_iterations * (trainset_size + valset_size)
        print(f"\nUsing default max_iterations={max_iterations}")
        print(f"  Calculated max_metric_calls = {max_iterations} × ({trainset_size} + {valset_size}) = {max_metric_calls}")

    # Run GEPA optimization
    print("\nStarting GEPA optimization...")
    print("-" * 60)

    # Save a copy of the config file to the run directory for reproducibility
    if run_dir:
        import shutil
        run_dir_path = Path(run_dir)
        run_dir_path.mkdir(parents=True, exist_ok=True)

        # Copy the original config file
        if config_path.exists():
            config_copy_path = run_dir_path / "config.yaml"
            shutil.copy(str(config_path), str(config_copy_path))
            print(f"Config saved to: {config_copy_path}")

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
        # Module selector controls how GEPA updates components:
        # - "round_robin": Updates ONE component per iteration, rotating through them (default)
        # - "all": Updates ALL components simultaneously in each iteration
        # When using modular prompts with multiple optimizable components,
        # "all" is recommended to update all components together
        module_selector=optimization.get("module_selector", "all"),
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nBest Score: {best_score:.4f}")
    print(f"Best Candidate Index: {result.best_idx}")
    print(f"Total Candidates Evaluated: {result.num_candidates}")

    # Assemble the best system_prompt from modular components
    # The best_candidate may contain either:
    # 1. A single "system_prompt" key (backward compatibility)
    # 2. Modular components like "role_description", "task_description", etc.
    best_candidate = result.best_candidate
    if "system_prompt" in best_candidate and len(best_candidate) == 1:
        # Backward compatibility: single system_prompt key
        best_system_prompt = best_candidate["system_prompt"]
    else:
        # Assemble from modular components using the same order as the adapter
        prompt_parts = []
        for component_name in component_order:
            if component_name in best_candidate:
                prompt_parts.append(best_candidate[component_name])
            elif component_name in fixed_components:
                prompt_parts.append(fixed_components[component_name])
        best_system_prompt = "\n".join(prompt_parts)

    print("\nBest System Prompt:")
    print("-" * 60)
    print(best_system_prompt)
    print("-" * 60)

    # Save results
    if run_dir:
        output_path = Path(run_dir) / "best_prompt.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(best_system_prompt)
        print(f"\nBest prompt saved to: {output_path}")

        # Also save the modular components separately for easier analysis
        components_path = Path(run_dir) / "best_prompt_components.json"
        with open(components_path, "w") as f:
            json.dump(best_candidate, f, indent=2)
        print(f"Best prompt components saved to: {components_path}")

        # Rename output folders from task_0, task_1 to actual task names
        rename_output_folders_to_task_names(run_dir, valset)

    return result


if __name__ == "__main__":
    main()
