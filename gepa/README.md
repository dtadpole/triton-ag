# GEPA Integration for Triton-AG

**GEPA** (Genetic-Pareto) is a framework for **optimizing text components of systems** (like AI prompts, instructions, or code snippets) using LLM-based reflection and evolutionary search with Pareto-efficient candidate selection.

This module integrates GEPA into the triton-ag project for **CUDA kernel generation prompt optimization**.

## Quick Start
```
# example to run the gepa code
python gepa/scripts/optimize_cuda_kernel.py --generate-reference-runtimes --run-dir shared/gepa_runs/t20/
```
### 1. Build and Run with Docker

```bash
cd triton-ag/gepa

# Build the Docker image
docker build -t gepa:latest .

# Run with your vLLM server
docker run -it --rm \
    -v $(pwd)/..:/workspace \
    gepa:latest bash
```

### 2. Run CUDA Kernel Prompt Optimization

```bash
# Using default config
python scripts/optimize_cuda_kernel.py

# With specific provider and model
python scripts/optimize_cuda_kernel.py \
    --provider h100_8_5_a \
    --model qwen3-32b \
    --max-metric-calls 100

# With custom config
python scripts/optimize_cuda_kernel.py --config configs/cuda_kernel.yaml
```

## Project Structure

```
gepa/
├── __init__.py
├── README.md
├── Dockerfile                    # Docker image (extends triton_ag)
├── requirements.txt
├── adapters/
│   ├── __init__.py
│   └── cuda_kernel_adapter.py    # CUDA kernel generation adapter
├── scripts/
│   ├── __init__.py
│   ├── test_dependencies.py      # Dependency verification
│   ├── vllm_integration.py       # vLLM integration scripts
│   └── optimize_cuda_kernel.py   # Main optimization script
└── configs/
    └── cuda_kernel.yaml          # CUDA optimization config
```

## How It Works

### GEPA Optimization Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    GEPA Optimization                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Seed Prompt ─────► 2. Evaluate on Training Data         │
│         │                         │                         │
│         │                         ▼                         │
│         │              3. Reflection LM Analyzes Failures   │
│         │                         │                         │
│         │                         ▼                         │
│         │              4. Propose Improved Prompt           │
│         │                         │                         │
│         └──────────────◄──────────┘                         │
│                                                             │
│  5. Repeat until budget exhausted or convergence            │
│                                                             │
│  6. Return best prompt from Pareto frontier                 │
└─────────────────────────────────────────────────────────────┘
```

### CudaKernelAdapter

The adapter connects GEPA to your CUDA kernel generation task:

```python
from gepa.adapters import CudaKernelAdapter, CudaKernelDataInst

# Create adapter using your InferenceClient config
adapter = CudaKernelAdapter(
    model_name="qwen3-32b",
    provider_name="h100_8_5_a",
    config_file="inferenceClient.yaml",
)

# Define training data
trainset = [
    CudaKernelDataInst(
        task_id="elementwise_add",
        reference_code="...",  # PyTorch reference code
    ),
]

# Optimize!
result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a CUDA expert..."},
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    reflection_lm=reflection_lm,
    max_metric_calls=150,
)
```

## Configuration

### configs/cuda_kernel.yaml

```yaml
defaults:
  task_model: "qwen3-32b"
  reflection_model: "qwen3-32b"
  task_provider: "h100_8_5_a"
  reflection_provider: "h100_8_5_a"

optimization:
  max_metric_calls: 150
  candidate_selection_strategy: "pareto"
  use_merge: true

seed_candidate:
  system_prompt: |
    You are an experienced CUDA developer...
```

### Available Providers (from inferenceClient.yaml)

| Provider | Description |
|----------|-------------|
| `h100_8_5_a` | Local H100 8-GPU (port 8001) |
| `h100_8_5_b` | Local H100 8-GPU (port 8002) |
| `h100_8b` | Local H100 8-GPU (port 8091) |
| `fireworks` | Fireworks AI API |
| `deepseek` | DeepSeek API |

### Available Models

| Model | Description |
|-------|-------------|
| `qwen3-32b` | Qwen3 32B - Good balance |
| `deepseek-r1` | DeepSeek R1 - Strong reasoning |
| `deepseek-v3` | DeepSeek V3 - Fast inference |

## Custom Evaluator

You can provide a custom evaluator for more sophisticated code evaluation:

```python
def custom_evaluator(data: CudaKernelDataInst, generated_code: str):
    """
    Custom evaluator that compiles and tests the CUDA code.

    Returns:
        (score, output, feedback)
    """
    # 1. Try to compile the code
    compile_success, compile_error = try_compile(generated_code)
    if not compile_success:
        return (0.0, output, f"Compilation failed: {compile_error}")

    # 2. Test for correctness
    correct, correctness_error = test_correctness(generated_code, data)
    if not correct:
        return (0.5, output, f"Correctness failed: {correctness_error}")

    # 3. Measure speedup
    speedup = measure_speedup(generated_code, data)

    # Score based on speedup (1.0 = matches reference, >1.0 = faster)
    score = min(speedup, 2.0) / 2.0  # Cap at 2x speedup
    feedback = f"Success! Speedup: {speedup:.2f}x"

    return (score, output, feedback)

# Use with adapter
adapter = CudaKernelAdapter(
    model_name="qwen3-32b",
    provider_name="h100_8_5_a",
    evaluator_fn=custom_evaluator,
)
```

## References

- **GEPA Paper**: [Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)
- **GEPA GitHub**: [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- **DSPy Integration**: [https://dspy.ai/tutorials/gepa_ai_program/](https://dspy.ai/tutorials/gepa_ai_program/)
