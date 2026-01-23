# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Triton-ag is an agent-based system for generating optimized CUDA/Triton kernels from PyTorch code. It uses LLM agents to iteratively write, evaluate, and optimize GPU kernels, with a reinforcement learning training loop (GRPO/GSPO) to improve the model's kernel generation capabilities.

**Key Components:**
- **Workflow System**: Queue-based orchestration (`workflowServer.py`, `workflowClient.py`)
- **Inference Pipeline**: Code generation with log probability collection (`inferenceComposer.py`)
- **Kernel Evaluation**: GPU compilation and benchmarking (`kbEvalServer.py`)
- **Training**: GRPO/GSPO, RFT, UFT trainers (`trainerGRPO.py`, `trainerRFT.py`, `trainerUFT.py`)
- **Sync**: LoRA hot-swap to vLLM servers (`workflowSync.py`)

For detailed architecture, see `claude/CODEBASE_OVERVIEW.md`.

## Common Commands

```bash
# Environment setup (uses Docker)
make build_docker          # Build the Docker image
make env_start             # Start Docker container in background
make env                   # Enter the running container

# Required directories (create on host before container)
mkdir -p ~/.workflow ~/.trainer ~/.inference ~/.keys ~/.cache ~/.kbeval

# Run the kernel coder agent (standalone mode)
python agent_kernel_coder.py -p <provider> -m <model>
# Example: python agent_kernel_coder.py -p deepseek -m deepseek-chat

# Background services
make mlflow                # Start MLflow server for experiment tracking
make kbEval                # Start kernel evaluation server
make workflow_server       # Start workflow orchestration server

# Fine-tuning
make finetune              # Data parallel fine-tuning on 4 GPUs
make finetune-single       # Single GPU fine-tuning
make finetune-2gpu         # Data parallel on 2 GPUs

# vLLM inference servers (various configurations in Makefile)
make vllm-qwen3-32b-devserver   # Start local Qwen3-32B server

# Jupyter notebooks
make jupyter               # Start Jupyter on port 8086
```

## Full RL Workflow Commands

### Workflow Server
```bash
# Start workflow server (typically in tmux session "workflow")
while true; do python ./workflowServer.py --host :: --port 8488; sleep 5; done
```

### Kernel Evaluation Server
```bash
# Start kbEval server (typically in tmux session "kbEval")
while true; do python ./kbEvalServer.py --local_host --port 5676 --device 7; sleep 1; done
```

### Inference Workers (run multiple)
```bash
# Worker 1
while true; do python ./inferenceComposer.py --prefix_tag my_tag.a --use_global_queue codeGenEval.base --proc_id 01; sleep 5; done

# Worker 2
while true; do python ./inferenceComposer.py --prefix_tag my_tag.a --use_global_queue codeGenEval.base --proc_id 02; sleep 5; done
```

### Trainer
```bash
CUDA_VISIBLE_DEVICES=0 python trainerMain.py --prefix_tag my_tag.a
```

### Sync Worker
```bash
while true; do python ./workflowSync.py --prefix_tag my_tag.a --module_file workflow/sync.module.vllm+logp.yaml; sleep 5; done
```

### Log Probability Servers (for GRPO)
```bash
CUDA_VISIBLE_DEVICES=1 python inferenceCustomServer.py --prefix_tag my_tag.a
CUDA_VISIBLE_DEVICES=2 python inferenceCustomServer.py --prefix_tag my_tag.a --port 8003
```

## Standalone Commands (without workflow)

```bash
# Inference with local vLLM (has log probs, can be used for GRPO)
python ./inferenceComposer.py --prefix_tag my_tag.a --input_dir kernel_bench/level1/ --provider local --model qwen3-14b

# Inference with 3rd party API (no log probs, SFT only)
python ./inferenceComposer.py --prefix_tag my_tag.a --input_dir kernel_bench/level1/ --module_file inference/codeGenEval.module.chat.yaml --provider deepseek --model deepseek-chat

# GRPO training on collected data
python ./trainerGRPO.py --input_dir ~/.inference/output/ --input_tag my_tag.a_20250815_201330

# SFT training
python ./trainerSFT.py --input_dir ~/.inference/output --input_tag my_tag.a_20250813_034658

# Initialize a new workflow
python ./workflowInit.py --prefix_tag my_tag.a
```

## Architecture

### Core Components

**Agent System** (`agent_*.py`, `agentUtil.py`)
- `agent_kernel_coder.py`: Main agent that generates CUDA kernels iteratively
- Uses the `agents` library with MCP (Model Context Protocol) for tool integration
- Agents are configured via `agent.yaml` with provider/model settings

**Kernel Evaluation** (`kbEval*.py`, `kbEvalUtil.py`)
- `kbEvalServer.py`: FastAPI server that compiles and benchmarks generated kernels
- `kbEvalClient.py`: Client for submitting kernel evaluation requests
- `kbEvalUtil.py`: Core utilities for kernel execution, timing, and correctness checking
- Evaluates kernels using `torch.utils.cpp_extension.load_inline`

**Workflow System** (`workflow*.py`)
- `workflowServer.py`: FastAPI server managing training/inference workflows
- `workflowRegistry.py`: Tracks workflow configurations and queues
- `workflowClient.py`: Client for submitting inference/training jobs
- `workflowUtil.py`: Block definitions (InferenceBlock, TrainerBlock, WorkflowSyncBlock)
- Workflows are defined in `workflow/*.yaml` files

**Training** (`trainer*.py`, `engine*.py`)
- `trainerMain.py`: Training coordinator managing RFT, GRPO, UFT trainers
- `trainerGRPO.py`: GRPO/GSPO reinforcement learning trainer
- `trainerSFT.py`, `trainerUFT.py`: Supervised fine-tuning variants
- `engineBase.py`, `engineFSDP.py`, `engineDeepspeed.py`: Training backends
- Uses LoRA for parameter-efficient fine-tuning

**Inference** (`inference*.py`)
- `inferenceComposer.py`: Main orchestrator with parallel workers
- `inferenceClient.py`: Client for LLM inference requests
- `inferenceCustomServer.py`: Custom inference server for log probability computation
- `configInterpreter.py`: YAML-driven execution engine
- Supports vLLM, SGLang, llama.cpp backends

### Configuration Files

- `model.yaml`: LLM provider configurations (API keys, base URLs, model settings)
- `agent.yaml`: Agent configurations (which model/provider each agent uses)
- `workflow.yaml`: Workflow registry and server settings
- `kbEval.yaml`: Kernel evaluation server/client configurations
- `inferenceClient.yaml`: Inference provider settings
- `engineBase.yaml`, `trainerGRPO.yaml`, `trainerRFT.yaml`: Training configs
- API keys are stored in `${HOME}/.keys/{provider}.api.key`

### Data Flow

1. Agent receives a PyTorch model to optimize
2. Agent generates CUDA kernel code iteratively (multi-turn with TreeTurns)
3. `kbEvalServer` compiles and benchmarks each iteration
4. Results feed back to agent for next iteration
5. Successful rollouts are collected for RL training via workflow system
6. `trainerGRPO` trains the model using GSPO algorithm
7. `workflowSync` hot-swaps LoRA adapters to vLLM for next epoch

### Data Directories

| Path | Content |
|------|---------|
| `~/.workflow/` | Workflow registry data, queue persistence |
| `~/.inference/output/` | Generated code, log probs, eval results |
| `~/.inference/sft/` | SFT-formatted training data |
| `~/.trainer/` | Model checkpoints |
| `~/.kbeval/` | Evaluation cache |

## Code Conventions

- Use `logger` from `logger.py` instead of `print` statements
- Test files go in `generated_test/` subdirectories with `_test.py` suffix
- Follow DRY principle but avoid unnecessary one-liner helper functions
- Do not change sequence length or model name in configs (user-controlled)
- Use `source .venv/bin/activate` for Python environment
- Use `export CUDA_VISIBLE_DEVICES=X,Y` to manage GPU allocation
- Use `export HF_TOKEN=$(cat ${HOME}/.keys/huggingface.api.key)` for HuggingFace access

## Key Dependencies

- `agents`: OpenAI Agents SDK for agent orchestration
- `torch`, `triton`: GPU kernel development
- `vllm`, `sglang`: LLM inference backends
- `unsloth`: Efficient fine-tuning
- `fastapi`, `uvicorn`: API servers
- `loguru`: Logging (wrapped in `logger.py`)
- `duckdb`: Data storage and querying
- `pydantic`: Configuration models

## Workflow Configuration

### Adding a New Workflow

1. Add to `workflow.yaml`:
```yaml
registry:
  my_tag.a:
    short_name: "ma"
    config_path: "workflow/my.a.yaml"
    data_dir: "~/.workflow"
```

2. Create `workflow/my.a.yaml` (copy from existing template like `14B.n.yaml`):
```yaml
global:
  prefix_tag: "my_tag.a"
  start_epoch: 0
  start_block: 0
  end_epoch: 20
  end_block: 24
```

3. Initialize: `python ./workflowInit.py --prefix_tag my_tag.a`

### Key Workflow Terminology

| Term | Definition |
|------|------------|
| `prefix_tag` | Unique identifier for a workflow run |
| `epoch_id` | Training epoch number |
| `block_id` | Block number within an epoch |
| `input_tag` | Combined: `{prefix_tag}_{epoch_id:03d}_{block_id:02d}` |
| `queue_name` | Task queue identifier (e.g., `inference.codeGenEval.1`) |

## Useful Tmux Aliases

```bash
alias tn='tmux new-session -s'
alias ta='tmux attach -t'
alias tl='tmux list-sessions'
alias tk='tmux kill-session -t'
```
