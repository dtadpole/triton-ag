# Triton-AG Codebase Overview

A detailed guide to understanding the triton-ag codebase structure, data flow, and how components interact.

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [System Architecture](#system-architecture)
3. [Component Reference Table](#component-reference-table)
4. [Workflow Orchestration System](#workflow-orchestration-system)
5. [Phase 1: Inference Pipeline](#phase-1-inference-pipeline)
6. [Phase 2: Training Pipeline](#phase-2-training-pipeline)
7. [Phase 3: Sync & LoRA Hot-Swap](#phase-3-sync--lora-hot-swap)
8. [Data Flow Diagrams](#data-flow-diagrams)
9. [Configuration System](#configuration-system)
10. [GEPA: Prompt Optimization](#gepa-prompt-optimization)
11. [File-by-File Reference](#file-by-file-reference)
12. [Typical Workflows](#typical-workflows)

---

## What This Project Does

Triton-ag is an **automated GPU kernel code generation and RL training system** that:

1. **Generates CUDA/Triton kernels** using LLMs (Qwen, DeepSeek, Claude, etc.)
2. **Evaluates generated kernels** for correctness and performance (speedup)
3. **Iterates** based on evaluation feedback using multi-turn conversations (TreeTurns)
4. **Collects generation traces** with log probabilities for training
5. **Fine-tunes models** using GRPO/GSPO, RFT (rejection fine-tuning), or UFT
6. **Synchronizes LoRA adapters** to vLLM servers for hot-swapping during training
7. **Optimizes prompts** using evolutionary algorithms (GEPA)

The end goal: train specialized models that can generate high-quality GPU kernels efficiently.

### Quick Reference: Running the Full RL Workflow

Before diving into details, here's the practical workflow from the operational guide:

```bash
# Setup directories (on host, before container)
mkdir -p ~/.workflow ~/.trainer ~/.inference ~/.keys ~/.cache ~/.kbeval

# Start container with mounts
docker run -it --gpus all --net=host \
  -v ~/.workflow/:/root/.workflow/ \
  -v ~/.trainer/:/root/.trainer/ \
  -v ~/.inference/:/root/.inference/ \
  -v ~/.keys/:/root/.keys/ \
  -v ~/.cache/:/root/.cache/ \
  -v ~/.kbeval/:/root/.kbeval/ \
  -v ${PWD}:/workspace/ \
  localhost/triton_ag /bin/bash

# Start workflow server (Terminal 1)
while true; do python ./workflowServer.py --host :: --port 8488; sleep 5; done

# Start kbEval server (Terminal 2)
while true; do python ./kbEvalServer.py --local_host --port 5676 --device 7; sleep 1; done

# Start inference workers (Terminal 3-4, multiple workers)
while true; do python ./inferenceComposer.py --prefix_tag my_tag.a --use_global_queue codeGenEval.base --proc_id 01; sleep 5; done

# Start trainer (Terminal 5)
CUDA_VISIBLE_DEVICES=0 python trainerMain.py --prefix_tag my_tag.a

# Start sync (Terminal 6)
while true; do python ./workflowSync.py --prefix_tag my_tag.a --module_file workflow/sync.module.vllm+logp.yaml; sleep 5; done
```

---

## System Architecture

### High-Level Component Overview

```
+==================================================================================+
|                              TRITON-AG RL TRAINING SYSTEM                        |
+==================================================================================+
|                                                                                  |
|  +----------------------------------------------------------------------------+  |
|  |                     WORKFLOW ORCHESTRATION LAYER                           |  |
|  |                                                                            |  |
|  |   workflowServer.py  <-->  workflowClient.py  <-->  workflowRegistry.py   |  |
|  |          |                        |                        |               |  |
|  |          +------------------------+------------------------+               |  |
|  |                                   |                                        |  |
|  |            Queue-based task distribution via FastAPI REST                  |  |
|  |              (inference, trainer, sync queues)                             |  |
|  |                                                                            |  |
|  +----------------------------------------------------------------------------+  |
|                                      |                                           |
|            +-------------------------+-------------------------+                 |
|            |                         |                         |                 |
|            v                         v                         v                 |
|  +-----------------+      +-----------------+      +-----------------+           |
|  |   INFERENCE     |      |    TRAINING     |      |      SYNC       |           |
|  |                 |      |                 |      |                 |           |
|  | inferenceComposer  |      | trainerMain.py |      | workflowSync.py |           |
|  |       .py       |      |                 |      |                 |           |
|  |                 |      | +-------------+ |      | LoRA adapter    |           |
|  | +-------------+ |      | | trainerGRPO | |      | sync to vLLM    |           |
|  | |inferenceClient | |      | | trainerRFT  | |      | servers         |           |
|  | |  kbEvalClient  | |      | | trainerUFT  | |      |                 |           |
|  | +-------------+ |      | +-------------+ |      +-----------------+           |
|  +-----------------+      +-----------------+                                    |
|         |                        |                                               |
|         v                        v                                               |
|  +-----------------+      +-----------------+                                    |
|  |  vLLM Server    |      |  Unsloth/HF    |                                    |
|  |  (generation +  |      |  Training      |                                    |
|  |   log probs)    |      |  Engine        |                                    |
|  +-----------------+      +-----------------+                                    |
|         |                        |                                               |
|         v                        v                                               |
|  +-----------------+      +-----------------+                                    |
|  | kbEvalServer.py |      | Checkpoint     |                                    |
|  | (GPU eval)      |      | Output         |                                    |
|  +-----------------+      +-----------------+                                    |
|                                                                                  |
+==================================================================================+
```

---

## Component Reference Table

This table maps each major component to its source code, configuration, and data directories.

| Component | Source Code | Config File | Data Directory |
|-----------|------------|-------------|----------------|
| **Workflow** | `workflowServer.py`, `workflowClient.py`, `workflowRegistry.py` | `workflow.yaml` | `~/.workflow/*` |
| **Inference** | `inferenceComposer.py`, `inferenceClient.py` | `inferenceClient.yaml`, `inference/*.yaml` | `~/.inference/**` |
| **KbEval** | `kbEvalServer.py`, `kbEvalClient.py` | `kbEval.yaml` | `~/.kbeval` |
| **Training** | `trainerMain.py`, `trainerGRPO.py`, `trainerRFT.py`, `trainerUFT.py` | `trainerGRPO.yaml`, `trainerRFT.yaml`, `engineBase.yaml` | `~/.trainer/**` |
| **Sync** | `workflowSync.py`, `workflowRsync.py` | `workflow/sync.module.*.yaml` | N/A |
| **Config Interpreter** | `configInterpreter.py`, `configEndpoints.py` | `*.module.yaml` files | N/A |

---

## Workflow Orchestration System

### Core Concepts

The workflow system uses a **queue-based architecture** to coordinate between inference, training, and sync components.

#### Key Terminology

| Term | Definition |
|------|------------|
| **prefix_tag** | Unique identifier for a workflow run (e.g., `TC_0.1.0_32B.c`) |
| **epoch_id** | Training epoch number (increments each full cycle) |
| **block_id** | Block number within an epoch |
| **input_tag** | Combined identifier: `{prefix_tag}_{epoch_id:03d}_{block_id:02d}` |
| **queue_name** | Task queue identifier (e.g., `inference.codeGenEval.1`, `trainer.grpo.1`) |

#### Block Types

Three Pydantic models define task blocks (from `workflowUtil.py`):

```python
class InferenceBlock(BaseModel):
    queue_type: str = "inference"
    queue_name: str
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    model_name: str
    num_samples: int = 20
    num_generations: int = 8
    num_turns_per_generation: int = 4
    parallel_workers: int = 32
    vllm_providers: list[str] = ["local"]
    logp_providers: list[str] = ["local"]
    kbeval_providers: list[str] = ["local"]
    module_file: str = "inference/codeGen.module.vllm+logp.yaml"
    prompt_file: str = "inference/triton.prompt.yaml"
    example_file: str = "inference/triton.example.yaml"
    input_dir: str = "~/KernelBench/KernelBench"
    output_dir: str = "~/.inference/output"
    sft_dir: str = "~/.inference/sft"
    context: dict = {}
    # TreeTurns parameters (new in latest version)
    hint_length_prob: float = 0        # Hint length probability (https://arxiv.org/pdf/2505.16984)
    min_num_generations: int = 3       # Wait for min generations before next turn
    selection_strategy: str = "random" # "random" or "best_speedup"

class TrainerBlock(BaseModel):
    queue_type: str = "trainer"
    queue_name: str
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str
    output_dir: str
    ...

class WorkflowSyncBlock(BaseModel):
    queue_type: str = "sync"
    queue_name: str
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    module_file: str
    ...
```

### WorkflowServer Architecture

The `WorkflowServer` (workflowServer.py:18) manages multiple registries, each with its own queue storage:

```
WorkflowServer
    |
    +-- registries: Dict[prefix_tag, WorkflowRegistry]
    |       |
    |       +-- WorkflowRegistry (prefix_tag="TC_0.1.0_32B.c")
    |       |       +-- QUEUE_PREFIX + "inference.codeGenEval.1" --> asyncio.Queue
    |       |       +-- QUEUE_PREFIX + "trainer.grpo.1" --> asyncio.Queue
    |       |       +-- QUEUE_PREFIX + "sync.sync.1" --> asyncio.Queue
    |       |       +-- ADAPTER_PREFIX + "checkpoint-100" --> adapter_data
    |       |
    |       +-- WorkflowRegistry (prefix_tag="TC_0.2.0_14B.a")
    |               +-- ...
    |
    +-- FastAPI endpoints:
            /queue/enqueue/{prefix_tag}/{queue_name}
            /queue/dequeue/{prefix_tag}/{queue_name}
            /queue/qsize/{prefix_tag}/{queue_name}
            /get/{prefix_tag}/{key}
            /put/{prefix_tag}/{key}
```

### Callback Chain

When a task completes, it triggers a callback that may enqueue the next task:

```
InferenceBlock completes
    |
    +-- workflowClient.callback(callback_kind="completion", ...)
            |
            +-- Checks workflow config for next step
            +-- Enqueues TrainerBlock if configured
            +-- Or enqueues next InferenceBlock
```

---

## Phase 1: Inference Pipeline

### Overview

The inference pipeline generates code and collects training data with log probabilities.

### Flow Diagram

```
+--------------------------------------------------------------------------+
|                        INFERENCE PIPELINE                                 |
+--------------------------------------------------------------------------+

  InferenceBlock from queue
        |
        v
  +------------------+
  | ComposerClient   |  (inferenceComposer.py:42)
  | - input_processor|
  | - queue_workers  |
  +--------+---------+
           |
           |  for each (task, generation, turn):
           v
  +------------------+     +------------------+     +------------------+
  | InferenceClient  | --> | vLLM Server      | --> | Generated Code   |
  | (multi-provider) |     | (local/remote)   |     | + Log Probs      |
  +--------+---------+     +------------------+     +--------+---------+
           |                                                 |
           v                                                 v
  +------------------+     +------------------+     +------------------+
  | KbEvalClient     | --> | KbEvalServer     | --> | Eval Results:    |
  |                  |     | (GPU execution)  |     | - compiled       |
  +------------------+     +------------------+     | - correctness    |
                                                   | - runtime        |
                                                   | - ref_runtime    |
                                                   +--------+---------+
                                                            |
                                                            v
                                                   +------------------+
                                                   | DuckDB Storage   |
                                                   | (Parquet files)  |
                                                   +------------------+
```

### Key Files

| File | Lines | Role |
|------|-------|------|
| `inferenceComposer.py` | 427 | Main orchestrator. Creates ComposerClient, manages parallel workers |
| `inferenceClient.py` | - | Multi-provider LLM client wrapper |
| `kbEvalClient.py` | - | Client for kernel evaluation server |
| `kbEvalServer.py` | - | FastAPI server for GPU kernel compilation and benchmarking |
| `configInterpreter.py:74` | - | YAML-driven workflow execution engine |

### ConfigInterpreter: YAML-Driven Execution

The system uses YAML configuration files (`.module.yaml`) to define execution pipelines:

```yaml
# Example: inference/codeGenEval.module.vllm+logp.yaml
module:
  context_vars:
    duckdb: "`DuckDBClient()`"

  input_processor:
    generate_samples:
      steps:
        - type: "python_async"
          code: |
            # Generate code samples
            ...
        - type: "endpoint_vllm"
          endpoint: "{vllm_provider}"
          prompt: "{prompt}"
          ...
        - type: "endpoint_kbeval"
          endpoint: "{kbeval_provider}"
          code: "{generated_code}"
          ...
```

The `ConfigInterpreter` (configInterpreter.py:74) processes these YAML configs:
- `_process_variable()`: Evaluates `` `backtick expressions` `` or `{format strings}`
- `_process_steps()`: Executes step sequences (python, python_async, endpoint_*, etc.)
- `execute()`: Main entry point for running a config block

### Data Collection for Training

Each inference run collects:

1. **Prompt tokens** (from vLLM tokenization)
2. **Completion tokens** (generated code)
3. **Log probabilities** (from vLLM)
4. **Evaluation results** (compiled, correctness, runtime, speedup)

Stored as Parquet files in `~/.inference/output/{input_tag}/`:
```
~/.inference/output/TC_0.1.0_32B.c_001_02/
    +-- task_001/
    |       +-- gen_00/
    |       |       +-- turn_00.parquet
    |       |       +-- turn_01.parquet
    |       |       +-- turn_02.parquet
    |       +-- gen_01/
    |               +-- ...
    +-- task_002/
            +-- ...
```

---

## Phase 2: Training Pipeline

### Overview

The training pipeline consumes inference data and trains the model using GRPO/GSPO, RFT, or UFT.

### TrainerMain: Training Coordinator

The `TrainerMain` class (trainerMain.py:23) coordinates multiple trainers:

```python
class TrainerMain:
    def __init__(self, engine: EngineBase, prefix_tag: str):
        self.engine = engine
        self.prefix_tag = prefix_tag
        self.reg_client = WorkflowClient(prefix_tag=prefix_tag)

    async def main_loop_task(self):
        # Initialize all three trainers
        rft_trainer = rft_get_trainer(...)
        grpo_trainer = grpo_get_trainer(...)
        uft_trainer = uft_get_trainer(...)

        while True:
            # Check queue sizes
            rft_qsize = await self.reg_client.qsize('trainer.rft.1')
            grpo_qsize = await self.reg_client.qsize('trainer.grpo.1')
            uft_qsize = await self.reg_client.qsize('trainer.uft.1')

            # Probabilistically select queue based on size
            # (weighted random selection with ALPHA=1.1 enhancement)
            if random.random() < grpo_prob:
                grpo_item = await self.reg_client.dequeue('trainer.grpo.1')
                await grpo_trainer.train_grpo_block(...)
```

### Training Types

#### 1. GRPO/GSPO Trainer (trainerGRPO.py)

Group Relative Policy Optimization with sequence-level clipping (GSPO variant):

```python
class GRPOConfig(BaseModel):
    """GRPO configuration from trainerGRPO.py"""
    max_seq_length: int = 16384
    ave_seq_length: int = 8000  # For constant length normalizer (Dr. GRPO fix)
    mask_non_assistant_tokens: bool = True
    discard_long_conversations: bool = True

    # Clipping parameters
    clip_ratio_epsilon_lower: float = 0.2
    clip_ratio_epsilon_upper: float = 0.3
    gspo_clip_ratio_epsilon_lower: float = 3e-4
    gspo_clip_ratio_epsilon_upper: float = 4e-4
    bound_advantage_range: float = 3.0

    # KL and regularization
    beta: float = 0.0  # KL divergence coefficient
    entropy_coeff: float = 0.0  # Entropy regularization

    # Reward configuration
    reward_scale: bool = True
    reward_scale_value: Optional[float] = None  # Use this instead of std if set
    reward_epsilon: float = 1e-3
    reward_noise: float = 1e-2

    # Reward components
    speedup_reward: float = 0.3
    correctness_reward: float = 0.3
    improvement_bonus: float = 0.2
    good_reward_threshold: float = 0.3
    bad_reward_threshold: float = 0.05

    # Loss type: "episode", "token", "seq_max", or "gspo"
    loss_type: str = "gspo"
    gamma: float = 0.5

    # Importance sampling
    use_truncated_is: bool = False
    truncated_is_ratio: float = 2.0
    clip_gradient_scale: float = 0.0
    pkpo_advantages_k: int = 1

    # Dr. GRPO length bias fix (from "Understanding R1-Zero-Like Training")
    # When True, uses constant normalizer instead of response length
    gspo_use_constant_length_normalizer: bool = False
```

**GSPO vs GRPO:**
- GRPO: Token-level probability ratios
- GSPO: Sequence-level probability ratios with length normalization

**Loss Computation (trainerGRPO.py:200-514):**
```python
def _compute_mini_batch_loss(self, batch, clip_metrics, ...):
    for i in range(len(advantages)):
        # Get log probs from forward pass
        forward_completion_log_probs = self._calculate_log_probs(...)

        # Compute ratio: exp(log_pi_new - log_pi_old)
        raw_log_ratio = forward_completion_log_probs - generation_completion_log_probs

        if self.grpo_config.loss_type == "gspo":
            # Sequence-level ratio with length normalization
            sequence_log_ratio = torch.sum(log_ratio) / self.grpo_config.ave_seq_length
            sequence_ratio = torch.exp(sequence_log_ratio)

            # Clip and compute advantage-weighted loss
            clamped_sequence_ratio = torch.clamp(sequence_ratio,
                1 - effective_eps_lower, 1 + effective_eps_upper)
            sequence_ratio_advantage = torch.min(
                sequence_ratio * advantages[i],
                clamped_sequence_ratio * advantages[i])
            loss = -final_sequence_ratio_advantage.mean()
```

#### 2. RFT Trainer (trainerRFT.py)

Rejection Fine-Tuning - selects top-performing responses for SFT:

```python
class RFTConfig(BaseModel):
    max_seq_length: int = 4096
    mask_non_assistant_tokens: bool = True
    mask_non_last_assistant_tokens: bool = True
    discard_long_conversations: bool = True
    return_top_percentile: float = 0.25  # Keep top 25%
```

#### 3. UFT Trainer (trainerUFT.py)

Unified Fine-Tuning - combines GRPO and SFT losses:

```python
class UFTConfig(BaseModel):
    beta_coef: float = 0.1  # Total = (1-beta)*grpo_loss + beta*sft_loss
```

### Reward Calculation

For tree-based multi-turn conversations (trainer/grpo_reward_kb_treeturns.py:17):

```python
def grpo_compute_rewards_v6_treeturns(
    query_result: list[dict],
    speedup_threshold_: float = 1.3,
    improvement_bonus_: float = 0.2,
    compile_reward_: float = 0.1,
    correctness_reward_: float = 0.2,
    speedup_reward_: float = 0.3,
):
    for turn in trajectory:
        # Correctness reward
        if not turn["compiled"]:
            correctness_reward = 0.0
        elif turn["compiled"] and not turn["correctness"]:
            correctness_reward = compile_reward_
        elif turn["compiled"] and turn["correctness"]:
            correctness_reward = correctness_reward_

        # Speedup reward (capped at speedup_reward_)
        speedup = turn["ref_runtime"] / turn["runtime"]
        speedup_reward = min(speedup_reward_,
            (speedup / speedup_threshold_)**4 * speedup_reward_)

        # Improvement bonus (only once per trajectory)
        if i > 0 and not had_improvement:
            if speedup > selected_speedup >= speedup_threshold_:
                improvement_bonus = improvement_bonus_
                had_improvement = True

        turn["reward"] = correctness_reward + speedup_reward + improvement_bonus
```

### Advantage Calculation

Advantages are computed using group normalization:

```python
# Group generations by (task_tag, turn_only_tag)
for group in groups:
    rewards = [gen["reward"] for gen in group]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) + eps

    for gen in group:
        gen["advantage"] = (gen["reward"] - mean_reward) / std_reward
```

---

## Phase 3: Sync & LoRA Hot-Swap

### Overview

After training produces new checkpoints, they must be synced to vLLM servers for the next inference round.

### Sync Flow

```
Training completes checkpoint
        |
        v
  +------------------+
  | Callback enqueues|
  | WorkflowSyncBlock|
  +--------+---------+
           |
           v
  +------------------+
  | WorkflowSync     |  (workflowSync.py:16)
  | dequeues task    |
  +--------+---------+
           |
           v
  +------------------+     +------------------+
  | ConfigInterpreter| --> | sync_task steps: |
  | executes module  |     | 1. rsync files   |
  +------------------+     | 2. reload vLLM   |
                           | 3. update logp   |
                           +------------------+
```

### Sync Module Configuration

```yaml
# workflow/sync.module.vllm+logp.yaml
sync_task:
  steps:
    - type: "endpoint_rsync"
      source: "{trainer_checkpoint_path}"
      targets: "{vllm_providers}"

    - type: "endpoint_vllm_reload"
      providers: "{vllm_providers}"
      adapter_name: "{checkpoint_name}"

    - type: "endpoint_logp_reload"
      providers: "{logp_providers}"
      adapter_name: "{checkpoint_name}"
```

### vLLM LoRA Hot-Swap

The vLLM server supports dynamic LoRA adapter loading:

```python
# inferenceClient.py - vLLM endpoint with LoRA
response = await client.chat.completions.create(
    model=model_name,  # e.g., "qwen3-32b"
    extra_body={
        "lora_adapter": adapter_name,  # e.g., "TC_0.1.0_32B.c/checkpoint-100"
    },
    messages=messages,
    logprobs=True,
    ...
)
```

---

## Data Flow Diagrams

### Complete RL Training Loop

```
+==========================================================================+
|                        COMPLETE RL TRAINING LOOP                          |
+==========================================================================+

  EPOCH 0                    EPOCH 1                    EPOCH 2
  +-----+                    +-----+                    +-----+
  |     |                    |     |                    |     |
  v     |                    v     |                    v     |
+-------+----+         +----------+----+         +----------+----+
| INFERENCE  |         | INFERENCE     |         | INFERENCE     |
| (base model)         | (checkpoint-50)          | (checkpoint-100)
+------+-----+         +------+--------+         +------+--------+
       |                      |                         |
       v                      v                         v
+------+-----+         +------+--------+         +------+--------+
| TRAINING   |         | TRAINING      |         | TRAINING      |
| GRPO/RFT   |         | GRPO/RFT      |         | GRPO/RFT      |
+------+-----+         +------+--------+         +------+--------+
       |                      |                         |
       v                      v                         v
+------+-----+         +------+--------+         +------+--------+
| checkpoint |         | checkpoint    |         | checkpoint    |
|    -50     |-------->|    -100       |-------->|    -150       |
+------+-----+   sync  +------+--------+   sync  +------+--------+
       |                      |                         |
       +----------------------+-------------------------+
                              |
                              v
                    +---------+---------+
                    | SYNC to vLLM      |
                    | Hot-swap adapter  |
                    +-------------------+
```

### Data Flow Within a Single Epoch

```
                         InferenceBlock
                              |
                              v
     +------------------------+------------------------+
     |                        |                        |
     v                        v                        v
+---------+            +---------+              +---------+
| Task 1  |            | Task 2  |    ...       | Task N  |
+---------+            +---------+              +---------+
     |                        |                        |
     v                        v                        v
+----+----+             +----+----+              +----+----+
|Gen 0..7 |             |Gen 0..7 |              |Gen 0..7 |
+---------+             +---------+              +---------+
     |                        |                        |
     +------------------------+------------------------+
                              |
                              v
                    +---------+---------+
                    | DuckDB Query      |
                    | (join all results)|
                    +---------+---------+
                              |
                              v
                    +---------+---------+
                    | Compute Rewards   |
                    | Group by turn     |
                    +---------+---------+
                              |
                              v
                    +---------+---------+
                    | Compute Advantages|
                    | (group normalize) |
                    +---------+---------+
                              |
                              v
                    +---------+---------+
                    | TrainerBlock      |
                    | (GRPO/RFT/UFT)    |
                    +---------+---------+
                              |
                              v
                    +---------+---------+
                    | New Checkpoint    |
                    +-------------------+
```

---

## Configuration System

### Configuration Hierarchy

```
+-----------------------------+     +-----------------------------+
|       workflow.yaml         |     |     inferenceClient.yaml    |
+-----------------------------+     +-----------------------------+
| registry:                   |     | providers:                  |
|   TC_0.1.0_32B.c:           |     |   local:                    |
|     config_path: workflow/  |     |     base_url: http://...    |
|       tc_32b.yaml           |     |     models:                 |
|     data_dir: ~/.workflow   |     |       qwen3-32b: ...        |
|     short_name: tc          |     |   h8_3:                     |
+-----------------------------+     |     base_url: http://...    |
                                    +-----------------------------+

+-----------------------------+     +-----------------------------+
|      engineBase.yaml        |     |     trainerGRPO.yaml        |
+-----------------------------+     +-----------------------------+
| model:                      |     | grpo:                       |
|   name: Qwen/Qwen3-32B      |     |   loss_type: gspo           |
|   max_seq_length: 32768     |     |   speedup_reward: 0.3       |
| training:                   |     |   clip_ratio_epsilon: 0.2   |
|   learning_rate: 3e-5       |     |   use_truncated_is: true    |
|   micro_batch_size: 1       |     +-----------------------------+
| lora:                       |
|   r: 128                    |
|   alpha: 128                |
+-----------------------------+
```

### Module Files (YAML-Driven Execution)

Module files define execution pipelines that `ConfigInterpreter` processes:

```yaml
# inference/codeGenEval.module.vllm+logp.yaml
module:
  context_vars:
    duckdb: "`DuckDBClient()`"
    vllm_client: "`InferenceClient(provider='vllm')`"

  input_processor:
    setup:
      steps:
        - type: "python"
          code: |
            tasks = load_kernel_bench_tasks(input_dir)

  queue_worker:
    generate_and_eval:
      context_vars:
        task: "`context['task']`"
      steps:
        - type: "endpoint_vllm"
          provider: "{vllm_provider}"
          messages: "{messages}"

        - type: "endpoint_kbeval"
          provider: "{kbeval_provider}"
          code: "{generated_code}"

        - type: "python"
          code: |
            save_to_parquet(result, output_path)
```

---

## GEPA: Prompt Optimization

GEPA (Genetic Evolution for Prompt Adaptation) optimizes system prompts using evolutionary algorithms.

### GEPA Flow

```
+--------------------------------------------------------------------------+
|                      GEPA PROMPT OPTIMIZATION                             |
+--------------------------------------------------------------------------+

  Seed Prompt                      Evaluation
  "You are an expert..."           on training set
        |                               |
        v                               v
  +------------+    mutate    +------------+    evaluate    +------------+
  | Prompt 1   | -----------> | Prompt 1'  | ------------> | Score: 0.7 |
  +------------+              +------------+               +------------+
        |                           |
        |    crossover              |
        v                           v
  +------------+              +------------+               +------------+
  | Prompt 2   | <----------  | Prompt 2'  | ------------> | Score: 0.8 |
  +------------+              +------------+               +------------+
        |                                                        |
        |                                                        |
        v                                                        v
  +------------------------------------------------------------------+
  |                    Pareto Selection                               |
  |  Keep prompts on efficiency frontier (score vs length tradeoff)  |
  +------------------------------------------------------------------+
        |
        v
  Repeat N generations --> Best prompt found!
```

### GEPA Key Files

| File | Role |
|------|------|
| `gepa/adapters/cuda_kernel_adapter.py` | Adapter connecting GEPA to kernel generation |
| `gepa/scripts/optimize_cuda_kernel.py` | Main optimization script |
| `gepa/README.md` | Integration documentation |

---

## File-by-File Reference

### Root Directory - Workflow & Orchestration

| File | Lines | Description |
|------|-------|-------------|
| `workflowServer.py` | 323 | FastAPI server managing queues and registries |
| `workflowClient.py` | - | Client for interacting with workflow server |
| `workflowRegistry.py` | - | Storage backend for workflow data (queues, adapters) |
| `workflowUtil.py` | 125 | Block definitions (InferenceBlock, TrainerBlock, WorkflowSyncBlock) |
| `workflowSync.py` | - | LoRA adapter synchronization to vLLM servers |

### Root Directory - Inference

| File | Lines | Description |
|------|-------|-------------|
| `inferenceComposer.py` | 427 | Main inference orchestrator with parallel workers |
| `inferenceClient.py` | - | Multi-provider LLM client (vLLM, API providers) |
| `kbEvalServer.py` | - | FastAPI server for kernel evaluation |
| `kbEvalClient.py` | - | Client for kernel evaluation |
| `configInterpreter.py` | - | YAML-driven execution engine |
| `configEndpoints.py` | - | Endpoint implementations (vllm, kbeval, duckdb) |

### Root Directory - Training

| File | Lines | Description |
|------|-------|-------------|
| `trainerMain.py` | 173 | Training coordinator - manages RFT, GRPO, UFT trainers |
| `trainerGRPO.py` | 600+ | GRPO/GSPO trainer with clipping and IS weighting |
| `trainerRFT.py` | 184 | Rejection Fine-Tuning trainer |
| `trainerUFT.py` | 200+ | Unified Fine-Tuning (GRPO + SFT) |
| `trainerUtil.py` | - | Training utilities (formatting, checkpointing) |
| `engineBase.py` | - | Base training engine with Unsloth/HF support |

### Root Directory - Agents (Standalone Generation)

| File | Description |
|------|-------------|
| `agent_kernel_coder.py` | CUDA kernel generation agent (single run mode) |
| `agent_triton_coder.py` | Triton-specific kernel generation |
| `agent_planner.py` | Task planning agent |
| `agentUtil.py` | Agent helper functions |

### `/trainer/` Directory

| File | Description |
|------|-------------|
| `grpo_reward_kb.py` | Reward calculation for single-turn |
| `grpo_reward_kb_treeturns.py` | Reward calculation for tree-based multi-turn |
| `pkpo.py` | Advantage computation functions |
| `uft_dataset.py` | UFT dataset preparation |

### `/inference/` Directory

| Pattern | Description |
|---------|-------------|
| `*.module.*.yaml` | Module configs for different inference modes |
| `*.prompt.yaml` | System prompts for code generation |
| `*.example.yaml` | Few-shot examples |

### `/workflow/` Directory

| Pattern | Description |
|---------|-------------|
| `*.yaml` | Workflow configurations for different experiments |
| `sync.module.*.yaml` | Sync module configurations |

### Data Directories

| Path | Content |
|------|---------|
| `~/.workflow/` | Workflow registry data, queue persistence |
| `~/.inference/output/` | Generated code, log probs, eval results |
| `~/.inference/sft/` | SFT-formatted training data |
| `~/.trainer/` | Model checkpoints |
| `~/.kbeval/` | Evaluation cache |

---

## Typical Workflows

### Workflow 1: Full RL Training Loop

```bash
# Terminal 1: Start workflow server
python workflowServer.py --config_path workflow.yaml

# Terminal 2: Start kbEval server
python kbEvalServer.py

# Terminal 3: Start vLLM server
make vllm-qwen3-32b

# Terminal 4: Start inference workers
python inferenceComposer.py --prefix_tag TC_0.1.0_32B.c --use_global_queue inference.1

# Terminal 5: Start trainer
python trainerMain.py --prefix_tag TC_0.1.0_32B.c --engine unsloth

# Terminal 6: Start sync worker
python workflowSync.py --prefix_tag TC_0.1.0_32B.c
```

### Workflow 2: Single Kernel Generation (Agent Mode)

```bash
# Start eval server
make kbEval

# Run agent
python agent_kernel_coder.py -p deepseek -m deepseek-chat
```

### Workflow 3: Manual Training

```bash
# Run GRPO training on collected data
python trainerGRPO.py \
    --prefix_tag TC_0.1.0_32B.c \
    --input_tag TC_0.1.0_32B.c_001_02 \
    --engine unsloth
```

---

## Appendix: Key Concepts

### GRPO vs GSPO

| Aspect | GRPO | GSPO |
|--------|------|------|
| Ratio calculation | Token-level | Sequence-level (sum/avg) |
| Length normalization | Per-token | Constant normalizer |
| Clip bounds | Fixed epsilon | Length-adjusted epsilon |

### Log Probability Flow

```
vLLM Generation
    |
    +-- logprobs=True
    |
    v
+-----------------+
| For each token: |
| - token_id      |
| - logprob       |
| - top_logprobs  |
+-----------------+
    |
    v
Stored in Parquet --> Used in GRPO training
```

### TreeTurns Selection Strategy

For multi-turn conversations with multiple parallel generations:

```
Turn 0: Generate N kernels
    |
    +-- Select best (by speedup)
    |
    v
Turn 1: Continue from selected context
    |
    +-- Generate N kernels
    |
    +-- Select best
    v
Turn 2: Continue...
```

The `selection_strategy` in InferenceBlock controls this:
- `"random"`: Random selection from successful generations
- `"best_speedup"`: Select highest speedup

---

## Appendix: Practical Setup Reference

### Tmux Aliases for Workflow Management

The project recommends using tmux for running multiple workflow components. Useful aliases:

```bash
alias t='tmux'
alias tl='tmux list-sessions'
alias tn='tmux new-session -s'
alias td='tmux detach'
alias ta='tmux attach -t'
alias tk='tmux kill-session -t'
alias tr='tmux rename-session -t'
alias tlw='tmux list-windows'
alias tnw='tmux new-window -n'
alias tkw='tmux kill-window -t'
alias trw='tmux rename-window'
```

### vLLM Server Configuration

Example vLLM server command with LoRA support:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --port 8091 \
    --host 0.0.0.0 \
    --api-key dummy \
    --enable-lora \
    --max-lora-rank 128 \
    --max-loras 8 \
    --gpu-memory-utilization 0.9 \
    --max_model_len 24576 \
    --enable_auto_tool_choice \
    --tool_call_parser hermes \
    --enable_chunked_prefill \
    --max_num_batched_tokens 8192 \
    --max_num_seqs 16 \
    --enable_prefix_caching \
    --generation-config vllm \
    --return-tokens-as-token-ids \
    --trust_remote_code
```

Key vLLM environment variables:
```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### Log Probability Server (for GRPO training)

Run separate log probability servers for computing reference log probs:

```bash
# Server 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python inferenceCustomServer.py --prefix_tag my_tag.a

# Server 2 on GPU 2
CUDA_VISIBLE_DEVICES=2 python inferenceCustomServer.py --prefix_tag my_tag.a --port 8003
```

### Workflow Initialization

Before starting a workflow run:

```bash
# Initialize workflow state
python ./workflowInit.py --prefix_tag my_tag.a

# Add workflow to registry in workflow.yaml:
# registry:
#   my_tag.a:
#     short_name: "ma"
#     config_path: "workflow/my.a.yaml"
#     data_dir: "~/.workflow"
```

### Standalone Mode Commands

Run inference without workflow orchestration:

```bash
# With local vLLM (has log probs, for GRPO)
python ./inferenceComposer.py \
    --prefix_tag my_tag.a \
    --input_dir kernel_bench/level1/ \
    --provider local \
    --model qwen3-14b

# With 3rd party API (no log probs, for SFT only)
python ./inferenceComposer.py \
    --prefix_tag my_tag.a \
    --input_dir kernel_bench/level1/ \
    --module_file inference/codeGenEval.module.chat.yaml \
    --provider deepseek \
    --model deepseek-chat
```

Run training without workflow orchestration:

```bash
# GRPO training
python ./trainerGRPO.py \
    --input_dir ~/.inference/output/ \
    --input_tag my_tag.a_20250815_201330

# SFT training
python ./trainerSFT.py \
    --input_dir ~/.inference/output \
    --input_tag my_tag.a_20250813_034658
```
