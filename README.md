# Triton Multi-Agent Code Generation

## Setup Environment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run Agent Triton Coder

```bash
python agent_triton_coder.py -i "Implement Triton kernel for the forward pass of nn.Linear, use autotune for the tiling parameters"
```

This will create a new working directory under `_run_{ddd}` and generate the triton kernel implementation.

## Run Agent Planner

```bash
python agent_planner.py -g "Generate triton kernel for nn.linear, no bias, both forward and backward, compare to PyTorch implementation, verify correctness, do _not_ benchmark performance"
```

This will create a new working directory under `_run_{ddd}` and generate the triton kernel implementation.
