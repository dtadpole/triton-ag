# Triton Multi-Agent Code Generation

## Setup Environment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Prepare API key for LLM access

create your API keys with relevant LLM provider and store them in ${HOME}/.keys/<provider>.api.key

e.g.

```
-- for Claude, store your API key in file ${HOME}/.keys/anthropic.api.key
-- for Deepseek, store your API key in file ${HOME}/.keys/deepseek.api.key
-- for Gemini, store your API key in file ${HOME}/.keys/gemini.api.key
-- for OpenAI, store your API key in file ${HOME}/.keys/gemini.api.key

use model.yaml to add and/or configure models.

use agentl.yaml to add and/or configure agents.
```

## Prepare background services

Run Observability Service:

```bash
make mlflow
```

Run Evaluator Service (adapted from Kernel Bench):

```bash
make kbEval
```

Clone the KernelBench git repo under ${HOME} to access various KernelBench test cases.

```bash
git clone git@github.com:ScalingIntelligence/KernelBench.git
```

## Run Agent Kernel Coder

```bash
python agent_kernel_coder.py
```

This will create a new working directory under `_run_{ddd}` and generate kernel implementation.

