# Triton and CUDA Kernel Agent based Code Generation

## Setup Environment

```bash
## devserver specific: install nvidia-container-toolkit
## When the devserver don't have nvidia-container-toolkit installed, install it with the following commands:
sudo dnf install -y nvidia-container-toolkit
sudo mkdir /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

#Check the proxy configuration, follow through all those steps if it's not working.
https://www.internalfb.com/wiki/Traffic/Proxygen_Services/ForwardProxy/Forward_Proxy_User/Devservers/#it-s-not-working-how-do

## Devserver use podman, and the docker is only a mirror of podman.
## The main challenge of using dev server is the proxy configuration and IPv6 only network.
## Proxy setup
## https://www.internalfb.com/wiki/Traffic/Proxygen_Services/ForwardProxy/Forward_Proxy_User/Devservers/
## Docker setup
## https://www.internalfb.com/wiki/Users/emilian/Docker_containers_on_a_devserver/
## 1. Download the docker tar from the google drive https://drive.google.com/file/d/1QAQkJ-7AKGEMy9cUkHRU8QZHY2XSrgxz/view?usp=sharing
## 2. Run docker load -i triton_ag.tar
# Run the following commands to build the docker environment

make env
make dev_setup
```

## Prepare API key for LLM access

create your API keys with relevant LLM provider and store them in ${HOME}/.keys/{provider}.api.key

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

Run Observability Service on another terminal or tmux session

```bash
make env
make mlflow
```

Run Evaluator Service on another terminal or tmux session (adapted from Kernel Bench):

```bash
make env
make kbEval
```

Clone the KernelBench git repo under ${HOME} to access various KernelBench test cases.

```bash
cd ${HOME}
git clone git@github.com:dtadpole/KernelBench.git
```
Note that our own version of KernelBench has increased dimension sizes for simple kernels [level 1, 19-87], to increas run time to be meaningfully higher than just the kernel launch time (4-8 us).

## Run Agent Kernel Coder

```bash
make env
make dev_setup
python agent_kernel_coder.py
```

This will create a new working directory under `_run_{ddd}` and generate kernel implementation.
