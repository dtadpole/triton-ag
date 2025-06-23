# Triton and CUDA Kernel Agent based Code Generation

## Setup Environment

```bash
## devserver specific: install nvidia-container-toolkit
## When the devserver don't have nvidia-container-toolkit installed, install it with the following commands:

# Add Proxy settings for Meta's devserver (P1848880359) to your ~/.bashrc and then run below commands
# source ~/.bashrc

## To prune docker images, sometime docker build fails with proxy related error which is due to caching
# docker builder prune

sudo dnf install -y nvidia-container-toolkit
sudo mkdir /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

#Check the proxy configuration, follow through all those steps if it's not working.
https://www.internalfb.com/wiki/Traffic/Proxygen_Services/ForwardProxy/Forward_Proxy_User/Devservers/#it-s-not-working-how-do

## Devserver use podman, and the docker is only a mirror of podman.
## The main challenge of using dev server is the proxy configuration and IPv6 only network.
## Proxy setup
## https://www.internalfb.com/wiki/Traffic/Proxygen_Services/ForwardProxy/Forward_Proxy_User/Devservers/
## !!!! Follow the link above to install ttls_fwdproxy
## Add the following export into ~/.bashrc
export https_proxy=http://fwdproxy:8080
export http_proxy=http://fwdproxy:8080
export ftp_proxy=http://fwdproxy:8080
export http_no_proxy='\''\'\'''\''.facebook.com|.tfbnw.net|*.fb.com'\''\'\'
## Docker setup
## Install podman and docker. In meta, docker is a simulator of podman.
## https://www.internalfb.com/wiki/Users/emilian/Docker_containers_on_a_devserver/

## clone the repo to devserver
git clone https://github.com/dtadpole/triton-ag
cd triton-ag

## Manifold: Meta's version of s3
## The manifold bucket for this project: llm_models/tree/huggingface/
## Basic commands https://www.internalfb.com/wiki/Manifold/Getting_Started/Manifold_CLI/
## It only works on the meta devserver !!!!
manifold ls llm_models/tree/huggingface/hub/

## After setting up proxy and docker on devserver, build the docker image using the following commands:
make build_docker

## start a docker container with all the required dependencies
make env
```

## Prepare API key for LLM access

create your API keys with relevant LLM provider and store them in ${HOME}/.keys/{provider}.api.key

e.g.

```
-- for Claude, store your API key in file ${HOME}/.keys/anthropic.api.key
-- for Deepseek, store your API key in file ${HOME}/.keys/deepseek.api.key
-- for Gemini, store your API key in file ${HOME}/.keys/gemini.api.key
-- for OpenAI, store your API key in file ${HOME}/.keys/openai.api.key

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
## Copy the level 1-4 folders in KernelBench/KernelBench into kernel_bench/ folder in this repo.
```
Note that our own version of KernelBench has increased dimension sizes for simple kernels [level 1, 19-87], to increas run time to be meaningfully higher than just the kernel launch time (4-8 us).

## Run Agent Kernel Coder For Demo Purpose

```bash
make env
python agent_kernel_coder.py -p deepseek -m deepseek-chat
```

This will create a new working directory under `_run_{ddd}` and generate kernel implementation.
