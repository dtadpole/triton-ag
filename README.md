# Triton and CUDA Kernel Agent based Code Generation

## Setup Environment

```bash
## devserver specific: install nvidia-container-toolkit
## When the devserver don't have nvidia-container-toolkit installed, install it with the following commands:

# Add Proxy settings for Meta's devserver (P1848880359) to your ~/.bashrc and then run below commands
source ~/.bashrc

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
# To speed up the evaluation, we can spin up 3 local servers and run 3 evaluators in parallel.
# each command will launch a docker container with the local kbeval service
make kbEvalLocal1
make kbEvalLocal2
make kbEvalLocal3
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
with-proxy python agent_kernel_coder.py -p deepseek -m deepseek-chat
```

This will create a new working directory under `_run_{ddd}` and generate kernel implementation.

## Setup shared drive on meta's devserver

We use sshfs to mount a shared drive on devserver. We use the devserver devgpu139.cco2.facebook.com to host the shared drive, and other devservers can mount the shared drive to read and write model artifacts.

```
ssh host: devgpu139.cco2.facebook.com
ssh port: 8081
```
To mount the shared drive on your devserver, run the following command on your devserver:
```
cd ~/.ssh
ssh-keygen -t rsa -b 4096 -f id_rsa_{mykey_name}
scp -P 8081 id_rsa_{mykey_name}.pub devgpu139.cco2.facebook.com:/tmp/

# log on the host devserver devgpu139.cco2.facebook.com
ssh devgpu139.cco2.facebook.com
docker exec -it ssh-server /bin/bash
cat /tmp/id_rsa_{mykey_name}.pub >> /home/testuser/.ssh/authorized_keys
chown codegen:codegen /home/codegen/.ssh/authorized_keys
chmod 600 /home/codegen/.ssh/authorized_keys

# use make command to mount the shared drive on your devserver

# in the client host, this command will start the container and mount the shared drive
make env_start
# in the docker environment
make mount_shared_drive
```
The shared drive will be shared/ in the current working directory.

## How to host your own LLM service on devserver
```
# create an empty API key
echo "EMPTY" >> ${HOME}/.keys/local.api.key
# Download the huggingface model to your devserver D76999058
# Save the model to a folder under /data/users/${USER}/huggingface/hub
# start local LLM server
make vllm-qwen3-32b-devserver
# Follow the same flow the same as with provider's API
# start local agent kernel coder
python agent_kernel_coder.py -p cmgdev -m qwen-3-32b
```
