#!/bin/bash

sudo docker run -it --runtime nvidia --gpus all --shm-size 32g -p 8089:8089 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.ssh/id_rsa:/root/.ssh/id_rsa -v ~/triton-ag:/root/triton-ag -v ~/.keys:/root/.keys -v ~/.aws:/root/.aws -v ~/.config/wandb:/root/.config/wandb --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" --ipc=host dtadpole/grpo:v0.2
