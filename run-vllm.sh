#!/bin/bash

sudo docker run -it --runtime nvidia --gpus all --shm-size 32g --network host -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.trainer:/root/.trainer --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" --ipc=host dtadpole/vllm:v0.7
