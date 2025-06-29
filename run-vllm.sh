#!/bin/bash

sudo docker run -it --runtime nvidia --gpus all --shm-size 32g -p 8091:8091 -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" --ipc=host dtadpole/vllm:v0.6
