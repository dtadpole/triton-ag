#!/bin/bash

sudo docker run -it --gpus all --shm-size 32g -p 8091:8091 -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" --ipc=host vllm:v0.1
