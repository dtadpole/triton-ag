#!/bin/bash

sudo docker run -it --gpus all --shm-size 32g -p 8080:8080 -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.ssh/id_rsa:/root/.ssh/id_rsa --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" --ipc=host dtadpole/vllm:v0.2
