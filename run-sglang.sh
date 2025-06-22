#!/bin/bash

sudo docker run -it --runtime nvidia --gpus all --shm-size 32g -p 8081:8081 -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" --ipc=host dtadpole/sglang:v0.7
