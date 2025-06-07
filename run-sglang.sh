#!/bin/bash

sudo docker run -it --gpus all --shm-size 32g -p 8082:8082 -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" --ipc=host sglang:v0.4
