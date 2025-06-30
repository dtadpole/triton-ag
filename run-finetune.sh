#!/bin/bash

sudo docker run -it --runtime nvidia --gpus all --shm-size 32g --network host -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.ssh/id_rsa:/root/.ssh/id_rsa --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" --ipc=host dtadpole/finetune:v0.7
