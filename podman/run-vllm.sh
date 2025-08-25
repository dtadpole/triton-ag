#!/bin/bash

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

podman run -it \
    --security-opt=label=disable \
    --device nvidia.com/gpu=all \
    --network host \
    --shm-size=32g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.trainer:/root/.trainer \
    --env "HF_TOKEN=`cat ~/.keys/huggingface.api.key`" \
    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
    -e http_proxy -e https_proxy -e no_proxy \
    docker://dtadpole/vllm:v0.8
