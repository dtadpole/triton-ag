# CUDA_VISIBLE_DEVICES = ${GPU}
ENV_VARS ?= PYTHONNOUSERSITE=1 \
        PYTHONPATH=${PYTHONPATH}:${PWD}

## docker build is blocked by proxy errors, no software update/installation can be done within docker
# build_docker: Dockerfile
# 	HTTPS_PROXY=fwdproxy:8080 docker build --network=host --progress=plain  -t triton_ag .
# Download from google drive link: https://drive.google.com/file/d/1QAQkJ-7AKGEMy9cUkHRU8QZHY2XSrgxz/view?usp=sharing

.PHONY: help finetune finetune-single finetune-2gpu finetune-debug

help:
	@echo "Available targets:"
	@echo "  env       	     - enter into dock container"
	@echo "  dev_setup       - Set up dev environment for devserver"
	@echo "  finetune        - Run data parallel fine-tuning on 4 GPUs"
	@echo "  finetune-single - Run single GPU fine-tuning"
	@echo "  finetune-2gpu   - Run data parallel fine-tuning on 2 GPUs"
	@echo "  finetune-debug  - Run data parallel fine-tuning with debug logging"
	@echo "  finetune-safe   - Run safe multi-GPU fine-tuning with model parallelism"
	@echo "  mlflow          - Start MLflow server"
	@echo "  kbEval          - Run knowledge base evaluation server"
	@echo "  codeRunServer   - Run code execution server"

env:
	docker run -it  --gpus all --net=host -p 8081:8081 -v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ triton_ag /bin/bash

vllm_env:
	docker run -it  --gpus all --net=host -p 8081:8081 -v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ triton_ag /bin/bash

dev_setup:
	npm config set proxy http://fwdproxy:8080
	npm config set https-proxy http://fwdproxy:8080

mlflow:
	mlflow server --host localhost --port 5051

kbEval:
	uv run kbEvalRemoteServer.py

codeRunServer:
	mcp dev codeRunServer.py

# Fine-tuning targets
finetune:
	@echo "Starting data parallel fine-tuning on 4 GPUs..."
	bash -c "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 finetune.py"

finetune-single:
	@echo "Starting single GPU fine-tuning..."
	bash -c "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python finetune.py"

finetune-2gpu:
	@echo "Starting data parallel fine-tuning on 2 GPUs..."
	bash -c "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29500 finetune.py"

vllm-qwen3-8b:
	vllm serve unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit \
	--max_model_len 40960 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes

vllm-qwen3-32b:
	vllm serve unsloth/Qwen3-32B-bnb-4bit \
	--max_model_len 40960 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes

sglang-qwen3-8b:
	sglang serve qwen/qwen3-8b-instruct \
	--max_model_len 40960 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes

llama.cpp-qwen3-32b:
	python -m llama_cpp.server \
	--model models/Qwen3-32B-Q4_K_M.gguf \
	--n_ctx 40960 \
	--n_gpu_layers 128 \
	--n_batch 32 \
	--n_threads 8

llama.cpp-server-qwen3-32b:
	CUDA_VISIBLE_DEVICES=0,1,2,3 ../llama.cpp/build/bin/llama-server \
	--jinja -fa \
	--model models/Qwen3-32B-Q4_K_M.gguf \
	--model-draft models/Qwen3-1.7B-Q4_K_M.gguf \
	--flash-attn \
	--n_gpu_layers 65 \
	--tensor-split 3,2 \
	--cont-batching \
	-c 20480 \
	-np 2 \
	-ngld 99 \
	--draft-max 16 \
	--draft-min 4 \
	--draft-p-min 0.4 \
	--device-draft CUDA1 \
	--cache-type-k q8_0 \
	--cache-type-v q8_0 \
	--ctx-size 131072 \
	-np 4 \
	--n_gpu_layers 65 \
	--temp 0.6 \
	--top-k 40 \
	--top-p 0.95 \
	--min-p 0.05 \
	--host 0.0.0.0

vllm_mistral_7b_dev:
	docker run --gpus all \
	--network=host \
    -v /data/users/jingbo25/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8005:8005 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
