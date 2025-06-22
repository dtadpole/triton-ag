# CUDA_VISIBLE_DEVICES = ${GPU}
ENV_VARS ?= PYTHONNOUSERSITE=1 \
        PYTHONPATH=${PYTHONPATH}:${PWD}
HOST=$(shell hostname)
IS_DEVSERVER=$(shell hostname | grep -E -c "dev.*\.facebook\.com")
META_PROXY := https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 ftp_proxy=http://fwdproxy:8080 http_no_proxy='\''\'\'''\''.facebook.com|.tfbnw.net|*.fb.com'\''\'\'

.PHONY: help finetune finetune-single finetune-2gpu finetune-debug

help:
	@echo "Available targets:"
	@echo "  build_docker    - build docker image"
	@echo "  env       	     - enter into dock container"
	@echo "  finetune        - Run data parallel fine-tuning on 4 GPUs"
	@echo "  finetune-single - Run single GPU fine-tuning"
	@echo "  finetune-2gpu   - Run data parallel fine-tuning on 2 GPUs"
	@echo "  finetune-debug  - Run data parallel fine-tuning with debug logging"
	@echo "  finetune-safe   - Run safe multi-GPU fine-tuning with model parallelism"
	@echo "  mlflow          - Start MLflow server"
	@echo "  kbEval          - Run knowledge base evaluation server"
	@echo "  codeRunServer   - Run code execution server"

host_check:
	@echo "Current hostname is "${HOST}
	@echo "Is it meta devserver "${IS_DEVSERVER}

build_docker: Dockerfile
ifeq (${IS_DEVSERVER}, 1)
	$(META_PROXY) docker build --network=host --progress=plain  -t triton_ag .
else
	docker build --network=host --progress=plain  -t triton_ag .
endif

env:
	docker run -it  --gpus all --net=host -p 8081:8081 -p 8082:8082 -v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ -v /data/users/${USER}/:/root/.cache/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ localhost/triton_ag /bin/bash

vllm_env:
	docker run -it  --gpus all --net=host -p 8081:8081 -v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ localhost/triton_ag /bin/bash

mlflow:
	mlflow server --host localhost --port 5051

kbEval:
	python kbEvalRemoteServer.py

codeRunServer:
	mcp dev codeRunServer.py

# Fine-tuning targets
finetune:
	@echo "Starting data parallel fine-tuning on 4 GPUs..."
	bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 finetune_unsloth.py"

finetune-manual:
	@echo "Starting data parallel fine-tuning on 4 GPUs..."
	bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 finetune_manual.py"

finetune-single:
	@echo "Starting single GPU fine-tuning..."
	bash -c "CUDA_VISIBLE_DEVICES=1 python finetune_unsloth.py"

finetune-2gpu:
	@echo "Starting data parallel fine-tuning on 2 GPUs..."
	bash -c "CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29500 finetune_unsloth.py"

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

vllm-qwen3-32b_devserver:
	vllm serve Qwen/Qwen3-32B \
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

jupyter:
	echo ${ENV_VARS}
	env ${ENV_VARS} jupyter notebook --allow-root --port 8082 --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
