# CUDA_VISIBLE_DEVICES = ${GPU}
ENV_VARS ?= PYTHONNOUSERSITE=1 \
        PYTHONPATH=${PYTHONPATH}:${PWD}
HOST=$(shell hostname)
IS_DEVSERVER=$(shell hostname | grep -E -c "dev.*\.facebook\.com")
META_PROXY := https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 ftp_proxy=http://fwdproxy:8080 http_no_proxy='\''\'\'''\''.facebook.com|.tfbnw.net|*.fb.com'\''\'\'
NO_PROXY := http_proxy=""  https_proxy=""

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
	@echo "Is it meta devserver 1:yes; 0:no ?  Ans: "${IS_DEVSERVER}

build_nginx: Dockerfile
ifeq (${IS_DEVSERVER}, 1)
	$(META_PROXY) docker build . -f Dockerfile.nginx --tag nginx-lb
else
	docker build . -f Dockerfile.nginx --tag nginx-lb
endif

build_docker: Dockerfile
ifeq (${IS_DEVSERVER}, 1)
	echo "Building docker image on devserver"
	$(META_PROXY) docker build --network=host --progress=plain  -t triton_ag .
else
	docker build --network=host --progress=plain  -t triton_ag .
endif

env:
	docker run -it  --gpus all --rm --net=host --ipc host -p 8081:8081 -p 8082:8082 -v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ -v /data/users/${USER}/:/root/.cache/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ localhost/triton_ag /bin/bash

mlflow:
	docker run -itd  \
	--gpus all \
	--net=host -p 5051:5051 \
	-v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ \
	--name mlflow \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "mlflow server --host localhost --port 5051 --backend-store-uri sqlite:///mlflow.sqlite"

kbEval:
	while true; do python kbEvalServer.py; sleep 1; done

launch_kbEval:
	make kbEvalLocal1
	make kbEvalLocal2
	make kbEvalLocal3

kbEvalLocal1:
	docker run -itd  \
	--gpus all \
	--net=host -p 8081:8081 -p 8082:8082 \
	-v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ \
	--name kblocal_1 \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "python kbEvalServer.py --local_host --port 5676 --device 5"

kbEvalLocal2:
	docker run -itd  \
	--gpus all \
	--net=host -p 8081:8081 -p 8082:8082 \
	-v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ \
	--name kblocal_2 \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "python kbEvalServer.py --local_host --port 5677 --device 6"

kbEvalLocal3:
	docker run -itd  \
	--gpus all \
	--net=host -p 8081:8081 -p 8082:8082 \
	-v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ \
	--name kblocal_3 \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "python kbEvalServer.py --local_host --port 5678 --device 7"

codeRunServer:
	mcp dev codeRunServer.py

# Fine-tuning targets
finetune:
	@echo "Starting data parallel fine-tuning on 4 GPUs..."
	bash -c "CUDA_VISIBLE_DEVICES=2,3,6,7 torchrun --nproc_per_node=4 --master_port=29500 finetune_unsloth.py"

finetune-manual:
	@echo "Starting data parallel fine-tuning on 4 GPUs..."
	bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 finetune_manual.py"


finetune-single:
	@echo "Starting single GPU fine-tuning..."
ifeq (${IS_DEVSERVER}, 1)
	bash -c "$(META_PROXY) CUDA_VISIBLE_DEVICES=1 python finetune_unsloth.py"
else
	bash -c "CUDA_VISIBLE_DEVICES=1 python finetune_unsloth.py"
endif


finetune-2gpu:
	@echo "Starting data parallel fine-tuning on 2 GPUs..."
ifeq (${IS_DEVSERVER}, 1)
	bash -c "$(META_PROXY) CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29500 finetune_unsloth.py"
else
	bash -c "CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29500 finetune_unsloth.py"
endif

vllm-qwen3-8b:
	vllm serve unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit \
	--max_model_len 40960 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes


vllm-qwen3-14b-devserver:
	CUDA_VISIBLE_DEVICES=2,5 vllm serve Qwen/Qwen3-14B \
	--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes \
	--tensor-parallel-size 2 \
	--host "::" \
	--port 8086

vllm-qwen25-7b-devserver:
	CUDA_VISIBLE_DEVICES=2,5 vllm serve unsloth/Qwen2.5-7B \
	--max_model_len 40960 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes \
	--tensor-parallel-size 2 \
	--host "::" \
	--port 8086

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

MODEL_TO_SERVE ?= finetune_model_output/sft_t2/qwen3_32b_awq

# MODEL_TO_SERVE ?= Qwen/Qwen3-32B-AWQ

launch_vllm_servers:
	make launch_vllm1
	make launch_vllm2
	make launch_vllm3
	make vllm_serve

vllm_serve:
	docker run -itd --ipc host --rm --net=host -v ./nginx_conf/:/etc/nginx/conf.d/ --name vllm_service --replace nginx-lb:latest

launch_vllm1:
	${META_PROXY} docker run -itd  \
	--gpus all \
	--ipc host \
	--net=host \
	-p 8001:8001 \
	-v ~/.bashrc:/root/.bashrc \
    -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ \
	-v ~/.kbeval:/root/.kbeval/ \
	-v ${PWD}:/workspace/ \
	--name vllm1 \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "make serve_model1"

launch_vllm2:
	${META_PROXY} docker run -itd  \
	--gpus all \
	--ipc host \
	--net=host \
	-p 8002:8002 \
	-v ~/.bashrc:/root/.bashrc \
    -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ \
	-v ~/.kbeval:/root/.kbeval/ \
	-v ${PWD}:/workspace/ \
	--name vllm2 \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "make serve_model2"

launch_vllm3:
	${META_PROXY} docker run -itd  \
	--gpus all \
	--ipc host \
	--net=host \
	-p 8003:8003 \
	-v ~/.bashrc:/root/.bashrc \
    -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ \
	-v ~/.kbeval:/root/.kbeval/ \
	-v ${PWD}:/workspace/ \
	--name vllm3 \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "make serve_model3"

serve_model1:
	CUDA_VISIBLE_DEVICES=2 vllm serve ${MODEL_TO_SERVE} \
	--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 65536 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes \
	--host "::" \
	--port 8001

serve_model2:
	CUDA_VISIBLE_DEVICES=3 vllm serve ${MODEL_TO_SERVE} \
	--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 65536 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes \
	--host "::" \
	--port 8002

serve_model3:
	CUDA_VISIBLE_DEVICES=4 vllm serve ${MODEL_TO_SERVE} \
	--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 65536 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes \
	--host "::" \
	--port 8003

vllm-qwen3-32b:
	CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-32B \
	--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 65536 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes \
	--tensor-parallel-size 2 \
	--host "::" \
	--port 8001

run_eval1:
	${NO_PROXY} docker run -itd  \
	--gpus all \
	--net=host -p 8081:8081 -p 8082:8082 \
	-v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ \
	--name run_eval1 \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "source run_eval1.sh"

run_eval2:
	${NO_PROXY} docker run -itd  \
	--gpus all \
	--net=host -p 8081:8081 -p 8082:8082 \
	-v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ \
	--name run_eval2 \
	--replace \
	--rm \
	localhost/triton_ag \
	/bin/bash -c "source run_eval2.sh"
