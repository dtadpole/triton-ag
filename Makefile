# CUDA_VISIBLE_DEVICES = ${GPU}
ENV_VARS ?= PYTHONNOUSERSITE=1 PYTHONPATH=${PYTHONPATH}:${PWD}
HOST=$(shell hostname)
IS_DEVSERVER=$(shell hostname | grep -E -c "dev.*\.facebook\.com")
META_PROXY := https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 ftp_proxy=http://fwdproxy:8080 no_proxy='\''\'\'''\''.facebook.com|.tfbnw.net|*.fb.com'\''\'\'
VLLM_SETTING := VLLM_ALLOW_RUNTIME_LORA_UPDATING=True HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0

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

build_docker: Dockerfile
ifeq (${IS_DEVSERVER}, 1)
	$(META_PROXY) docker build --network=host --progress=plain  -t triton_ag .
else
	docker build --network=host --progress=plain  -t triton_ag .
endif

build_nginx: Dockerfile
ifeq (${IS_DEVSERVER}, 1)
	$(META_PROXY) docker build . -f Dockerfile.nginx --tag nginx-lb
else
	docker build . -f Dockerfile.nginx --tag nginx-lb
endif

build_vllm_gptoss: Dockerfile_vllm_gpt_oss
ifeq (${IS_DEVSERVER}, 1)
	echo "Current hostname is "${HOST}
	$(META_PROXY) docker build . -f Dockerfile_vllm_gpt_oss --format=docker --tag vllm_gptoss
else
	docker build . -f Dockerfile_vllm_gpt_oss --format=docker --tag vllm_gptoss
endif


build_vllm: Dockerfile_vllm
ifeq (${IS_DEVSERVER}, 1)
	echo "Current hostname is "${HOST}
	$(META_PROXY) docker build . -f Dockerfile_vllm --no-cache --net=host --format=docker --tag vllm
else
	docker build . -f Dockerfile_vllm --net=host --format=docker --tag vllm
endif

build_docker_autoawq: Dockerfile_autoawq
ifeq (${IS_DEVSERVER}, 1)
	$(META_PROXY) docker build -f Dockerfile_autoawq --network=host --progress=plain  -t autoawq .
else
	docker build -f Dockerfile_autoawq --network=host --progress=plain  -t autoawq .
endif


env_autoawq:
	docker run -it  --gpus all --net=host -p 8081:8081 -p 8082:8082 -v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ -v /data/users/${USER}/:/root/.cache/ -v ~/.inference/:/root/.inference/ -v ~/.kbeval:/root/.kbeval/ -v ${PWD}:/workspace/ localhost/autoawq /bin/bash

env_start:
	$(META_PROXY) docker run -d \
		--name codegen \
		--replace \
		--gpus all \
		--cap-add SYS_ADMIN \
		--net=host \
		--shm-size=128g \
		--pids-limit -1 \
		--ulimit nofile=65536:65536 \
		--ulimit nproc=-1:-1\
		--ulimit memlock=-1:-1 \
		-v ~/.ssh/:/root/.ssh \
		-v ~/.bashrc:/root/.bashrc \
		-v ~/.netrc:/root/.netrc \
		-v ~/.gitconfig:/root/.gitconfig \
		-v ~/.keys/:/root/.keys/ \
		-v /data/users/${USER}/:/root/.cache/ \
		-v ${PWD}:/workspace/ \
		--cap-add SYS_ADMIN \
		--device /dev/fuse \
		--security-opt apparmor:unconfined \
		--privileged \
		localhost/triton_ag \
		/bin/bash -c "make mount_shared_drive && make wandb_login && tail -f /dev/null"

env:
	docker exec -it codegen /bin/bash

env_vllm:
	docker exec -it vllm /bin/bash

env_vllm_start:
	docker run -d \
		--name vllm \
		--replace \
		--gpus all \
		--cap-add SYS_ADMIN \
		--net=host \
		--shm-size=128g \
		--pids-limit -1 \
		--ulimit nofile=65536:65536 \
		--ulimit nproc=-1:-1\
		--ulimit memlock=-1:-1 \
		-v ~/.ssh/:/root/.ssh \
		-v ~/.netrc:/root/.netrc \
		-v ~/.gitconfig:/root/.gitconfig \
		-v ~/.keys/:/root/.keys/ \
		-v /data/users/${USER}/:/root/.cache/ \
		-v ${PWD}:/workspace/ \
		--cap-add SYS_ADMIN \
		--device /dev/fuse \
		--security-opt apparmor:unconfined \
		--privileged \
		localhost/vllm \
		/bin/bash -c "make mount_shared_drive && tail -f /dev/null"

wandb_login:
	wandb login --host=https://fairwandb.org

mount_shared_drive:
	sshfs -o IdentityFile=/root/.ssh/id_rsa_shared -p 8081 codegen@devvm8492.cco0.facebook.com:/shared/ shared/

mount_shared_code:
	sshfs -o IdentityFile=/root/.ssh/id_rsa_shared -p 8082 codegen@devvm8492.cco0.facebook.com:/shared/ /workspace/

mlflow:
	# mlflow server --host localhost --port 5051
	mlflow server --host localhost --port 5051 --backend-store-uri sqlite:///mlflow.sqlite

workflow_server:
	while true; do python ./workflowServer.py --host :: --port 8488; sleep 5; done

sync_config:
	cp workflow.yaml shared/config/workflow.yaml
	cp workflow/* shared/config/workflow/

snapshot_config:
	@if [ -z "$(prefix_tag)" ]; then \
		echo "Error: prefix_tag is required. Usage: make snapshot_config prefix_tag=<tag_name>"; \
		exit 1; \
	fi
	@echo "Creating config snapshot with prefix: $(prefix_tag)"
	@mkdir -p shared/config/snapshot/$(prefix_tag)
	@find . -maxdepth 2 -name "*.yaml" -not -path "./shared/*" -type f | while read file; do \
		rel_path=$$(echo $$file | sed 's|^\./||'); \
		dest_dir=shared/config/snapshot/$(prefix_tag)/$$(dirname $$rel_path); \
		mkdir -p $$dest_dir; \
		cp $$file shared/config/snapshot/$(prefix_tag)/$$rel_path; \
		echo "Copied $$file -> shared/config/snapshot/$(prefix_tag)/$$rel_path"; \
	done
	@echo "Config snapshot completed in shared/config/snapshot/$(prefix_tag)/"

lora_merge_compress_autoawq:
	CUDA_VISIBLE_DEVICES=2 python lora_merge_awq.py

lora_merge_compress:
	CUDA_VISIBLE_DEVICES=4 python lora_merge_llmcomp_awq.py

kbEval:
	while true; do python kbEvalServer.py; sleep 1; done

kbeval_local:
	python kbEvalServer.py --local_host --port 5676 --device 7

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
	bash -c "CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 finetune_unsloth.py"


finetune-4gpu:
	@echo "Starting data parallel fine-tuning on 4 GPUs..."
	bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 finetune_unsloth.py"


vllm-qwen3-8b:
	vllm serve unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit \
	--max_model_len 40960 \
	--enable-auto-tool-choice \
	--tool-call-parser hermes

# vllm-qwen3-32b-devserver:
# 	CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen3-32B-AWQ \
# 	--max-model-len 40960 \
# 	--enable-auto-tool-choice \
# 	--tool-call-parser hermes \
# 	--dtype bfloat16 \
# 	--return-tokens-as-token-ids \
# 	--host "::" \
# 	--port 8091


vllm_env:
	${VLLM_SETTING} docker run -it \
    --security-opt=label=disable \
    --device nvidia.com/gpu=all \
    --network host \
    --shm-size=32g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.trainer:/root/.trainer \
    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
    -e http_proxy -e https_proxy -e no_proxy \
	-v ~/.inference/:/root/.inference/ \
	-v ~/.bashrc:/root/.bashrc -v ~/.netrc:/root/.netrc \
	-v ~/.gitconfig:/root/.gitconfig -v ~/.keys/:/root/.keys/ \
	-v /data/users/${USER}/:/root/.cache/ \
	-v ~/.kbeval:/root/.kbeval/ \
	-v ${PWD}:/workspace/ \
    docker://dtadpole/vllm:v0.7 \
	/bin/bash -c "source /root/.venv/bin/activate && cd /workspace/ && /bin/bash"

vllm-qwen3-14b-devserver:
	${VLLM_SETTING} CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --port 8091 --host :: \
    --api-key dummy \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-lora --max-lora-rank 128 --max-loras 6 \
    --gpu-memory-utilization 0.95 --max_model_len 24576 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 144 \
    --enable_prefix_caching --prefix-caching-hash-algo builtin \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids \
    --enforce-eager


vllm-qwen3-32b-devserver:
	${VLLM_SETTING} CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8091 --host :: \
    --api-key dummy \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --enable-lora --max-lora-rank 128 --max-loras 6 \
    --gpu-memory-utilization 0.95 --max_model_len 24576 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 144 \
    --enable_prefix_caching --prefix-caching-hash-algo builtin \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids \
    --enforce-eager

vllm-qwen3-32b-devserver_a:
	${VLLM_SETTING} CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8001 --host :: \
    --api-key dummy \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --enable-lora --max-lora-rank 128 --max-loras 6 \
    --gpu-memory-utilization 0.80 --max_model_len 24576 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 128 \
    --enable_prefix_caching --prefix-caching-hash-algo builtin \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids

vllm-qwen3-32b-devserver_b:
	${VLLM_SETTING} CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8002 --host :: \
    --api-key dummy \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --enable-lora --max-lora-rank 128 --max-loras 6 \
    --gpu-memory-utilization 0.80 --max_model_len 24576 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 128 \
    --enable_prefix_caching --prefix-caching-hash-algo builtin \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids


vllm-qwen3-32b-sft-devserver:
	${VLLM_SETTING} CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8091 --host :: \
    --api-key dummy \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --enable-lora --max-lora-rank 128 --max-loras 6 \
	--lora-modules  \
		qwen3_32b_sft_t2=shared/finetune_model_output/sft_t2/checkpoint-289  \
		qwen3_32b_sft_t5=shared/finetune_model_output/sft_t5/checkpoint-181  \
		qwen3_32b_sft_t6=shared/finetune_model_output/sft_t6/checkpoint-362  \
		qwen3_32b_sft_t7=shared/finetune_model_output/sft_t7/checkpoint-724  \
    --gpu-memory-utilization 0.95 --max_model_len 24576 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 144 \
    --enable_prefix_caching --prefix-caching-hash-algo builtin \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids \
    --enforce-eager


vllm-qwen3-32b-awq-sft-devserver:
	${VLLM_SETTING} CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --port 8091 --host :: \
    --api-key dummy \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-lora --max-lora-rank 128 --max-loras 6 \
    --gpu-memory-utilization 0.95 --max_model_len 24576 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 144 \
    --enable_prefix_caching --prefix-caching-hash-algo builtin \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids \
    --enforce-eager


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
	${ENV_VARS} jupyter notebook --allow-root --port 8086 --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''


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

vllm_gptoss_a:
	CUDA_VISIBLE_DEVICES=0,1 docker run -it \
		--name vllm_gptoss_a \
		--replace \
		--gpus all \
		--cap-add SYS_ADMIN \
		--net=host \
		--shm-size=128g \
		--pids-limit -1 \
		--ulimit nofile=65536:65536 \
		--ulimit nproc=-1:-1\
		--ulimit memlock=-1:-1 \
		-v ~/.ssh/:/root/.ssh \
		-v ~/.bashrc:/root/.bashrc \
		-v ~/.netrc:/root/.netrc \
		-v ~/.gitconfig:/root/.gitconfig \
		-v ~/.keys/:/root/.keys/ \
		-e CUDA_VISIBLE_DEVICES=0,1 \
		-e HF_HOME=/root/.cache/huggingface \
		-e HTTP_PROXY="http://fwdproxy:8080" \
		-e HTTPS_PROXY="http://fwdproxy:8080" \
		-v /data/users/${USER}/huggingface:/root/.cache/huggingface \
		-v ${PWD}:/workspace/ \
		--cap-add SYS_ADMIN \
		--device /dev/fuse \
		--security-opt apparmor:unconfined \
		--privileged \
		vllm/vllm-openai:latest \
		--model openai/gpt-oss-120b \
		--port 8001 --host :: \
		--api-key dummy \
		--data-parallel-size 1 \
		--tensor-parallel-size 2 \
		--pipeline-parallel-size 1 \
		--gpu-memory-utilization 0.9 --max_model_len 24576 \
		--load_format safetensors \
		--trust_remote_code \
		--guided_decoding_backend guidance --guided-decoding-disable-fallback \
		--enable_auto_tool_choice --tool_call_parser hermes \
		--scheduling_policy priority \
		--enable_chunked_prefill --max_num_batched_tokens 2048 \
		--max_log_len 0 --max_num_seqs 144 \
		--enable_prefix_caching --prefix-caching-hash-algo builtin \
		--generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
		--return-tokens-as-token-ids \
		--async-scheduling


vllm_gptoss_b:
	docker run -it \
		--name vllm_gptoss_b \
		--replace \
		--gpus all \
		--cap-add SYS_ADMIN \
		--net=host \
		--shm-size=128g \
		--pids-limit -1 \
		--ulimit nofile=65536:65536 \
		--ulimit nproc=-1:-1\
		--ulimit memlock=-1:-1 \
		-v ~/.ssh/:/root/.ssh \
		-v ~/.bashrc:/root/.bashrc \
		-v ~/.netrc:/root/.netrc \
		-v ~/.gitconfig:/root/.gitconfig \
		-v ~/.keys/:/root/.keys/ \
		-e CUDA_VISIBLE_DEVICES=2,3 \
		-e HF_HOME=/root/.cache/huggingface \
		-e HTTP_PROXY="http://fwdproxy:8080" \
		-e HTTPS_PROXY="http://fwdproxy:8080" \
		-v /data/users/${USER}/huggingface:/root/.cache/huggingface \
		-v ${PWD}:/workspace/ \
		--cap-add SYS_ADMIN \
		--device /dev/fuse \
		--security-opt apparmor:unconfined \
		--privileged \
		vllm/vllm-openai:latest \
		--model openai/gpt-oss-120b \
		--port 8002 --host :: \
		--api-key dummy \
		--data-parallel-size 1 \
		--tensor-parallel-size 2 \
		--pipeline-parallel-size 1 \
		--gpu-memory-utilization 0.9 --max_model_len 24576 \
		--load_format safetensors \
		--trust_remote_code \
		--guided_decoding_backend guidance --guided-decoding-disable-fallback \
		--enable_auto_tool_choice --tool_call_parser hermes \
		--scheduling_policy priority \
		--enable_chunked_prefill --max_num_batched_tokens 2048 \
		--max_log_len 0 --max_num_seqs 144 \
		--enable_prefix_caching --prefix-caching-hash-algo builtin \
		--generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
		--return-tokens-as-token-ids \
		--async-scheduling
