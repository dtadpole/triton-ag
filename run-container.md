docker run --gpus all -it ubuntu:24.04

########################################
SGLang
########################################

apt update
apt install -y vim curl wget
apt install -y build-essential python3 python3-dev libxml2-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv
. .venv/bin/activate

uv pip install "sglang[all]>=0.4.6.post5" "vllm" "bitsandbytes"
# uv pip install "vllm==0.8.4"
# uv pip install "bitsandbytes>=0.45.3"

wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sh cuda_12.9.0_575.51.03_linux.run
rm cuda_12.9.0_575.51.03_linux.run

### run.sh

#!/bin/bash

cd $HOME
source .venv/bin/activate
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_ATTENTION_BACKEND=FLASHINFER

"$@"

### install vLLM

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install flashinfer-python
uv pip install 'sglang>=0.4.9.post4'
uv pip install pydantic pybase64 orjson uvicorn uvloop fastapi psutil zmq pillow huggingface huggingface_hub transformers sentencepiece sgl_kernel
###

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

###

/root/run.sh python -m sglang.launch_server --model-path Qwen/Qwen3-32B --host 0.0.0.0 --port 8081 --tool-call-parser qwen25  --context-length 16384 --max-prefill-tokens 2048 --max-total-tokens 16384 --max-running-requests 32


# this is optimal, but will require hours to quantize model as AWQ

/root/run.sh /root/.venv/bin/python3 -m sglang.launch_server --model-path Qwen/Qwen3-14B-AWQ --host 0.0.0.0 --port 8081 --tool-call-parser qwen25  --context-length 16384 --max-prefill-tokens 2048 --max-total-tokens 57344 --max-running-requests 32 --dp 4

### this is ideal setup (FP8) ###

/root/run.sh /root/.venv/bin/python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B-FP8 --host 0.0.0.0 --port 8081 --tool-call-parser qwen25  --context-length 16384 --max-prefill-tokens 2048 --max-total-tokens 65536 --max-running-requests 32 --dp 4

/root/run.sh /root/.venv/bin/python3 -m sglang.launch_server --model-path Qwen/Qwen3-14B-FP8 --host 0.0.0.0 --port 8081 --tool-call-parser qwen25  --context-length 16384 --max-prefill-tokens 2048 --max-total-tokens 20480 --max-running-requests 32 --dp 4

# below is not working for Qwen3

/root/run.sh /root/.venv/bin/python3 -m sglang.launch_server --model-path Qwen/Qwen3-14B --quantization bitsandbytes --load-format bitsandbytes --host 0.0.0.0 --port 8081 --tool-call-parser qwen25  --context-length 16384 --max-prefill-tokens 2048 --max-total-tokens 65536 --max-running-requests 32 --dp 1


 --mem-fraction-static 0.7 --max-running-requests 32 

--quantization bitsandbytes --load-format bitsandbytes



--mem-fraction-static 0.7  # Default is often 0.9
--max-running-requests 32  # Reduce concurrent requests

--max-prefill-tokens 2048  # Reduce from default
--max-total-tokens 4096    # Limit total sequence length

python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3-32B-AWQ \
        --context-length 32768 \
        --host 0.0.0.0 \
        --port 8081 \
        --tool-call-parser qwen25 \
        --pp 1 \
        --dp 1

        --quantization bitsandbytes \
        --load-format bitsandbytes \

        --tp-size 4 \
        --reasoning-parser qwen3 \


/root/run.sh /root/.venv/bin/python3 -m sglang.launch_server --model-path Qwen/Qwen3-4B-AWQ     --lora-paths my_adapter=dtadpole/KernelCoder-4B-AWQ_20250621-160317 --max-loras-per-batch 1 --lora-backend triton --disable-radix-cache --context-length 32768 --host 0.0.0.0 --port 8081 --tool-call-parser qwen25 --dp 1


/root/run.sh /root/.venv/bin/python3 -m sglang.launch_server --model-path Qwen/Qwen3-32B-AWQ     --lora-paths my_adapter=dtadpole/KernelCoder-32B-AWQ_20250621-161329 --max-loras-per-batch 1 --lora-backend triton --disable-radix-cache --context-length 32768 --host 0.0.0.0 --port 8081 --tool-call-parser qwen25 --dp 1

========================================

docker ps

docker commit <pid> sglang:vX.Y

docker tag sglang:vX.Y dtadpole/sglang:vX.Y

########################################
VLLM
########################################

apt update
apt install -y vim curl wget
apt install -y build-essential python3 python3-dev libxml2-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv
. .venv/bin/activate

uv pip install "vllm" "bitsandbytes"

wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sh cuda_12.9.0_575.51.03_linux.run
rm cuda_12.9.0_575.51.03_linux.run


### run.sh

#!/bin/bash

cd $HOME
source .venv/bin/activate
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_ATTENTION_BACKEND=FLASHINFER

"$@"

### install vLLM

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install 'vllm==0.10.0'

export VLLM_ATTENTION_BACKEND=FLASHINFER

 --revision 9216db5

###

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --port 8091 --host 0.0.0.0 --api-key dummy --dtype bfloat16 --kv-cache-dtype auto --gpu-memory-utilization 0.95 --max_model_len 16384 --load_format safetensors --guided_decoding_backend guidance --enable_auto_tool_choice --tool_call_parser hermes --scheduling_policy fcfs --enable_prefix_caching --prefix-caching-hash-algo sha256 --enable_chunked_prefill --max_num_batched_tokens 2048 --max_num_seqs 16 --max_log_len 0 --trust_remote_code --generation-config vllm --override-generation-config '{"temperature":0,"top_p":1,"top_k":0}' --enforce-eager

--no-enable-prefix-caching 

--enable_prefix_caching --prefix-caching-hash-algo sha256 

###

--max_num_batched_tokens 8192 --max_num_seqs 20

###

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --port 8091 --host 0.0.0.0  --api-key dummy --gpu-memory-utilization 0.95 --max_model_len 16384 --load_format safetensors --guided_decoding_backend guidance --guided-decoding-disable-fallback --enable_auto_tool_choice --tool_call_parser hermes --scheduling_policy fcfs --enable_chunked_prefill --max_num_batched_tokens 2048 --max_num_seqs 16 --max_log_len 0 --trust_remote_code --enable_prefix_caching --prefix-caching-hash-algo sha256 --generation-config vllm --override-generation-config '{"temperature":0,"top_p":1,"top_k":0}' --enforce-eager

--no-enable-prefix-caching 

--enable_prefix_caching --prefix-caching-hash-algo sha256 

###


/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --enable-lora --max-loras 4 --max-lora-rank 128 --gpu-memory-utilization 0.9 --max-model-len 16384 --port 8091 --host 0.0.0.0 --trust-remote-code --tool-call-parser hermes --max_log_len 0 --trust_remote_code

--rope-scaling '{"rope_type":"yarn","factor":0.5,"original_max_position_embeddings":32768}'

--default-sampling-params '{"temperature": 0.7, "top_p": 0.9, "max_tokens": 14336}'

### load and unload lora adapter

curl -X POST http://localhost:8091/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
        "lora_name": "KC_0.1.0_520",
        "lora_path": "/root/.trainer/KC_0.1.0/checkpoint-520"
      }'

### This is super fast (AWQ)

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-AWQ --host 0.0.0.0  --port 8091 --max-model-len 14336 --max-num-batched-tokens 2048 --max-num-seqs 16 --gpu-memory-utilization 0.75 --pipeline-parallel-size 1 --data-parallel-size 1 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

### This is ideal setup (FP8) ###

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-FP8 --host 0.0.0.0  --port 8091 --max-model-len 16384 --max-num-batched-tokens 2048 --max-num-seqs 32 --gpu-memory-utilization 0.75 --pipeline-parallel-size 1 --data-parallel-size 4 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-FP8 --host 0.0.0.0  --port 8091 --max-model-len 16384 --max-num-batched-tokens 2048 --max-num-seqs 32 --gpu-memory-utilization 0.8 --pipeline-parallel-size 1 --data-parallel-size 4 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B-FP8 --host 0.0.0.0  --port 8091 --max-model-len 16384 --max-num-batched-tokens 2048 --max-num-seqs 32 --gpu-memory-utilization 0.8 --pipeline-parallel-size 1 --data-parallel-size 4 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

# This is too slow 
/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --host 0.0.0.0  --port 8091 --quantization bitsandbytes  --load-format bitsandbytes --max-model-len 14336 --max-num-batched-tokens 2048 --max-num-seqs 16 --gpu-memory-utilization 0.75 --pipeline-parallel-size 1 --data-parallel-size 1 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

# This is too slow

/root/.venv/bin/python3 -m vllm.entrypoints.openai.api_server  --model Qwen/Qwen3-14B --enable-lora --lora-modules my_adapter=dtadpole/KernelCoder-4B_20250621-071556 --max-lora-rank 64 --host 0.0.0.0  --port 8091  --dtype bfloat16  --trust-remote-code  --quantization bitsandbytes  --load-format bitsandbytes  --max-model-len 32768  --gpu-memory-utilization 0.9  --pipeline-parallel-size 1  --data-parallel-size 1  --tensor-parallel-size 1 --enable-auto-tool-choice  --tool-call-parser hermes  --reasoning-parser qwen3  --disable-log-requests

# try this:

/root/.venv/bin/python3 -m vllm.entrypoints.openai.api_server  --model Qwen/Qwen3-14B --top-k 40 --host 0.0.0.0 --port 8091 --max-model-len 16384  --gpu-memory-utilization 0.9 --pipeline-parallel-size 1 --data-parallel-size 1 --tensor-parallel-size 1 --enable-auto-tool-choice  --tool-call-parser hermes --reasoning-parser qwen3 --disable-log-requests

/root/.venv/bin/python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --top-k 40 --enable-lora --lora-modules my_adapter=dtadpole/KernelCoder-32B_20250621-013349 --max-lora-rank 64 --host 0.0.0.0 --port 8091 --max-model-len 16384 --gpu-memory-utilization 0.9  --pipeline-parallel-size 1 --data-parallel-size 1 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --disable-log-requests



========================================

docker ps

docker commit <pid> vllm:vX.Y

docker tag vllm:vX.Y dtadpole/vllm:vX.Y

########################################
