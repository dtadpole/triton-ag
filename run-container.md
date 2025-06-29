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

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

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

### This is super fast (AWQ)

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-AWQ --host 0.0.0.0  --port 8091 --max-model-len 16384 --max-num-batched-tokens 2048 --max-num-seqs 16 --gpu-memory-utilization 0.75 --pipeline-parallel-size 1 --data-parallel-size 1 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

### This is ideal setup (FP8) ###

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-FP8 --host 0.0.0.0  --port 8091 --max-model-len 16384 --max-num-batched-tokens 2048 --max-num-seqs 32 --gpu-memory-utilization 0.75 --pipeline-parallel-size 1 --data-parallel-size 1 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-FP8 --host 0.0.0.0  --port 8091 --max-model-len 16384 --max-num-batched-tokens 2048 --max-num-seqs 32 --gpu-memory-utilization 0.75 --pipeline-parallel-size 1 --data-parallel-size 1 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

# This is too slow 
/root/run.sh python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --host 0.0.0.0  --port 8091 --quantization bitsandbytes  --load-format bitsandbytes --max-model-len 16384 --max-num-batched-tokens 2048 --max-num-seqs 16 --gpu-memory-utilization 0.9 --pipeline-parallel-size 1 --data-parallel-size 1 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests

# This is too slow

/root/.venv/bin/python3 -m vllm.entrypoints.openai.api_server  --model Qwen/Qwen3-14B --enable-lora --lora-modules my_adapter=dtadpole/KernelCoder-4B_20250621-071556 --max-lora-rank 64 --host 0.0.0.0  --port 8091  --dtype bfloat16  --trust-remote-code  --quantization bitsandbytes  --load-format bitsandbytes  --max-model-len 32768  --gpu-memory-utilization 0.9  --pipeline-parallel-size 1  --data-parallel-size 1  --tensor-parallel-size 1 --enable-auto-tool-choice  --tool-call-parser hermes  --reasoning-parser qwen3  --disable-log-requests


========================================

docker ps

docker commit <pid> vllm:vX.Y

docker tag vllm:vX.Y dtadpole/vllm:vX.Y

########################################
