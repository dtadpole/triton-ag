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

python3 -m sglang.launch_server \
        --model-path unsloth/Qwen3-32B-bnb-4bit \
        --context-length 40960 \
        --host 0.0.0.0 \
        --port 8081 \
        --tool-call-parser qwen25 \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --pp 4 \
        --dp 1


        --tp-size 4 \
        --reasoning-parser qwen3 \

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


python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3-32B-bnb-4bit \
  --host 0.0.0.0 \
  --port 8091 \
  --dtype bfloat16 \
  --trust-remote-code \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --pipeline-parallel-size 4 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --disable-log-requests


========================================

docker ps

docker commit <pid> vllm:vX.Y

docker tag vllm:vX.Y dtadpole/vllm:vX.Y

########################################
