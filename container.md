docker run --gpus all -it ubuntu:24.04

apt update
apt install -y vim curl wget
apt install -y build-essential python3 python3-dev libxml2-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv
. .venv/bin/activate

uv pip install "sglang[all]>=0.4.6.post5"
uv pip install vllm==0.8.4

wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sh cuda_12.9.0_575.51.03_linux.run
rm cuda_12.9.0_575.51.03_linux.run

python3 -m sglang.launch_server --model-path Qwen/Qwen3-32B-AWQ --context-length 40960 --host 0.0.0.0 --port 8081 --reasoning-parser qwen3 --tool-call-parser qwen25 --tp-size 2 -dp 1
