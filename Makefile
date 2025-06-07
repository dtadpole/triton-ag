# CUDA_VISIBLE_DEVICES = ${GPU}

mlflow:
	mlflow server --host localhost --port 5050

kbEval:
	uv run kbEvalRemoteServer.py 

codeRunServer:
	mcp dev codeRunServer.py

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
