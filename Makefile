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