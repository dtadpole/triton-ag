
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

cd ~/.trainer

    --data-parallel-size 4 \
    --port 8091 --host 0.0.0.0 
    --quantization fp8 \

H100_96G (4 GPUs)

export CUDA_VISIBLE_DEVICES=4,5,6,7

/root/run.sh python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8091 --host :: \
    --api-key dummy \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --enable-lora --max-lora-rank 128 --max-loras 6 \
    --gpu-memory-utilization 0.92 --max_model_len 24576 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 128 \
    --enable_prefix_caching \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids \
    --enforce-eager 

A100_40G:

/root/run.sh python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --port 8091 --host :: \
    --api-key dummy \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --enable-lora --max-lora-rank 64 --max-loras 2 \
    --gpu-memory-utilization 0.90 --max_model_len 8192 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 64 \
    --enable_prefix_caching \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids \
    --enforce-eager 


#    --enable-lora --max-lora-rank 128 --max-loras 6 \

/root/run.sh python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-20b \
    --port 8091 --host 0.0.0.0 \
    --api-key dummy \
    --gpu-memory-utilization 0.95 --max_model_len 24576 \
    --load_format safetensors \
    --trust_remote_code \
    --guided_decoding_backend guidance --guided-decoding-disable-fallback \
    --enable_auto_tool_choice --tool_call_parser hermes \
    --scheduling_policy priority \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --max_log_len 0 --max_num_seqs 16 \
    --enable_prefix_caching --prefix-caching-hash-algo sha256 \
    --generation-config vllm --override-generation-config '{"temperature":0.6,"top_p":1.0,"top_k":0,"repetition_penalty":1.0}' \
    --return-tokens-as-token-ids \
    --enforce-eager 


#    --kv-cache-dtype fp8_e5m2 --calculate-kv-scales true \
