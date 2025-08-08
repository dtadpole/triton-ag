from vllm import LLM, SamplingParams
import torch
import os

def detect_model_type(model_path):
    """Detect if model is quantized and what type"""
    if os.path.isdir(model_path):
        # Check for quantized model files
        files = os.listdir(model_path)
        if any('awq' in f.lower() for f in files):
            return "awq"
        elif any('gptq' in f.lower() for f in files):
            return "gptq"
        elif any('ggml' in f.lower() or 'gguf' in f.lower() for f in files):
            return "ggml"
        elif any('int4' in f.lower() or 'int8' in f.lower() for f in files):
            return "int_quantized"

    # Check model name for quantization hints
    model_name = model_path.lower()
    if 'awq' in model_name:
        return "awq"
    elif 'gptq' in model_name:
        return "gptq"
    elif 'ggml' in model_name or 'gguf' in model_name:
        return "ggml"

    return "full_precision"

def run_batch_inference(prompts, max_model_length, max_new_tokens, group_size, model_path):

    # Detect model type
    model_type = detect_model_type(model_path)
    print(f"Detected model type: {model_type}")

    # Configure based on model type
    if model_type == "awq":
        # AWQ quantized model
        llm = LLM(
            model=model_path,
            quantization="awq",
            tensor_parallel_size=1,
            dtype=torch.float16,  # AWQ typically uses float16
            trust_remote_code=True,
            max_model_len=max_model_length,
            max_num_seqs=64,
            gpu_memory_utilization=0.95,  # Can use more memory with quantized models
        )
    elif model_type == "gptq":
        # GPTQ quantized model
        llm = LLM(
            model=model_path,
            quantization="gptq",
            tensor_parallel_size=1,
            dtype=torch.float16,
            trust_remote_code=True,
            max_model_len=max_model_length,
            max_num_seqs=64,
            gpu_memory_utilization=0.95,
        )
    elif model_type == "ggml":
        print("GGML/GGUF models are not directly supported by vLLM")
        print("Consider using llama.cpp or converting to HF format")
        return
    else:
        # Full precision model
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype=torch.float16,     # Use float16 for memory efficiency
            # dtype=torch.bfloat16,  # Alternative: use bfloat16 if supported
            # dtype="auto",          # Let vLLM decide the best dtype
            trust_remote_code=True,
            max_model_len=max_model_length,
            gpu_memory_utilization=0.9,  # Use less memory for full precision
        )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_new_tokens,
        n=group_size
    )

    # Run batch inference
    print("Running batch inference...")
    outputs = llm.generate(prompts, sampling_params)

    output_dict = {}

    # Process and display results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        output_dict[prompt] = []
        for o_ in output.outputs:
            output_dict[prompt].append(o_.text)

    return output_dict


if __name__ == "__main__":
    # Run the basic batch inference example
    prompts = ["hello world!"]
    result = run_batch_inference(prompts, 2048, 1024, 4, "Qwen/Qwen3-8B-AWQ")
    print(result)
