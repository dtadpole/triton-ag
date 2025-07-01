import predibase

# Deploy LoRA adapter
adapter = predibase.Adapter.upload(
    "dtadpole/KernelCoder-32B_20250621-013349",
    base_model="Qwen/Qwen3-32B"
)

# Serverless inference
response = adapter.predict({
    "prompt": "Write a CUDA kernel that computes the matrix multiplication of two matrices A and B, and stores the result in matrix C. The matrices are of size NxN.",
    "max_tokens": 8192
})

