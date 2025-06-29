# Using Together AI for LoRA inference
import os
import requests
from together import Together

with open(os.path.join(os.path.expanduser("~"), ".keys", "together.api.key"), "r") as f:
    together_api_key = f.read().strip()

with open(os.path.join(os.path.expanduser("~"), ".keys", "huggingface.api.key"), "r") as f:
    huggingface_api_key = f.read().strip()

client = Together(api_key=together_api_key)

PRESIGNED_ADAPTER_URL="https://huggingface.co/dtadpole/KernelCoder-32B_20250621-013349"

MODEL_TYPE="adapter"
BASE_MODEL="Qwen/Qwen3-32B"
DESCRIPTION="KernelCoder-32B"
ADAPTER_MODEL_NAME="KernelCoder-32B-20250621"

# curl -v https://api.together.xyz/v0/models \
#   -H 'Content-Type: application/json' \
#   -H "Authorization: Bearer $TOGETHER_API_KEY" \
#   -d '{
#   "model_name": "'${ADAPTER_MODEL_NAME}'",
#   "model_source": "'${PRESIGNED_ADAPTER_URL}'",
#   "model_type": "'${MODEL_TYPE}'",
#   "base_model": "'${BASE_MODEL}'",
#   "description": "'${DESCRIPTION}'"
# }'

# convert the above to a python request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {together_api_key}"
}
data = {
    "model_name": ADAPTER_MODEL_NAME,
    "model_source": PRESIGNED_ADAPTER_URL,
    "model_type": MODEL_TYPE,
    "description": DESCRIPTION,
    "hf_token": huggingface_api_key,
    # "base_model": BASE_MODEL,
}

upload_response = requests.post("https://api.together.xyz/v0/models", headers=headers, json=data)

print(upload_response.json())


# Run inference with LoRA
# inference_response = client.completions.create(
#     model="dtadpole/KernelCoder-32B_20250621-013349",  # Uses your LoRA
#     prompt="Write a CUDA kernel that computes the matrix multiplication of two matrices A and B, and stores the result in matrix C. The matrices are of size NxN.",
#     max_tokens=8192
# )

# print(inference_response.text)