# Modal Labs LoRA hosting
import modal

app = modal.App("lora-server")

@app.function(
    image=modal.Image.debian_slim().pip_install("vllm", "transformers"),
    gpu="A100"
)
def serve_lora(prompt: str, lora_adapter: str):
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    
    llm = LLM(model="Qwen/Qwen3-32B", enable_lora=True)
    lora_request = LoRARequest("dtadpole/KernelCoder-32B_20250621-013349", 1, lora_adapter)
    
    outputs = llm.generate([prompt], lora_request=lora_request)
    return outputs[0].outputs[0].text

@app.function()
@modal.fastapi_endpoint()
def api(prompt: str, adapter: str = "default"):
    return {"response": serve_lora.remote(prompt, adapter)}
