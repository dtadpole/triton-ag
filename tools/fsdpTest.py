# fsdp_qwen3_32b_offload.py
import os, torch, torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload, MixedPrecision, ShardingStrategy,
)

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)

    # bfloat16 if available, otherwise float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    model_id = "Qwen/Qwen3-32B"
    tok = AutoTokenizer.from_pretrained(model_id)

    # Load on CPU to avoid GPU spikes before FSDP shards things
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": "cpu"}
    )

    # FSDP: FULL_SHARD + CPU offload (params+grads live on CPU when idle)
    model = FSDP(
        model,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        cpu_offload=CPUOffload(offload_params=True),
        use_orig_params=True,
        limit_all_gathers=True,  # helps peak memory during all-gathers
    )

    # Simple prompt
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Say hi in one short sentence."}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(device)

    # Inference: keep it lean
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        # small max_new_tokens keeps KV-cache memory modest
        out = model.module.generate(**inputs, max_new_tokens=32)

    if dist.get_rank() == 0:
        print(tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
