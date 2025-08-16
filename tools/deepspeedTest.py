import torch, deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B",
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
model.config.use_cache = False
model.gradient_checkpointing_enable()

ds_cfg = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

# ...or pass a torch optimizer instead of the "optimizer" block:
# opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    config=ds_cfg,
    # optimizer=opt,  # <- use this if you comment out the "optimizer" block above
)

inputs = tok("hello world", return_tensors="pt").to(engine.device)
labels = inputs["input_ids"].clone()
out = engine(**inputs, labels=labels)
loss = out.loss
engine.backward(loss)
engine.step()
if engine.global_rank == 0:
    print("loss:", loss.item())
