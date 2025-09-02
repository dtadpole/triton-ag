import torch, deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen3-32B"   # or "Qwen/Qwen3-32B-Instruct"

def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📜 Model [{model.__class__.__name__}] loaded - Total: {total_params:,}, Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

# 1) Tokenizer & base model
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.config.use_cache = False
model.gradient_checkpointing_enable()
print_model_info(model)

# 2) Add tiny LoRA adapters (train only a few projection layers)
lora = LoraConfig(
    r=128, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj", "gate_proj", "up_proj", "down_proj"]  # minimal & effective
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()  # sanity check
print_model_info(model)

# 3) DeepSpeed config (keep it tiny)
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

# 4) Wrap with DeepSpeed (only train LoRA params)
engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    config=ds_cfg,
)

# 5) One toy step (replace with your real batch/loader)
prompt = "Summarize: DeepSpeed with LoRA on Qwen3-32B."
batch = tok(prompt, return_tensors="pt").to(engine.device)
labels = batch["input_ids"].clone()
out = engine(**batch, labels=labels)
loss = out.loss
engine.backward(loss)
engine.step()

if engine.global_rank == 0:
    print("loss:", loss.item())
