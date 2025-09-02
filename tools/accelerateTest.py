import os, torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True

MODEL_ID = "Qwen/Qwen3-32B"

def main():
    accelerator = Accelerator()                          # uses your fsdp config when launched
    device = accelerator.device

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": "cpu"}
    )
    model.train()

    # toy text dataset -> replace with yours
    ds = load_dataset("imdb", split="train[:1%]")
    def encode(batch):
        txt = tok(batch["text"], truncation=True, max_length=1024)
        txt["labels"] = txt["input_ids"].copy()
        return txt
    ds = ds.map(encode, batched=True, remove_columns=ds.column_names)

    train_loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=default_data_collator)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_loader, model, optim = accelerator.prepare(train_loader, model, optim)

    for step, batch in enumerate(train_loader, start=1):
        with accelerator.accumulate(model):              # gradient accumulation
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            loss = outputs.loss
            accelerator.backward(loss)
            optim.step()
            optim.zero_grad()

        if accelerator.is_main_process and step % 50 == 0:
            accelerator.print(f"step {step} | loss {loss.item():.4f}")
        if step == 200:                                  # demo run
            break

    # FSDP-friendly save
    accelerator.save_state("ckpt")                       # sharded save
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            "qwen3_32b_finetuned",
            is_main_process=True,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

if __name__ == "__main__":
    main()
