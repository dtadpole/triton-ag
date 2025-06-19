import torch
import json
from transformers import DataCollatorForLanguageModeling

texts = [
  "The quick brown fox jumps over the lazy dog.",
  "I am learning about AI today"  
]
# Tokenize
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B')
tokens = [tokenizer(t) for t in texts]

# Default collate function 
collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Pass it to dataloader
dataloader = torch.utils.data.DataLoader(dataset=tokens, collate_fn=collate_fn, batch_size=2) 

# this will end in error
for batch in dataloader:
    # recursively convert batch.data from Tensor to list
    for k, v in batch.data.items():
        if isinstance(v, torch.Tensor):
            batch.data[k] = v.tolist()
    print(json.dumps(batch.data, indent=4))

print(tokenizer.special_tokens_map)
