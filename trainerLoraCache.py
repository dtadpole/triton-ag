import torch
import torch.nn as nn
import os
from unsloth import FastLanguageModel
from peft import PeftModel
import argparse
from logger import logger
from typing import Optional

class TrainerLoraCache:
    def __init__(self,
                 prefix_tag: str,
                 base_model: nn.Module,
                 lora_model: nn.Module,
                 extra_cache_size: int = 2,
                 cache_dir: str = "~/.trainer",
    ):
        self.prefix_tag = prefix_tag
        self.base_model = base_model
        self.lora_model = lora_model
        self.cache_folder = os.path.expanduser(cache_dir)
        self.extra_cache_size = extra_cache_size

    def load_checkpoint(self, checkpoint_name: Optional[str]=None):
        if checkpoint_name is None:
            # return base model if no checkpoint name is provided
            logger.info(f"Using base model as checkpoint is [None]")
            return self.base_model
        elif checkpoint_name in self.lora_model.peft_config.keys():
            logger.info(f"LoRA checkpoint {checkpoint_name} already loaded")
            self.lora_model.set_adapter(checkpoint_name)
            # self.lora_model.eval()
            return self.lora_model
        # load new lora model if not already loaded
        lora_path = os.path.join(self.cache_folder, self.prefix_tag, checkpoint_name)
        logger.info(f"Loading LoRA checkpoint from {lora_path}")
        self.lora_model.load_adapter(lora_path, adapter_name=checkpoint_name)
        self.lora_model.set_adapter(checkpoint_name)
        # self.lora_model.eval()
        logger.info(f"Loaded LoRA checkpoint from {lora_path}")
        self._clean_cache(keep_checkpoint=checkpoint_name)
        return self.lora_model

    def _clean_cache(self, keep_checkpoint: str=None):
        checkpoint_numbers = [int(key.split("-")[-1]) for key in self.lora_model.peft_config.keys() if key.startswith("checkpoint") and key != "checkpoint-latest" and key != 'default' and key != keep_checkpoint]
        checkpoint_numbers.sort()
        # logger.info(f"Cleaning cache for {self.prefix_tag} with checkpoint numbers {checkpoint_numbers}")
        for checkpoint_number in checkpoint_numbers[:-self.extra_cache_size]:
            checkpoint_name = f"checkpoint-{checkpoint_number}"
            self.lora_model.delete_adapter(checkpoint_name)
            logger.info(f"Removed LoRA model from {checkpoint_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, required=True)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--cache_dir", type=str, default="~/.trainer")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--checkpoint_name_1", type=str, default="checkpoint-2000")
    parser.add_argument("--checkpoint_name_2", type=str, default="checkpoint-2050")
    parser.add_argument("--checkpoint_name_3", type=str, default="checkpoint-2100")
    parser.add_argument("--checkpoint_name_4", type=str, default="checkpoint-2150")
    parser.add_argument("--checkpoint_name_5", type=str, default="checkpoint-2200")
    parser.add_argument("--checkpoint_name_6", type=str, default="checkpoint-2250")
    parser.add_argument("--checkpoint_name_7", type=str, default="checkpoint-2300")
    args = parser.parse_args()


    checkpoint_latest = "checkpoint-latest"
    base_model, tokenizer = FastLanguageModel.from_pretrained(args.base_model)
    lora_path = os.path.join(os.path.expanduser(args.cache_dir), args.prefix_tag, checkpoint_latest)
    lora_model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=checkpoint_latest)

    trainer_lora_cache = TrainerLoraCache(
        args.prefix_tag,
        base_model,
        lora_model,
        extra_cache_size=args.cache_size,
        cache_dir=args.cache_dir
    )

    trainer_lora_cache.load_checkpoint(args.checkpoint_name_3)
    trainer_lora_cache.load_checkpoint(args.checkpoint_name_4)
    trainer_lora_cache.load_checkpoint(args.checkpoint_name_5)
    trainer_lora_cache.load_checkpoint(args.checkpoint_name_6)
    trainer_lora_cache.load_checkpoint(args.checkpoint_name_7)
    trainer_lora_cache.load_checkpoint(args.checkpoint_name_7) # load same checkpoint again to test if it is already loaded
    trainer_lora_cache.load_checkpoint(args.checkpoint_name_1) # put lower number at the end for testing
    trainer_lora_cache.load_checkpoint(args.checkpoint_name_2) # put lower number at the end for testing

    print(list(trainer_lora_cache.lora_model.peft_config.keys()))
