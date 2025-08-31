#!/usr/bin/env python3
"""
fsdp_to_hf.py — Convert a sharded FSDP checkpoint (saved via torch.distributed.checkpoint.save)
into Hugging Face `save_pretrained` format for **causal language models**.

Assumptions (simplified):
1) The checkpoint was saved with `torch.distributed.checkpoint.save(...)` into a DIRECTORY.
2) The checkpoint is **always sharded** (multiple files + a manifest).
3) Target arch is **causal LM**. We instantiate `transformers.AutoModelForCausalLM` from a
   provided config (either `--from-pretrained` or `--config`).

High-level logic:
- Build an HF `AutoModelForCausalLM` from config (no weights).
- **Fast path (default):** Construct a name→tensor mapping that points **directly to the model's parameters/buffers** (no intermediate copy), then let `torch.distributed.checkpoint.load` write shards straight into those tensors.
- Optional dtype cast, then `save_pretrained()` (optionally safetensors).
- Profiling mode: measure elapsed time for each phase (load, dtype cast, save) to diagnose bottlenecks.
- Only model parameter/buffer keys are loaded; optimizer states are ignored.

Typical usage:
  python fsdp_to_hf.py \
    --checkpoint ./fsdp_dir \
    --outdir ./hf_model_out \
    --from-pretrained meta-llama/Llama-2-7b-hf \
    --safe
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Dict

import torch
from torch import nn

# PyTorch distributed checkpoint (TDC)
try:
    # Latest API (PyTorch ≥ 2.4)
    from torch.distributed.checkpoint import load as tdc_load
    try:
        # Optional: file-system reader with adjustable threads (if available)
        from torch.distributed.checkpoint import FileSystemReader  # type: ignore
    except Exception:
        FileSystemReader = None  # type: ignore
except Exception as e:
    raise RuntimeError(
        "This script requires torch.distributed.checkpoint.load (PyTorch ≥ 2.4).\n"
        "Please upgrade PyTorch and ensure 'torch.distributed.checkpoint' is available."
    ) from e

# Transformers (Hugging Face)
try:
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    raise RuntimeError(
        "This script requires 'transformers'. Install with: pip install transformers"
    ) from e

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_hf_causal_lm(from_pretrained: Optional[str], config_src: Optional[str]) -> nn.Module:
    if from_pretrained:
        logging.info(f"Loading model from: {from_pretrained}")
        # Load config first to preserve original torch_dtype
        original_config = AutoConfig.from_pretrained(from_pretrained)
        original_dtype = original_config.torch_dtype
        logging.info(f"Original model dtype: {original_dtype}")
        
        # Load model with weights from HF - this is actually faster than building from config
        model = AutoModelForCausalLM.from_pretrained(from_pretrained, torch_dtype=torch.float32)
        # Restore original config to preserve the original torch_dtype
        model.config.torch_dtype = original_dtype
    elif config_src:
        logging.info(f"Loading config from: {config_src}")
        cfg = AutoConfig.from_pretrained(config_src)
        logging.info("Building model from config (no weights)...")
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=None)
    else:
        raise ValueError("Provide either --from-pretrained or --config to supply config.json")
    
    logging.info("Model loaded successfully")
    return model


def load_fsdp_shards_into(target_state: Dict[str, torch.Tensor], checkpoint_dir: Path, threads: int | None = None) -> None:
    """Load FSDP shards into target_state using torch.distributed.checkpoint.load API.

    If FileSystemReader is available, use it to optionally increase IO parallelism.
    Only the keys present in target_state (model params/buffers) are materialized; other
    collections such as optimizer state are ignored automatically.
    """
    if FileSystemReader is not None and threads and threads > 0:
        try:
            reader = FileSystemReader(str(checkpoint_dir), thread_count=threads)  # type: ignore[arg-type]
            tdc_load(state_dict=target_state, storage_reader=reader)
            return
        except Exception as e:
            logging.warning("Falling back to default reader (FileSystemReader thread tuning failed): %s", e)
    # Default path: let TDC open from directory
    # For single-process loading, we need to use FileSystemReader
    reader = FileSystemReader(str(checkpoint_dir)) if FileSystemReader else None
    if reader:
        tdc_load(state_dict=target_state, storage_reader=reader)
    else:
        # Fallback: try the old API
        tdc_load(target_state, str(checkpoint_dir))


def save_hf(model: nn.Module, outdir: Path, safe: bool, max_shard_size: str | None = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir, safe_serialization=safe, max_shard_size=max_shard_size)


def copy_tokenizer_if_requested(tokenizer_from: Optional[str], outdir: Path) -> None:
    if not tokenizer_from:
        return
    tok = AutoTokenizer.from_pretrained(tokenizer_from, use_fast=True)
    tok.save_pretrained(outdir)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert sharded FSDP checkpoint to HF causal LM")
    p.add_argument("--checkpoint", required=True, help="Directory produced by torch.distributed.checkpoint.save")
    p.add_argument("--outdir", required=True, help="Output directory for Hugging Face model files")
    p.add_argument("--from-pretrained", dest="from_pretrained", default=None, help="Config/tokenizer source on Hub or local dir (no weights)")
    p.add_argument("--config", dest="config_src", default=None, help="Local config dir if not using --from-pretrained")
    p.add_argument("--tokenizer-from", dest="tokenizer_from", default=None, help="Optional: copy tokenizer files from this model name/path to outdir")
    p.add_argument("--dtype", choices=DTYPE_MAP.keys(), default=None, help="Cast model to dtype before saving")
    p.add_argument("--safe", action="store_true", help="Save with safetensors (recommended)")
    p.add_argument("--ignore-missing", action="store_true", help="Allow missing/unexpected keys when finalizing into the HF model")
    p.add_argument("--threads", type=int, default=None, help="Optional IO threads for FileSystemReader (if available)")
    p.add_argument("--cpu-threads", type=int, default=None, help="Set torch.set_num_threads / set_num_interop_threads for CPU parallelism")
    p.add_argument("--max-shard-size", default="4GB", help="Optional max shard size for HF save_pretrained (e.g., 4GB); matches original HF model sharding")
    p.add_argument("--show-threads", action="store_true", help="Print effective torch/OpenMP thread settings for debugging")
    p.add_argument("--two-step", action="store_true", help="Use slower two-step load (via model.state_dict copy) instead of direct in-place load")
    p.add_argument("--profile", action="store_true", help="Measure time spent in each phase (load/cast/save)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")

    # Configure CPU thread pools if requested
    if args.cpu_threads and args.cpu_threads > 0:
        try:
            torch.set_num_threads(args.cpu_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(max(1, args.cpu_threads // 2))
        except Exception:
            pass

    if args.show_threads:
        try:
            logging.info("torch.get_num_threads() = %s", torch.get_num_threads())
        except Exception:
            pass
        try:
            logging.info("torch.get_num_interop_threads() = %s", torch.get_num_interop_threads())
        except Exception:
            pass
        import os
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_AFFINITY", "KMP_BLOCKTIME", "MKL_DYNAMIC"):
            if var in os.environ:
                logging.info("env %s=%s", var, os.environ[var])

    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"--checkpoint must be a directory (got: {ckpt_dir})")

    # 1) Build empty HF causal LM from config
    t0 = time.time()
    logging.info("Starting model building phase...")
    model = build_hf_causal_lm(args.from_pretrained, args.config_src)
    model.eval()
    t1 = time.time()
    logging.info("Model building phase completed")

    # 2) Prepare TARGET mapping on CPU
    # First, get the available keys from the checkpoint
    reader = FileSystemReader(str(ckpt_dir)) if FileSystemReader else None
    if reader:
        metadata = reader.read_metadata()
        available_keys = set(metadata.state_dict_metadata.keys())
        logging.info(f"Found {len(available_keys)} keys in checkpoint")
    else:
        available_keys = set()
    
    with torch.no_grad():
        if not args.two_step:
            target: Dict[str, torch.Tensor] = {}
            for name, p in model.named_parameters():
                # Map from HF model keys to FSDP checkpoint keys
                if name.startswith("model."):
                    # model.* -> app.model.model.*
                    checkpoint_key = f"app.model.{name}"
                    if checkpoint_key in available_keys:
                        target[checkpoint_key] = p.detach()
                    else:
                        logging.info(f"Skipping parameter not in checkpoint: {name}")
                elif name.startswith("lm_head."):
                    # lm_head.* -> app.model.lm_head.*
                    checkpoint_key = f"app.model.{name}"
                    if checkpoint_key in available_keys:
                        target[checkpoint_key] = p.detach()
                    else:
                        logging.info(f"Skipping parameter not in checkpoint: {name}")
                else:
                    # Other keys -> app.model.*
                    checkpoint_key = f"app.model.{name}"
                    if checkpoint_key in available_keys:
                        target[checkpoint_key] = p.detach()
                    else:
                        logging.info(f"Skipping parameter not in checkpoint: {name}")
            for name, b in model.named_buffers():
                # Skip buffers that are computed dynamically (like rotary_emb.inv_freq)
                if "rotary_emb.inv_freq" in name:
                    logging.info(f"Skipping computed buffer: {name}")
                    continue
                # Map from HF model keys to FSDP checkpoint keys
                if name.startswith("model."):
                    # model.* -> app.model.model.*
                    checkpoint_key = f"app.model.{name}"
                    if checkpoint_key in available_keys:
                        target[checkpoint_key] = b
                    else:
                        logging.info(f"Skipping buffer not in checkpoint: {name}")
                elif name.startswith("lm_head."):
                    # lm_head.* -> app.model.lm_head.*
                    checkpoint_key = f"app.model.{name}"
                    if checkpoint_key in available_keys:
                        target[checkpoint_key] = b
                    else:
                        logging.info(f"Skipping buffer not in checkpoint: {name}")
                else:
                    # Other keys -> app.model.*
                    checkpoint_key = f"app.model.{name}"
                    if checkpoint_key in available_keys:
                        target[checkpoint_key] = b
                    else:
                        logging.info(f"Skipping buffer not in checkpoint: {name}")
        else:
            target = model.state_dict()
            # Map all keys to have correct 'app.model.' prefix
            mapped_target = {}
            for k, v in target.items():
                # Skip buffers that are computed dynamically
                if "rotary_emb.inv_freq" in k:
                    logging.info(f"Skipping computed buffer: {k}")
                    continue
                if k.startswith("model."):
                    # model.* -> app.model.model.*
                    checkpoint_key = f"app.model.{k}"
                    if checkpoint_key in available_keys:
                        mapped_target[checkpoint_key] = v
                    else:
                        logging.info(f"Skipping key not in checkpoint: {k}")
                elif k.startswith("lm_head."):
                    # lm_head.* -> app.model.lm_head.*
                    checkpoint_key = f"app.model.{k}"
                    if checkpoint_key in available_keys:
                        mapped_target[checkpoint_key] = v
                    else:
                        logging.info(f"Skipping key not in checkpoint: {k}")
                else:
                    # Other keys -> app.model.*
                    checkpoint_key = f"app.model.{k}"
                    if checkpoint_key in available_keys:
                        mapped_target[checkpoint_key] = v
                    else:
                        logging.info(f"Skipping key not in checkpoint: {k}")
            target = mapped_target

    # 3) Load shards
    logging.info("Loading sharded FSDP checkpoint via torch.distributed.checkpoint.load ...")
    t2 = time.time()
    load_fsdp_shards_into(target, ckpt_dir, threads=args.threads)
    t3 = time.time()

    # 4) If we used two-step, load into model (only model params/buffers keys considered)
    # Convert back from FSDP checkpoint keys to HF model keys
    if not args.two_step:
        # For direct loading, we need to create a mapping back to the original model keys
        hf_target = {}
        for key, tensor in target.items():
            if key.startswith("app.model.model."):
                # app.model.model.* -> model.*
                hf_key = key[10:]  # Remove 'app.model.' prefix
                hf_target[hf_key] = tensor
            elif key.startswith("app.model.lm_head."):
                # app.model.lm_head.* -> lm_head.*
                hf_key = key[10:]  # Remove 'app.model.' prefix
                hf_target[hf_key] = tensor
            elif key.startswith("app.model."):
                # app.model.* -> * (for other keys)
                hf_key = key[10:]  # Remove 'app.model.' prefix
                hf_target[hf_key] = tensor
        missing, unexpected = model.load_state_dict(hf_target, strict=not args.ignore_missing)
    else:
        # For two-step, target already has the correct keys
        missing, unexpected = model.load_state_dict(target, strict=not args.ignore_missing)
    if missing:
        logging.warning("Missing keys count: %d (first 20): %s", len(missing), missing[:20])
    if unexpected:
        logging.warning("Unexpected keys count: %d (first 20): %s", len(unexpected), unexpected[:20])
    if (missing or unexpected) and not args.ignore_missing:
        logging.error("State dict did not load cleanly. Re-run with --ignore-missing if acceptable.")
        return 2
    t4 = time.time()

    # 5) Optional dtype cast
    if args.dtype:
        model.to(DTYPE_MAP[args.dtype])
        logging.info("Cast model to %s", args.dtype)
    t5 = time.time()

    # 6) Save HF model
    # Convert to the same dtype as the original model if available
    if hasattr(model.config, 'torch_dtype') and model.config.torch_dtype:
        original_dtype = model.config.torch_dtype
        current_dtype = next(model.parameters()).dtype
        if original_dtype != current_dtype:
            logging.info("Converting model from %s to original dtype: %s", current_dtype, original_dtype)
            model = model.to(original_dtype)
    
    save_hf(model, Path(args.outdir), safe=args.safe, max_shard_size=args.max_shard_size or "4GB")
    logging.info("Saved Hugging Face model to: %s", args.outdir)
    t6 = time.time()

    # 7) Copy tokenizer
    try:
        copy_tokenizer_if_requested(args.tokenizer_from or args.from_pretrained, Path(args.outdir))
    except Exception as e:
        logging.warning("Tokenizer copy skipped/failed: %s", e)
    t7 = time.time()

    # Print detailed timing information
    logging.info("=== TIMING BREAKDOWN ===")
    logging.info("Build model from config: %.2fs", t1-t0)
    logging.info("Load FSDP checkpoint: %.2fs", t3-t2)
    logging.info("Load state dict into model: %.2fs", t4-t3)
    logging.info("Dtype conversion: %.2fs", t5-t4)
    logging.info("Save Hugging Face model: %.2fs", t6-t5)
    logging.info("Copy tokenizer: %.2fs", t7-t6)
    logging.info("TOTAL TIME: %.2fs", t7-t0)
    
    if args.profile:
        logging.info("DETAILED TIMINGS: build=%.2fs, load=%.2fs, state_dict=%.2fs, cast=%.2fs, save=%.2fs, tokenizer=%.2fs, total=%.2fs",
                     t1-t0, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t7-t0)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
