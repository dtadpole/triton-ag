#!/usr/bin/env python3
"""
Utility script to prepare a checkpoint from a previous training for fresh training.

This script loads a checkpoint, extracts only the model weights (either LoRA adapter
or full model weights), and saves a new checkpoint with:
- Model weights preserved
- Global step reset to 0
- Optimizer state removed
- Scheduler state removed

This allows you to start fresh training from a pretrained model without carrying
over training state.

Usage:
    python prepare_checkpoint_for_fresh_training.py \
        --input /path/to/checkpoint-1000 \
        --output /path/to/checkpoint-fresh

    # Or specify exact training_state.pt file
    python prepare_checkpoint_for_fresh_training.py \
        --input /path/to/checkpoint-1000/training_state.pt \
        --output /path/to/checkpoint-fresh/training_state.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from logger import logger


def prepare_checkpoint_for_fresh_training(
    input_checkpoint: str, output_checkpoint: str, force: bool = False
):
    """
    Prepare checkpoint for fresh training by keeping only model weights.

    Args:
        input_checkpoint: Path to input checkpoint directory or training_state.pt
        output_checkpoint: Path to output checkpoint directory or training_state.pt
        force: If True, overwrite existing output checkpoint
    """
    # Resolve paths
    input_path = Path(input_checkpoint)
    output_path = Path(output_checkpoint)

    # Get the training state file paths
    if input_path.is_dir():
        input_state_file = input_path / "training_state.pt"
        output_dir = output_path
        output_state_file = output_path / "training_state.pt"
    else:
        input_state_file = input_path
        output_dir = output_path.parent
        output_state_file = output_path

    # Validate input
    if not input_state_file.exists():
        logger.error(f"❌ Input checkpoint not found: {input_state_file}")
        sys.exit(1)

    # Check if output already exists
    if output_state_file.exists() and not force:
        logger.error(
            f"❌ Output checkpoint already exists: {output_state_file}\n"
            f"   Use --force to overwrite"
        )
        sys.exit(1)

    # Load the checkpoint
    logger.info(f"📂 Loading checkpoint from: {input_state_file}")
    try:
        checkpoint = torch.load(input_state_file, map_location="cpu")
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        sys.exit(1)

    # Display checkpoint info
    logger.info(f"📊 Checkpoint contents: {list(checkpoint.keys())}")
    if "global_step" in checkpoint:
        logger.info(f"📍 Original global step: {checkpoint['global_step']}")

    # Determine checkpoint type (LoRA or full model)
    has_lora = "lora_state_dict" in checkpoint
    has_model = "model_state_dict" in checkpoint

    if not has_lora and not has_model:
        logger.error(
            "❌ No model weights found in checkpoint\n"
            f"   Available keys: {list(checkpoint.keys())}"
        )
        sys.exit(1)

    # Create new checkpoint with only model weights
    new_checkpoint = {"global_step": 0}

    if has_lora:
        new_checkpoint["lora_state_dict"] = checkpoint["lora_state_dict"]
        logger.info(
            f"✅ Extracted LoRA weights "
            f"({len(checkpoint['lora_state_dict'])} parameters)"
        )

    if has_model:
        new_checkpoint["model_state_dict"] = checkpoint["model_state_dict"]
        logger.info(
            f"✅ Extracted model weights "
            f"({len(checkpoint['model_state_dict'])} parameters)"
        )

    # Include config if present
    if "config" in checkpoint:
        new_checkpoint["config"] = checkpoint["config"]
        logger.info("✅ Config preserved")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the new checkpoint
    logger.info(f"💾 Saving fresh checkpoint to: {output_state_file}")
    try:
        torch.save(new_checkpoint, output_state_file)
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")
        sys.exit(1)

    # If working with directories, also copy other files
    if input_path.is_dir() and output_path.is_dir():
        # Copy config file if it exists
        input_config = input_path / "training_config.yaml"
        if input_config.exists():
            output_config = output_path / "training_config.yaml"
            try:
                with open(input_config, "r") as f:
                    config_data = yaml.safe_load(f)

                # Update config to reflect fresh training
                if "training" in config_data:
                    if "latest_checkpoint_name" in config_data["training"]:
                        config_data["training"]["latest_checkpoint_name"] = None

                with open(output_config, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False)

                logger.info(f"📄 Config copied to: {output_config}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to copy config: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("✅ Checkpoint prepared successfully!")
    logger.info(f"📥 Input:  {input_state_file}")
    logger.info(f"📤 Output: {output_state_file}")
    logger.info("")
    logger.info("🔄 Changes made:")
    logger.info("   ✓ Model weights preserved")
    logger.info("   ✓ Global step reset to 0")
    logger.info("   ✓ Optimizer state removed")
    logger.info("   ✓ Scheduler state removed")
    logger.info("")
    logger.info("💡 You can now use this checkpoint for fresh training:")
    logger.info(f"   config.training.latest_checkpoint_name = '{output_path.name}'")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare checkpoint for fresh training by removing optimizer/scheduler state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare from checkpoint directory
  python prepare_checkpoint_for_fresh_training.py \\
      --input checkpoints/checkpoint-1000 \\
      --output checkpoints/checkpoint-fresh

  # Prepare from specific training_state.pt file
  python prepare_checkpoint_for_fresh_training.py \\
      --input checkpoints/checkpoint-1000/training_state.pt \\
      --output checkpoints/checkpoint-fresh/training_state.pt

  # Force overwrite existing output
  python prepare_checkpoint_for_fresh_training.py \\
      --input checkpoints/checkpoint-1000 \\
      --output checkpoints/checkpoint-fresh \\
      --force
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input checkpoint directory or training_state.pt file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output checkpoint directory or training_state.pt file",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwrite if output checkpoint already exists",
    )

    args = parser.parse_args()

    # Run the preparation
    prepare_checkpoint_for_fresh_training(args.input, args.output, args.force)


if __name__ == "__main__":
    main()
