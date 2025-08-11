import os
import sys
import os
from datetime import datetime
from collections import defaultdict
import duckdb
import unsloth
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import yaml
import asyncio
import argparse
import json
from pydantic import BaseModel
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from trainerBase import BaseTrainer, TrainerConfig, TrainerStatus, train_async
from trainerUtil import format_conversation, SimpleCollator
from logger import logger
import random
import math
import shutil
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import traceback
from configEndpoints import DuckDBClient, StatsClient
from configInterpreter import ConfigInterpreter
from workflowUtil import TrainerGRPOBlock
from workflowClient import WorkflowClient
from workflowServer import WorkflowServer

LATEST_REFERENCE_NAME = "reference_state_latest.pt"

CLIP_RATIO_LOWER_PERCENTAGE = "clip_ratio_lower_pct"
CLIP_RATIO_UPPER_PERCENTAGE = "clip_ratio_upper_pct"
BOUND_ADVANTAGE_LOWER_PERCENTAGE = "bound_adv_lower_pct"
BOUND_ADVANTAGE_UPPER_PERCENTAGE = "bound_adv_upper_pct"

class GRPOConfig(BaseModel):
    """GRPO configuration"""
    max_seq_length: int = 16384
    mask_non_assistant_tokens: bool = True
    discard_long_conversations: bool = True
    clip_ratio_epsilon_lower: float = 0.2
    clip_ratio_epsilon_upper: float = 0.3
    gspo_clip_ratio_epsilon_lower: float = 3e-4
    gspo_clip_ratio_epsilon_upper: float = 4e-4
    bound_advantage_range: float = 3.0
    beta: float = 0.0  # KL divergence coefficient
    reward_scale: bool = True
    reward_epsilon: float = 1e-3
    reward_noise: float = 1e-2
    # loss_type: str = "token" # "episode" or "token" or "seq_max" or "group_max" or "gspo"
    loss_type: str = "gspo" # "episode" or "token" or "seq_max" or "group_max" or "gspo"
    gamma: float = 0.5

    @classmethod
    def from_yaml(cls, file_path: str) -> "GRPOConfig":
        """Load GRPO configuration from YAML file"""
        with open(os.path.expanduser(file_path), 'r') as f:
            config = yaml.safe_load(f)
        grpo_config = config.get('grpo', {})
        return cls(**grpo_config)


class GRPOTrainer(BaseTrainer):
    """GRPO (Generalized Preference Optimization) trainer for preference learning"""
    
    def __init__(self,
        prefix_tag: str,
        grpo_config: GRPOConfig,
        base_config: TrainerConfig,
        module_file: str = "trainer/grpo.module.yaml",
        status: Optional[TrainerStatus] = None,
        base_trainer: BaseTrainer = None,
        # reference_model: Optional[torch.nn.Module] = None,
    ):
        """Initialize GRPO trainer"""
        super().__init__(prefix_tag, base_config, status, base_trainer)
        self.configInterpreter = ConfigInterpreter()
        self.duckdbClient = DuckDBClient()
        self.grpo_config = grpo_config
        self.module_file = module_file
        self.module_config = yaml.safe_load(open(module_file, 'r')).get('module', {})
        self.context_vars = {
            # built-in context vars
            "self": self,
            "os": os,
            "json": json,
            "yaml": yaml,
            "grpo_config": self.grpo_config,
        }
        context_var_config = self.module_config.get('context_vars', {})
        self.context_vars = self.configInterpreter.prepare_context_vars(self, context_var_config, self.context_vars)
        

        logger.info(f"⭐ [GRPOTrainer] Initialized with GRPOConfig: {grpo_config}")

    def short_name(self):
        return 'grpo'

    def _update_grpo_config(self, grpo_config: GRPOConfig):
        """Update GRPO config"""
        self.grpo_config = grpo_config

    def log_raw_data(self, data: Any, context_vars: Dict[str, Any]):
        """Log raw data"""
        logger.info(f"🔍 [GRPOTrainer] DuckDB search has found [{len(data)}] rows.\n{data}")

    def _compute_mini_batch_loss(self, batch: Dict[str, Any], clip_metrics: Dict[str, List[float]], group_max_length: Optional[int] = None):
        """Compute loss for the generated tokens"""
        # turn_tags = batch['turn_tag']
        # rewards = batch['reward']
        advantages = batch['advantage']
        prompt_token_ids = batch['prompt_token_ids']
        completion_token_ids = batch['completion_token_ids']
        completion_log_probs = batch['completion_log_probs']
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # get logits from model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        batch_loss = 0.0
        for i in range(len(advantages)): # for each generation result in the batch
            # get the logits for the completion tokens only, remove the prompt tokens
            prompt_token_len = len(prompt_token_ids[i])
            output_completion_logits = outputs.logits[i, prompt_token_len-1:-1, :]
            new_log_probs = F.log_softmax(output_completion_logits, dim=-1) # dim: (completion_len, vocab_size)
            # get log probabilities for the completion tokens
            labels = torch.tensor(completion_token_ids[i], device=self.device)
            new_action_log_probs = new_log_probs[:len(completion_token_ids[i]), :].gather(
                dim=-1, 
                index=labels.unsqueeze(-1)
            ).squeeze(-1)  # (completion_len)

            # calculate log ratio (in log space is subtraction)
            completion_log_probs_tensor = torch.tensor(completion_log_probs[i], device=self.device)
            if completion_log_probs_tensor.shape != new_action_log_probs.shape:
                raise ValueError(f"❌ [GRPOTrainer] Completion log probabilities and new action log probabilities have different shapes: {completion_log_probs_tensor.shape} != {new_action_log_probs.shape}")
            # log probs is calculated in log space, so we need to subtract the log probabilities
            log_ratio = new_action_log_probs - completion_log_probs_tensor

            if self.grpo_config.loss_type == "gspo":

                sequence_log_ratio = torch.sum(log_ratio) / len(new_action_log_probs)
                sequence_ratio = torch.exp(sequence_log_ratio)

                # calculate clipped upper and lower percentage
                clip_metrics[CLIP_RATIO_UPPER_PERCENTAGE].append(torch.sum(sequence_ratio > 1+self.grpo_config.gspo_clip_ratio_epsilon_upper).item() * 100.0)
                clip_metrics[CLIP_RATIO_LOWER_PERCENTAGE].append(torch.sum(sequence_ratio < 1-self.grpo_config.gspo_clip_ratio_epsilon_lower).item() * 100.0)

                clamped_sequence_ratio = torch.clamp(sequence_ratio, 1-self.grpo_config.gspo_clip_ratio_epsilon_lower, 1+self.grpo_config.gspo_clip_ratio_epsilon_upper)

                sequence_ratio_advantage = torch.min(sequence_ratio * advantages[i], clamped_sequence_ratio * advantages[i])

                # calculate clipped upper and lower percentage
                clip_metrics[BOUND_ADVANTAGE_UPPER_PERCENTAGE].append(torch.sum(sequence_ratio_advantage > self.grpo_config.bound_advantage_range).item() * 100.0)
                clip_metrics[BOUND_ADVANTAGE_LOWER_PERCENTAGE].append(torch.sum(sequence_ratio_advantage < -self.grpo_config.bound_advantage_range).item() * 100.0)

                final_sequence_ratio_advantage = torch.clamp(sequence_ratio_advantage, -self.grpo_config.bound_advantage_range, self.grpo_config.bound_advantage_range)

                loss = -final_sequence_ratio_advantage.mean()

            else:
                ratio = torch.exp(log_ratio)

                # calculate clipped upper and lower percentage
                clip_metrics[CLIP_RATIO_UPPER_PERCENTAGE].append(torch.sum(ratio > 1+self.grpo_config.clip_ratio_epsilon_upper).item() * 100.0 / len(new_action_log_probs))
                clip_metrics[CLIP_RATIO_LOWER_PERCENTAGE].append(torch.sum(ratio < 1-self.grpo_config.clip_ratio_epsilon_lower).item() * 100.0 / len(new_action_log_probs))

                clamped_ratio = torch.clamp(ratio, 1-self.grpo_config.clip_ratio_epsilon_lower, 1+self.grpo_config.clip_ratio_epsilon_upper)

                # min ratio advantage is min of ratio_advantage and clamped_ratio_advantage
                ratio_advantage = torch.min(ratio * advantages[i], clamped_ratio * advantages[i]) # dim: (completion_len)

                # calculate clipped upper and lower percentage
                clip_metrics[BOUND_ADVANTAGE_UPPER_PERCENTAGE].append(torch.sum(ratio_advantage > self.grpo_config.bound_advantage_range).item() * 100.0 / len(new_action_log_probs))
                clip_metrics[BOUND_ADVANTAGE_LOWER_PERCENTAGE].append(torch.sum(ratio_advantage < -self.grpo_config.bound_advantage_range).item() * 100.0 / len(new_action_log_probs))

                # calculate clipped upper and lower percentage
                final_ratio_advantage = torch.clamp(ratio_advantage, -self.grpo_config.bound_advantage_range, self.grpo_config.bound_advantage_range)

                # compute loss
                if self.grpo_config.loss_type == "episode":
                    loss = -final_ratio_advantage.mean()
                elif self.grpo_config.loss_type == "token":
                    loss = -torch.sum(final_ratio_advantage) / len(new_action_log_probs)
                elif self.grpo_config.loss_type == "group_max":
                    loss = -torch.sum(final_ratio_advantage) / group_max_length
                elif self.grpo_config.loss_type == "seq_max":
                    loss = -torch.sum(final_ratio_advantage) / self.grpo_config.max_seq_length
                else:
                    raise ValueError(f"❌ [GRPOTrainingGroup] Invalid loss type: {self.grpo_config.loss_type}")
            
            batch_loss += loss

        return batch_loss

    def train_group(self, group_dataset: list[dict], callback: Optional[Callable] = None, total_groups: int = 0):
        """Train the model for one group"""
        dataloader = DataLoader(
            group_dataset,
            batch_size=self.config.training.micro_batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=True,
            collate_fn=SimpleCollator(tokenizer=self.tokenizer)
        )
        
        accumulated_loss = 0.0

        group_reward_mean = np.mean([result["reward"] for result in group_dataset])
        group_reward_std = np.std([result["reward"] for result in group_dataset])

        group_reward_items = {}
        for result in group_dataset:
            for key, value in result["reward_items"].items():
                if key not in group_reward_items:
                    group_reward_items[key] = []
                group_reward_items[key].append(value)
        group_reward_items_mean = {k: np.mean(v) for k, v in group_reward_items.items()}
        group_reward_items_std = {k: np.std(v) for k, v in group_reward_items.items()}

        # group_max_length = max(len(result['input_ids']) for result in group_dataset)
        group_max_length = max(len(result['completion_token_ids']) for result in group_dataset)

        self.model.train()
        clip_metrics = {
            CLIP_RATIO_UPPER_PERCENTAGE: [],
            CLIP_RATIO_LOWER_PERCENTAGE: [],
            BOUND_ADVANTAGE_UPPER_PERCENTAGE: [],
            BOUND_ADVANTAGE_LOWER_PERCENTAGE: [],
        }
        for batch_idx, batch in enumerate(dataloader):
            # Training step
            mini_batch_loss = self._compute_mini_batch_loss(batch, clip_metrics, group_max_length=group_max_length)
            
            # Scale loss for gradient accumulation
            mini_batch_loss = mini_batch_loss * self.config.training.loss_multiplier / len(dataloader) # divide by the group size
            mini_batch_loss.backward()

            # del outputs
            torch.cuda.empty_cache()

            accumulated_loss += mini_batch_loss.item()
            
        # Optimization step (only after entire group is processed, this changes the model parameters)
        grad_norm = self._optimization_step()
        
        # Calculate average loss
        avg_loss = accumulated_loss
        accumulated_loss = 0.0

        self.trainer_status.global_step += 1

        # Log metrics
        current_lr = self.scheduler.get_last_lr()[0]
        metrics = {
            "train/loss": avg_loss,
            "train/learning_rate": current_lr,
            "train/grad_norm": grad_norm,
            f"train_{self.short_name()}/loss": avg_loss,
            f"train_{self.short_name()}/learning_rate": current_lr,
            f"train_{self.short_name()}/grad_norm": grad_norm,
            f"train_{self.short_name()}/num_trainable_groups": total_groups,
            f"train_{self.short_name()}/num_group_results": len(group_dataset),
            f"reward/total_mean": group_reward_mean,
            f"reward/total_std": group_reward_std,
        }
        for key, value in group_reward_items_mean.items():
            metrics[f"reward/item_{key}_mean"] = value
        for key, value in group_reward_items_std.items():
            metrics[f"reward/item_{key}_std"] = value
        for key, value in clip_metrics.items():
            metrics[f"clip/{key}"] = np.mean(value)
        self._log_metrics(metrics, self.trainer_status.global_step)
        
        # Save checkpoint
        if self.trainer_status.global_step % self.config.training.save_steps == 0:
            self._save_checkpoint(self.trainer_status.global_step, callback=callback)


    async def train_grpo_block(self, block: TrainerGRPOBlock, callback: Optional[Callable] = None):
        """Train the model for one block"""
        # Create data loader
        context_vars = self.configInterpreter.prepare_context_vars(
            runtime=self,
            context_config=self.module_config.get('context_vars', {}),
            context_vars=self.context_vars | {
                "block": block
            },
        )
        success = await self.configInterpreter.execute(
            runtime=self,
            config=self.module_config.get('input_processor', {}),
            context_vars=context_vars,
        )
        if not success:
            logger.error(f"❌ [GRPOTrainer] [{block.input_tag}] Failed to execute module config: {self.module_config.get('processor', {}).get('grpo_trainer', {})}")
            return

        group_datasets = context_vars.get('__result__', {})

        logger.info(f"👉 [{self.__class__.__name__}] [{block.input_tag}] Block started with [{len(group_datasets)}] groups, Initial global step: [{self.trainer_status.global_step}]")

        start_time = time.time()
        # Create progress bar
        progress_bar = tqdm(
            total=len(group_datasets),
            desc=block.input_tag,
            initial=0
        )

        for group_tag, group_dataset in group_datasets.items():
            self.train_group(group_dataset, callback=callback, total_groups=len(group_datasets))
            # Update step counter
            progress_bar.update(1)
                
            # Check if training is complete
            if self.trainer_status.global_step >= self.config.training.max_steps:
                break

        total_time = time.time() - start_time
        logger.info(f"🎉 [{self.__class__.__name__}] [{block.input_tag}] Block completed in [{total_time:.1f}s] - Final global step: [{self.trainer_status.global_step}]")
        progress_bar.close()

def grpo_get_trainer(base_trainer: BaseTrainer, prefix_tag: str, base_config_file: str = "trainerBase.yaml", grpo_config_file: str = "trainerGRPO.yaml"):
    """Get a GRPO trainer"""
    try:
        base_config = TrainerConfig.from_yaml(base_config_file, override_yaml_path=grpo_config_file)
        logger.info(f"⚙️ [GRPOTrainer] [{prefix_tag}] Base configuration loaded from [{base_config_file}]")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] [{prefix_tag}] Failed to load base configuration: {e}")
        raise e

    try:
        grpo_config = GRPOConfig.from_yaml(grpo_config_file)
        logger.info(f"⚙️ [GRPOTrainer] [{prefix_tag}] GRPO configuration loaded from [{grpo_config_file}]")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] [{prefix_tag}] Failed to load GRPO configuration: {e}")
        raise e
    
    try:
        trainer = GRPOTrainer(prefix_tag, grpo_config, base_config, base_trainer=base_trainer)
        logger.info(f"⭐ [GRPOTrainer] [{prefix_tag}] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] [{prefix_tag}] Initialization failed: {e}")
        raise e
    
    return trainer

async def main():
    """Main function for GRPO training"""
    parser = argparse.ArgumentParser(description="Train a model using GRPOTrainer")
    parser.add_argument("--prefix_tag", type=str, default="TC_0.1.0_14B.m")
    parser.add_argument("--epoch_id", type=int, default=0)
    parser.add_argument("--block_id", type=int, default=0)
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_14B.m_005_06")
    parser.add_argument("--input_dir", type=str, default="~/.codeGenEval")
    parser.add_argument("--output_dir", type=str, default="~/.trainer/grpo")
    parser.add_argument("--base_config", type=str, default="trainerBase.yaml")
    parser.add_argument("--grpo_config", type=str, default="trainerGRPO.yaml")
    parser.add_argument("--module_file", type=str, default="trainer/grpo.module.yaml")
    args = parser.parse_args()

    trainer = grpo_get_trainer(None, args.prefix_tag, args.base_config, args.grpo_config)
    grpo_block = TrainerGRPOBlock(
        prefix_tag=args.prefix_tag,
        epoch_id=args.epoch_id,
        block_id=args.block_id,
        input_tag=args.input_tag,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    grpo_block.input_dir = os.path.expanduser(grpo_block.input_dir)
    grpo_block.output_dir = os.path.expanduser(grpo_block.output_dir)

    await trainer.train_grpo_block(grpo_block)

if __name__ == "__main__":
    asyncio.run(main())
