import copy
import os
import sys
from collections import defaultdict
import duckdb
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
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
from globalUtils import TrainerGRPOBlock
from globalRegClient import GlobalRegClient
from globalWorkflow import GlobalWorkflow

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
    bound_advantage_range: float = 3.0
    beta: float = 0.0  # KL divergence coefficient
    reference_model_update_steps: int = 100
    reward_scale: bool = True
    reward_epsilon: float = 1e-3
    reward_noise: float = 1e-2
    loss_type: str = "group_max" # "episode" or "token" or "seq_max" or "group_max"

    @classmethod
    def from_yaml(cls, file_path: str) -> "GRPOConfig":
        """Load GRPO configuration from YAML file"""
        with open(os.path.expanduser(file_path), 'r') as f:
            config = yaml.safe_load(f)
        grpo_config = config.get('grpo', {})
        return cls(**grpo_config)

class GenerationResult(BaseModel):
    """Result of a single generation"""
    turn_tag: str
    reward: float # total reward
    reward_items: Dict[str, float] = {} # individual reward components
    prompt_token_ids: List[int] = [] # prompt tokens
    completion_token_ids: List[int] = [] # completion tokens
    completion_log_probs: List[float] = [] # completion log probabilities

class GenerationResultGroup(BaseModel):
    """Set of generation results"""
    task_tag: str
    turn_id: int
    results: List[GenerationResult]
    metadata: Dict[str, Any] = {} # metadata

    def get_dataset(self, config: GRPOConfig):
        """Get a dataset for the result group"""
        advantages = self._compute_advantages(config)
        return [
            {
                'turn_tag': result.turn_tag,
                'reward': result.reward,
                'advantage': advantage,
                'prompt_token_ids': result.prompt_token_ids,
                'completion_token_ids': result.completion_token_ids,
                'completion_log_probs': result.completion_log_probs,
                'input_ids': result.prompt_token_ids + result.completion_token_ids,
                'attention_mask': torch.ones_like(torch.tensor(result.prompt_token_ids + result.completion_token_ids)),
            }
            for result, advantage in zip(self.results, advantages)
        ]

    def _compute_advantages(self, config: GRPOConfig):
        """Compute advantages for the generated tokens"""
        # calculate mean and stdev of the rewards
        rewards = [result.reward for result in self.results]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        # whether to scale the rewards
        if config.reward_scale:
            advantages = [(r - mean_reward) / (std_reward + config.reward_epsilon) for r in rewards]
        else:
            advantages = [r - mean_reward for r in rewards]
        # add noise to the advantages
        advantages = [a + np.random.normal(0, config.reward_noise) for a in advantages]
        # return the advantages
        return advantages

class GenerationDataset(Dataset):
    """Dataset for generation results"""
    def __init__(self, result_groups: List[GenerationResultGroup]):
        self.result_groups = result_groups

    def __len__(self):
        return len(self.result_groups)

    def __getitem__(self, idx):
        return self.result_groups[idx]


class GRPOTrainer(BaseTrainer):
    """GRPO (Generalized Preference Optimization) trainer for preference learning"""

    def __init__(self, prefix_tag: str, grpo_config: GRPOConfig, base_config: TrainerConfig, status: Optional[TrainerStatus] = None, base_trainer: BaseTrainer = None):
        """Initialize GRPO trainer"""
        super().__init__(prefix_tag, base_config, status, base_trainer)
        self.grpo_config = grpo_config
        self.reference_model = None

        logger.info(f"⭐ [GRPOTrainer] Initialized with GRPOConfig: {grpo_config}")

    def short_name(self):
        return 'grpo'

    def _update_grpo_config(self, grpo_config: GRPOConfig):
        """Update GRPO config"""
        self.grpo_config = grpo_config

    def compute_ref_log_probs(self, batch: Dict[str, Any]):
        """Compute log probabilities for the generated tokens"""
        if self.reference_model is None:
            raise ValueError("The reference_model is None, can't compute log probs for reference model")
        with torch.no_grad():
            prompt_token_ids = batch['prompt_token_ids']
            outputs = self.reference_model(batch['input_ids'].to(self.reference_model.device),
                                           batch['attention_mask'].to(self.reference_model.device))
            logits = outputs.logits[:, len(prompt_token_ids)-1:-1, :]
            log_probs = F.log_softmax(logits, dim=-1).to(self.model.device)

            return log_probs # dim: (batch_size, seq_len, vocab_size)

    def _deepcopy_reference_model(self, device=None):
        self.reference_model = copy.deepcopy(self.model)
        if device is not None:
            self.reference_model.to(device)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def _update_reference_model(self):
        logger.info(f"🔄 [GRPOTrainer] Updating reference model (LoRA mode)")
        if self.reference_model is None:
            logger.info("The reference model is None, no need to update state")
            return

        if not self.config.lora.use_lora:
            # Full model case (your existing code handles this)
            self.reference_model.load_state_dict(self.model.state_dict())
        else:
            # Get current LoRA adapter state
            current_lora_state = get_peft_model_state_dict(self.model)

            # Set the reference model to use the same adapter weights
            set_peft_model_state_dict(self.reference_model, current_lora_state)

            logger.info(f"✅ [GRPOTrainer] LoRA adapter weights copied to reference model")
            logger.info(f"📊 [GRPOTrainer] Updated {len(current_lora_state)} adapter parameters")

        # Ensure reference model is in eval mode and frozen
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        logger.info(f"🔒 [GRPOTrainer] Reference model frozen and set to eval mode")

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

        # check that all the prompt_token_ids are the same
        # prompt_token_ids_0 = prompt_token_ids[0]
        # prompt_token_len = len(prompt_token_ids_0)
        # if len(prompt_token_ids) > 1:
        #     for i in range(1, len(prompt_token_ids)):
        #         if len(prompt_token_ids[i]) != prompt_token_len:
        #              raise ValueError(f"❌ [GRPOTrainer] Prompt token ids are not the same length: [idx.{i} != idx.{0}]")
        #         elif prompt_token_ids[i] != prompt_token_ids_0:
        #             raise ValueError(f"❌ [GRPOTrainer] Prompt token ids are not the same: [idx.{i} != idx.{0}]")

        # get logits from model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # add kl penalty
        ref_log_prob = None
        if self.grpo_config.beta > 0:
            if self.reference_model is None:
                logger.info("Making a deep copy of current model for reference")
                self._deepcopy_reference_model()
            with torch.no_grad():
                ref_log_prob = self.compute_ref_log_probs(batch)

        batch_loss = 0.0
        total_kl_divergence = 0.0
        total_policy_loss = 0.0

        for i in range(len(advantages)): # for each generation result in the batch
            # get the logits for the completion tokens only, remove the prompt tokens
            prompt_token_len = len(prompt_token_ids[i])
            output_completion_logits = outputs.logits[i, prompt_token_len-1:-1, :]
            new_log_probs = F.log_softmax(output_completion_logits, dim=-1) # dim: (completion_len, vocab_size)
            # get log probabilities for the completion tokens
            labels = torch.tensor(completion_token_ids[i], device=self.device)
            completion_length = len(completion_token_ids[i])
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
                policy_loss = -final_ratio_advantage.mean()
            elif self.grpo_config.loss_type == "token":
                policy_loss = -torch.sum(final_ratio_advantage) / len(new_action_log_probs)
            elif self.grpo_config.loss_type == "group_max":
                policy_loss = -torch.sum(final_ratio_advantage) / group_max_length
            elif self.grpo_config.loss_type == "seq_max":
                policy_loss = -torch.sum(final_ratio_advantage) / self.grpo_config.max_seq_length
            else:
                raise ValueError(f"❌ [GRPOTrainingGroup] Invalid loss type: {self.grpo_config.loss_type}")
            total_policy_loss += policy_loss

            kl_loss = 0.0
            if self.grpo_config.beta > 0 and ref_log_prob is not None:
                # Extract reference model log probabilities for actual completion tokens
                ref_action_log_probs = ref_log_prob[i, :completion_length, :].gather(
                    dim=-1,
                    index=labels.unsqueeze(-1)
                ).squeeze(-1)  # Shape: (completion_length)
                # KL divergence: KL(π_θ || π_ref) = log π_θ(a|s) - log π_ref(a|s)
                token_kl_divergences = new_action_log_probs - ref_action_log_probs  # Shape: (completion_length)

                # Aggregate KL divergence based on loss type (same as policy loss)
                if self.grpo_config.loss_type == "episode":
                    kl_loss = self.grpo_config.beta * token_kl_divergences.mean()
                elif self.grpo_config.loss_type == "token":
                    kl_loss = self.grpo_config.beta * torch.sum(token_kl_divergences) / completion_length
                elif self.grpo_config.loss_type == "group_max":
                    kl_loss = self.grpo_config.beta * torch.sum(token_kl_divergences) / group_max_length
                elif self.grpo_config.loss_type == "seq_max":
                    kl_loss = self.grpo_config.beta * torch.sum(token_kl_divergences) / self.grpo_config.max_seq_length
                else:
                    raise ValueError(f"❌ [GRPOTrainingGroup] Invalid loss type: {self.grpo_config.loss_type}")

                total_kl_divergence += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            sample_loss = policy_loss + kl_loss
            batch_loss += sample_loss

        return batch_loss

    def train_block(self, run_tag: str, dataset: GenerationDataset, eval_dataset: Optional[GenerationDataset] = None, callback: Optional[Callable] = None):
        """Train the model for one block"""
        # Create data loader
        logger.info(f"👉 [{self.__class__.__name__}] [{run_tag}] Block started with [{len(dataset.result_groups)}] groups, Initial global step: [{self.trainer_status.global_step}]")

        start_time = time.time()
        # Create progress bar
        progress_bar = tqdm(
            total=len(dataset.result_groups),
            desc=run_tag,
            initial=0
        )

        for group in dataset.result_groups:
            group_dataset = group.get_dataset(self.grpo_config)
            dataloader = DataLoader(
                group_dataset,
                batch_size=self.config.training.micro_batch_size,
                shuffle=True,
                num_workers=self.config.training.dataloader_num_workers,
                pin_memory=True,
                collate_fn=SimpleCollator(tokenizer_pad_token_id=self.tokenizer.pad_token_id)
            )

            accumulated_loss = 0.0

            group_reward_mean = np.mean([result.reward for result in group.results])
            group_reward_std = np.std([result.reward for result in group.results])

            group_reward_items = {}
            for result in group.results:
                for key, value in result.reward_items.items():
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

            # Update step counter
            self.trainer_status.global_step += 1
            progress_bar.update(1)

            # Log metrics
            current_lr = self.scheduler.get_last_lr()[0]
            metrics = {
                "train/loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/grad_norm": grad_norm,
                f"train_{self.short_name()}/loss": avg_loss,
                f"train_{self.short_name()}/learning_rate": current_lr,
                f"train_{self.short_name()}/grad_norm": grad_norm,
                f"train_{self.short_name()}/num_trainable_groups": len(dataset.result_groups),
                f"train_{self.short_name()}/num_group_results": len(group.results),
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

            # Evaluation
            if eval_dataset and self.trainer_status.global_step % self.config.training.eval_steps == 0:
                self._evaluate(eval_dataset)

            # update reference model
            if self.reference_model is not None and self.trainer_status.global_step % self.grpo_config.reference_model_update_steps == 0:
                self._update_reference_model()

            # Check if training is complete
            if self.trainer_status.global_step >= self.config.training.max_steps:
                break

        total_time = time.time() - start_time
        logger.info(f"🎉 [{self.__class__.__name__}] [{run_tag}] Block completed in [{total_time:.1f}s] - Final global step: [{self.trainer_status.global_step}]")
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

def grpo_train_block(block: TrainerGRPOBlock, trainer: GRPOTrainer, callback: Optional[Callable] = None):
    """Train the model for one block"""
    logger.info(f"👉 [GRPOTrainer] [{block.input_tag}] GRPO Training started for block...")

    # Load dataset
    search_path = os.path.expanduser(f"{block.input_dir}/{block.input_tag}")
    if not os.path.exists(search_path):
        error_msg = f"❌ [GRPOTrainer] [{block.input_tag}] Input directory [{search_path}] does not exist"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Query conversation files
    result = duckdb.sql(f"""SELECT filename, compiled, correctness, metadata, runtime, runtime_stats
                        FROM read_json_auto('{search_path}/**/reference_eval.json', sample_size=-1, ignore_errors=true)
                    """)

    result_df = result.df()

    # for each row in the result_df, create a generation result group
    result_groups = []
    for index, row in result_df.iterrows():
        if not row['compiled'] or not row['correctness']:
            # this should not happen, but just in case
            logger.error(f"❌ [GRPOTrainer] [{block.input_tag}] Invalid Reference Evaluation - Skipping [{row['filename']}] Compiled: [{row['compiled']}] Correctness: [{row['correctness']}]")
            continue

        folder = os.path.dirname(row['filename'])
        ref_runtime = row['runtime']

        # get task tag from metadata
        if 'task_tag' not in row['metadata']:
            logger.warning(f"⚠️ [GRPOTrainer] [{block.input_tag}] Skipping [{row['filename']}] No task tag in metadata")
            continue
        task_tag = row['metadata']['task_tag']

        generation_results_by_turn = defaultdict(list)
        # read in all the gen_xx_completion.json files in the same folder (non-recursive)
        completion_files = [f for f in os.listdir(folder) if f.startswith('gen_') and f.endswith('_completion.json')]
        for completion_file in completion_files:
            with open(os.path.join(folder, completion_file), 'r') as f:
                completion_data = json.load(f)
            turn_tag = completion_file.replace('_completion.json', '')
            turn_id = int(turn_tag.split('_')[-1][1:]) # turn_id is the last number in the turn_tag, e.g. gen_01_t03 -> 3
            # read corresponding gen_xx_eval.json
            eval_file = completion_file.replace('_completion.json', '_eval.json')
            if not os.path.exists(os.path.join(folder, eval_file)):
                logger.warning(f"⚠️ [GRPOTrainer] [{block.input_tag}] No eval file found for [{completion_file}]")
                continue
            with open(os.path.join(folder, eval_file), 'r') as f:
                eval_data = json.load(f)
            # ok, now compile all the information together
            compiled = eval_data['compiled']
            correctness = eval_data['correctness']
            runtime = eval_data['runtime']
            # reward_compiled = 0.0 # do NOT use compiled reward
            reward_correctness = 0.3 if correctness else 0.0
            reward_speedup = 0.0 if runtime < 0 else ref_runtime / runtime
            reward = reward_correctness + reward_speedup
            # create a generation result group
            prompt = completion_data['prompt']
            prompt_token_ids = trainer.tokenizer.encode(prompt)
            # split completion_data['logprobs'] into a list of completion ids and logprobs
            completion_token_ids = [logprob['token_id'] for logprob in completion_data['logprobs']]
            completion_log_probs = [logprob['logprob'] for logprob in completion_data['logprobs']]
            # create the generation result object
            result = GenerationResult(
                turn_tag=turn_tag,
                reward=reward,
                reward_items={
                    "correctness": reward_correctness,
                    "speedup": reward_speedup,
                },
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                completion_log_probs=completion_log_probs,
            )
            generation_results_by_turn[turn_id].append(result)

        # iterate over the generation_results_by_turn and create a generation result group for each turn
        for turn_id, generation_results in generation_results_by_turn.items():
            # create a generation result group
            if len(generation_results) < 2:
                logger.warning(f"⚠️ [GRPOTrainer] [{block.input_tag}] Skipping [{folder}] that has only [{len(generation_results)}] results for turn [{turn_id}]")
                continue
            # check if all the reward are the same, if so, skip
            if all(result.reward == generation_results[0].reward for result in generation_results):
                logger.warning(f"⚠️ [GRPOTrainer] [{block.input_tag}] Skipping [{folder}] All rewards are the same: [{generation_results[0].reward}] for [{task_tag}] turn [{turn_id}]")
                continue
            result_group = GenerationResultGroup(task_tag=task_tag, turn_id=turn_id, results=generation_results)
            result_groups.append(result_group)

    # Create group dataset
    dataset = GenerationDataset(result_groups)
    if len(dataset) == 0:
        logger.warning(f"🗑️ [GRPOTrainer] [{block.input_tag}] No tasks for GRPO in [{search_path}]")
        return

    logger.info(f"📊 [GRPOTrainer] [{block.input_tag}] Dataset prepared - loaded [{len(dataset)}] groups")

    # Train the block
    try:
        trainer.train_block(block.input_tag, dataset, callback=callback)
        logger.info(f"🎯 [GRPOTrainer] [{block.input_tag}] Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] [{block.input_tag}] Training failed: {e}")
        traceback.print_exc()
        raise e

def grpo_get_sample_dataset(tokenizer: AutoTokenizer, size: int = 20) -> GenerationDataset:
    """Get a sample dataset for testing"""
    prompts = [
        "<|im_start|>user\nWrite a program to print 'Hello, World!'<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nShow me how to print 'Hello, All!' in python<|im_end|>\n<|im_start|>assistant\n"
    ]
    completions = [
        [
            """
            ```python
            print('Hello, World!')
            ```<|im_end|>
            """,
            """
            I need to print 'Hello, World!', I will use following code:
            ```python
            print('Hello,' + ' World!')
            ```<|im_end|>
            """,
            """
            I need to print 'Hello, World!', I will use following code to do it in python:
            ```python
            print(', '.join(['Hello', 'World!']))
            ```<|im_end|>
            """,
            """
            I will use the following code to print 'Hello, World!':
            ```python
            output = ', '.join(['Hello', 'World!'])
            print(output)
            ```<|im_end|>
            """,
        ],
        [
            """
            I will use the following code to print 'Hello, All!':
            ```python
            print('Hello, All!')
            ```<|im_end|>
            """,
            """
            Usually, you can print 'Hello, All!' in python by using the following code:
            ```python
            print('Hello,' + ' All!')
            ```<|im_end|>
            """,
            """
            I need to print 'Hello, All!' in python, I will use the following code:
            ```python
            print(', '.join(['Hello', 'All!']))
            ```<|im_end|>
            """,
            """
            I need to print 'Hello, All!', I will use the following code to do it in python:
            ```python
            output = ', '.join(['Hello', 'All!'])
            print(output)
            ```<|im_end|>
            """,
        ]
    ]
    rewards = [
        [1.0, 0.8, 0.5, 0.7],
        [0.5, 1.0, 0.7, 0.8],
    ]

    # create a generation dataset
    result_groups = []
    for prompt, completions, rewards in zip(prompts, completions, rewards):
        # create a result group
        result_group = GenerationResultGroup(task_tag="test", turn_id=0, results=[])
        id = 0
        for completion, reward in zip(completions, rewards):
            prompt_token_ids = tokenizer.encode(prompt)
            completion_token_ids = tokenizer.encode(completion)
            id += 1
            # create a random normal distribution of log probabilities
            completion_log_probs = np.random.normal(0.0, 0.1, len(completion_token_ids)).tolist()
            # generation result
            result = GenerationResult(
                turn_tag=f"gen_{id:02d}",
                reward=reward,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                completion_log_probs=completion_log_probs,
            )
            # add the result to the result group
            result_group.results.append(result)
        # add the result group to the generation dataset
        result_groups.append(result_group)

    repeated_groups = (result_groups * (size // len(result_groups) + 1))[:size]

    generation_dataset = GenerationDataset(repeated_groups)
    return generation_dataset

async def main():
    """Main function for GRPO training"""
    parser = argparse.ArgumentParser(description="Train a model using GRPOTrainer")
    parser.add_argument("--prefix_tag", type=str, default="TC_0.1.0_14B.test")
    parser.add_argument("--epoch_id", type=int, default=0)
    parser.add_argument("--block_id", type=int, default=0)
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_14B_20250801_233446")
    parser.add_argument("--input_dir", type=str, default="~/.codeGenEval")
    parser.add_argument("--output_dir", type=str, default="~/.trainer")
    parser.add_argument("--base_config", type=str, default="trainerBase.yaml")
    parser.add_argument("--grpo_config", type=str, default="trainerGRPO.yaml")
    parser.add_argument("--use_sample_dataset", action="store_true")
    args = parser.parse_args()

    if args.use_sample_dataset:
        trainer = grpo_get_trainer(None, args.prefix_tag, args.base_config, args.grpo_config)
        dataset = grpo_get_sample_dataset(trainer.tokenizer)
        logger.info(f"📊 [GRPOTrainer] Use Simple Mode - loaded [{len(dataset)}] groups")
        try:
            run_tag = f"{args.prefix_tag}_simple_mode"
            trainer.train_block(run_tag, dataset)
            logger.info("🎯 [GRPOTrainer] Training completed successfully!")
        except Exception as e:
            logger.error(f"❌ [GRPOTrainer] Training failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        trainer = grpo_get_trainer(None, args.prefix_tag, args.base_config, args.grpo_config)
        grpo_block = TrainerGRPOBlock(
            prefix_tag=args.prefix_tag,
            epoch_id=args.epoch_id,
            block_id=args.block_id,
            input_tag=args.input_tag,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        grpo_train_block(grpo_block, trainer)

if __name__ == "__main__":
    asyncio.run(main())
