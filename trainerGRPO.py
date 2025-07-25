import os
import sys
import duckdb
import unsloth
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Any, Tuple
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

LATEST_REFERENCE_NAME = "reference_state_latest.pt"

class GRPOConfig(BaseModel):
    """GRPO configuration"""
    max_seq_length: int = 16384
    mask_non_assistant_tokens: bool = True
    discard_long_conversations: bool = True
    clip_epsilon_lower: float = 0.2
    clip_epsilon_upper: float = 0.3
    beta: float = 0.0  # KL divergence coefficient
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
    gen_tag: str
    reward: float # total reward
    reward_items: Dict[str, float] = {} # individual reward components
    prompt_token_ids: List[int] = [] # prompt tokens
    completion_token_ids: List[int] = [] # completion tokens
    completion_log_probs: List[float] = [] # completion log probabilities

class GenerationResultGroup(BaseModel):
    """Set of generation results"""
    task_tag: str
    results: List[GenerationResult]
    metadata: Dict[str, Any] = {} # metadata

    def get_dataset(self, config: GRPOConfig):
        """Get a dataset for the result group"""
        advantages = self._compute_advantages(config)
        return [
            {
                'gen_tag': result.gen_tag,
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
    
    def __init__(self, prefix_tag: str, grpo_config: GRPOConfig, base_config: TrainerConfig, status: Optional[TrainerStatus] = None):
        """Initialize GRPO trainer"""
        super().__init__(prefix_tag, base_config, status)
        self.grpo_config = grpo_config
        self.reference_model = None

        self.reference_model = self.base_model
        if self.config.lora.use_lora:
            self.reference_model = self._setup_lora(self.reference_model)

        # Setup reference directory
        self.reference_path = Path(os.path.expanduser(self.config.training.checkpoint_path)) / self.prefix_tag / 'reference'
        self.reference_path.mkdir(parents=True, exist_ok=True)

        if self._reference_exists(self.reference_path):
            self.reference_model = self._load_reference(self.reference_path)

        logger.info(f"🎯 [GRPOTrainer] Initialized for preference optimization with GRPOConfig: {grpo_config}")

    def _reference_exists(self, reference_path: str) -> bool:
        """Check if reference exists"""
        reference_path_obj = Path(reference_path)
        
        # Check for reference_state.pt file
        if reference_path_obj.is_dir():
            return (reference_path_obj / LATEST_REFERENCE_NAME).exists()
        else:
            return reference_path_obj.exists()
    
    def _load_reference(self, reference_location: str):
        """Load reference for resuming training"""
        logger.info(f"🔄 [{self.__class__.__name__}] Loading reference from: {reference_location}")
        
        # Get the training state file path
        reference_path_obj = Path(reference_location)
        reference_state_path = reference_path_obj / LATEST_REFERENCE_NAME if reference_path_obj.is_dir() else reference_path_obj
        
        # Load reference checkpoint
        reference_state = torch.load(reference_state_path, map_location='cpu')
        
        # Load model state
        if self.config.lora.use_lora and 'lora_state_dict' in reference_state:
            set_peft_model_state_dict(self.reference_model, reference_state['lora_state_dict'])
        elif not self.config.lora.use_lora and 'model_state_dict' in reference_state:
            self.reference_model.load_state_dict(reference_state['model_state_dict'])
        else:
            logger.warning(f"⚠️ [{self.__class__.__name__}] Model state not found or incompatible in reference checkpoint")

        reference_global_step = reference_state.get('global_step', 0)
        reference_epoch_id = reference_state.get('epoch_id', 0)
        reference_block_id = reference_state.get('block_id', 0)
        
        logger.info(f"✅ [{self.__class__.__name__}] Reference loaded - Step: [{reference_global_step}], Epoch: [{reference_epoch_id}], Block: [{reference_block_id}]")

        return self.reference_model
    
    def _save_reference(self, global_step: int):
        """Save reference"""
        reference_path = self.checkpoint_path / f"reference-{global_step}"
        reference_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(reference_path)
        self.tokenizer.save_pretrained(reference_path)
        
        # Save reference state
        reference_state = {
            'reference_global_step': global_step,
            'reference_epoch_id': self.trainer_status.epoch_id,
            'reference_block_id': self.trainer_status.block_id,
            'config': self.config.model_dump(),
        }
        
        if self.config.lora.use_lora:
            reference_state['lora_state_dict'] = get_peft_model_state_dict(self.reference_model)
        else:
            reference_state['model_state_dict'] = self.reference_model.state_dict()
        
        reference_state_file = reference_path / f"reference_{global_step:04d}.pt"
        torch.save(reference_state, reference_state_file)
        
        # Save config
        with open(reference_path / f"reference_config_{global_step:04d}.yaml", 'w') as f:
            yaml.dump(self.config.model_dump(), f, default_flow_style=False)
        
        # Update latest reference link
        self._update_latest_reference_link(reference_state_file)
        
        logger.info(f"💾 [{self.__class__.__name__}] Reference saved: {reference_state_file}")
    
    def _update_latest_reference_link(self, reference_state_file: Path):
        """Update latest reference link"""
        latest_path = self.reference_path / LATEST_REFERENCE_NAME
        
        # Remove existing link/directory
        if latest_path.exists():
            if latest_path.is_symlink():
                latest_path.unlink()
            else:
                shutil.rmtree(latest_path)
        
        # Create symlink or copy
        try:
            latest_path.symlink_to(reference_state_file.name)
        except OSError:
            shutil.copy(reference_state_file, latest_path)

    def compute_ref_log_probs(self, batch: Dict[str, Any]):
        """Compute log probabilities for the generated tokens"""
        with torch.no_grad():
            outputs = self.reference_model(batch['input_ids'], batch['attention_mask'])
            logits = outputs.logits[0]
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs # dim: (batch_size, seq_len, vocab_size)

    def _compute_mini_batch_loss(self, batch: Dict[str, Any], group_max_length: Optional[int] = None):
        """Compute loss for the generated tokens"""
        # gen_tags = batch['gen_tag']
        # rewards = batch['reward']
        advantages = batch['advantage']
        prompt_token_ids = batch['prompt_token_ids']
        completion_token_ids = batch['completion_token_ids']
        completion_log_probs = batch['completion_log_probs']
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # check that all the prompt_token_ids are the same
        if len(prompt_token_ids) > 1:
            for i in range(1, len(prompt_token_ids)):
                if prompt_token_ids[i] != prompt_token_ids[0]:
                    raise ValueError(f"❌ [GRPOTrainer] Prompt token ids are not the same: [idx.{i} != idx.{0}]")
        
        # get logits from model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # get the logits for the completion tokens only, remove the prompt tokens
        output_completion_logits = outputs.logits[:, len(prompt_token_ids)-1:-1, :]
        new_log_probs = F.log_softmax(output_completion_logits, dim=-1) # dim: (batch_size, completion_len, vocab_size)

        batch_loss = 0.0
        for i in range(len(advantages)): # for each generation result in the batch
            # get log probabilities for the completion tokens
            labels = torch.tensor(completion_token_ids[i], device=self.device)
            new_action_log_probs = new_log_probs[i, :len(completion_token_ids[i]), :].gather(
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

            ratio_advantage = ratio * advantages[i]
            clamped_ratio_advantage = torch.clamp(ratio_advantage, 1-self.grpo_config.clip_epsilon_lower, 1+self.grpo_config.clip_epsilon_upper)
            # final ratio advantage is min of ratio_advantage and clamped_ratio_advantage
            final_ratio_advantage = torch.min(ratio_advantage, clamped_ratio_advantage) # dim: (completion_len)

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

    def train_block(self, run_tag: str, dataset: GenerationDataset, eval_dataset: Optional[GenerationDataset] = None):
        """Train the model for one block"""
        # Create data loader
        logger.info(f"🏋️ [{self.__class__.__name__}] [{run_tag}] Block started with [{len(dataset.result_groups)}] groups, Initial global step: [{self.trainer_status.global_step}]")

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
                collate_fn=SimpleCollator(tokenizer=self.tokenizer)
            )
            
            accumulated_loss = 0.0

            # group_max_length = max(len(result['input_ids']) for result in group_dataset)
            group_max_length = max(len(result['completion_token_ids']) for result in group_dataset)

            self.model.train()
            for batch_idx, batch in enumerate(dataloader):
                # Training step
                mini_batch_loss = self._compute_mini_batch_loss(batch, group_max_length)
                
                # Scale loss for gradient accumulation
                mini_batch_loss = mini_batch_loss / len(dataloader) # divide by the group size
                mini_batch_loss.backward()

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
            self._log_metrics(avg_loss, current_lr, self.trainer_status.global_step, grad_norm)
            
            # Save checkpoint
            if self.trainer_status.global_step % self.config.training.save_steps == 0:
                self._save_checkpoint(self.trainer_status.global_step)
            
            # Evaluation
            if eval_dataset and self.trainer_status.global_step % self.config.training.eval_steps == 0:
                self._evaluate(eval_dataset)
            
            # Check if training is complete
            if self.trainer_status.global_step >= self.config.training.max_steps:
                break

        total_time = time.time() - start_time
        logger.info(f"🎉 [{self.__class__.__name__}] [{run_tag}] Block completed in [{total_time:.1f}s] - Final global step: [{self.trainer_status.global_step}]")
        progress_bar.close()


def get_sample_dataset(tokenizer: AutoTokenizer, size: int = 20) -> GenerationDataset:
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
        result_group = GenerationResultGroup(task_tag="test", results=[])
        id = 0
        for completion, reward in zip(completions, rewards):
            prompt_token_ids = tokenizer.encode(prompt)
            completion_token_ids = tokenizer.encode(completion)
            id += 1
            # create a random normal distribution of log probabilities
            completion_log_probs = np.random.normal(0.0, 0.1, len(completion_token_ids)).tolist()
            # generation result
            result = GenerationResult(
                gen_tag=f"gen_{id:02d}",
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
    parser.add_argument("--simple_mode", type=bool, default=False)
    parser.add_argument("--input_dir", type=str, default="~/.codeGenEval")
    parser.add_argument("--run_tag", type=str, default="v0.1_20250724_223559")
    parser.add_argument("--base-config", type=str, default="trainerBase.yaml")
    parser.add_argument("--config", type=str, default="trainerGRPO.yaml")
    args = parser.parse_args()
    
    # Load configuration
    try:
        base_config = TrainerConfig.from_yaml(args.base_config)
        logger.info(f"✅ [GRPOTrainer] Base configuration loaded from {args.base_config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load base configuration: {e}")
        sys.exit(1)

    try:
        grpo_config = GRPOConfig.from_yaml(args.config)
        logger.info(f"✅ [GRPOTrainer] GRPO configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Failed to load GRPO configuration: {e}")
        sys.exit(1)
    
    try:
        trainer = GRPOTrainer(args.run_tag, grpo_config, base_config)
        logger.info("✅ [GRPOTrainer] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Initialization failed: {e}")
        sys.exit(1)
    
    if args.simple_mode:
        dataset = get_sample_dataset(trainer.tokenizer)
        logger.info(f"📊 [GRPOTrainer] Use Simple Mode - loaded [{len(dataset)}] groups")
    else:
        # Load dataset
        search_path = os.path.expanduser(f"{args.input_dir}/{args.run_tag}")
        if not os.path.exists(search_path):
            logger.error(f"❌ [GRPOTrainer] [{args.run_tag}] Input directory [{search_path}] does not exist")
            return

        # Query conversation files
        result = duckdb.sql(f"""SELECT filename, compiled, correctness, metadata, runtime, runtime_stats
                            FROM read_json_auto('{search_path}/**/reference_eval.json', sample_size=-1, ignore_errors=true) 
                        """)
        
        result_df = result.df()

        # for each row in the result_df, create a generation result group
        result_groups = []
        for index, row in result_df.iterrows():
            if not row['compiled'] or not row['correctness']:
                logger.warning(f"⚠️ [GRPOTrainer] Skipping [{row['filename']}] Compiled: [{row['compiled']}] Correctness: [{row['correctness']}]")
                continue

            folder = os.path.dirname(row['filename'])
            ref_runtime = row['runtime']

            # get task tag from metadata
            if 'task_tag' not in row['metadata']:
                logger.warning(f"⚠️ [GRPOTrainer] Skipping [{row['filename']}] No task tag in metadata")
                continue
            task_tag = row['metadata']['task_tag']

            generation_results = []
            prompt = None
            prompt_token_ids = []
            # read in all the gen_xx_completion.json files in the same folder (non-recursive)
            completion_files = [f for f in os.listdir(folder) if f.startswith('gen_') and f.endswith('_completion.json')]
            for completion_file in completion_files:
                with open(os.path.join(folder, completion_file), 'r') as f:
                    completion_data = json.load(f)
                gen_tag = completion_file.replace('_completion.json', '')
                # read corresponding gen_xx_eval.json
                eval_file = completion_file.replace('_completion.json', '_eval.json')
                if not os.path.exists(os.path.join(folder, eval_file)):
                    logger.warning(f"⚠️ [GRPOTrainer] No eval file found for [{completion_file}]")
                    continue
                with open(os.path.join(folder, eval_file), 'r') as f:
                    eval_data = json.load(f)
                # ok, now compile all the information together
                compiled = eval_data['compiled']
                correctness = eval_data['correctness']
                runtime = eval_data['runtime']
                reward_compiled = 0.0 if compiled else -0.5
                reward_correctness = 0.0 if correctness else -0.5
                reward_runtime = 0.0 if runtime < 0 else ref_runtime / runtime
                reward = reward_compiled + reward_correctness + reward_runtime
                # create a generation result group
                if prompt is None:
                    prompt = completion_data['prompt']
                    prompt_token_ids = trainer.tokenizer.encode(prompt)
                else:
                    if prompt != completion_data['prompt']:
                        logger.error(f"❌ [GRPOTrainer] Prompt mismatch for [{gen_tag}] - [{prompt}] != [{completion_data['prompt']}]")
                        # ignore this generation result and continue
                        continue
                # split completion_data['logprobs'] into a list of completion ids and logprobs
                completion_token_ids = [logprob['token_id'] for logprob in completion_data['logprobs']]
                completion_log_probs = [logprob['logprob'] for logprob in completion_data['logprobs']]
                # create the generation result object
                result = GenerationResult(
                    gen_tag=gen_tag,
                    reward=reward,
                    reward_items={
                        "compiled": reward_compiled,
                        "correctness": reward_correctness,
                        "runtime": reward_runtime,
                    },
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    completion_log_probs=completion_log_probs,
                )
                generation_results.append(result)

            # create a generation result group only if we have at least 2 results
            if len(generation_results) < 2:
                logger.warning(f"⚠️ [GRPOTrainer] Skipping [{folder}] that has only [{len(generation_results)}] results")
                continue
            # check if all the reward are the same, if so, skip
            if all(result.reward == generation_results[0].reward for result in generation_results):
                logger.warning(f"⚠️ [GRPOTrainer] Skipping [{folder}] All rewards are the same: [{generation_results[0].reward}]")
                continue
            # we are here because we have at least 2 results and the rewards are not the same
            # so we can create a generation result group
            result_group = GenerationResultGroup(task_tag=task_tag, results=generation_results)
            result_groups.append(result_group)


        # Create group dataset
        dataset = GenerationDataset(result_groups)
        logger.info(f"✅ [GRPOTrainer] Dataset prepared from [{args.run_tag}] - loaded [{len(dataset)}] groups")
    
    # Train the block
    try:
        trainer.train_block(args.run_tag, dataset)
        logger.info("🎉 [GRPOTrainer] Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
