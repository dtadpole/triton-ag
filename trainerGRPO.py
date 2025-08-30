import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import yaml
import asyncio
import argparse
import json
from pydantic import BaseModel
from engineBase import EngineBase, EngineConfig
from trainerUtil import SimpleCollator, make_checkpoint_callback
from logger import logger
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from configEndpoints import DuckDBClient
from configInterpreter import ConfigInterpreter
from workflowUtil import TrainerBlock
from workflowSync import WorkflowSync

LATEST_REFERENCE_NAME = "reference_state_latest.pt"

CLIP_RATIO_LOWER_PERCENTAGE = "clip_ratio_lower_pct"
CLIP_RATIO_UPPER_PERCENTAGE = "clip_ratio_upper_pct"
BOUND_ADVANTAGE_LOWER_PERCENTAGE = "bound_adv_lower_pct"
BOUND_ADVANTAGE_UPPER_PERCENTAGE = "bound_adv_upper_pct"
IS_RATIO_TRUNCATED_PERCENTAGE = "is_ratio_truncated_pct"
LOG_PROB_AVERAGE_VALUE = "log_prob_avg_value"
LOG_PROB_AVERAGE_RATIO = "log_prob_avg_ratio"
LOG_PROB_FORWARD_GENERATION_DIFF = "log_prob_forward_generation_diff"
LOG_PROB_GENERATION_VLLM_DIFF = "log_prob_generation_vllm_diff"
LOG_PROB_FORWARD_VLLM_DIFF = "log_prob_forward_vllm_diff"

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
    # loss_type: str = "token" # "episode" or "token" or "seq_max" or "gspo"
    loss_type: str = "gspo" # "episode" or "token" or "seq_max" or "gspo"
    gamma: float = 0.5
    use_truncated_is: bool = False
    truncated_is_ratio: float = 2.0

    @classmethod
    def from_yaml(cls, file_path: str) -> "GRPOConfig":
        """Load GRPO configuration from YAML file"""
        with open(os.path.expanduser(file_path), 'r') as f:
            config = yaml.safe_load(f)
        grpo_config = config.get('grpo', {})
        return cls(**grpo_config)


class GRPOTrainer():
    """GRPO (Generalized Preference Optimization) trainer for preference learning"""

    def __init__(self,
        engine: EngineBase,
        prefix_tag: str,
        grpo_config: GRPOConfig,
        module_file: str = "trainer/grpo.module.yaml",
    ):
        """Initialize GRPO trainer"""
        self.engine = engine
        self.tokenizer = self.engine.tokenizer
        self.status = self.engine.status
        self.configInterpreter = ConfigInterpreter()
        self.duckdbClient = DuckDBClient()
        self.grpo_config = grpo_config
        self.module_file = module_file
        self.module_config = yaml.safe_load(open(module_file, 'r')).get('module', {})
        # self.lora_cache = TrainerLoraCache(
        #     prefix_tag,
        #     self.engine.model,
        #     self.engine.model,
        #     extra_cache_size=self.engine.config.lora.extra_cache_size,
        #     cache_dir=self.engine.config.training.checkpoint_path,
        # )
        self.context_vars = {
            # built-in context vars
            "self": self,
            "os": os,
            "json": json,
            "yaml": yaml,
            "torch": torch,
            "gc": gc,
            "time": time,
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

    def _calculate_log_probs(
        self,
        logits: torch.Tensor,
        prompt_token_len: int,
        completion_token_ids: torch.Tensor,
    ):
        """Get log probabilities for the completion tokens"""
        output_completion_logits = logits[prompt_token_len-1:-1, :]
        log_probs = F.log_softmax(output_completion_logits, dim=-1) # dim: (completion_len, vocab_size)
        # get log probabilities for the completion tokens
        labels = torch.tensor(completion_token_ids, device=self.engine.device) \
            if isinstance(completion_token_ids, list) \
            else completion_token_ids.to(self.engine.device)
        action_log_probs = log_probs[:len(completion_token_ids), :].gather(
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (completion_len)
        return action_log_probs

    '''
    def _calculate_override_log_probs(self,
                                      input_ids: torch.Tensor,
                                      attention_mask: torch.Tensor,
                                      prompt_token_len: int,
                                      completion_token_ids: torch.Tensor,
                                      checkpoint_name: str,
                                      ):
        """Get log probabilities for the completion tokens using old model"""
        # gc.collect()
        # torch.cuda.empty_cache()
        checkpoint_model = self.lora_cache.load_checkpoint(checkpoint_name)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        # print('input_ids.shape: ', input_ids.shape, input_ids.dtype, input_ids.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        # print('attention_mask.shape: ', attention_mask.shape, attention_mask.dtype, attention_mask.device)
        checkpoint_model.eval()
        with torch.inference_mode():
            model_outputs = checkpoint_model(input_ids=input_ids, attention_mask=attention_mask)
            # print('model_outputs.logits.shape: ', model_outputs.logits.shape)
        action_log_probs = self._calculate_log_probs(
            model_outputs.logits[0],
            prompt_token_len,
            completion_token_ids,
        )
        return action_log_probs
    '''

    def _compute_mini_batch_loss(
        self,
        batch: Dict[str, Any],
        clip_metrics: Dict[str, List[float]],
        max_tokens_in_group: int = 0,
        total_tokens_in_group: int = 0,
    ):
        """Compute loss for the generated tokens"""
        # turn_tags = batch['turn_tag']
        # rewards = batch['reward']
        advantages = batch['advantage']
        prompt_token_ids = batch['logp_server_prompt_ids']
        completion_token_ids = batch['logp_server_completion_ids']
        generation_log_probs = batch['logp_server_logps']
        vllm_completion_log_probs = batch['vllm_completion_log_probs']
        input_ids = batch['input_ids'].to(self.engine.device)
        attention_mask = batch['attention_mask'].to(self.engine.device)

        model_outputs = self.engine.model(input_ids=input_ids, attention_mask=attention_mask)

        batch_loss = 0.0
        for i in range(len(advantages)): # for each generation result in the batch
            # get the logits for the completion tokens only, remove the prompt tokens
            prompt_token_len = len(prompt_token_ids[i])
            forward_completion_log_probs = self._calculate_log_probs(
                model_outputs.logits[i],
                prompt_token_len,
                completion_token_ids[i],
            )

            # if 'checkpoint_name' in batch:
            #     completion_log_probs_override = self._calculate_override_log_probs(
            #         input_ids[i],
            #         attention_mask[i],
            #         prompt_token_len,
            #         completion_token_ids[i],
            #         checkpoint_names[i],
            #     )
            #     completion_log_probs_tensor = completion_log_probs_override
            #     log_prob_override_mse = torch.mean(torch.abs(torch.tensor(completion_log_probs[i], device=self.device) - completion_log_probs_tensor))
            # else:
            generation_completion_log_probs = torch.tensor(generation_log_probs[i][len(prompt_token_ids[i])-1:len(prompt_token_ids[i])-1+len(completion_token_ids[i])], device=self.engine.device)
            vllm_completion_log_probs_i = torch.tensor(vllm_completion_log_probs[i], device=self.engine.device)
            # calculate log ratio (in log space is subtraction)
            if generation_completion_log_probs.shape != forward_completion_log_probs.shape:
                raise ValueError(f"❌ [GRPOTrainer] Generation completion log probabilities and forward completion log probabilities have different shapes: {generation_completion_log_probs.shape} != {forward_completion_log_probs.shape}")
            if vllm_completion_log_probs_i.shape != generation_completion_log_probs.shape:
                raise ValueError(f"❌ [GRPOTrainer] VLLM completion log probabilities and generation completion log probabilities have different shapes: {vllm_completion_log_probs_i.shape} != {generation_completion_log_probs.shape}")

            # log_prob_override_mse is always zero here
            log_prob_forward_generation_diff = torch.mean(torch.abs(forward_completion_log_probs - generation_completion_log_probs))
            log_prob_generation_vllm_diff = torch.mean(torch.abs(generation_completion_log_probs - vllm_completion_log_probs_i))
            log_prob_forward_vllm_diff = torch.mean(torch.abs(forward_completion_log_probs - vllm_completion_log_probs_i))

            # log probs is calculated in log space, so we need to subtract the log probabilities
            raw_log_ratio = forward_completion_log_probs - generation_completion_log_probs

            clip_metrics[LOG_PROB_AVERAGE_VALUE].append(torch.mean(forward_completion_log_probs).item())
            clip_metrics[LOG_PROB_FORWARD_GENERATION_DIFF].append(log_prob_forward_generation_diff.item())
            clip_metrics[LOG_PROB_GENERATION_VLLM_DIFF].append(log_prob_generation_vllm_diff.item())
            clip_metrics[LOG_PROB_FORWARD_VLLM_DIFF].append(log_prob_forward_vllm_diff.item())

            if self.grpo_config.loss_type == "gspo":

                if self.grpo_config.use_truncated_is:
                    is_ratio = torch.exp(generation_completion_log_probs - vllm_completion_log_probs_i).detach()
                    clip_metrics[IS_RATIO_TRUNCATED_PERCENTAGE].append(torch.sum(is_ratio > self.grpo_config.truncated_is_ratio).item() * 100.0 / len(forward_completion_log_probs))
                    truncated_is_ratio = torch.clamp(is_ratio, max=self.grpo_config.truncated_is_ratio)
                    log_ratio = truncated_is_ratio * raw_log_ratio
                else:
                    log_ratio = raw_log_ratio

                sequence_log_ratio = torch.sum(log_ratio) / len(forward_completion_log_probs)
                sequence_ratio = torch.exp(sequence_log_ratio)

                clip_metrics[LOG_PROB_AVERAGE_RATIO].append(sequence_ratio.item())

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
                ratio = torch.exp(raw_log_ratio)

                clip_metrics[LOG_PROB_AVERAGE_RATIO].append(ratio.mean().item())

                # calculate clipped upper and lower percentage
                clip_metrics[CLIP_RATIO_UPPER_PERCENTAGE].append(torch.sum(ratio > 1+self.grpo_config.clip_ratio_epsilon_upper).item() * 100.0 / len(forward_completion_log_probs))
                clip_metrics[CLIP_RATIO_LOWER_PERCENTAGE].append(torch.sum(ratio < 1-self.grpo_config.clip_ratio_epsilon_lower).item() * 100.0 / len(forward_completion_log_probs))

                clamped_ratio = torch.clamp(ratio, 1-self.grpo_config.clip_ratio_epsilon_lower, 1+self.grpo_config.clip_ratio_epsilon_upper)

                # min ratio advantage is min of ratio_advantage and clamped_ratio_advantage
                ratio_advantage = torch.min(ratio * advantages[i], clamped_ratio * advantages[i]) # dim: (completion_len)

                # calculate clipped upper and lower percentage
                clip_metrics[BOUND_ADVANTAGE_UPPER_PERCENTAGE].append(torch.sum(ratio_advantage > self.grpo_config.bound_advantage_range).item() * 100.0 / len(forward_completion_log_probs))
                clip_metrics[BOUND_ADVANTAGE_LOWER_PERCENTAGE].append(torch.sum(ratio_advantage < -self.grpo_config.bound_advantage_range).item() * 100.0 / len(forward_completion_log_probs))

                # calculate clipped upper and lower percentage
                bounded_ratio_advantage = torch.clamp(ratio_advantage, -self.grpo_config.bound_advantage_range, self.grpo_config.bound_advantage_range)

                if self.grpo_config.use_truncated_is:
                    is_ratio = torch.exp(generation_completion_log_probs - vllm_completion_log_probs_i).detach()
                    clip_metrics[IS_RATIO_TRUNCATED_PERCENTAGE].append(torch.sum(is_ratio > self.grpo_config.truncated_is_ratio).item() * 100.0 / len(forward_completion_log_probs))
                    truncated_is_ratio = torch.clamp(is_ratio, max=self.grpo_config.truncated_is_ratio)
                    final_ratio_advantage = truncated_is_ratio * bounded_ratio_advantage
                else:
                    final_ratio_advantage = bounded_ratio_advantage

                # compute loss
                if self.grpo_config.loss_type == "episode":
                    loss = -final_ratio_advantage.mean()
                elif self.grpo_config.loss_type == "token":
                    loss = -torch.sum(final_ratio_advantage) / total_tokens_in_group
                elif self.grpo_config.loss_type == "group_max":
                    loss = -torch.sum(final_ratio_advantage) / max_tokens_in_group
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
            batch_size=self.engine.config.training.micro_batch_size,
            shuffle=True,
            num_workers=self.engine.config.training.dataloader_num_workers,
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

        max_tokens_in_group = max([len(result["vllm_completion_ids"]) for result in group_dataset])
        total_tokens_in_group = sum([len(result["vllm_completion_ids"]) for result in group_dataset])

        self.engine.model.train()
        clip_metrics = {
            CLIP_RATIO_UPPER_PERCENTAGE: [],
            CLIP_RATIO_LOWER_PERCENTAGE: [],
            BOUND_ADVANTAGE_UPPER_PERCENTAGE: [],
            BOUND_ADVANTAGE_LOWER_PERCENTAGE: [],
            IS_RATIO_TRUNCATED_PERCENTAGE: [],
            LOG_PROB_AVERAGE_VALUE: [],
            LOG_PROB_AVERAGE_RATIO: [],
            LOG_PROB_FORWARD_GENERATION_DIFF: [],
            LOG_PROB_GENERATION_VLLM_DIFF: [],
            LOG_PROB_FORWARD_VLLM_DIFF: [],
        }
        for batch_idx, batch in enumerate(dataloader):
            # Training step
            mini_batch_loss = self._compute_mini_batch_loss(
                batch,
                clip_metrics,
                max_tokens_in_group=max_tokens_in_group,
                total_tokens_in_group=total_tokens_in_group,
            )

            # Scale loss for gradient accumulation
            if self.grpo_config.loss_type != "token": # for token level loss, the loss has already been scaled by the group size
                mini_batch_loss = mini_batch_loss * self.engine.config.training.loss_multiplier / len(dataloader) # divide by the group size
            # run backward step
            self.engine._backward_step(mini_batch_loss)

            # del outputs
            torch.cuda.empty_cache()

            accumulated_loss += mini_batch_loss.item()

        # Optimization step (only after entire group is processed, this changes the model parameters)
        grad_norm = self.engine._optimization_step()

        # Calculate average loss
        avg_loss = accumulated_loss
        accumulated_loss = 0.0

        self.engine.status.global_step += 1

        # Log metrics
        current_lr = self.engine._get_current_lr()
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
        self.engine._log_metrics(metrics, self.engine.status.global_step)

        # Save checkpoint
        if self.engine.status.global_step % self.engine.config.training.save_steps == 0:
            self.engine._save_checkpoint(self.engine.status.global_step, callback=callback)


    async def train_grpo_block(self, block: TrainerBlock, callback: Optional[Callable] = None):
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

        logger.info(f"👉 [{self.__class__.__name__}] [{block.input_tag}] Block started with [{len(group_datasets)}] groups, Initial global step: [{self.engine.status.global_step}]")

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
            await asyncio.sleep(0.1)

            # Check if training is complete
            if self.engine.status.global_step >= self.engine.config.training.max_steps:
                break

        try:
            # always save checkpoint at the end of the block
            self.engine._save_checkpoint(self.engine.status.global_step)
        except Exception as e:
            logger.error(f"❌ [GRPOTrainer] [{block.input_tag}] Failed to save checkpoint: {e}")

        total_time = time.time() - start_time
        logger.info(f"🎉 [{self.__class__.__name__}] [{block.input_tag}] Block completed in [{total_time:.1f}s] - Final global step: [{self.engine.status.global_step}]")
        progress_bar.close()
        await asyncio.sleep(0.1)


def grpo_get_trainer(
    engine: EngineBase,
    prefix_tag: str,
    engine_config_file: str = "engineBase.yaml",
    grpo_config_file: str = "trainerGRPO.yaml",
    module_file: str = "trainer/grpo.module.yaml",
):
    """Get a GRPO trainer"""
    try:
        engine._update_config(EngineConfig.from_yaml(engine_config_file, override_yaml_path=grpo_config_file))
        logger.info(f"⚙️ [GRPOTrainer] [{prefix_tag}] Engine config [{engine.__class__.__name__}] loaded from [{engine_config_file}]")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] [{prefix_tag}] Engine config [{engine.__class__.__name__}] failed to load: [{type(e)}] {e}")
        raise e

    try:
        grpo_config = GRPOConfig.from_yaml(grpo_config_file)
        logger.info(f"⚙️ [GRPOTrainer] [{prefix_tag}] GRPO configuration loaded from [{grpo_config_file}]")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] [{prefix_tag}] Failed to load GRPO configuration: {e}")
        raise e

    try:
        trainer = GRPOTrainer(engine=engine, prefix_tag=prefix_tag, grpo_config=grpo_config, module_file=module_file)
        logger.info(f"⭐ [GRPOTrainer] [{prefix_tag}] Trainer initialized")
    except Exception as e:
        logger.error(f"❌ [GRPOTrainer] [{prefix_tag}] Initialization failed: {e}")
        raise e

    return trainer

async def main():
    """Main function for GRPO training"""
    parser = argparse.ArgumentParser(description="Train a model using GRPOTrainer")
    parser.add_argument("--queue_name", type=str, default="grpo.1")
    parser.add_argument("--engine", type=str, default="unsloth")
    parser.add_argument("--engine_config", type=str, default="engineBase.yaml")
    parser.add_argument("--prefix_tag", type=str, default="auto.trainer.grpo")
    parser.add_argument("--epoch_id", type=int, default=0)
    parser.add_argument("--block_id", type=int, default=0)
    parser.add_argument("--input_tag", type=str, default="TC_0.1.0_14B.n_004_01")
    parser.add_argument("--input_dir", type=str, default=INFERENCE_DIR + "/codeGenEval")
    parser.add_argument("--output_dir", type=str, default=TRAINER_DIR + "/grpo")
    parser.add_argument("--grpo_config", type=str, default="trainerGRPO.yaml")
    parser.add_argument("--module_file", type=str, default="trainer/grpo.module.yaml")
    args = parser.parse_args()

    if args.engine == "unsloth":
        import unsloth

    engine_config = EngineConfig.from_yaml(args.engine_config)
    engine_config.model.engine = args.engine
    engine = EngineBase.create_engine(args.prefix_tag, engine_config) # no status for testing
    trainer = grpo_get_trainer(engine, args.prefix_tag, args.engine_config, args.grpo_config)
    grpo_block = TrainerBlock(
        queue_name=args.queue_name,
        prefix_tag=args.prefix_tag,
        epoch_id=args.epoch_id,
        block_id=args.block_id,
        input_tag=args.input_tag,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    grpo_block.input_dir = os.path.expanduser(grpo_block.input_dir)
    grpo_block.output_dir = os.path.expanduser(grpo_block.output_dir)

    callback_func = make_checkpoint_callback(
        prefix_tag=args.prefix_tag,
        trainer_block=grpo_block,
        workflow_provider="default",
    )

    await trainer.train_grpo_block(grpo_block, callback=callback_func)
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
