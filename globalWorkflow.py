import yaml
import asyncio
import argparse
from datetime import datetime
from logger import logger
from globalRegClient import GlobalRegClient
from globalUtils import CodeGenEvalBlock, CritiqueBlock, ExemplarBlock, ReflectionBlock, TrainerSFTBlock, TrainerRFTBlock, TrainerGRPOBlock, get_prefix_tag, ComposerBlock

TASK_TYPE_COMPOSER = "inference.composer"
TASK_TYPE_GRPO = "trainer.grpo"
TASK_TYPE_SFT = "trainer.sft"
TASK_TYPE_RFT = "trainer.rft"

VALID_TASK_TYPES = [
    TASK_TYPE_COMPOSER,
    TASK_TYPE_GRPO,
    TASK_TYPE_SFT,
    TASK_TYPE_RFT,
]

class GlobalWorkflow:
    def __init__(self, prefix_tag: str, config_path: str = "globalWorkflow.yaml"):
        self.config = self.from_yaml(config_path)
        self.prefix_tag = get_prefix_tag(prefix_tag, config_path)
        self.global_config = self.config.get("global", {})
        self.global_reg_client = GlobalRegClient()

    def from_yaml(self, config_path):
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        return yaml_data

    def _get_task_default(self, task_type: str, task_name: str):
        return self.config.get(task_type, {}).get(task_name, {}).get("default", {})
    
    async def init_vars(self):
        init_vars = self.global_config.get("init_vars", []) # list of dicts
        for init_var in init_vars:
            await self.global_reg_client.put(init_var.get("name"), init_var.get("value"))
            logger.info(f"🔢 [GlobalWorkflow] [{self.prefix_tag}] Initialized [{init_var.get('name')}] = [{init_var.get('value')}]")

    async def init_tasks(self, start_epoch: int = 0, start_block: int = 0):
        global_config = self.config.get("global", {})
        num_epochs = global_config.get("num_epochs", 100)
        num_blocks_per_epoch = global_config.get("num_blocks_per_epoch", 16)

        init_tasks = global_config.get("init_tasks", [])

        for epoch_id in range(start_epoch, num_epochs):
            for block_id in range(start_block, num_blocks_per_epoch):
                # iterate through tasks and blocks
                env_vars = {
                    "prefix_tag": self.prefix_tag,
                    "epoch_id": epoch_id,
                    "block_id": block_id,
                }
                for init_task in init_tasks:
                    task_name = init_task.get("name", None)
                    if task_name is None:
                        raise ValueError(f"Task [{init_task}] has no name")
                    task_type = init_task.get("type", None)
                    if task_type is None or task_type not in VALID_TASK_TYPES:
                        raise ValueError(f"Task [{task_name}] has invalid task type: [{task_type}]")
                    # clone task_config
                    queue_name = f"{task_type}:{task_name}"
                    task_config = init_task.copy()
                    # evaluate everything in the task_config
                    for key, value in task_config.items():
                        if isinstance(value, str):
                            # format string with ${key}
                            task_config[key] = value.format(**env_vars)
                        else:
                            task_config[key] = value
                    # if task is composer, then we need to create a composerBlock
                    if task_type == TASK_TYPE_COMPOSER:
                        # use self.composer_config as default
                        composerBlock = ComposerBlock(**(self._get_task_default(task_type, task_name) | task_config))
                        await self.global_reg_client.enqueue(queue_name, composerBlock.model_dump(), create_queue=True)
                        logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued [{queue_name}] [{composerBlock.epoch_id:03d}_{composerBlock.block_id:02d}], content: [{composerBlock.model_dump()}]")
                    else:
                        raise ValueError(f"Task [{task_name}] has unknown task type: [{task_type}]")

    async def _enqueue_post_task(self, task_type: str, task_name: str, task_config: dict, env_vars: dict):
        # clone task_config
        task_config = task_config.copy()
        task_type = task_config.get("type", None)
        if task_type is None or task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Task [{task_name}] has invalid task type: [{task_type}]")
        # evaluate everything in the task_config
        for key, value in task_config.items():
            if isinstance(value, str):
                task_config[key] = value.format(**env_vars)
        # enqueue the task
        queue_name = f"{task_type}:{task_name}"
        if task_type == TASK_TYPE_COMPOSER:
            composerBlock = ComposerBlock(**(self._get_task_default(task_type, task_name) | task_config))
            await self.global_reg_client.enqueue(queue_name, composerBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}:{task_name}], content: [{composerBlock.model_dump()}]")
        elif task_type == TASK_TYPE_SFT:
            trainerSFTBlock = TrainerSFTBlock(**(self._get_task_default(task_type, task_name) | task_config))
            await self.global_reg_client.enqueue(queue_name, trainerSFTBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}:{task_name}], content: [{trainerSFTBlock.model_dump()}]")
        elif task_type == TASK_TYPE_RFT:
            trainerRFTBlock = TrainerRFTBlock(**(self._get_task_default(task_type, task_name) | task_config))
            await self.global_reg_client.enqueue(queue_name, trainerRFTBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}:{task_name}], content: [{trainerRFTBlock.model_dump()}]")
        elif task_type == TASK_TYPE_GRPO:
            trainerGRPOBlock = TrainerGRPOBlock(**(self._get_task_default(task_type, task_name) | task_config))
            await self.global_reg_client.enqueue(queue_name, trainerGRPOBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}:{task_name}], content: [{trainerGRPOBlock.model_dump()}]")
        else:
            raise ValueError(f"Task [{task_name}] has unknown task type: [{task_type}]")

    def _get_run_tag(self, epoch_id: int, block_id: int):
        return f"{self.prefix_tag}_{epoch_id:03d}_{block_id:02d}"

    async def post_composer(self, task_name: str, composerBlock: ComposerBlock):
        logger.info(f"⏳ [GlobalWorkflow] [{self.prefix_tag}] Posting composer block: [{composerBlock.model_dump()}]")
        env_vars = {
            "prefix_tag": self.prefix_tag,
            "epoch_id": composerBlock.epoch_id,
            "block_id": composerBlock.block_id,
            "run_tag": self._get_run_tag(composerBlock.epoch_id, composerBlock.block_id),
        }
        post_tasks = self.config.get(TASK_TYPE_COMPOSER, {}).get(task_name, {}).get("post_tasks", [])
        for post_task in post_tasks:
            task_type = post_task.get("type", None)
            task_name = post_task.get("name", None)
            if task_type is None or task_name is None:
                raise ValueError(f"Task [{post_task}] has no name or type")
            if task_type not in VALID_TASK_TYPES:
                raise ValueError(f"Task [{task_name}] has invalid task type: [{task_type}]")
            await self._enqueue_post_task(task_type, task_name, post_task, env_vars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="auto")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--start_block", type=int, default=0)
    args = parser.parse_args()

    globalWorkflow = GlobalWorkflow(prefix_tag=args.prefix_tag)
    asyncio.run(globalWorkflow.init_vars())
    asyncio.run(globalWorkflow.init_tasks(start_epoch=args.start_epoch, start_block=args.start_block))
