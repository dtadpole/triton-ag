import yaml
import asyncio
import argparse
from logger import logger
from globalRegClient import GlobalRegClient
from globalUtils import CodeGenEvalBlock, CritiqueBlock, ExemplarBlock, ReflectionBlock, TrainerSFTBlock, TrainerRFTBlock, TrainerGRPOBlock

TASK_TYPE_CODEGENEVAL = "inference.codeGenEval"
TASK_TYPE_EXEMPLAR = "inference.exemplar"
TASK_TYPE_CRITIQUE = "inference.critique"
TASK_TYPE_GRPO = "trainer.grpo"
TASK_TYPE_SFT = "trainer.sft"
TASK_TYPE_RFT = "trainer.rft"

VALID_TASK_TYPES = [
    TASK_TYPE_CODEGENEVAL,
    TASK_TYPE_EXEMPLAR,
    TASK_TYPE_CRITIQUE,
    TASK_TYPE_GRPO,
    TASK_TYPE_SFT,
    TASK_TYPE_RFT,
]

class GlobalWorkflow:
    def __init__(self, prefix_tag: str, config_path: str = "globalWorkflow.yaml"):
        self.prefix_tag = prefix_tag
        self.config = self.from_yaml(config_path)
        self.global_config = self.config.get("global", {})
        self.codeGenEval_default = self.config.get(TASK_TYPE_CODEGENEVAL, {}).get("default", {})
        self.exemplar_default = self.config.get(TASK_TYPE_EXEMPLAR, {}).get("default", {})
        self.critique_default = self.config.get(TASK_TYPE_CRITIQUE, {}).get("default", {})
        self.trainerSFT_default = self.config.get(TASK_TYPE_SFT, {}).get("default", {})
        self.trainerRFT_default = self.config.get(TASK_TYPE_RFT, {}).get("default", {})
        self.trainerGRPO_default = self.config.get(TASK_TYPE_GRPO, {}).get("default", {})
        self.global_reg_client = GlobalRegClient()

    def from_yaml(self, config_path):
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        return yaml_data
    
    async def init_vars(self):
        init_vars = self.global_config.get("init_vars", {})
        for key, value in init_vars.items():
            await self.global_reg_client.put(key, value)

    async def init_tasks(self, start_epoch: int = 0, start_block: int = 0):
        global_config = self.config.get("global", {})
        num_epochs = global_config.get("num_epochs", 100)
        num_blocks_per_epoch = global_config.get("num_blocks_per_epoch", 16)

        init_tasks = global_config.get("init_tasks", {})

        for epoch_id in range(start_epoch, num_epochs):
            for block_id in range(start_block, num_blocks_per_epoch):
                # iterate through tasks and blocks
                env_vars = {
                    "prefix_tag": self.prefix_tag,
                    "epoch_id": epoch_id,
                    "block_id": block_id,
                }
                for task_name, task_config in init_tasks.items():
                    task_type = task_config.get("type", None)
                    if task_type is None or task_type not in VALID_TASK_TYPES:
                        raise ValueError(f"Task [{task_name}] has invalid task type: [{task_type}]")
                    # clone task_config
                    task_config = task_config.copy()
                    # evaluate everything in the task_config
                    for key, value in task_config.items():
                        if isinstance(value, str):
                            # format string with ${key}
                            task_config[key] = value.format(**env_vars)
                        else:
                            task_config[key] = value
                    # if task is codeGenEval, then we need to create a codeGenEvalBlock
                    if task_type == TASK_TYPE_CODEGENEVAL:
                        # use self.codeGenEval_config as default
                        codeGenEvalBlock = CodeGenEvalBlock(**(self.codeGenEval_default | task_config))
                        await self.global_reg_client.enqueue(task_type, codeGenEvalBlock.model_dump())
                    else:
                        raise ValueError(f"Task [{task_name}] has unknown task type: [{task_type}]")

    async def _enqueue_post_task(self, task_name: str, task_config: dict, env_vars: dict):
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
        if task_type == TASK_TYPE_EXEMPLAR:
            exemplarBlock = ExemplarBlock(**(self.exemplar_default | task_config))
            await self.global_reg_client.enqueue(task_type, exemplarBlock.model_dump())
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}], content: [{exemplarBlock.model_dump()}]")
        elif task_type == TASK_TYPE_CRITIQUE:
            critiqueBlock = CritiqueBlock(**(self.critique_default | task_config))
            await self.global_reg_client.enqueue(task_type, critiqueBlock.model_dump())
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}], content: [{critiqueBlock.model_dump()}]")
        elif task_type == TASK_TYPE_SFT:
            trainerSFTBlock = TrainerSFTBlock(**(self.trainerSFT_default | task_config))
            await self.global_reg_client.enqueue(task_type, trainerSFTBlock.model_dump())
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}], content: [{trainerSFTBlock.model_dump()}]")
        elif task_type == TASK_TYPE_RFT:
            trainerRFTBlock = TrainerRFTBlock(**(self.trainerRFT_default | task_config))
            await self.global_reg_client.enqueue(task_type, trainerRFTBlock.model_dump())
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}], content: [{trainerRFTBlock.model_dump()}]")
        elif task_type == TASK_TYPE_GRPO:
            trainerGRPOBlock = TrainerGRPOBlock(**(self.trainerGRPO_default | task_config))
            await self.global_reg_client.enqueue(task_type, trainerGRPOBlock.model_dump())
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}], content: [{trainerGRPOBlock.model_dump()}]")
        else:
            raise ValueError(f"Task [{task_name}] has unknown task type: [{task_type}]")

    def _get_run_tag(self, epoch_id: int, block_id: int):
        return f"{self.prefix_tag}_{epoch_id:03d}_{block_id:02d}"

    async def post_codeGenEval(self, codeGenEvalBlock: CodeGenEvalBlock):
        env_vars = {
            "prefix_tag": self.prefix_tag,
            "epoch_id": codeGenEvalBlock.epoch_id,
            "block_id": codeGenEvalBlock.block_id,
            "run_tag": self._get_run_tag(codeGenEvalBlock.epoch_id, codeGenEvalBlock.block_id),
        }
        post_tasks = self.config.get(TASK_TYPE_CODEGENEVAL, {}).get("post_tasks", {})
        for task_name, task_config in post_tasks.items():
            await self._enqueue_post_task(task_name, task_config, env_vars)

    async def post_exemplar(self, exemplarBlock: ExemplarBlock):
        env_vars = {
            "prefix_tag": self.prefix_tag,
            "epoch_id": exemplarBlock.epoch_id,
            "block_id": exemplarBlock.block_id,
            "run_tag": self._get_run_tag(exemplarBlock.epoch_id, exemplarBlock.block_id),
        }
        post_tasks = self.config.get(TASK_TYPE_EXEMPLAR, {}).get("post_tasks", {})
        for task_name, task_config in post_tasks.items():
            await self._enqueue_post_task(task_name, task_config, env_vars)

    async def post_critique(self, critiqueBlock: CritiqueBlock):
        env_vars = {
            "prefix_tag": self.prefix_tag,
            "epoch_id": critiqueBlock.epoch_id,
            "block_id": critiqueBlock.block_id,
            "run_tag": self._get_run_tag(critiqueBlock.epoch_id, critiqueBlock.block_id),
        }
        post_tasks = self.config.get(TASK_TYPE_CRITIQUE, {}).get("post_tasks", {})
        for task_name, task_config in post_tasks.items():
            await self._enqueue_post_task(task_name, task_config, env_vars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="KC_0.1.0_14B")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--start_block", type=int, default=0)
    args = parser.parse_args()

    globalWorkflow = GlobalWorkflow(prefix_tag=args.prefix_tag)
    asyncio.run(globalWorkflow.init_vars())
    asyncio.run(globalWorkflow.init_tasks(start_epoch=args.start_epoch, start_block=args.start_block))
