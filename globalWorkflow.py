import yaml
import asyncio
import argparse
from globalRegClient import GlobalRegClient
from globalUtils import CodeGenEvalBlock, CritiqueBlock, ExamplarBlock, ReflectionBlock

class GlobalWorkflow:
    def __init__(self, prefix_tag: str, config_path: str = "globalWorkflow.yaml"):
        self.prefix_tag = prefix_tag
        self.config = self.from_yaml(config_path)
        self.global_config = self.config.get("global", {})
        self.codeGenEval_default = self.config.get("codeGenEval", {}).get("default", {})
        self.critique_default = self.config.get("critique", {}).get("default", {})
        self.examplar_default = self.config.get("examplar", {}).get("default", {})
        self.reflection_default = self.config.get("reflection", {}).get("default", {})
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
                    if task_name == "codeGenEval":
                        # use self.codeGenEval_config as default
                        codeGenEvalBlock = CodeGenEvalBlock(**(self.codeGenEval_default | task_config))
                        await self.global_reg_client.enqueue(f"inference.codeGenEval", codeGenEvalBlock.model_dump())
                    else:
                        raise ValueError(f"Unknown task: [{task_name}]")

    async def post_codeGenEval(self, codeGenEvalBlock: CodeGenEvalBlock):
        pass

    async def post_critique(self, critiqueBlock: CritiqueBlock):
        pass

    async def post_examplar(self, examplarBlock: ExamplarBlock):
        pass

    async def post_reflection(self, reflectionBlock: ReflectionBlock):
        pass
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="KC_0.1.0_14B")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--start_block", type=int, default=0)
    args = parser.parse_args()

    globalWorkflow = GlobalWorkflow(prefix_tag=args.prefix_tag)
    asyncio.run(globalWorkflow.init_vars())
    asyncio.run(globalWorkflow.init_tasks(start_epoch=args.start_epoch, start_block=args.start_block))