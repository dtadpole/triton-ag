import argparse
import asyncio
from loguru import logger
from workflowUtil import InferenceBlock, WorkflowSyncBlock, TrainerBlock, merge_dicts, deep_format
from workflowClient import WorkflowClient
from configInterpreter import ConfigInterpreter

class WorkflowInit:
    def __init__(self, prefix_tag: str):
        self.prefix_tag = prefix_tag
        self.workflowClient = WorkflowClient(prefix_tag)

    def log_qsize(self, result: dict, context_vars: dict) -> dict:
        prefix_tag = context_vars.get('prefix_tag', None)
        epoch_id = context_vars.get('epoch_id', None)
        block_id = context_vars.get('block_id', None)
        logger.info(f"🔍 [WorkflowInit] [{prefix_tag}] [{epoch_id}] [{block_id}] [qsize: {result}]")

    async def _get_queue_default(self, queue_type: str, queue_name: str):
        workflow_config = await self.workflowClient.get_workflow_config(self.prefix_tag)
        return workflow_config.get(queue_type, {}).get(queue_name, {}).get("default", {})

    async def init(self, args: argparse.Namespace):
        # get workflow config
        workflow_config = await self.workflowClient.get_workflow_config(args.prefix_tag)
        global_config = workflow_config.get("global", {})
        init_config = workflow_config.get("init", {})

        # get start and end epoch and block
        start_epoch = global_config.get("start_epoch", 0) if args.start_epoch == -1 else args.start_epoch
        start_block = global_config.get("start_block", 0) if args.start_block == -1 else args.start_block
        end_epoch = global_config.get("end_epoch", 50) if args.end_epoch == -1 else args.end_epoch
        end_block = global_config.get("end_block", 10) if args.end_block == -1 else args.end_block

        # get pairs of epoch and block
        pairs = [(e, b)
                for e in range(start_epoch, end_epoch)
                for b in range(start_block if e == start_epoch else 0, (end_block) if e == end_epoch else (end_block))]

        for e, b in pairs:
            try:
                workitems_config = init_config.get(args.queue_name, [])
                for workitem_config in workitems_config:
                    logger.info(f"🔍 [WorkflowInit] [{self.prefix_tag}] [{e}] [{b}] [{workitem_config}]")
                    queue_type = workitem_config.get("queue_type")
                    queue_name = workitem_config.get("queue_name")
                    queue_full_name = f"{queue_type}.{queue_name}"
                    if queue_type == "inference":
                        task_data = deep_format(workitem_config, env={
                            "prefix_tag": self.prefix_tag,
                            "epoch_id": e,
                            "block_id": b,
                            "input_tag": f"{self.prefix_tag}_{e:03d}_{b:02d}",
                        })
                        inferenceBlock = InferenceBlock(**merge_dicts(await self._get_queue_default(queue_type, queue_name), task_data))
                        await self.workflowClient.enqueue(queue_full_name, inferenceBlock.model_dump(), create_queue=True)
                        qsize = await self.workflowClient.qsize(queue_full_name)
                        self.log_qsize(qsize, {
                            "prefix_tag": self.prefix_tag,
                            "epoch_id": e,
                            "block_id": b,
                        })
                    else:
                        raise ValueError(f"Unknown queue type: {workitem_config.get('queue_type')}")
            except Exception as ex:
                logger.error(f"Error initializing workflow [{e}] [{b}]: {ex}")
                raise ex

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="auto") # "TC_0.1.0_14B.m") # "auto"
    parser.add_argument("--queue_name", type=str, default="codeGenEval.base")
    parser.add_argument("--start_epoch", type=int, default=-1)
    parser.add_argument("--start_block", type=int, default=-1)
    parser.add_argument("--end_epoch", type=int, default=-1)
    parser.add_argument("--end_block", type=int, default=-1)
    args = parser.parse_args()

    if args.prefix_tag.startswith("auto"):
        logger.error(f"❌ [WorkflowInit] --prefix_tag is required!")
        return

    workflow_init = WorkflowInit(args.prefix_tag)
    await workflow_init.init(args)

if __name__ == "__main__":
    asyncio.run(main())