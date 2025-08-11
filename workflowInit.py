import argparse
import asyncio
from loguru import logger
from workflowClient import WorkflowClient
from configInterpreter import ConfigInterpreter

class WorkflowInit:
    def __init__(self, workflow_client: WorkflowClient):
        self.workflowClient = workflow_client
        self.configInterpreter = ConfigInterpreter()

    def log_qsize(self, result: dict, context_vars: dict) -> dict:
        prefix_tag = context_vars.get('prefix_tag', None)
        epoch_id = context_vars.get('epoch_id', None)
        block_id = context_vars.get('block_id', None)
        logger.info(f"🔍 [WorkflowInit] [{prefix_tag}] [{epoch_id}] [{block_id}] [qsize: {result}]")

    async def init(self, args: argparse.Namespace):
        # get workflow config
        workflow_config = await self.workflowClient.get_workflow_config(args.prefix_tag)
        global_config = workflow_config.get("global", {})

        # get start and end epoch and block
        start_epoch = global_config.get("start_epoch", 0) if args.start_epoch == -1 else args.start_epoch
        start_block = global_config.get("start_block", 0) if args.start_block == -1 else args.start_block
        end_epoch = global_config.get("end_epoch", 16) if args.end_epoch == -1 else args.end_epoch
        end_block = global_config.get("end_block", 10) if args.end_block == -1 else args.end_block

        # get pairs of epoch and block
        pairs = [(e, b)
                for e in range(start_epoch, end_epoch + 1)
                for b in range(start_block if e == start_epoch else 0, (end_block + 1) if e == end_epoch else (end_block + 1))]

        for e, b in pairs:
            success = False
            while not success:
                try:
                    workflow_config = await self.workflowClient.get_workflow_config(args.prefix_tag)
                    workitem_config = workflow_config.get("workitems", {}).get(args.workitem, {})
                    success = await self.configInterpreter.execute(
                        runtime=self,
                        config=workitem_config,
                        context_vars={
                            "prefix_tag": args.prefix_tag,
                            "epoch_id": e,
                            "block_id": b,
                        }
                    )
                    if success:
                        break
                except Exception as e:
                    logger.error(f"Error initializing workflow: {e}")
                    raise e

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="auto") # "TC_0.1.0_14B.m") # "auto"
    parser.add_argument("--workitem", type=str, default="codeGenEval.init")
    parser.add_argument("--start_epoch", type=int, default=-1)
    parser.add_argument("--start_block", type=int, default=-1)
    parser.add_argument("--end_epoch", type=int, default=-1)
    parser.add_argument("--end_block", type=int, default=-1)
    args = parser.parse_args()

    if args.prefix_tag == "auto":
        logger.error(f"❌ [WorkflowInit] --prefix_tag is required!")
        return

    workflow_client = WorkflowClient(args.prefix_tag)
    workflow_init = WorkflowInit(workflow_client)
    await workflow_init.init(args)

if __name__ == "__main__":
    asyncio.run(main())