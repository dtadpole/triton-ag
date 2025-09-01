import yaml
import asyncio
import httpx
from typing import Any, Dict, Optional
from logger import logger
from workflowUtil import InferenceBlock, TrainerBlock, WorkflowSyncBlock, deep_format, merge_dicts
from pydantic import BaseModel

TASK_TYPE_INFERENCE = "inference"
TASK_TYPE_TRAINER = "trainer"
TASK_TYPE_SYNC = "sync"

VALID_TASK_TYPES = [
    TASK_TYPE_INFERENCE,
    TASK_TYPE_TRAINER,
    TASK_TYPE_SYNC,
]

class WorkflowClient:
    def __init__(self, prefix_tag: str, provider_name: str = "default", config_path: str = "workflow.yaml"):
        self.prefix_tag = prefix_tag
        self.provider_name = provider_name
        self.config_path = config_path
        self.config = self._load_config(self.config_path)
        self.registry_config = self.config.get("registry", {})
        if self.prefix_tag not in self.registry_config:
            if self.prefix_tag.startswith("auto"):
                logger.warning(f"⚠️ [WorkflowClient] [{self.prefix_tag}] Prefix tag is auto-generated, continuing...")
            else:
                error_msg = f"❌ [WorkflowClient] [{self.prefix_tag}] Prefix tag [{self.prefix_tag}] not found in workflow registry! Please check your [{config_path}] file."
                logger.error(error_msg)
                raise ValueError(error_msg)
        self.client_config = self.config.get("providers", {}).get(self.provider_name, {})
        self.host = self.client_config.get("host", "localhost")
        self.port = self.client_config.get("port", 8488)
        self.base_url = f"http://{self.host}:{self.port}"
        self.retries = self.client_config.get("retries", 5)
        self.timeout = self.client_config.get("timeout", 300)
        logger.info(f"🔍 [WorkflowClient] Initialized: [{self.host}:{self.port}]")

    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    async def get_workflow_config(self, prefix_tag: str):
        url = f"{self.base_url}/workflow/get/{prefix_tag}"
        retry_count = 0
        while retry_count < self.retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error getting workflow config: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error getting workflow config: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e
        return None

    async def keys(self):
        url = f"{self.base_url}/keys/{self.prefix_tag}"
        retry_count = 0
        while retry_count < self.retries:
            try:
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error getting keys: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error getting keys: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e
        return []

    async def exists(self, key: str):
        url = f"{self.base_url}/exists/{self.prefix_tag}/{key}"
        retry_count = 0
        while retry_count < self.retries:
            try:
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error checking if key exists: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error checking if key exists: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e
        return False

    async def get(self, key: str, last_modified_within: Optional[int]=None, return_none_if_not_found: bool=False):
        url = f"{self.base_url}/get/{self.prefix_tag}/{key}"
        if last_modified_within is not None:
            url += f"?last_modified_within={last_modified_within}"
        retry_count = 0
        while retry_count < self.retries:
            try:
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(url, timeout=self.timeout)
                    if response.status_code == 404 and return_none_if_not_found:
                        return None
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error getting key: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error getting key: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e
        return None

    async def put(self, key: str, value: Any):
        url = f"{self.base_url}/put/{self.prefix_tag}/{key}"
        retry_count = 0
        while retry_count < self.retries:
            try:
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.post(url, json={"value": value}, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error putting key: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error putting key: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e
        return None

    async def get_queues(self):
        url = f"{self.base_url}/queue/list/{self.prefix_tag}"
        retry_count = 0
        while retry_count < self.retries:
            try:
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error getting queues: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error getting queues: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e
        return []

    async def enqueue(self, queue_name: str, item: Dict[str, Any], create_queue: bool = False):
        retry_count = 0
        while retry_count < self.retries:
            try:
                url = f"{self.base_url}/queue/enqueue/{self.prefix_tag}/{queue_name}"
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.post(url, json={
                        "item": item,
                        "create_queue": create_queue
                    }, timeout=self.timeout)
                    response.raise_for_status()
                    logger.info(f"🔍 [WorkflowClient] [{self.prefix_tag}] Enqueued to [{queue_name}], content: [{item}]")
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error enqueuing item: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error enqueuing item: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e

    async def dequeue(self, queue_name: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                url = f"{self.base_url}/queue/dequeue/{self.prefix_tag}/{queue_name}"
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    logger.info(f"🔍 [WorkflowClient] [{self.prefix_tag}] Dequeued from [{queue_name}], content: [{response.json()}]")
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error dequeuing item: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error dequeuing item: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e

    async def qsize(self, queue_name: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                url = f"{self.base_url}/queue/qsize/{self.prefix_tag}/{queue_name}"
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error getting queue size: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error getting queue size: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e

    async def peek(self, queue_name: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                url = f"{self.base_url}/queue/peek/{self.prefix_tag}/{queue_name}"
                limits = httpx.Limits(
                    max_keepalive_connections=0,
                    max_connections=100,
                    keepalive_expiry=0,
                )
                async with httpx.AsyncClient(
                    limits=limits,
                    headers={"Connection": "close"},
                    http2=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                retry_count += 1
                if retry_count < self.retries:
                    logger.warning(f"Error peeking queue: {e}, retrying in {2 ** retry_count} seconds [{retry_count}/{self.retries}]")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Error peeking queue: [{e}] after [{retry_count}/{self.retries}] retries")
                    raise e
        return None

    def _get_queue_default(self, queue_type: str, queue_name: str):
        return self.config.get(queue_type, {}).get(queue_name, {}).get("default", {})

    async def _enqueue_callback_task(self, queue_type: str, queue_name: str, task_config: dict, env: dict):
        # clone task_config
        task_config = task_config.copy()
        queue_type = task_config.get("queue_type", None)
        if queue_type is None or queue_type not in VALID_TASK_TYPES:
            raise ValueError(f"Task [{queue_name}] has invalid queue type: [{queue_type}] in [{task_config}]")
        # evaluate everything in the task_config
        task_data = deep_format(task_config, env)
        # enqueue the task
        full_queue_name = f"{queue_type}.{queue_name}"
        if queue_type == TASK_TYPE_INFERENCE:
            inferenceBlock = InferenceBlock(**merge_dicts(self._get_queue_default(queue_type, queue_name), task_data))
            await self.enqueue(full_queue_name, inferenceBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [WorkflowClient] [{self.prefix_tag}] Enqueued to [{full_queue_name}], content: [{inferenceBlock.model_dump()}]")
        elif queue_type == TASK_TYPE_TRAINER:
            trainerBlock = TrainerBlock(**merge_dicts(self._get_queue_default(queue_type, queue_name), task_data))
            await self.enqueue(full_queue_name, trainerBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [WorkflowClient] [{self.prefix_tag}] Enqueued to [{full_queue_name}], content: [{trainerBlock.model_dump()}]")
        elif queue_type == TASK_TYPE_SYNC:
            syncBlock = WorkflowSyncBlock(**merge_dicts(self._get_queue_default(queue_type, queue_name), task_data))
            await self.enqueue(full_queue_name, syncBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [WorkflowClient] [{self.prefix_tag}] Enqueued to [{full_queue_name}], content: [{syncBlock.model_dump()}]")
        else:
            raise ValueError(f"Task [{queue_name}] has unknown task type: [{queue_type}]")

    def _get_run_tag(self, epoch_id: int, block_id: int):
        return f"{self.prefix_tag}_{epoch_id:03d}_{block_id:02d}"

    async def callback(
        self,
        callback_kind: str,
        queue_type: str,
        queue_name: str,
        block: BaseModel,
        env: dict = {},
    ):
        logger.info(f"⏳ [WorkflowClient] [{self.prefix_tag}] [{queue_type}.{queue_name}] {callback_kind}. Block: [{block.model_dump()}] Env vars: [{env}]")
        env = {
            "queue_type": queue_type,
            "queue_name": queue_name,
            "prefix_tag": self.prefix_tag,
            "epoch_id": block.epoch_id,
            "block_id": block.block_id,
            "run_tag": self._get_run_tag(block.epoch_id, block.block_id),
            "block": block.model_dump(),
            "env": env,
        }
        workflow_config = await self.get_workflow_config(self.prefix_tag)
        if workflow_config is None:
            logger.error(f"❌ [WorkflowClient] [{self.prefix_tag}] Workflow config not found")
            return
        callback_configs = workflow_config.get(queue_type, {}).get(queue_name, {}).get("callbacks", {})
        if callback_kind not in callback_configs:
            logger.info(f"🔍 [WorkflowClient] [{self.prefix_tag}] [{queue_type}.{queue_name}] [{callback_kind}] not found, skipping...")
            return
        callback_tasks = callback_configs.get(callback_kind, [])
        for callback_task in callback_tasks:
            queue_type = callback_task.get("queue_type", None)
            queue_name = callback_task.get("queue_name", None)
            if queue_type is None or queue_name is None:
                raise ValueError(f"Callback task [{callback_task}] has no queue_type or queue_name")
            if queue_type not in VALID_TASK_TYPES:
                raise ValueError(f"Callback task [{queue_name}] has invalid queue type: [{queue_type}] in [{callback_task}]")
            await self._enqueue_callback_task(queue_type, queue_name, callback_task, env)

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_tag", type=str, default="test")
    parser.add_argument("--provider_name", type=str, default="default")
    parser.add_argument("--queue_name", type=str, default="test.test.1")
    parser.add_argument("--test_callback", action="store_true")
    args = parser.parse_args()

    client = WorkflowClient(prefix_tag=args.prefix_tag, provider_name=args.provider_name)

    if args.test_callback:
        block = InferenceBlock(
            queue_name="codeGenEval.base",
            prefix_tag=args.prefix_tag,
            epoch_id=0,
            block_id=0,
            input_tag=f"{args.prefix_tag}_000_00",
            model_name="test",
            num_samples=2,
            num_generations=2,
            num_turns_per_generation=2,
            parallel_workers=1,
        )
        await client.callback(
            callback_kind="completion",
            queue_type="inference",
            queue_name="codeGenEval.base",
            block=block,
            context={},
        )
    else:
        print(await client.get_queues())
        print(await client.enqueue(args.queue_name, {"testKey1": "testValue1"}))
        print(await client.enqueue(args.queue_name, {"testKey2": "testValue2"}))
        print(await client.qsize(args.queue_name))
        print(await client.dequeue(args.queue_name))
        print(await client.qsize(args.queue_name))

if __name__ == "__main__":
    asyncio.run(main())