import yaml
import asyncio
import httpx
from typing import Any, Dict, Optional
from logger import logger
from workflowUtil import get_global_registry_port, CodeGenEvalBlock, CritiqueBlock, ExemplarBlock, ReflectionBlock, TrainerSFTBlock, TrainerRFTBlock, TrainerGRPOBlock, ComposerBlock
from pydantic import BaseModel

TASK_TYPE_INFERENCE = "inference"
TASK_TYPE_GRPO = "trainer.grpo"
TASK_TYPE_SFT = "trainer.sft"
TASK_TYPE_RFT = "trainer.rft"

VALID_TASK_TYPES = [
    TASK_TYPE_INFERENCE,
    TASK_TYPE_GRPO,
    TASK_TYPE_SFT,
    TASK_TYPE_RFT,
]

class WorkflowClient:
    def __init__(self, prefix_tag: str, config_path: str = "workflow.yaml"):
        self.prefix_tag = prefix_tag
        self.config_path = config_path
        self.config = self._load_config(self.config_path)
        self.registry_config = self.config.get("registry", {})
        if self.prefix_tag not in self.registry_config:
            error_msg = f"Prefix tag [{self.prefix_tag}] not found in workflow registry! Please check your [{config_path}] file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.client_config = self.config.get("client", {})
        self.host = self.client_config.get("host", "localhost")
        self.port = self.client_config.get("port", 8488)
        self.base_url = f"http://{self.host}:{self.port}"
        self.retries = self.client_config.get("retries", 5)
        self.timeout = self.client_config.get("timeout", 300)
        logger.info(f"🔍 [GlobalRegClient] Initialized with host: {self.host}, port: {self.port}, retries: {self.retries}, timeout: {self.timeout}")

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
                async with httpx.AsyncClient() as client:
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
                async with httpx.AsyncClient() as client:
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

    async def get(self, key: str, last_modified_within: Optional[int]=None):
        url = f"{self.base_url}/get/{self.prefix_tag}/{key}"
        if last_modified_within is not None:
            url += f"?last_modified_within={last_modified_within}"
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
                async with httpx.AsyncClient() as client:
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
                async with httpx.AsyncClient() as client:
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
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json={
                        "item": item,
                        "create_queue": create_queue
                    }, timeout=self.timeout)
                    response.raise_for_status()
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
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=self.timeout)
                    response.raise_for_status()
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
                async with httpx.AsyncClient() as client:
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
                async with httpx.AsyncClient() as client:
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

    def _get_task_default(self, queue_type: str, queue_name: str):
        return self.config.get(queue_type, {}).get(queue_name, {}).get("default", {})

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
        if task_type == TASK_TYPE_INFERENCE:
            inferenceBlock = ComposerBlock(**(self._get_task_default(task_type, task_name) | task_config))
            await self.enqueue(queue_name, inferenceBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}:{task_name}], content: [{inferenceBlock.model_dump()}]")
        elif task_type == TASK_TYPE_SFT:
            trainerSFTBlock = TrainerSFTBlock(**(self._get_task_default(task_type, task_name) | task_config))
            await self.enqueue(queue_name, trainerSFTBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}:{task_name}], content: [{trainerSFTBlock.model_dump()}]")
        elif task_type == TASK_TYPE_RFT:
            trainerRFTBlock = TrainerRFTBlock(**(self._get_task_default(task_type, task_name) | task_config))
            await self.enqueue(queue_name, trainerRFTBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}:{task_name}], content: [{trainerRFTBlock.model_dump()}]")
        elif task_type == TASK_TYPE_GRPO:
            trainerGRPOBlock = TrainerGRPOBlock(**(self._get_task_default(task_type, task_name) | task_config))
            await self.enqueue(queue_name, trainerGRPOBlock.model_dump(), create_queue=True)
            logger.info(f"🎢 [GlobalWorkflow] [{self.prefix_tag}] Enqueued to [{task_type}:{task_name}], content: [{trainerGRPOBlock.model_dump()}]")
        else:
            raise ValueError(f"Task [{task_name}] has unknown task type: [{task_type}]")

    def _get_run_tag(self, epoch_id: int, block_id: int):
        return f"{self.prefix_tag}_{epoch_id:03d}_{block_id:02d}"

    async def post_block(self, queue_type: str, queue_name: str, block: BaseModel):
        logger.info(f"⏳ [GlobalWorkflow] [{self.prefix_tag}] [{queue_type}.{queue_name}] Posting block: [{block.model_dump()}]")
        env_vars = {
            "prefix_tag": self.prefix_tag,
            "epoch_id": block.epoch_id,
            "block_id": block.block_id,
            "run_tag": self._get_run_tag(block.epoch_id, block.block_id),
        }
        workflow_config = await self.get_workflow_config(self.prefix_tag)
        if workflow_config is None:
            logger.error(f"❌ [GlobalWorkflow] [{self.prefix_tag}] Workflow config not found")
            return
        post_workitems = workflow_config.get(queue_type, {}).get(queue_name, {}).get("post_workitems", [])
        for post_workitem in post_workitems:
            task_type = post_workitem.get("type", None)
            task_name = post_workitem.get("name", None)
            if task_type is None or task_name is None:
                raise ValueError(f"Task [{post_workitem}] has no name or type")
            if task_type not in VALID_TASK_TYPES:
                raise ValueError(f"Task [{task_name}] has invalid task type: [{task_type}]")
            await self._enqueue_post_task(task_type, task_name, post_workitem, env_vars)

if __name__ == "__main__":
    client = WorkflowClient(prefix_tag="test")
    print(asyncio.run(client.get_queues()))
    print(asyncio.run(client.enqueue("test", {"testKey1": "testValue1"})))
    print(asyncio.run(client.enqueue("test", {"testKey2": "testValue2"})))
    print(asyncio.run(client.qsize("test")))
    print(asyncio.run(client.dequeue("test")))
    print(asyncio.run(client.qsize("test")))
    print(asyncio.run(client.dequeue("test")))
    print(asyncio.run(client.qsize("test")))
