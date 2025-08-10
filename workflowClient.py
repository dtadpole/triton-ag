import yaml
import asyncio
import httpx
from typing import Any, Dict, Optional
from logger import logger
from workflowUtil import get_global_registry_port


class WorkflowClient:
    def __init__(self, prefix_tag: str = "auto", trainer_dir: str = "~/.trainer"):
        self.config = self._load_config().get("client", {})
        self.host = self.config.get("host", "localhost")
        self.port = get_global_registry_port(prefix_tag, trainer_dir)
        self.base_url = f"http://{self.host}:{self.port}"
        self.retries = self.config.get("retries", 5)
        self.timeout = self.config.get("timeout", 300)
        logger.info(f"🔍 [GlobalRegClient] Initialized with host: {self.host}, port: {self.port}, retries: {self.retries}, timeout: {self.timeout}")

    def _load_config(self):
        with open("globalRegistry.yaml", "r") as f:
            return yaml.safe_load(f)

    async def keys(self):
        url = f"{self.base_url}/keys"
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

    async def get(self, key: str, last_modified_within: Optional[int]=None):
        url = f"{self.base_url}/get/{key}"
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
        url = f"{self.base_url}/put"
        retry_count = 0
        while retry_count < self.retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json={"key": key, "value": value}, timeout=self.timeout)
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
        url = f"{self.base_url}/queue/list"
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
                url = f"{self.base_url}/queue/enqueue"
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json={
                        "queue_name": queue_name,
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
                url = f"{self.base_url}/queue/dequeue/{queue_name}"
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
                url = f"{self.base_url}/queue/qsize/{queue_name}"
                async with httpx.AsyncClient() as client:
                    logger.info(f"🔍 [GlobalRegClient] Getting queue size via [{url}]")
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

    def _get_task_default(self, task_type: str, task_name: str):
        return self.config.get(task_type, {}).get(task_name, {}).get("default", {})
    
    async def init_vars(self):
        init_vars = self.global_config.get("init_vars", []) # list of dicts
        for init_var in init_vars:
            await self.global_reg_client.put(init_var.get("name"), init_var.get("value"))
            logger.info(f"🔢 [GlobalWorkflow] [{self.prefix_tag}] Initialized [{init_var.get('name')}] = [{init_var.get('value')}]")

    async def init_tasks(self,
                         start_epoch: Optional[int] = None,
                         start_block: Optional[int] = None,
                         end_epoch: Optional[int] = None,
                         end_block: Optional[int] = None,
                         ):
        global_config = self.config.get("global", {})
        start_epoch = start_epoch if start_epoch is not None else global_config.get("start_epoch", 0)
        start_block = start_block if start_block is not None else global_config.get("start_block", 0)
        end_epoch = end_epoch if end_epoch is not None else global_config.get("end_epoch", 10)
        end_block = end_block if end_block is not None else global_config.get("end_block", 16)

        init_tasks = global_config.get("init_tasks", [])

        for epoch_id in range(start_epoch, end_epoch):
            for block_id in range(start_block, end_block):
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
    client = WorkflowClient()
    print(asyncio.run(client.get_queues()))
    print(asyncio.run(client.enqueue("test", {"testKey1": "testValue1"})))
    print(asyncio.run(client.enqueue("test", {"testKey2": "testValue2"})))
    print(asyncio.run(client.qsize("test")))
    print(asyncio.run(client.dequeue("test")))
    print(asyncio.run(client.qsize("test")))
    print(asyncio.run(client.dequeue("test")))
    print(asyncio.run(client.qsize("test")))
