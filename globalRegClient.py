import yaml
import asyncio
import httpx
from typing import Any, Dict, Optional
from logger import logger
from globalUtils import get_global_registry_port


class GlobalRegClient:
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

if __name__ == "__main__":
    client = GlobalRegClient()
    print(asyncio.run(client.get_queues()))
    print(asyncio.run(client.enqueue("test", {"testKey1": "testValue1"})))
    print(asyncio.run(client.enqueue("test", {"testKey2": "testValue2"})))
    print(asyncio.run(client.qsize("test")))
    print(asyncio.run(client.dequeue("test")))
    print(asyncio.run(client.qsize("test")))
    print(asyncio.run(client.dequeue("test")))
    print(asyncio.run(client.qsize("test")))
