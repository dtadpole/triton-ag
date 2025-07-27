import yaml
import asyncio
import httpx
from typing import Any, Dict
from logger import logger


class GlobalRegClient:
    def __init__(self, host: str = "localhost", port: int = 8084):
        self.config = self._load_config().get("client", {})
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 8084)
        self.base_url = f"http://{self.host}:{self.port}"
        self.retries = self.config.get("retries", 5)
        self.timeout = self.config.get("timeout", 300)

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

    async def get(self, key: str):
        url = f"{self.base_url}/get/{key}"
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
    
    async def enqueue(self, queue_name: str, item: Dict[str, Any]):
        retry_count = 0
        while retry_count < self.retries:
            try:
                url = f"{self.base_url}/queue/enqueue"
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json={"queue_name": queue_name, "item": item}, timeout=self.timeout)
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
