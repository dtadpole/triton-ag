import os
import json
import yaml
import time
import asyncio
import uvicorn
from typing import Any, Dict, Optional, Annotated
from fastapi import FastAPI, HTTPException, Body, Query
from loguru import logger
from globalUtils import GlobalUtils

global_utils = GlobalUtils()
fastapi = global_utils.fastapi

QUEUE_PREFIX = "queue."
GLOBAL_REGISTRY_DIR = "globalRegistry"

# create a singleton class to store global variables
class GlobalRegistry:
    _instance = None # class variable to store the instance

    # singleton pattern
    def __new__(cls):
        if cls._instance is None:
            # If no instance exists, create one using the superclass's __new__
            cls._instance = super(GlobalRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # __init__ will be called every time, but only the first time will
        # actually initialize the instance if we add a flag.
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.config = self._load_config()
            self.registry = {}  # registry of global variables, or object registry ==> {key: value}
            self.tasks = {}  # registry of tasks, or task registry ==> {task_name: task_coroutine}
            self.put("reg.fastapi", fastapi) # this is a reference to the fastapi instance

    def _load_config(self):
        with open("globalRegistry.yaml", "r") as f:
            return yaml.safe_load(f)

    async def _refresh_config_task(self):
        while True:
            try:
                interval = self.config.get("_refresh_config_task", {}).get("interval", 10)
                # logger.info(f"🔄 Refreshing config every {interval} seconds")
                self.config = self._load_config()
                await self._load_queues()
            except Exception as e:
                logger.error(f"Error refreshing config: {e}")
            finally:
                await asyncio.sleep(interval)

    async def _load_queues(self):
        """
        Update the queues in the config
        """
        try:
            # load previous queued items from storage
            queue_storage = {}
            try:
                # if globalRegistry/queues.yaml exists, load it
                if os.path.exists(f"{GLOBAL_REGISTRY_DIR}/queues.json"):
                    with open(f"{GLOBAL_REGISTRY_DIR}/queues.json", "r") as f:
                        queue_storage = json.load(f) # queue_storage is a dictionary of queue names and their items
            except Exception as e:
                # load from globalRegistry/queues.yaml.bak
                if os.path.exists(f"{GLOBAL_REGISTRY_DIR}/queues.json.bak"):
                    with open(f"{GLOBAL_REGISTRY_DIR}/queues.json.bak", "r") as f:
                        queue_storage = json.load(f)
            # load the queues from the config
            queues = self.config.get("queues", [])
            for queue in queues:
                object_name = f"{QUEUE_PREFIX}{queue.get('name')}"
                if object_name not in self.registry:
                    logger.info(f"🎢 Creating queue [{object_name}]")
                    self.put(object_name, asyncio.Queue())
                    if object_name in queue_storage:
                        for item in queue_storage[object_name]:
                            await self.get(object_name).put(item)
                        logger.info(f"📦 Loaded {len(queue_storage[object_name])} items into queue [{object_name}]")
        except Exception as e:
            logger.error(f"Error updating queues: {e}")

    async def _save_queue_task(self):
        """
        Save the queues to the config
        """
        while True:
            try:
                interval = self.config.get("_save_queue_task", {}).get("interval", 10)
                queue_storage = {}
                queues = self.config.get("queues", [])
                for queue in queues:
                    queue_name = queue.get("name")
                    object_name = f"{QUEUE_PREFIX}{queue_name}"
                    q = self.get(object_name)
                    if q is not None:
                        queue_storage[object_name] = list(q._queue)
                # if folder globalRegistry does not exist, create it
                if not os.path.exists(GLOBAL_REGISTRY_DIR):
                    os.makedirs(GLOBAL_REGISTRY_DIR)
                # if globalRegistry/queues.yaml exists, move it to globalRegistry/queues.yaml.bak
                if os.path.exists(f"{GLOBAL_REGISTRY_DIR}/queues.json"):
                    os.rename(f"{GLOBAL_REGISTRY_DIR}/queues.json", f"{GLOBAL_REGISTRY_DIR}/queues.json.bak")
                # save the queues to the config
                with open(f"{GLOBAL_REGISTRY_DIR}/queues.json", "w") as f:
                    json.dump(queue_storage, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving queues: {e}")
            finally:
                await asyncio.sleep(interval)

    def keys(self):
        """
        Return a list of keys in the object registry
        """
        try:
            return list(self.registry.keys())
        except Exception as e:
            logger.error(f"Error getting keys: {e}")
            return []

    def get(self, key, last_modified_within: Optional[int]=None):
        """
        Get the value of a key in the object registry
        last_modified_within: if not None, return the value if the last modified time is within the last_modified_within seconds
        """
        try:
            item = self.registry.get(key, None)
            if item is None:
                return None
            if last_modified_within is not None:
                if time.time() - item["timestamp"] > last_modified_within:
                    return None
                else:
                    return item["value"]
            else:
                return item["value"]
        except Exception as e:
            logger.error(f"Error getting {key}: {e}")
            return None
    
    def put(self, key, value):
        """
        Put a value into the object registry
        """
        try:
            if key not in self.registry:
                item = {
                    "version": 1,
                    "timestamp": time.time(),
                    "value": value
                }
                self.registry[key] = item
                return None
            else:
                item = self.registry[key]
                old_value = item["value"]
                item["version"] += 1
                item["timestamp"] = time.time()
                item["value"] = value
                self.registry[key] = item
                return old_value
        except Exception as e:
            logger.error(f"Error getting {key}: {e}")
            return None
        
    def delete(self, key):
        """
        Delete a key from the object registry
        """
        try:
            del self.registry[key]
        except Exception as e:
            logger.error(f"Error deleting {key}: {e}")

    def task_names(self):
        """
        Return a list of task names in the task registry
        """
        try:
            return list(self.tasks.keys())
        except Exception as e:
            logger.error(f"Error getting task names: {e}")
            return []

    def get_task(self, task_name):
        """
        Get a task from the task registry
        """
        try:
            task_item = self.tasks.get(task_name, None)
            if task_item is None:
                return None
            return task_item["task"]
        except Exception as e:
            logger.error(f"Error getting task {task_name}: {e}")
            return None
    
    def put_task(self, task_name, task_coroutine):
        """
        Put a task into the task registry
        """
        try:
            # TODO: we can enhance task_coroutine here to add error handling etc.
            task = asyncio.create_task(task_coroutine) # run the task in the event loop
            if task_name not in self.tasks:
                task_item = {
                    "version": 1,
                    "task": task
                }
                self.tasks[task_name] = task_item
            else:
                task_item = self.tasks[task_name]
                task_item["version"] += 1
                task_item["task"] = task
                self.tasks[task_name] = task_item
        except Exception as e:
            logger.error(f"Error putting task {task_name}: {e}")

    def delete_task(self, task_name, cancel_task=True):
        """
        Cancel a task in the task registry
        cancel_task: if True, the task will be canceled in the event loop
        """
        try:
            task_item = self.tasks.get(task_name, None)
            if task_item is None:
                return
            if cancel_task:
                task_item["task"].cancel()
            del self.tasks[task_name]
        except Exception as e:
            logger.error(f"Error canceling task {task_name}: {e}")            

    def get_working_dir(self):
        """
        Get the working directory of the server
        """
        return os.getcwd()

    
    async def run(self):
        """
        Run the server and refresh the config in parallel
        """
        try:
            host = self.config.get("fastapi", {}).get("host", "0.0.0.0")
            port = self.config.get("fastapi", {}).get("port", 8000)
            server = uvicorn.Server(uvicorn.Config(self.get("reg.fastapi"), host=host, port=port))
            logger.info(f"Configuring fastapi server at [{host}:{port}]")

            # create a task to run the server
            self.put_task("reg.fastapi", server.serve())

            # create a task to refresh the config
            self.put_task("reg.refresh_config", self._refresh_config_task())

            # create a task to save the queues
            self.put_task("reg.save_queue", self._save_queue_task())

            # now we need to start the asyncio loop
            # by this time, there could be other tasks running in the event loop
            # so we need to wait for the tasks to complete
            tasks = []
            for task_name, task_item in self.tasks.items():
                logger.info(f"🚀 Starting task [{task_name}]")
                tasks.append(task_item["task"])
            # run the tasks
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error running server: {e}")
            return

@fastapi.get("/keys")
async def keys():
    return reg.keys()

@fastapi.get("/get/{key}")
async def get(key: str,
              last_modified_within: Annotated[int | None, Query(gt=0)] = None
              ):
    return reg.get(key, last_modified_within)

@fastapi.post("/put")
async def put(key: str = Body(...), value: Any = Body(None)):
    old_value = reg.put(key, value)
    return old_value

@fastapi.delete("/delete/{key}")
async def delete(key: str):
    reg.delete(key)
    return {"message": f"Key [{key}] deleted"}


@fastapi.get("/queue/list")
async def qlist():
    return {
        "queues": [key.replace(QUEUE_PREFIX, "") for key in reg.keys() if key.startswith(QUEUE_PREFIX)]
    }

@fastapi.post("/queue/enqueue")
async def enqueue(queue_name: str = Body(...), item: Dict[str, Any] = Body(...)):
    queue = reg.get(f"{QUEUE_PREFIX}{queue_name}")
    if queue is None:
        raise HTTPException(status_code=404, detail=f"Queue [{queue_name}] not found")
    try:
        await queue.put(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enqueuing item [{item}] into queue [{queue_name}]: {e}")
    return {"message": f"Item [{item}] enqueued into queue [{queue_name}]"}

@fastapi.get("/queue/dequeue/{queue_name}")
async def dequeue(queue_name: str):
    queue = reg.get(f"{QUEUE_PREFIX}{queue_name}")
    if queue is None:
        raise HTTPException(status_code=404, detail=f"Queue [{queue_name}] not found")
    # get timeout from config
    timeout = reg.config.get("fastapi", {}).get("dequeue_timeout", 5)
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"Queue [{queue_name}] timed out")

@fastapi.get("/queue/qsize/{queue_name}")
async def qsize(queue_name: str):
    queue = reg.get(f"{QUEUE_PREFIX}{queue_name}")
    if queue is None:
        raise HTTPException(status_code=404, detail=f"Queue [{queue_name}] not found")
    return queue.qsize()


if __name__ == "__main__":
    # initialize the global registry singleton
    reg = GlobalRegistry()
    # add repl server to the fastapi app, and initialize/reset the repl namespace
    from replServer import init_vars
    init_vars(reg)
    # run the main loop
    asyncio.run(reg.run())
