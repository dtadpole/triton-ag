import os
import yaml
import asyncio
import uvicorn
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Body
from loguru import logger
from globalUtils import GlobalUtils

global_utils = GlobalUtils()
fastapi = global_utils.fastapi

QUEUE_PREFIX = "queue."

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

    async def _refresh_config(self):
        interval = self.config.get("_refresh_config", {}).get("interval", 10)
        while True:
            try:
                # logger.info("🔄 Refreshing config")
                self.config = self._load_config()
                await self._update_queues()
            except Exception as e:
                logger.error(f"Error refreshing config: {e}")
            finally:
                await asyncio.sleep(interval)

    async def _update_queues(self):
        """
        Update the queues in the config
        """
        try:
            queues = self.config.get("queues", [])
            for queue in queues:
                object_name = f"{QUEUE_PREFIX}{queue.get('name')}"
                if object_name not in self.registry:
                    logger.info(f"🎢 Creating queue [{object_name}]")
                    self.put(object_name, asyncio.Queue())
                else:
                    pass
        except Exception as e:
            logger.error(f"Error updating queues: {e}")

    def keys(self):
        """
        Return a list of keys in the object registry
        """
        try:
            return list(self.registry.keys())
        except Exception as e:
            logger.error(f"Error getting keys: {e}")
            return []

    def get(self, key):
        """
        Get the value of a key in the object registry
        """
        try:
            item = self.registry.get(key, None)
            if item is None:
                return None
            return item["value"]
        except Exception as e:
            logger.error(f"Error getting {key}: {e}")
            return None
    
    def put(self, key, value,):
        """
        Put a value into the object registry
        """
        try:
            if key not in self.registry:
                item = {
                    "version": 1,
                "value": value
                }
                self.registry[key] = item
            else:
                item = self.registry[key]
                item["version"] += 1
                item["value"] = value
                self.registry[key] = item
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
            self.put_task("reg.refresh_config", self._refresh_config())

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
    return {"size": queue.qsize()}


if __name__ == "__main__":
    # initialize the global registry singleton
    reg = GlobalRegistry()
    # add repl server to the fastapi app, and initialize/reset the repl namespace
    from replServer import reset_vars
    reset_vars()
    # run the main loop
    asyncio.run(reg.run())
