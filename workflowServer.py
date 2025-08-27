import yaml
import asyncio
import argparse
import socket
import uvicorn
import traceback
import os
from typing import Optional, Any, Dict, Annotated
from fastapi import FastAPI, HTTPException, Body, Query, APIRouter
from datetime import datetime
from logger import logger
from workflowRegistry import WorkflowRegistry, QUEUE_PREFIX, ADAPTER_PREFIX
from workflowUtil import get_prefix_tag, CodeGenEvalBlock, CritiqueBlock, ExemplarBlock, ReflectionBlock, TrainerSFTBlock, TrainerRFTBlock, TrainerGRPOBlock, ComposerBlock
from replServer import ReplServer
from util import WORKFLOW_DIR


class WorkflowServer:
    def __init__(self, config_path: str = "workflow.yaml"):
        self.router = APIRouter()
        self.config_path = config_path
        self.registries = {}
        self.short_vars = {}
        self._enable_api_endpoints()
        self._load_config()

    def _load_config(self):
        self.config = self.from_yaml(self.config_path)
        for prefix_tag, registry_item in self.config.get("registry", {}).items():
            if prefix_tag not in self.registries:
                # we found a new prefix tag, so we need to create a new registry
                workflow_config_path = registry_item.get("config_path", "workflow/example.yaml")
                workflow_data_dir = registry_item.get("data_dir", WORKFLOW_DIR)
                workflow_registry = WorkflowRegistry(
                    workflow_config_path=workflow_config_path,
                    prefix_tag=prefix_tag,
                    data_dir=workflow_data_dir
                )
                self.registries[prefix_tag] = workflow_registry
                logger.info(f"🗂️ [WorkflowServer] Loaded registry [{prefix_tag}] with config [{workflow_config_path}]")
        # reset short vars
        self.short_vars = self._reset_short_vars()

    def _reset_short_vars(self):
        # process short name in separate loop
        short_vars = {}
        for prefix_tag, registry_item in self.config.get("registry", {}).items():
            short_name = registry_item.get("short_name", None)
            if short_name is not None:
                registry = self.registries.get(prefix_tag, None)
                if registry is not None:
                    short_vars[short_name] = registry
        return short_vars

    def from_yaml(self, config_path):
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        return yaml_data

    def _enable_api_endpoints(self):
        """
        Add the API endpoints to the router
        """
        @self.router.get("/workflow/get/{prefix_tag}")
        async def workflow_get(prefix_tag: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            return self.registries[prefix_tag].get_workflow_config()

        @self.router.delete("/workflow/delete/{prefix_tag}")
        async def workflow_delete(prefix_tag: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            # Also remove from short_vars if present
            short_names_to_remove = [k for k, v in self.short_vars.items() if v is None or v == self.registries.get(prefix_tag)]
            for short_name in short_names_to_remove:
                self.short_vars.pop(short_name, None)
            del self.registries[prefix_tag]
            return {"message": f"Registry with prefix tag [{prefix_tag}] deleted"}

        @self.router.get("/keys/{prefix_tag}")
        async def keys(prefix_tag: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            return self.registries[prefix_tag].keys()

        @self.router.get("/exists/{prefix_tag}/{key}")
        async def exists(prefix_tag: str, key: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            return self.registries[prefix_tag].exists(key)

        @self.router.get("/get/{prefix_tag}/{key}")
        async def get(prefix_tag: str, key: str,
                    last_modified_within: Annotated[int | None, Query(gt=0)] = None
                    ):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            return self.registries[prefix_tag].get(key, last_modified_within)

        @self.router.post("/put/{prefix_tag}/{key}")
        async def put(prefix_tag: str, key: str, value: Any = Body(None)):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            value_data = value.get('value', None)
            old_value = self.registries[prefix_tag].put(key, value_data)
            return old_value

        @self.router.delete("/delete/{prefix_tag}/{key}")
        async def delete(prefix_tag: str, key: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            self.registries[prefix_tag].delete(key)
            return {"message": f"Key [{key}] deleted"}

        @self.router.get("/queue/list/{prefix_tag}")
        async def qlist(prefix_tag: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            return {
                "queues": [key.replace(QUEUE_PREFIX, "") for key in self.registries[prefix_tag].keys() if key.startswith(QUEUE_PREFIX)]
            }

        @self.router.post("/queue/enqueue/{prefix_tag}/{queue_name}")
        async def enqueue(
            prefix_tag: str,
            queue_name: str,
            item: Dict[str, Any] = Body(...),
            create_queue: bool = Body(default=False)
        ):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            object_name = f"{QUEUE_PREFIX}{queue_name}"
            queue = self.registries[prefix_tag].get(object_name)
            if queue is None:
                if create_queue:
                    logger.info(f"🎢 Creating queue [{object_name}]")
                    queue = asyncio.Queue()
                    self.registries[prefix_tag].put(object_name, queue)
                else:
                    raise HTTPException(status_code=404, detail=f"Queue [{queue_name}] not found")
            try:
                await queue.put(item)
                logger.info(f"🎢 Enqueued item [{item}] into queue [{queue_name}]")
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error enqueuing item [{item}] into queue [{queue_name}]: {e}")
            return {"message": f"Enqueued [{item}] into queue [{queue_name}]"}

        @self.router.get("/queue/dequeue/{prefix_tag}/{queue_name}")
        async def dequeue(prefix_tag: str, queue_name: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            object_name = f"{QUEUE_PREFIX}{queue_name}"
            queue = self.registries[prefix_tag].get(object_name)
            if queue is None:
                raise HTTPException(status_code=404, detail=f"Queue [{queue_name}] not found")
            # get timeout from config
            timeout = self.config.get("fastapi", {}).get("dequeue_timeout", 5)
            try:
                item = await asyncio.wait_for(queue.get(), timeout=timeout)
                logger.info(f"🎢 Dequeued item [{item}] from queue [{queue_name}]")
                return item
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail=f"Queue [{queue_name}] timed out")

        @self.router.get("/queue/qsize/{prefix_tag}/{queue_name}")
        async def qsize(prefix_tag: str, queue_name: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            object_name = f"{QUEUE_PREFIX}{queue_name}"
            queue = self.registries[prefix_tag].get(object_name)
            if queue is None:
                raise HTTPException(status_code=404, detail=f"Queue [{queue_name}] not found")
            return queue.qsize()

        @self.router.get("/queue/peek/{prefix_tag}/{queue_name}")
        async def peek(prefix_tag: str, queue_name: str):
            if prefix_tag not in self.registries:
                raise HTTPException(status_code=404, detail=f"Prefix tag [{prefix_tag}] not found")
            object_name = f"{QUEUE_PREFIX}{queue_name}"
            queue = self.registries[prefix_tag].get(object_name)
            if queue is None:
                raise HTTPException(status_code=404, detail=f"Queue [{queue_name}] not found")
            if queue.qsize() == 0:
                return None
            else:
                return queue._queue[0]

    async def _refresh_config_task(self):
        """
        Refresh the config
        """
        while True:
            try:
                interval = self.config.get("_refresh_config_task", {}).get("interval", 15)
                self._load_config()
            except Exception as e:
                logger.error(f"Error refreshing config: {e}")
            finally:
                await asyncio.sleep(interval)

    async def _refresh_workflow_task(self):
        """
        Refresh the workflow
        """
        while True:
            try:
                interval = self.config.get("_refresh_workflow_task", {}).get("interval", 15)
                for prefix_tag, registry in self.registries.items():
                    try:
                        registry._load_workflow_config()
                    except Exception as e:
                        logger.error(f"Error refreshing workflow for prefix tag [{prefix_tag}]: [{type(e).__name__}] {e}")
            except Exception as e:
                logger.error(f"Error refreshing workflow: {e}")
            finally:
                await asyncio.sleep(interval)

    async def _save_adapter_task(self):
        """
        Save the adapters to the config
        """
        while True:
            try:
                interval = self.config.get("_save_adapter_task", {}).get("interval", 15)
                for prefix_tag, registry in self.registries.items():
                    try:
                        await registry._save_adapter_task()
                    except Exception as e:
                        logger.error(f"Error saving adapters for prefix tag [{prefix_tag}]: [{type(e).__name__}] {e}")
            except Exception as e:
                logger.error(f"Error saving adapters: {e}")
            finally:
                await asyncio.sleep(interval)

    async def _save_queue_task(self):
        """
        Save the queues to the config
        """
        while True:
            try:
                interval = self.config.get("_save_queue_task", {}).get("interval", 15)
                for prefix_tag, registry in self.registries.items():
                    try:
                        await registry._save_queue_task()
                    except Exception as e:
                        logger.error(f"Error saving queues for prefix tag [{prefix_tag}]: [{type(e).__name__}] {e}")
            except Exception as e:
                logger.error(f"Error saving queues: {e}")
            finally:
                await asyncio.sleep(interval)

    async def run(self, fastapi: FastAPI, host: str = "0.0.0.0", port: int = 8488):
        """
        Run the server and refresh the config in parallel
        """
        try:
            # get the port from the config
            # host = self.config.get("fastapi", {}).get("host", "0.0.0.0")
            # port = self.config.get("fastapi", {}).get("port", 0)
            # s = socket.socket(); s.bind((host, port)); s.listen(2048)
            # listen_host, listen_port = s.getsockname()
            # server = uvicorn.Server(uvicorn.Config(self.get("reg.fastapi"), fd=s.fileno()))
            # logger.info(f"FastAPI server listening on: [{listen_host}:{listen_port}]")

            # write port number to {self.global_registry_dir}/{REG_PORT_FILE}
            # port_file = os.path.join(self.global_registry_dir, REG_PORT_FILE)
            # with open(port_file, "w") as f:
            #     f.write(str(listen_port))

            hostname = socket.gethostname()
            hostname_config = self.config.get("fastapi", {}).get(hostname, {})
            host = hostname_config.get("host", "0.0.0.0")
            port = hostname_config.get("port", port)
            server = uvicorn.Server(uvicorn.Config(fastapi, host=host, port=port))
            logger.info(f"FastAPI server listening on: [{host}:{port}]")

            # create a task to run the server
            tasks = []
            tasks.append(server.serve())
            logger.info(f"🚀 Starting task [server.serve]")

            tasks.append(self._refresh_config_task())
            logger.info(f"🚀 Starting task [self._refresh_config_task]")

            tasks.append(self._refresh_workflow_task())
            logger.info(f"🚀 Starting task [self._refresh_workflow_task]")

            tasks.append(self._save_queue_task())
            logger.info(f"🚀 Starting task [self._save_queue_task]")

            tasks.append(self._save_adapter_task())
            logger.info(f"🚀 Starting task [self._save_adapter_task]")

            # run the tasks
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Error running server: [{type(e).__name__}] {e}")
            return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="workflow.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8488)
    args = parser.parse_args()

    # initialize the global registry singleton
    workflowServer = WorkflowServer(config_path=args.config_path)
    replServer = ReplServer()
    replServer.init_default_vars(workflowServer.short_vars)

    fastapi = FastAPI()
    fastapi.include_router(workflowServer.router)
    fastapi.include_router(replServer.router)

    # run the main loop
    asyncio.run(workflowServer.run(fastapi, host=args.host, port=args.port))
