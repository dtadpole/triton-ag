import os
import yaml
from datetime import datetime
import json
import time
import asyncio
from typing import Optional
from loguru import logger

QUEUE_PREFIX = "queue."
ADAPTER_PREFIX = "adapter."

# create a singleton class to store global variables
class WorkflowRegistry:
    def __init__(self, prefix_tag: str, workflow_config_path: str, data_dir: str = "~/.workflow"):
        self.prefix_tag = prefix_tag
        self.workflow_config_path = workflow_config_path
        self.workflow_config = self._load_workflow_config()
        self.reg = {}  # registry of global variables, or object registry ==> {key: value}
        self.workflow_registry_dir = os.path.join(os.path.expanduser(data_dir), self.prefix_tag)
        os.makedirs(self.workflow_registry_dir, exist_ok=True)
        self._load_adapters()
        self._load_queues()
        logger.info(f"🗂️ [WorkflowRegistry] [{self.prefix_tag}] registry directory: [{self.workflow_registry_dir}]")

    def _load_workflow_config(self):
        """
        Load the workflow config
        """
        with open(self.workflow_config_path, "r") as f:
            self.workflow_config = yaml.safe_load(f)
        return self.workflow_config

    def _load_adapters(self):
        """
        Load the adapters from the config
        """
        try:
            if os.path.exists(f"{self.workflow_registry_dir}/adapters.json"):
                with open(f"{self.workflow_registry_dir}/adapters.json", "r") as f:
                    adapters = json.load(f)
            else:
                adapters = {}
            for adapter_name, adapter_value in adapters.items():
                object_name = f"{ADAPTER_PREFIX}{adapter_name}"
                if object_name not in self.reg:
                    self.put(object_name, adapter_value)
                    logger.info(f"🧩 [WorkflowRegistry] [{self.prefix_tag}] Loaded [{object_name}] with value [{adapter_value}]")
        except Exception as e:
            logger.error(f"Error loading adapters: [{type(e).__name__}] {e}")

    def _load_queues(self):
        """
        Update the queues in the config
        """
        try:
            # load previous queued items from storage
            queue_storage = {}
            try:
                # if globalRegistry/queues.json exists, load it
                queue_filename = f"{self.workflow_registry_dir}/queues.json"
                if os.path.exists(queue_filename):
                    with open(queue_filename, "r") as f:
                        queue_storage = json.load(f) # queue_storage is a dictionary of queue names and their items
            except Exception as e:
                # load from globalRegistry/queues.yaml.bak
                if os.path.exists(f"{self.workflow_registry_dir}/queues.json.bak"):
                    with open(f"{self.workflow_registry_dir}/queues.json.bak", "r") as f:
                        queue_storage = json.load(f)
            # load queues from the config
            queues_configs = self.workflow_config.get("queues", [])
            for queue_config in queues_configs:
                queue_name = queue_config.get("name")
                queue_object_name = f"{QUEUE_PREFIX}{queue_name}"
                if queue_object_name not in self.reg:
                    logger.info(f"🎢 [WorkflowRegistry] [{self.prefix_tag}] Creating queue [{queue_object_name}]")
                    self.put(queue_object_name, asyncio.Queue())
            for queue_name, queue_items in queue_storage.items():
                queue_object_name = f"{QUEUE_PREFIX}{queue_name}"
                if queue_object_name not in self.reg:
                    logger.info(f"🎢 [WorkflowRegistry] [{self.prefix_tag}] Creating queue [{queue_object_name}]")
                    self.put(queue_object_name, asyncio.Queue())
                    for item in queue_items:
                        self.get(queue_object_name).put_nowait(item)
                    logger.info(f"📦 [WorkflowRegistry] [{self.prefix_tag}] Loaded {len(queue_items)} items into queue [{queue_object_name}]")
        except Exception as e:
            logger.error(f"Error updating queues: [{type(e).__name__}] {e}")

    async def _save_adapter_task(self):
        """
        Save the adapters to the config
        """
        adapters = {}
        # if folder globalRegistry does not exist, create it
        if not os.path.exists(self.workflow_registry_dir):
            os.makedirs(self.workflow_registry_dir)
        adapter_filename = f"{self.workflow_registry_dir}/adapters.json"
        # if globalRegistry/adapters.json exists, move it to globalRegistry/adapters.json.bak
        if os.path.exists(adapter_filename):
            os.rename(adapter_filename, f"{adapter_filename}.bak")
        # save the adapters to the config
        with open(adapter_filename, "w") as f:
            for key in self.keys():
                if key.startswith(ADAPTER_PREFIX):
                    adapter_name = key.replace(ADAPTER_PREFIX, "")
                    adapter_value = self.get(key)
                    adapters[adapter_name] = adapter_value
            json.dump(adapters, f, indent=2)

    async def _save_queue_task(self):
        """
        Save the queues to the config
        """
        queue_storage = {}
        # if folder globalRegistry does not exist, create it
        if not os.path.exists(self.workflow_registry_dir):
            os.makedirs(self.workflow_registry_dir)
        # if globalRegistry/queues.json exists, move it to globalRegistry/queues.json.bak
        queue_filename = f"{self.workflow_registry_dir}/queues.json"
        if os.path.exists(queue_filename):
            os.rename(queue_filename, f"{queue_filename}.bak")
        # iterate over all the keys in the registry
        for key in self.keys():
            if key.startswith(QUEUE_PREFIX):
                queue_name = key.replace(QUEUE_PREFIX, "")
                queue_storage[queue_name] = list(self.get(key)._queue)
        # save the queues to the config
        with open(queue_filename, "w") as f:
            json.dump(queue_storage, f, indent=2)

    def keys(self):
        """
        Return a list of keys in the object registry
        """
        try:
            return list(self.reg.keys())
        except Exception as e:
            logger.error(f"Error getting keys: {e}")
            return []

    def get(self, key, last_modified_within: Optional[int]=None):
        """
        Get the value of a key in the object registry
        last_modified_within: if not None, return the value if the last modified time is within the last_modified_within seconds
        """
        try:
            item = self.reg.get(key, None)
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
            if key not in self.reg:
                item = {
                    "version": 1,
                    "timestamp": time.time(),
                    "value": value
                }
                self.reg[key] = item
                return None
            else:
                item = self.reg[key]
                old_value = item["value"]
                item["version"] += 1
                item["timestamp"] = time.time()
                item["value"] = value
                self.reg[key] = item
                return old_value
        except Exception as e:
            logger.error(f"Error getting {key}: {e}")
            return None
        
    def delete(self, key):
        """
        Delete a key from the object registry
        """
        try:
            del self.reg[key]
        except Exception as e:
            logger.error(f"Error deleting {key}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_dir", type=str, default="~/.workflow")
    parser.add_argument("--prefix_tag", type=str, default="auto")
    args = parser.parse_args()

    if args.prefix_tag == "auto":
        prefix_tag = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        prefix_tag = args.prefix_tag

    # initialize the workflow registry singleton
    reg = WorkflowRegistry(prefix_tag=prefix_tag, wd=args.workflow_dir)
