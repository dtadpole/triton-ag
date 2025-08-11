import os
import yaml
import asyncio
import traceback
from logger import logger
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from trainerUtil import rsync_file, VLLMClient

REG_PORT_FILE = ".reg.port"
REG_DIR = ".reg"

MODEL_OVERRIDE_KEY = "adapter.model_override"

class TrainerSFTBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default="~/.exemplar")
    output_dir: str = Field(default="~/.trainer")
    test_mode: bool = Field(default=False)

class TrainerRFTBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.trainer")
    test_mode: bool = Field(default=False)

class TrainerGRPOBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.trainer")
    test_mode: bool = Field(default=False)

class CodeGenEvalBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    provider_name: str
    model_name: str
    num_samples: int
    num_generations: int
    num_turns_per_generation: int = Field(default=4)
    parallel_tasks: int = Field(default=24)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/triton-ag/kernel_bench")
    output_dir: str = Field(default="~/.codeGenEval")
    template: str = Field(default="triton.1")
    logprobs: bool = Field(default=True)
    test_mode: bool = Field(default=False)

class ExemplarBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    num_generations: int
    parallel_tasks: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.exemplar")
    template: str = Field(default="triton.1")
    test_mode: bool = Field(default=False)

class CritiqueBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    parallel_tasks: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.critique")
    test_mode: bool = Field(default=False)

class ReflectionBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    parallel_tasks: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/.codeGenEval")
    output_dir: str = Field(default="~/.reflection")
    test_mode: bool = Field(default=False)

class ComposerBlock(BaseModel):
    prefix_tag: str
    epoch_id: int
    block_id: int
    input_tag: str
    provider_name: str
    model_name: str
    module_file: str
    prompt_file: str
    example_file: str
    num_samples: int
    num_generations: int
    num_turns_per_generation: int = Field(default=4)
    parallel_workers: int = Field(default=16)
    model_override: Optional[str] = Field(default=None)
    input_dir: str = Field(default="~/.inference/composer")
    output_dir: str = Field(default="~/.inference/composer")
    test_mode: bool = Field(default=False)

def get_prefix_tag(prefix_tag:str="auto", config_path:str="globalWorkflow.yaml"):
    if prefix_tag == "auto":
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        loaded_prefix_tag = yaml_data.get("global", {}).get("prefix_tag", f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # write to config_path
        return loaded_prefix_tag
    else:
        return prefix_tag

def get_global_registry_dir(prefix_tag:str="auto", trainer_dir:str="~/.trainer"):
    prefix_tag = get_prefix_tag(prefix_tag)
    return os.path.join(os.path.expanduser(trainer_dir), prefix_tag, REG_DIR)

def get_global_registry_port(prefix_tag:str="auto", trainer_dir:str="~/.trainer"):
    prefix_tag = get_prefix_tag(prefix_tag)
    port_file = os.path.join(get_global_registry_dir(prefix_tag, trainer_dir), REG_PORT_FILE)
    if os.path.exists(port_file):
        with open(port_file, "r") as f:
            return int(f.read())
    else:
        return None

class RsyncQueue:
    def __init__(self):
        from workflowClient import WorkflowClient
        self.rsync_queue = asyncio.Queue()
        self.rsync_config = self.load_config().get('rsync', {})
        self.retries = self.rsync_config.get('retries', 3)
        self.timeout = self.rsync_config.get('timeout', 10)
        self.vllm_client = VLLMClient()
        self.reg_client = WorkflowClient()

    async def get(self, timeout: int = 10):
        return await asyncio.wait_for(self.rsync_queue.get(), timeout=timeout)

    def put(self, checkpoint_path: str):
        self.rsync_queue.put_nowait(checkpoint_path)

    def load_config(self, config_path: str = "trainerMain.yaml"):
        """Load config from yaml file"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def enqueue(self, checkpoint_path: str):
        """Enqueue checkpoint path to the rsync_queue"""
        logger.info(f"📞 [RsyncQueue] Enqueue: [{checkpoint_path}]")
        self.rsync_queue.put_nowait(checkpoint_path)

    async def upload_lora_adapter(self, checkpoint_path: str):
        if not self.rsync_config.get('run_upload', False):
            logger.info(f"🔍 [RsyncQueue] Skipping upload because [run_upload] is [false]")
            return
        # get with timeout
        source_prefix = os.path.expanduser(self.rsync_config.get('rsync_source_prefix', '~/.trainer'))
        target_prefix = self.rsync_config.get('rsync_target_prefix', '192.168.1.205:.trainer')
        rsync_name = checkpoint_path.split('/')[-2] + '/' + checkpoint_path.split('/')[-1]
        if checkpoint_path.startswith(source_prefix):
            target_path = checkpoint_path.replace(source_prefix, target_prefix)
        else:
            target_path = target_prefix + checkpoint_path
        # rsync_path is the path to the rsync command
        rsync_path = self.rsync_config.get('rsync_path', 'rsync').format(rsync_name=rsync_name)
        # get a list of files to rsync
        file_list = self.rsync_config.get('file_list', [])
        for file in file_list:
            source_file = os.path.join(checkpoint_path, file)
            # target_file = os.path.join(target_path, file)
            logger.info(f"📞 [RsyncQueue] Rsyncing: [{source_file}] to [{target_path}]")
            await rsync_file(source_file, target_path + '/', rsync_path=rsync_path)
        # we are here if rsync is successful
        logger.info(f"🌐 [RsyncQueue] Rsynced: [{checkpoint_path}] to [{target_path}]")

    async def clean_up_lora_adapters(self, lora_path: str):
        retry_count = 0
        while retry_count < self.retries:
            try:
                retry_count += 1
                # find out all the loaded lora adapters
                models = await self.vllm_client.get_models()
                models = models.get('data', [])
                available_lora_adapters = [model['id'] for model in models if model['parent'] is not None]
                # fine out who is using the lora adapters
                used_lora_adapters = [lora_path]
                keys = await self.reg_client.keys()
                for key in keys:
                    if key.startswith("adapter."):
                        value = await self.reg_client.get(key, last_modified_within=LAST_MODIFIED_WITHIN)
                        used_lora_adapters.append(value)
                # unused lora adapters are the ones in available_lora_adapters but not in used_lora_adapters
                unused_lora_adapters = [lora_adapter for lora_adapter in available_lora_adapters if lora_adapter not in used_lora_adapters]
                # unload the unused lora adapters
                for lora_adapter in unused_lora_adapters:
                    await self.vllm_client.unload_lora_adapter(lora_adapter)
                    logger.info(f"🔍 [RsyncQueue] Unloaded unused lora adapter: {lora_adapter}")
                # we are here if everything executed successfully
                return
            except Exception as e:
                if retry_count < self.retries:
                    logger.info(f"🔍 [RsyncQueue] Retrying to load and unload lora adapter: {lora_path} in {2 ** retry_count} seconds")
                    await asyncio.sleep(2 ** retry_count)
                    continue
                else:
                    logger.error(f"❌ [RsyncQueue] Failed to load and unload lora adapter: {lora_path} after {self.retries} retries")
                    raise e

    async def rsync_task(self):
        """Push rsync task to the rsync_queue"""
        while True:
            try:
                checkpoint_path = await asyncio.wait_for(self.rsync_queue.get(), timeout=self.timeout)
                checkpoint_path = str(checkpoint_path)
                if self.rsync_queue.qsize() > 0:
                    logger.info(f"🔍 [RsyncQueue] Skipping [{checkpoint_path}] because queue size: [{self.rsync_queue.qsize()}]")
                    continue
                await self.upload_lora_adapter(checkpoint_path)
                logger.info(f"🔍 [RsyncQueue] Uploaded lora adapter: [{checkpoint_path}]")
                # get the last 2 parts of the checkpoint_path
                lora_path = checkpoint_path.split('/')[-2] + '/' + checkpoint_path.split('/')[-1]
                await self.vllm_client.load_lora_adapter(lora_path, lora_path)
                logger.info(f"🔍 [RsyncQueue] Loaded lora adapter: [{lora_path}]")
                # update the model_override in the global registry
                await self.reg_client.put(MODEL_OVERRIDE_KEY, lora_path)
                logger.info(f"🔍 [RsyncQueue] Model override [{MODEL_OVERRIDE_KEY}] updated to [{lora_path}]")
                # clean up the unused lora adapters
                await self.clean_up_lora_adapters(lora_path)
                logger.info(f"🔍 [RsyncQueue] Cleaned up lora adapters: [{lora_path}]")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"❌ [RsyncQueue] Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

