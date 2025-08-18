import os
import random
from collections import defaultdict
from typing import List, Dict, Any
from pydantic import BaseModel
import httpx
import torch
import uvicorn
import yaml
import json
import traceback
import asyncio
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from fastapi import APIRouter, HTTPException, FastAPI
from logger import logger
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class ProbRequest(BaseModel):
    model_name: str
    input_ids: List[int]

class ProbResponse(BaseModel):
    model_name: str
    input_ids: List[int]
    gen_logps: List[float]

class GenerateRequest(BaseModel):
    model_name: str
    prompt: str
    max_new_tokens: int
    temperature: float

class GenerateResponse(BaseModel):
    model_name: str
    text: str
    gen_token_ids: List[int]
    gen_logps: List[float]

class LoadLoraAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str

class UnloadLoraAdapterRequest(BaseModel):
    lora_name: str

def from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


class InferenceCustomServer:
    def __init__(self, config_path="inferenceCustom.yaml"):
        self.model_config = from_yaml(config_path).get("model", {})
        self.model_path = os.path.expanduser(self.model_config.get("model_path", "~/.trainer"))
        self.model_name = self.model_config.get("model_name", "Qwen/Qwen3-32B")
        self.tokenizer_name = self.model_config.get("tokenizer_name", self.model_name)
        self.max_seq_len = self.model_config.get("max_seq_len", 24576)
        self.mini_batch_size = self.model_config.get("mini_batch_size", 16)
        self.pad_to_multiple_of = self.model_config.get("pad_to_multiple_of", 8)
        self.check_interval = self.model_config.get("check_interval", 1)
        self.prob_timeout = self.model_config.get("prob_timeout", 120)
        self.router = APIRouter()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            _attn_implementation=self.model_config.get("_attn_implementation", "sdpa")
        ).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.lora_model = get_peft_model(self.model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **self.model_config.get("default_lora_config", {}))
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.worker_queue = asyncio.Queue()
        self.adapter_name_to_model_id = {}
        self._enable_api_endpoints()

    def _enable_api_endpoints(self):
        """
        Add the API endpoints to the router
        """
        @self.router.get("/models")
        async def get_models() -> dict[str, Any]:
            result = {
                "object": "list",
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "parent": None,
                }]
            }
            # add lora adapters
            for adapter_name in self.lora_model.peft_config.keys():
                if adapter_name in self.adapter_name_to_model_id:
                    model_id = self.adapter_name_to_model_id[adapter_name]
                else:
                    model_id = adapter_name
                if adapter_name != 'default':
                    result['data'].append({
                        "id": model_id,
                        "object": "model",
                        "parent": self.model_name,
                    })
            return result

        @self.router.get("/configs")
        async def get_configs() -> Dict[str, Any]:
            return {
                "base_model_name": self.model_name,
                "max_seq_len": self.max_seq_len,
                "mini_batch_size": self.mini_batch_size,
            }

        @self.router.post("/load_lora_adapter")
        async def load_lora_adapter(request: LoadLoraAdapterRequest):
            lora_path = os.path.join(self.model_path, request.lora_path)
            model_id = os.path.basename(os.path.dirname(lora_path)) + '/' + os.path.basename(lora_path)
            lora_adapter_name = model_id.replace('.', '_')
            self.adapter_name_to_model_id[lora_adapter_name] = model_id
            # get lora adapter name from path
            if lora_adapter_name in self.lora_model.peft_config.keys():
                logger.info(f"LoRA checkpoint {lora_adapter_name} already loaded")
                self.lora_model.set_adapter(lora_adapter_name)
                return {
                    'status': 'success',
                    'message': f'LoRA checkpoint [{lora_adapter_name}] already loaded'
                }
            
            # check that the path exists
            if not os.path.exists(lora_path):
                raise HTTPException(status_code=404, detail=f"LoRA checkpoint [{lora_path}] not found")
            # load new lora model if not already loaded
            logger.info(f"Loading LoRA checkpoint from {lora_path}")
            self.lora_model.load_adapter(lora_path, adapter_name=lora_adapter_name)
            # self.lora_model.eval()
            logger.info(f"Loaded LoRA checkpoint from {lora_path}")
            return {
                'status': 'success',
                'message': f'LoRA checkpoint [{lora_adapter_name}] loaded from [{lora_path}]'
            }
        
        @self.router.post("/unload_lora_adapter")
        async def unload_lora_adapter(request: UnloadLoraAdapterRequest):
            lora_adapter_name = request.lora_name.replace('.', '_')
            if lora_adapter_name not in self.lora_model.peft_config.keys():
                raise HTTPException(status_code=404, detail=f"LoRA checkpoint [{lora_adapter_name}] not found")
            self.lora_model.delete_adapter(lora_adapter_name)
            return {
                'status': 'success',
                'message': f'LoRA checkpoint [{lora_adapter_name}] unloaded'
            }

        @self.router.post("/logps")
        async def get_logps(request: ProbRequest) -> ProbResponse:
            model_or_adapter_name = request.model_name.replace('.', '_')
            if request.model_name != self.model_name and model_or_adapter_name not in self.lora_model.peft_config.keys():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model ID mismatch. Requested model ID: [{model_or_adapter_name}] Allowed model ID: [{self.model_name}]"
                )
            if len(request.input_ids) > self.max_seq_len:
                raise HTTPException(
                    status_code=500,
                    detail=f"Input length [{len(request.input_ids)}] exceeds max sequence length [{self.max_seq_len}]"
                )
            result_queue = asyncio.Queue()
            self.worker_queue.put_nowait({
                "model_name": model_or_adapter_name,
                "input_ids": request.input_ids,
                "result_queue": result_queue
            })
            # wait for the result from the result queue, with a timeout of x seconds
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=self.prob_timeout)
                if "error" in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                elif "status" in result and result['status'] != "success":
                    raise HTTPException(status_code=500, detail=result['status'])
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Probability inference timeout")
            return ProbResponse(model_name=request.model_name, input_ids=result['input_ids'], gen_logps=result['gen_logps'])

    def _calculate_prob(self, work_items):
        # calculate max_len for all work_items
        max_len = max([len(item['input_ids']) for item in work_items])
        max_len = (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
        # extend each input_ids to max_len
        input_ids_list = []
        for item in work_items:
            input_ids_list.append(item['input_ids'] + [0] * (max_len - len(item['input_ids'])))
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.model.device)  # (B, L)
        # assert that all the work_items have the same model_name
        assert all(item['model_name'] == work_items[0]['model_name'] for item in work_items), "All work items must have the same model_name"
        if work_items[0]['model_name'] == self.model_name:
            model = self.model
        elif work_items[0]['model_name'] in self.lora_model.peft_config.keys():
            model = self.lora_model
            self.lora_model.set_adapter(work_items[0]['model_name'])
        else:
            raise Exception(f"Model ID not found: [{work_items[0]['model_name']}]")
        # run the model
        with torch.inference_mode():
            per_token_logits = model(input_ids).logits

        # calculate log probabilities for each token
        per_token_logits = per_token_logits[:, :-1, :]           # (B, L-1, V)
        per_token_logps = per_token_logits.log_softmax(dim=-1)  # (B, L-1, V)
        per_token_logps = per_token_logps.gather(dim=2, index=input_ids[:, 1:].unsqueeze(2)).squeeze(2) # (B, L-1)
        # convert to fp32 first before moving to cpu
        per_token_logps = per_token_logps.cpu().float().numpy()        # (B, L-1)
        return per_token_logps

    async def prob_worker_task(self):
        while True:
            try:
                qsize = self.worker_queue.qsize()
                if qsize == 0:
                    await asyncio.sleep(self.check_interval)
                    continue
                work_items_by_model = defaultdict(list)
                for _ in range(qsize):
                    item = await self.worker_queue.get()
                    work_items_by_model[item['model_name']].append(item)

                while True:
                    # breat if all the values for the work_items_by_model are empty
                    if all(len(v) == 0 for v in work_items_by_model.values()):
                        break
                    # randomly pick a key from work_items, using probability from length of the value of corresponding key
                    model_item_lengths = [(k, len(v)) for k, v in work_items_by_model.items()]
                    selected_model_name, selected_length = random.choices(model_item_lengths, weights=[v for k, v in model_item_lengths])[0]

                    work_items = work_items_by_model[selected_model_name][:self.mini_batch_size]
                    work_items_by_model[selected_model_name] = work_items_by_model[selected_model_name][self.mini_batch_size:]
                    try:
                        per_token_logps = self._calculate_prob(work_items)
                        # for each work item, return the log probabilities for its input_ids, cut to the length of input_ids
                        for i, item in enumerate(work_items):
                            result = {
                                "input_ids": item['input_ids'],
                                "gen_logps": per_token_logps[i, :len(item['input_ids'])].tolist(),
                                "status": "success",
                            }
                            item['result_queue'].put_nowait(result)

                        logger.info(f"Probability worker [{selected_model_name}] completed [{len(work_items)}] items. [{[len(item['input_ids']) for item in work_items]}]")
                    except Exception as e:
                        for i, item in enumerate(work_items):
                            result = {
                                "input_ids": item['input_ids'],
                                "gen_logps": [],
                                "status": "error",
                                "error": f'[{type(e).__name__}]: {e}',
                            }
                            item['result_queue'].put_nowait(result)
                        logger.error(f"Probability worker [{selected_model_name}] task failed. [{[len(item['input_ids']) for item in work_items]}] [{type(e).__name__}] {e}")
                        logger.error(traceback.format_exc())
                        continue

            except Exception as e:
                # log the error and the traceback, then continue
                logger.error(f"Probability worker task had UNEXPECTED ERROR: [{type(e).__name__}] {e}")
                logger.error(traceback.format_exc())
                continue
            finally:
                await asyncio.sleep(self.check_interval)

    async def run(self, fastapi: FastAPI, host: str, port: int):
        server = uvicorn.Server(uvicorn.Config(fastapi, host=host, port=port))

        tasks = [
            asyncio.create_task(self.prob_worker_task()),
            asyncio.create_task(server.serve())
        ]
        # run the tasks
        await asyncio.gather(*tasks)

    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default="inferenceCustom.yaml")
    args.add_argument('--host', type=str, default=None)
    args.add_argument('--port', type=int, default=None)
    args = args.parse_args()

    # get host and port from config file, or from command line arguments if provided
    server_config = from_yaml(args.config).get("server", {})
    host = server_config.get("host", "0.0.0.0") if args.host is None else args.host
    port = server_config.get("port", 8092) if args.port is None else args.port

    fastapi = FastAPI()
    inference_server = InferenceCustomServer(args.config)
    fastapi.include_router(inference_server.router, prefix="/v1")

    asyncio.run(inference_server.run(fastapi, host=host, port=port))
    