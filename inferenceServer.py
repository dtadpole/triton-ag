import os
import random
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

def from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


class InferenceServer:
    def __init__(self, config_path="inferenceServer.yaml"):
        self.model_config = from_yaml(config_path).get("model", {})
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
        self.model.eval()
        self.model.requires_grad_(False)
        self.worker_queue = asyncio.Queue()
        self._enable_api_endpoints()

    def _enable_api_endpoints(self):
        """
        Add the API endpoints to the router
        """
        @self.router.get("/models")
        async def get_models() -> List[str]:
            return [self.model_name]

        @self.router.get("/configs")
        async def get_configs() -> Dict[str, Any]:
            return {
                "model_name": self.model_name,
                "max_seq_len": self.max_seq_len,
                "mini_batch_size": self.mini_batch_size,
            }

        @self.router.post("/prob")
        async def get_prob(request: ProbRequest) -> ProbResponse:
            if request.model_name != self.model_name:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model ID mismatch. Requested model ID: [{request.model_name}] Allowed model ID: [{self.model_name}]"
                )
            if len(request.input_ids) > self.max_seq_len:
                raise HTTPException(
                    status_code=500,
                    detail=f"Input length [{len(request.input_ids)}] exceeds max sequence length [{self.max_seq_len}]"
                )
            result_queue = asyncio.Queue()
            self.worker_queue.put_nowait({
                "input_ids": request.input_ids,
                "result_queue": result_queue
            })
            # wait for the result from the result queue, with a timeout of x seconds
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=self.prob_timeout)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Probability inference timeout")
            return ProbResponse(model_name=request.model_name, input_ids=result['input_ids'], gen_logps=result['gen_logps'])

    async def prob_worker_task(self):
        while True:
            try:
                qsize = self.worker_queue.qsize()
                if qsize == 0:
                    await asyncio.sleep(self.check_interval)
                    continue
                work_items = []
                for _ in range(min(qsize, self.mini_batch_size)):
                    work_items.append(await self.worker_queue.get())

                max_len = max([len(item['input_ids']) for item in work_items])
                max_len = (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

                # extend each input_ids to max_len
                input_ids_list = []
                for item in work_items:
                    input_ids_list.append(item['input_ids'] + [0] * (max_len - len(item['input_ids'])))
                input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.model.device)  # (B, L)
                with torch.inference_mode():
                    per_token_logits = self.model(input_ids).logits

                # calculate log probabilities for each token
                per_token_logits = per_token_logits[:, :-1, :]           # (B, L-1, V)
                per_token_logps = per_token_logits.log_softmax(dim=-1)  # (B, L-1, V)
                per_token_logps = per_token_logps.gather(dim=2, index=input_ids[:, 1:].unsqueeze(2)).squeeze(2) # (B, L-1)
                # convert to fp32 first before moving to cpu
                per_token_logps = per_token_logps.cpu().float().numpy()        # (B, L-1)

                # for each work item, return the log probabilities for its input_ids, cut to the length of input_ids
                for i, item in enumerate(work_items):
                    result = {
                        "input_ids": item['input_ids'],
                        "gen_logps": per_token_logps[i, :len(item['input_ids'])].tolist()
                    }
                    item['result_queue'].put_nowait(result)

                logger.info(f"Probability inference worker completed [{len(work_items)}] work items. [{[len(item['input_ids']) for item in work_items]}]")

            except Exception as e:
                # log the error and the traceback, then continue
                logger.error(f"Probability inference worker task failed: [{type(e).__name__}] {e}")
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

async def test_client(config_path="inferenceServer.yaml"):
    client_config = from_yaml(config_path).get("client", {})
    host = client_config.get("host", "localhost")
    port = client_config.get("port", 8499)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://{host}:{port}/models")
        # add success emoji
        logger.info(f"✅ Got models: {response.json()}")

    # now prepare the input_ids
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    system_prompt = "You are a helpful assistant."
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "What is machine learning?",
        "How does the human brain work?",
        "What is the meaning of life?",
        "Can you tell me a joke?",
        "Show me a picture of a cat.",
        "Does the moon have a face?",
        "What is the best way to learn programming?",
        "How AI will change the world?",
    ]

    async def test_prob_task():
        while True:
            try:
                # randomly select 1 prompt from prompts
                prompt = random.choice(prompts)
                formatted_prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    tokenize=False,
                )
                input_ids = tokenizer.encode(formatted_prompt)
                logger.info(f"🔍 Input IDs: [{len(input_ids)}] {input_ids}")
                async with httpx.AsyncClient() as client:
                    response = await client.post(f"http://{host}:{port}/prob", json={
                        "model_name": "Qwen/Qwen3-32B",
                        "input_ids": input_ids
                    },
                    timeout=120,
                    )
                    logger.info(f"✅ Got response: {response.json()}")
            except Exception as e:
                logger.error(f"❌ Error: [{type(e).__name__}] {e}")
            finally:
                await asyncio.sleep(random.uniform(0.1, 1))

    # create 10 tasks
    tasks = [test_prob_task() for _ in range(4)]
    await asyncio.gather(*tasks)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--test', action='store_true', default=False)
    args.add_argument('--config', type=str, default="inferenceServer.yaml")
    args.add_argument('--host', type=str, default=None)
    args.add_argument('--port', type=int, default=None)
    args = args.parse_args()

    if args.test:
        asyncio.run(test_client())
        exit()

    # get host and port from config file, or from command line arguments if provided
    server_config = from_yaml(args.config).get("server", {})
    host = server_config.get("host", "0.0.0.0") if args.host is None else args.host
    port = server_config.get("port", 8499) if args.port is None else args.port

    fastapi = FastAPI()
    inference_server = InferenceServer(args.config)
    fastapi.include_router(inference_server.router)

    asyncio.run(inference_server.run(fastapi, host=host, port=port))
    