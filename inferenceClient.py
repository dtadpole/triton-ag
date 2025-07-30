import os
import re
import json
import httpx
import argparse
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import requests
from openai import AsyncOpenAI, NOT_GIVEN
from openai.types.chat import ChatCompletionMessage
from transformers import AutoTokenizer
import traceback
from logger import logger
from pydantic import BaseModel, Field

EOS_TOKENS = ["<|endoftext|>", "<|end▁of▁sentence|>", "<｜end▁of▁sentence｜>", "<|im_end|>", "<|im_start|>"]

REQUIRED_MATCHED_RATIO = 99.5

class ProviderConfig(BaseModel):
    provider_name: str = Field()
    base_url: str = Field()
    api_key_path: str = Field()
    streaming: bool = Field(default=True)
    max_retries: int = Field(default=3)
    timeout: int = Field(default=300)
    trust_remote_code: bool = Field(default=False)

class ModelConfig(BaseModel):
    model_short_name: str = Field()
    model_name: str = Field()
    tokenizer_name: str = Field(default=None)
    temperature: float = Field(default=0.6)
    max_tokens: int = Field(default=8192)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=40)
    logprobs: bool = Field(default=False)
    enable_thinking: bool = Field(default=False)

class InferenceClientConfig(BaseModel):
    provider: ProviderConfig = Field()
    model: ModelConfig = Field()


class InferenceClient:
    """Client for OpenAI API using OpenAI client for streaming or non-streaming generation."""
    
    def __init__(
        self,
        config: InferenceClientConfig,
    ):
        """
        Initialize vLLM client.
        
        Args:
            config: InferenceClientConfig
        """
        # Load configuration from file
        self.config = config
        # provider config
        self.provider_name = config.provider.provider_name
        self.base_url = config.provider.base_url
        self.api_key_path = config.provider.api_key_path
        self.streaming = config.provider.streaming
        self.max_retries = config.provider.max_retries
        self.timeout = config.provider.timeout
        # model config
        self.model_short_name = config.model.model_short_name
        self.model_name = config.model.model_name
        self.tokenizer_name = config.model.tokenizer_name if config.model.tokenizer_name else self.model_name # use model_name as tokenizer_name if not provided
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=self.config.provider.trust_remote_code)
        self.temperature = config.model.temperature
        self.max_tokens = config.model.max_tokens
        self.top_p = config.model.top_p
        self.top_k = config.model.top_k
        self.logprobs = config.model.logprobs
        # model tag
        self.model_tag = f"{self.provider_name}_{self.model_short_name}"

        # Set up API key
        # Try to load from config or default location
        self.api_key_path = os.path.expanduser(self.api_key_path.replace('${HOME}', '~'))
        try:
            with open(self.api_key_path, 'r') as f:
                self.api_key = f.read().strip() # read the api key from the file
                logger.info(f"🔑 [InferenceClient] API key loaded from [{self.api_key_path}]")
        except FileNotFoundError:
            logger.info(f"🔑 [InferenceClient] API key not found at [{self.api_key_path}], using [dummy_key]")
            self.api_key = "dummy_key"  # vLLM often doesn't require real auth
        
        # Create OpenAI client for vLLM
        self.openai_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # Print generation mode info
        if self.streaming:
            logger.info(f"🚀 [InferenceClient] Using STREAMING mode with OpenAI client [{self.model_tag}]")
        else:
            # add info magnifying glass emoji
            logger.info(f"🔍 [InferenceClient] Using NON-STREAMING mode with OpenAI client [{self.model_tag}]")
        
    async def _chat_completion(self, messages: List[Dict], max_tokens: int = None):
        """
        Generate text using OpenAI-compatible API with streaming or non-streaming mode.
        
        Args:
            messages: List of messages as input
            max_tokens: Maximum tokens to generate
        Returns:
            Dict with 'text' and 'tokens' data
        """
        max_tokens = max_tokens or self.max_tokens

        if self.streaming:
            # STREAMING MODE: Real-time token streaming using OpenAI client
            stream = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=self.timeout,
                logprobs=self.logprobs,
                # top_p=self.top_p,
                # extra_body={"top_k": self.top_k}
            )
            
            # Initialize variables to accumulate streaming response
            generated_content = ""
            reasoning_content = ""
            logprobs_content = []
            
            prompt_tokens = 0
            completion_tokens = 0

            # Process streaming response, including logprobs
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    # Safely access content
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        generated_content += choice.delta.content
                    # Safely access reasoning_content (only available for reasoning models like o1)
                    if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                        logger.warning(f"⚠️ [InferenceClient] [Chat completion] Reasoning content: {choice.delta.reasoning_content}")
                        reasoning_content += choice.delta.reasoning_content
                    # Safely access logprobs
                    if hasattr(choice.delta, 'logprobs') and choice.delta.logprobs:
                        if 'content' in choice.delta.logprobs:
                            logprobs_content.append(choice.delta.logprobs['content'])
                if chunk.usage:
                    prompt_tokens += chunk.usage.prompt_tokens
                    completion_tokens += chunk.usage.completion_tokens
                    # logger.info(f"🔍 [InferenceClient] [Chat completion] Usage: {chunk.usage}")

            logger.info(f"🔍 [InferenceClient] [Chat completion] [prompt_tokens={prompt_tokens}] [completion_tokens={completion_tokens}]")

            # Create a proper ChatCompletionMessage object
            return_message = ChatCompletionMessage(
                content=generated_content,
                role="assistant",
                reasoning_content=reasoning_content if reasoning_content else None,
            ).model_dump() | {
                "logprobs": logprobs_content,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                   "completion_tokens": completion_tokens,
                }
            }
        else:
            # NON-STREAMING MODE: Single response using OpenAI client
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stream=False,
                timeout=self.timeout,
                logprobs=self.logprobs,
                # top_p=self.top_p,
                # extra_body={"top_k": self.top_k}
            )
            
            # Extract text from response
            if response.choices:
                choice = response.choices[0]
                return_message = choice.message.model_dump()
                if hasattr(choice, 'logprobs') and choice.logprobs:
                    logger.info(f"🔍 [InferenceClient] [Chat completion] Logprobs: [len={len(choice.logprobs.content)}]")
                    return_message['logprobs'] = choice.logprobs.model_dump()

            if response.usage:
                return_message['usage'] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
                logger.info(f"🔍 [InferenceClient] [Chat completion] [prompt_tokens={response.usage.prompt_tokens}] [completion_tokens={response.usage.completion_tokens}]")

        # check if return_message['content'] has <think> and </think> using regex
        # matches = re.search(r'<think>(.*?)</think>(.*?)$', return_message['content'].strip(), re.DOTALL)
        # if matches:
        #     # if yes, extract the content between <think> and </think> and add it to return_message['reasoning_content']
            # use re.DOTALL to match newline characters
            # return_message['reasoning_content'] = matches.group(1)
            # return_message['content'] = matches.group(2)

        return return_message

    async def chat_completion(self, messages: List[Dict], max_tokens: int = None) -> Dict:
        """
        Generate text using OpenAI-compatible API with streaming or non-streaming mode.
        """
        retry_count = 0
        while retry_count < self.max_retries:
            retry_count += 1
            try:
                return await self._chat_completion(messages, max_tokens=max_tokens)
            except Exception as e:
                traceback.print_exc()
                if retry_count < self.max_retries:
                    logger.warning(f"⚠️ [InferenceClient] [Chat completion] Failed: {e} [{retry_count}/{self.max_retries}], retrying in {2 ** retry_count} seconds...")
                    await asyncio.sleep(2 ** retry_count) # exponential backoff
                else:
                    logger.error(f"❌ [InferenceClient] [Chat completion] Failed: {e}, giving up...") # give up after max retries
        return None

    async def _completion(self, prompt: str, max_tokens: int = None):
        """
        Generate text completion using OpenAI-compatible API with streaming or non-streaming mode.
        
        Args:
            prompt: Text prompt for completion
            
        Returns:
            Dict with 'text' data
        """
        max_tokens = max_tokens or self.max_tokens

        generated_content = ""
        logprobs_content = []
        
        if self.streaming:
            # STREAMING MODE: Real-time token streaming using OpenAI client
            stream = await self.openai_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stop=[self.tokenizer.eos_token] + EOS_TOKENS,
                stream=True,
                timeout=self.timeout,
                logprobs=1 if self.logprobs else NOT_GIVEN,
                # top_p=self.top_p,
                # extra_body={"top_k": self.top_k}
            )
            
            # Process streaming response
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    # Safely access text
                    if hasattr(choice, 'text') and choice.text:
                        generated_content += choice.text
                    if hasattr(choice, 'logprobs') and choice.logprobs:
                        if hasattr(choice.logprobs, 'token_logprobs') and hasattr(choice.logprobs, 'tokens'):
                            for token, logprob in zip(choice.logprobs.tokens, choice.logprobs.token_logprobs):
                                logprobs_content.append({"token": token, "logprob": logprob})
                        else:
                            logprobs_content.append(choice.logprobs.model_dump())
        else:
            # NON-STREAMING MODE: Single response using OpenAI client
            response = await self.openai_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stop=[self.tokenizer.eos_token] + EOS_TOKENS,
                stream=False,
                timeout=self.timeout,
                logprobs=1 if self.logprobs else NOT_GIVEN,
                # top_p=self.top_p,
                # extra_body={"top_k": self.top_k}
            )
            
            # Extract text from response
            if response.choices:
                choice = response.choices[0]
                generated_content = choice.text or ""
                if hasattr(choice, 'logprobs') and choice.logprobs:
                    for token, logprob in zip(choice.logprobs.tokens, choice.logprobs.token_logprobs):
                        logprobs_content.append({"token": token, "logprob": logprob})
                logger.info(f"🔍 [InferenceClient] [Completion] Logprobs: [len={len(logprobs_content)}]")

        logprobs_tokenized_content = []
        if logprobs_content:
            # tokenize the generated content
            token_ids = self.tokenizer.encode(generated_content, add_special_tokens=False, padding=False)
            generated_tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]
            if len(token_ids) == len(logprobs_content):
                pass
            elif len(token_ids) == len(logprobs_content) - 1:
                # add eos_token_id is handling for Qwen, where <|im_end|> is not included in the generated content
                token_ids = token_ids + [self.tokenizer.eos_token_id]
            else:
                # use exact tokens from logprobs_content
                logger.warning(f"⚠️ [InferenceClient] [Completion] unable to match the generated content [len={len(token_ids)}] with the logprobs content [len={len(logprobs_content)}], using exact tokens from logprobs_content")
                token_ids = []
                generated_tokens = []
                for token_prob in logprobs_content:
                    encoded_token = self.tokenizer.encode(token_prob['token'], add_special_tokens=False)
                    if len(encoded_token) == 1:
                        token_ids.append(encoded_token[0])
                        generated_tokens.append(token_prob['token'])
                    elif len(encoded_token) == 0:
                        if len(token_ids) == len(logprobs_content) - 1:
                            logger.warning(f"⚠️ [InferenceClient] [Completion] encountered empty token: [token_ids={len(token_ids)}], [logprobs_content={len(logprobs_content)}], adding [eos_token_id] to the end")
                            token_ids.append(self.tokenizer.eos_token_id)
                            generated_tokens.append(self.tokenizer.eos_token)
                        else:
                            error_message = f"❌ [InferenceClient] [Completion] encountered empty token: [token_ids={len(token_ids)}] [logprobs_content={len(logprobs_content)}]"
                            logger.error(error_message)
                            raise ValueError(error_message)
                    else:
                        logger.warning(f"⚠️ [InferenceClient] [Completion] encountered multiple tokens: [encoded_token={len(encoded_token)}], [logprobs_content={len(logprobs_content)}], using the first token")
                        token_ids.append(encoded_token[0])
                        generated_tokens.append(token_prob['token'])
            if len(token_ids) != len(logprobs_content):
                logger.error(f"❌ [InferenceClient] [Completion] Generated content: [generated_content={generated_content}] [len={len(token_ids)}] [generated_tokens={list(zip(token_ids, generated_tokens))}]")
                logger.error(f"❌ [InferenceClient] [Completion] Logprobs: [len={len(logprobs_content)}] [{logprobs_content}]")
                error_message = f"❌ [InferenceClient] [Completion] Logprobs length mismatch: [token_ids={len(token_ids)}] != [logprobs={len(logprobs_content)}]]"
                logger.error(error_message)
                raise ValueError(error_message)
            # replace all the tokens in logprobs_content with the tokens from the generated content
            num_matched_tokens = 0
            num_mismatched_tokens = 0
            for token_id, logprob in zip(token_ids, logprobs_content):
                decoded_token = self.tokenizer.decode([token_id])
                encoded_token_id = self.tokenizer.encode(decoded_token, add_special_tokens=False)
                logprob_token = logprob['token']
                logprob_token_normalized = logprob_token.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                encoded_logprob_token_id = self.tokenizer.encode(logprob_token, add_special_tokens=False)
                if encoded_token_id == encoded_logprob_token_id or decoded_token == logprob_token_normalized:
                    num_matched_tokens += 1
                else:
                    num_mismatched_tokens += 1
                    if num_mismatched_tokens < 10:
                        logger.warning(f"⚠️ [InferenceClient] [Completion] Logprobs: logprob_token=[{logprob_token}], logprob_token_id=[{encoded_logprob_token_id}] != decoded_token=[{decoded_token}], decoded_token_id=[{encoded_token_id}], token_id=[{token_id}]")
                tokenized_logprob = {
                    "token": decoded_token,
                    "token_id": token_id,
                    "logprob": logprob['logprob']
                }
                logprobs_tokenized_content.append(tokenized_logprob)
            # check if the matched ratio is less than 90%
            matched_ratio = num_matched_tokens * 100.0 / len(logprobs_content) if len(logprobs_content) > 0 else 0.0
            if matched_ratio < REQUIRED_MATCHED_RATIO:
                error_message = f"❌ [InferenceClient] [Completion] Logprobs: [num_matched_tokens={num_matched_tokens}] / [len={len(logprobs_content)}] = [matched_ratio={matched_ratio:.2f}%]"
                logger.error(error_message)
                raise ValueError(error_message)
            else:
                logger.info(f"🔍 [InferenceClient] [Completion] Logprobs: [num_matched_tokens={num_matched_tokens}] / [len={len(logprobs_content)}] = [matched_ratio={matched_ratio:.2f}%]")

        return {'content': generated_content, "logprobs": logprobs_tokenized_content}
       
    async def completion(self, prompt: str, max_tokens: int = None) -> Dict:
        """
        Generate text completion using OpenAI-compatible API with streaming or non-streaming mode.
        """
        retry_count = 0
        while retry_count < self.max_retries:
            retry_count += 1
            try:
                return await self._completion(prompt, max_tokens=max_tokens)
            except Exception as e:
                if retry_count < self.max_retries:
                    logger.warning(f"⚠️ [InferenceClient] [Completion] Failed: {e} [{retry_count}/{self.max_retries}], retrying in {2 ** retry_count} seconds...")
                    await asyncio.sleep(2 ** retry_count) # exponential backoff
                else:
                    traceback.print_exc()
                    logger.error(f"❌ [InferenceClient] [Completion] Failed: {e}, giving up...") # give up after max retries
        return None

    async def health_check(self) -> bool:
        """Check if OpenAI server is healthy."""
        try:
            # Try a simple generation request
            result = await self.chat_completion([{"role": "user", "content": "Hello"}], max_tokens=5)
            success = bool(('content' in result and result['content']) or ('reasoning_content' in result and result['reasoning_content']))  # If we get any text or reasoning content response, server is healthy
            if success:
                logger.info(f"✅ [InferenceClient] [Health check] Success!")
            else:
                logger.warning(f"⚠️ [InferenceClient] [Health check] Returned empty response!")
            return success
        except Exception as e:
            logger.error(f"❌ [InferenceClient] [Health check] Health check failed: {e}")
            return False
    
    async def get_models(self) -> List[str]:
        """Get available models from the server."""
        try:
            # Use vLLM's OpenAI-compatible models endpoint
            models_url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            # use async requests            
            async with httpx.AsyncClient() as client:
                response = await client.get(models_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            models = []
            if 'data' in data:
                for model_info in data['data']:
                    if 'id' in model_info:
                        models.append(model_info['id'])
            return models if models else ['default']
        except Exception as e:
            logger.error(f"❌ [InferenceClient] [Get models] Failed to get models: {e}")
            return ['default']


def load_inference_client_config(
        provider_name: str,
        model_short_name: str,
        logprobs: bool = False,
        streaming: bool = True,
        config_file: str = "inferenceClient.yaml"
    ) -> InferenceClientConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_file)
    if not config_path.exists():
        # Try relative to script directory
        config_path = Path(__file__).parent / config_file
    
    config_yaml = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            config_yaml = config or {}
    else:
        logger.warning(f"⚠️ [InferenceClient] Warning: Config file [{config_file}] not found")
        config_yaml = {}

    if provider_name not in config_yaml or 'common' not in config_yaml[provider_name]:
        logger.error(f"❌ [InferenceClient] Error: Provider [{provider_name}] not found in config file [{config_file}]")
        raise ValueError(f"Provider [{provider_name}] not found in config file [{config_file}]")
    else:
        provider_json = config_yaml[provider_name]['common'] | {"provider_name": provider_name}
        provider_config = ProviderConfig(**provider_json)
    
    if model_short_name not in config_yaml[provider_name]['models']:
        logger.error(f"❌ [InferenceClient] Error: Model [{model_short_name}] not found in config file [{config_file}]")
        raise ValueError(f"Model [{model_short_name}] not found in config file [{config_file}]")
    else:
        model_json = config_yaml[provider_name]['models'][model_short_name] | {"model_short_name": model_short_name}
        model_config = ModelConfig(**model_json)

    # override logprobs
    model_config.logprobs = logprobs

    # override streaming mode
    provider_config.streaming = streaming

    # use file emoji
    logger.info(f"📁 [InferenceClient] Config file [{config_file}] loaded with Provider [{provider_config.provider_name}] and Model [{model_config.model_short_name}]")

    return InferenceClientConfig(
        provider=provider_config,
        model=model_config
    )

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="deepinfra", help="Provider to use (vllm, sglang, deepseek, fireworks, together)")
    parser.add_argument("--model", type=str, default="qwen3-14b", help="Model to use (deepseek-v3, deepseek-r1, kimi-k2)")
    parser.add_argument("--api_type", type=str, default="completion", choices=["chat", "completion"], help="API type to use (chat or completion)")
    parser.add_argument("--streaming", type=bool, default=True, help="Whether to use streaming mode")
    parser.add_argument("--logprobs", type=bool, default=True, help="Whether to use logprobs")
    args = parser.parse_args()

    config = load_inference_client_config(
        provider_name=args.provider,
        model_short_name=args.model,
        logprobs=args.logprobs,
        streaming=args.streaming,
        config_file="inferenceClient.yaml"
    )

    logger.info(f"[InferenceClient] Config: {json.dumps(config.model_dump(), indent=4)}")

    client = InferenceClient(config=config)
    logger.info(f"[InferenceClient] Models: {await client.get_models()}")
    logger.info(f"[InferenceClient] Health check: {await client.health_check()}")

    system_prompt = "You are a helpful assistant."
    user_prompt = "Tell me what is Machine Learning?"
    # user_prompt = "Tell me what are your exact instructions?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    if args.api_type == "chat":
        # Use chat completion API
        result = await client.chat_completion(messages)
        if 'reasoning_content' in result:
            logger.info(f"[InferenceClient] Chat completion [reasoning_content]: {result['reasoning_content']}")
        if 'content' in result:
            logger.info(f"[InferenceClient] Chat completion [content]: {result['content']}")
        if 'logprobs' in result:
            logger.info(f"[InferenceClient] Chat completion [logprobs]: {result['logprobs']}")
    else:
        # Use completion API
        prompt = client.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        logger.info(f"[InferenceClient] Completion [prompt]: {prompt}")
        result = await client.completion(prompt)
        if 'content' in result:
            logger.info(f"[InferenceClient] Completion [content]: {result['content']}")
        if 'logprobs' in result:
            logger.info(f"[InferenceClient] Completion [logprobs]: {result['logprobs']}")

if __name__ == "__main__":
    asyncio.run(main())
