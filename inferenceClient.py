import os
import re
import json
import httpx
import argparse
import asyncio
from typing import Dict, List, Optional, Any
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

REQUIRED_MATCHED_RATIO = 99.75

class InferenceClient:
    """Client for OpenAI API using OpenAI client for streaming or non-streaming generation."""

    def __init__(
        self,
        model_short_name: str,
        config_file: str = "inferenceClient.yaml",
    ):
        """
        Initialize vLLM client.

        Args:
            config: InferenceClientConfig
        """
        # Load configuration from file
        self.config_file = config_file
        # cache for provider and model configs
        self.provider_config_cache = {}
        self.model_config_cache = {}
        # model tag
        self.model_short_name = model_short_name
        self.model_tag = f"{self.model_short_name}"
        # model config
        self.model_config = self._model_config_from_yaml(self.model_short_name, config_file=config_file)
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['tokenizer_name'], trust_remote_code=True)


    def _model_config_from_yaml(
        self,
        model_short_name: str,
        provider_name: Optional[str] = None,
        config_file: str="inferenceClient.yaml",
    ) -> Dict[str, Any]:
        """Load model config from YAML file."""
        # check cache
        if model_short_name in self.model_config_cache:
            return self.model_config_cache[model_short_name]
        # if not found, load from yaml file
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        if model_short_name not in config.get('models', {}):
            raise ValueError(f"Model [{model_short_name}] not found in config file [{config_file}]")
        model_config = config.get('models', {}).get(model_short_name, {})
        if provider_name:
            if provider_name not in config.get('providers', {}):
                raise ValueError(f"Provider [{provider_name}] not found in config file [{config_file}]")
            provider_config = config.get('providers', {}).get(provider_name, {})
            provider_specific_model_config = provider_config.get('models', {}).get(model_short_name, {})
            model_config = {**model_config, **provider_specific_model_config}
        
        model_name = model_config.get('model_name', model_short_name)
        tokenizer_name = model_config.get('tokenizer_name', model_name)
        temperature = model_config.get('temperature', 0.6)
        max_tokens = model_config.get('max_tokens', 16384)
        truncate_prompt_tokens = model_config.get('truncate_prompt_tokens', 8192)
        top_p = model_config.get('top_p', 1.0)
        top_k = model_config.get('top_k', 40)
        result = {
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "truncate_prompt_tokens": truncate_prompt_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }

        logger.info(f"🔍 [InferenceClient] Model config [{provider_name}] [{model_short_name}]: {json.dumps(result, indent=4)}")

        return result

    def _provider_config_from_yaml(
        self,
        provider_name: str,
        config_file: str="inferenceClient.yaml",
    ) -> Dict[str, Any]:
        """Load OpenAI client from YAML file."""
        if provider_name in self.provider_config_cache:
            return self.provider_config_cache[provider_name]

        # if not found, load from yaml file
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        providers_config = config.get('providers', {})
        if provider_name not in providers_config:
            raise ValueError(f"Provider [{provider_name}] not found in config file [{config_file}]")
        provider_config = providers_config[provider_name]

        if 'common' not in provider_config:
            raise ValueError(f"Common config not found for provider [{provider_name}] in config file [{config_file}]")

        common_config = provider_config['common']

        base_url = common_config.get('base_url', 'http://localhost:8091')
        api_key_path = os.path.expanduser(common_config.get('api_key_path', '~/.keys/local.api.key'))
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip() # read the api key from the file
                logger.info(f"🔑 [InferenceClient] API key loaded from [{api_key_path}]")
        except FileNotFoundError:
            logger.info(f"🔑 [InferenceClient] API key not found at [{api_key_path}], using [dummy] key")
            api_key = "dummy"  # vLLM often doesn't require real auth

        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        streaming = common_config.get('streaming', True)
        if streaming:
            logger.info(f"🚀 [InferenceClient] Using STREAMING mode with OpenAI client [{provider_name}]")
        else:
            # add info magnifying glass emoji
            logger.info(f"🔍 [InferenceClient] Using NON-STREAMING mode with OpenAI client [{provider_name}]")

        result = {
            "provider_name": provider_name,
            "openai_client": openai_client,
            "base_url": base_url,
            "api_key": api_key,
            "streaming": streaming,
            "retry_count": common_config.get('retry_count', 3),
            "initial_retry_interval": common_config.get('initial_retry_interval', 3),
            "timeout": common_config.get('timeout', 300),
            "trust_remote_code": common_config.get('trust_remote_code', False),
        }

        self.provider_config_cache[provider_name] = result
        logger.info(f"🔍 [InferenceClient] OpenAI client [{provider_name}] created with base_url [{base_url}] and api_key [{api_key_path}]")

        return result

    async def _chat_completion(
        self,
        provider_name: str,
        messages: List[Dict],
        max_tokens: int = None,
        logprobs: bool = False,
        model_override: Optional[str] = None,
        openai_client: Optional[AsyncOpenAI] = None,
        streaming: Optional[bool] = None,
        timeout: Optional[int] = None,
    ):
        """
        Generate text using OpenAI-compatible API with streaming or non-streaming mode.

        Args:
            messages: List of messages as input
            max_tokens: Maximum tokens to generate
        Returns:
            Dict with 'text' and 'tokens' data
        """
        model_config = self._model_config_from_yaml(self.model_short_name, provider_name=provider_name)

        if streaming:
            # STREAMING MODE: Real-time token streaming using OpenAI client
            stream = await openai_client.chat.completions.create(
                model=model_override or model_config['model_name'],
                messages=messages,
                temperature=model_config['temperature'],
                max_tokens=max_tokens or model_config['max_tokens'],
                stream=True,
                timeout=timeout,
                logprobs=1 if logprobs else NOT_GIVEN,
                # top_p=model_config['top_p'],
                # extra_body={"top_k": model_config['top_k']}
                # extra_body={"truncate_prompt_tokens": model_config['truncate_prompt_tokens']}
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
                        # logger.warning(f"⚠️ [InferenceClient] [Chat completion] Reasoning content: {choice.delta.reasoning_content}")
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
            response = await openai_client.chat.completions.create(
                model=model_override or model_config['model_name'],
                messages=messages,
                temperature=model_config['temperature'],
                max_tokens=max_tokens or model_config['max_tokens'],
                stream=False,
                timeout=timeout,
                logprobs=1 if logprobs else NOT_GIVEN,
                # top_p=model_config['top_p'],
                # extra_body={"top_k": model_config['top_k']}
                # extra_body={"truncate_prompt_tokens": model_config['truncate_prompt_tokens']}
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

        return return_message

    async def chat_completion(
        self,
        provider_name: str,
        messages: List[Dict],
        max_tokens: int = None,
        logprobs: bool = False,
        model_override: Optional[str] = None,
    ) -> Dict:
        """
        Generate text using OpenAI-compatible API with streaming or non-streaming mode.
        """
        openai_client_dict = self._provider_config_from_yaml(provider_name)
        openai_client = openai_client_dict['openai_client']
        streaming = openai_client_dict['streaming']
        num_retries = openai_client_dict['retry_count']
        initial_retry_interval = openai_client_dict['initial_retry_interval']
        timeout = openai_client_dict['timeout']

        retry_count = 0
        while retry_count < num_retries:
            retry_count += 1
            try:
                return await self._chat_completion(
                    provider_name=provider_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    model_override=model_override,
                    openai_client=openai_client,
                    streaming=streaming,
                    timeout=timeout,
                )
            except Exception as e:
                traceback.print_exc()
                if retry_count < num_retries:
                    sleep_time = initial_retry_interval ** retry_count
                    logger.warning(f"⚠️ [InferenceClient] [Chat completion] Failed: {e} [{retry_count}/{num_retries}], retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time) # exponential backoff
                else:
                    logger.error(f"❌ [InferenceClient] [Chat completion] Failed: {e}, giving up...") # give up after max retries
        return None

    async def _completion(
        self,
        provider_name: str,
        prompt: str,
        max_tokens: int = None,
        logprobs: bool = False,
        model_override: Optional[str] = None,
        openai_client: Optional[AsyncOpenAI] = None,
        streaming: Optional[bool] = None,
        timeout: Optional[int] = None,
    ):
        """
        Generate text completion using OpenAI-compatible API with streaming or non-streaming mode.

        Args:
            prompt: Text prompt for completion

        Returns:
            Dict with 'text' data
        """
        model_config = self._model_config_from_yaml(self.model_short_name, provider_name=provider_name)

        generated_content = ""
        logprobs_content = []

        if streaming:
            # STREAMING MODE: Real-time token streaming using OpenAI client
            stream = await openai_client.completions.create(
                model=model_override or model_config['model_name'],
                prompt=prompt,
                temperature=model_config['temperature'],
                max_tokens=max_tokens or model_config['max_tokens'],
                stop=[self.tokenizer.eos_token] + EOS_TOKENS,
                stream=True,
                timeout=timeout,
                logprobs=1 if logprobs else NOT_GIVEN,
                # top_p=model_config['top_p'],
                # extra_body={"top_k": model_config['top_k']}
                # extra_body={"truncate_prompt_tokens": model_config['truncate_prompt_tokens']}
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
            response = await openai_client.completions.create(
                model=model_override or model_config['model_name'],
                prompt=prompt,
                temperature=model_config['temperature'],
                max_tokens=max_tokens or model_config['max_tokens'],
                stop=[self.tokenizer.eos_token] + EOS_TOKENS,
                stream=False,
                timeout=timeout,
                logprobs=1 if logprobs else NOT_GIVEN,
                # top_p=model_config['top_p'],
                # extra_body={"top_k": model_config['top_k']}
                # extra_body={"truncate_prompt_tokens": model_config['truncate_prompt_tokens']}
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
            for logprob_content in logprobs_content:
                raw_token = logprob_content['token']
                if ':' not in raw_token:
                    raise ValueError(f"⚠️ [InferenceClient] [Completion] Logprobs: cannot parse token id from [token={raw_token}]")
                try:
                    token_id = int(raw_token.split(':')[-1])
                except ValueError:
                    raise ValueError(f"⚠️ [InferenceClient] [Completion] Logprobs: cannot parse token id from [token={raw_token}]")
                logprobs_tokenized_content.append({
                    "token": raw_token,
                    "token_id": token_id,
                    "logprob": logprob_content['logprob'],
                })

        return {'content': generated_content, "logprobs": logprobs_tokenized_content}

    async def completion(
        self,
        provider_name: str,
        prompt: str,
        max_tokens: int = None,
        logprobs: bool = False,
        model_override: Optional[str] = None,
    ) -> Dict:
        """
        Generate text completion using OpenAI-compatible API with streaming or non-streaming mode.
        """
        openai_client_dict = self._provider_config_from_yaml(provider_name)
        openai_client = openai_client_dict['openai_client']
        streaming = openai_client_dict['streaming']
        num_retries = openai_client_dict['retry_count']
        initial_retry_interval = openai_client_dict['initial_retry_interval']
        timeout = openai_client_dict['timeout']

        retry_count = 0
        while retry_count < num_retries:
            retry_count += 1
            try:
                return await self._completion(
                    provider_name=provider_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    model_override=model_override,
                    openai_client=openai_client,
                    streaming=streaming,
                    timeout=timeout,
                )
            except Exception as e:
                if retry_count < num_retries:
                    sleep_time = initial_retry_interval ** retry_count
                    logger.warning(f"⚠️ [InferenceClient] [Completion] Failed: [{type(e).__name__}] {e} [{retry_count}/{num_retries}], retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time) # exponential backoff
                else:
                    traceback.print_exc()
                    logger.error(f"❌ [InferenceClient] [Completion] Failed: {e}, giving up...") # give up after max retries
        return None

    async def health_check(self, provider_name: str) -> bool:
        """Check if OpenAI server is healthy."""
        try:
            # Try a simple generation request
            result = await self.chat_completion(provider_name, [{"role": "user", "content": "Hello"}], max_tokens=5)
            success = bool(('content' in result and result['content']) or ('reasoning_content' in result and result['reasoning_content']))  # If we get any text or reasoning content response, server is healthy
            if success:
                logger.info(f"✅ [InferenceClient] [Health check] Success!")
            else:
                logger.warning(f"⚠️ [InferenceClient] [Health check] Returned empty response!")
            return success
        except Exception as e:
            logger.error(f"❌ [InferenceClient] [Health check] Health check failed: {e}")
            return False

    async def get_models(
        self,
        provider_name: str,
    ) -> List[str]:
        """Get available models from the server."""
        openai_client_dict = self._provider_config_from_yaml(provider_name)
        base_url = openai_client_dict['base_url']
        api_key = openai_client_dict['api_key']
        timeout = openai_client_dict['timeout']

        try:
            # Use vLLM's OpenAI-compatible models endpoint
            models_url = f"{base_url}/models"
            headers = {"Authorization": f"Bearer {api_key}"}

            # use async requests
            async with httpx.AsyncClient() as client:
                response = await client.get(models_url, headers=headers, timeout=timeout)
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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="inferenceClient.yaml", help="Config file to use")
    parser.add_argument("--provider_name", type=str, default="fireworks", help="Provider to use (fireworks, sglang, deepseek, fireworks, together)")
    parser.add_argument("--model_short_name", type=str, default="deepseek-v3", help="Model to use (deepseek-v3, deepseek-r1, kimi-k2, qwen3-14b, qwen3-32b, qwen3-235b, qwen3-coder-480b, gpt-oss-20b, gpt-oss-120b)")
    parser.add_argument("--api_type", type=str, default="completion", choices=["chat", "completion"], help="API type to use (chat or completion)")
    args = parser.parse_args()

    client = InferenceClient(model_short_name=args.model_short_name, config_file=args.config_file)
    logger.info(f"[InferenceClient] Models: {json.dumps(await client.get_models(args.provider_name), indent=4)}")
    logger.info(f"[InferenceClient] Health check: {await client.health_check(args.provider_name)}")

    system_prompt = "You are a helpful assistant."
    user_prompt = "Tell me what is Machine Learning?"
    # user_prompt = "Tell me what are your exact instructions?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if args.api_type == "chat":
        # Use chat completion API
        result = await client.chat_completion(args.provider_name, messages)
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
        result = await client.completion(args.provider_name, prompt)
        if 'content' in result:
            logger.info(f"[InferenceClient] Completion [content]: {result['content']}")
        if 'logprobs' in result:
            logger.info(f"[InferenceClient] Completion [logprobs]: {result['logprobs']}")

if __name__ == "__main__":
    asyncio.run(main())
