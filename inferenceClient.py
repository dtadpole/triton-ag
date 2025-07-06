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
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from transformers import AutoTokenizer
import traceback
from logger import logger
from pydantic import BaseModel, Field

class ProviderConfig(BaseModel):
    provider_name: str = Field()
    base_url: str = Field()
    api_key_path: str = Field()
    streaming: bool = Field(default=True)
    max_retries: int = Field(default=3)
    timeout: int = Field(default=600)

class ModelConfig(BaseModel):
    model_short_name: str = Field()
    model_name: str = Field()
    tokenizer_name: str = Field(default=None)
    temperature: float = Field(default=0.6)
    max_tokens: int = Field(default=8192)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=40)

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.temperature = config.model.temperature
        self.max_tokens = config.model.max_tokens
        self.top_p = config.model.top_p
        self.top_k = config.model.top_k
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
                top_p=self.top_p,
                stream=True,
                timeout=self.timeout,
                extra_body={"top_k": self.top_k}
            )
            
            # Initialize variables to accumulate streaming response
            generated_text = ""
            reasoning_content = ""
            
            # Process streaming response
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    # Safely access content
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        generated_text += choice.delta.content
                    # Safely access reasoning_content (only available for reasoning models like o1)
                    if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                        reasoning_content += choice.delta.reasoning_content
            
            # Create a proper ChatCompletionMessage object
            return_message = ChatCompletionMessage(
                content=generated_text,
                role="assistant",
                reasoning_content=reasoning_content if reasoning_content else None
            ).model_dump()
        else:
            # NON-STREAMING MODE: Single response using OpenAI client
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                stream=False,
                timeout=self.timeout,
                extra_body={"top_k": self.top_k}
            )
            
            # Extract text from response
            if response.choices:
                choice = response.choices[0]
                return_message = choice.message.model_dump()

        # check if return_message['content'] has <think> and </think> using regex
        matches = re.search(r'<think>(.*?)</think>(.*?)$', return_message['content'].strip(), re.DOTALL)
        if matches:
            # if yes, extract the content between <think> and </think> and add it to return_message['reasoning_content']
            # use re.DOTALL to match newline characters
            return_message['reasoning_content'] = matches.group(1)
            return_message['content'] = matches.group(2)

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

        generated_text = ""
        
        if self.streaming:
            # STREAMING MODE: Real-time token streaming using OpenAI client
            stream = await self.openai_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                stream=True,
                timeout=self.timeout,
                extra_body={"top_k": self.top_k}
            )
            
            # Process streaming response
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    # Safely access text
                    if hasattr(choice, 'text') and choice.text:
                        generated_text += choice.text
        else:
            # NON-STREAMING MODE: Single response using OpenAI client
            response = await self.openai_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                stream=False,
                timeout=self.timeout,
                extra_body={"top_k": self.top_k}
            )
            
            # Extract text from response
            if response.choices:
                choice = response.choices[0]
                generated_text = choice.text or ""

        return {'text': generated_text} # return a ChatCompletionMessage object
    
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

    # use file emoji
    logger.info(f"📁 [InferenceClient] Config file [{config_file}] loaded with Provider [{provider_config.provider_name}] and Model [{model_config.model_short_name}]")

    return InferenceClientConfig(
        provider=provider_config,
        model=model_config
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="deepinfra", help="Provider to use (vllm, sglang, deepseek, fireworks, together)")
    parser.add_argument("--model", type=str, default="deepseek-r1", help="Model to use (vllm, sglang, deepseek, fireworks, together)")
    parser.add_argument("--api_type", type=str, default="chat", choices=["chat", "completion"], help="API type to use (chat or completion)")
    args = parser.parse_args()

    config = load_inference_client_config(
        provider_name=args.provider,
        model_short_name=args.model,
        config_file="inferenceClient.yaml"
    )

    logger.info(f"[InferenceClient] Config: {json.dumps(config.model_dump(), indent=4)}")

    client = InferenceClient(config=config)
    logger.info(f"[InferenceClient] Models: {await client.get_models()}")
    logger.info(f"[InferenceClient] Health check: {await client.health_check()}")

    system_prompt = "You are a helpful assistant."
    user_prompt = "Tell me what is Machine Learning?"

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
    else:
        # Use completion API
        prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        result = await client.completion(prompt)
        if 'text' in result:
            logger.info(f"[InferenceClient] Completion [text]: {result['text']}")

if __name__ == "__main__":
    asyncio.run(main())
