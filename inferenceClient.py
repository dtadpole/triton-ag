import os
import re
import httpx
import argparse
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import requests
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
# from transformers import AutoTokenizer
import traceback
from logger import logger

class InferenceClient:
    """Client for OpenAI API using OpenAI client for streaming or non-streaming generation."""
    
    def __init__(
        self,
        client_type: str,
        streaming: bool = True,
        config_file: str = "inferenceClient.yaml",
    ):
        """
        Initialize vLLM client.
        
        Args:
            client_type: Type of client to use (vllm, sglang, deepseek, fireworks, together)
            streaming: Whether to use streaming mode
            config_file: Path to YAML config file with server settings
        """
        # Load configuration from file
        self.config = self._load_config(config_file)
        self.client_type = client_type
        self.streaming = streaming
        common_config = self.config.get(self.client_type, {}).get('common', {})
        
        # Set defaults from config
        self.model_tag = self.config.get(self.client_type, {}).get('generation', {}).get('model', 'default')
        self.base_url = common_config.get('base_url', 'http://localhost:8000/v1')

        # Set up API key
        # Try to load from config or default location
        api_key_path = common_config.get('api_key', "${HOME}/.keys/local.api.key")
        if api_key_path.startswith('${HOME}/'):
            api_key_path = os.path.expanduser(api_key_path.replace('${HOME}', '~'))
        try:
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()
                logger.info(f"🔑 [InferenceClient] API key loaded from [{api_key_path}]")
        except FileNotFoundError:
            logger.info(f"🔑 [InferenceClient] API key not found at [{api_key_path}], using [dummy_key]")
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
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            # Try relative to script directory
            config_path = Path(__file__).parent / config_file
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # use file emoji
                logger.info(f"📁 [InferenceClient] Config file {config_file} loaded")
                return config or {}
        else:
            logger.warning(f"⚠️ [InferenceClient] Warning: Config file {config_file} not found")
            return {}
    
    async def _chat_completion(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text using OpenAI-compatible API with streaming or non-streaming mode.
        
        Args:
            messages: List of messages as input
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'text' and 'tokens' data
        """
        # Get defaults from config
        generation_config = self.config.get(self.client_type, {}).get('generation', {})
        temperature = temperature if temperature is not None else generation_config.get('temperature', 0.6)
        max_tokens = max_tokens or generation_config.get('max_tokens', 8192)

        # tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        
        if self.streaming:
            # STREAMING MODE: Real-time token streaming using OpenAI client
            stream = await self.openai_client.chat.completions.create(
                model=self.model_tag,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get('top_p', 1.0),
                stream=True,
                timeout=timeout,
                extra_body={"top_k": 40}
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
                model=self.model_tag,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get('top_p', 1.0),
                stream=False,
                timeout=timeout,
                extra_body={"top_k": 40}
            )
            
            # Extract text from response
            if response.choices:
                choice = response.choices[0]
                return_message = choice.message.model_dump()
                # generated_text = choice.message.content or ""

        # check if return_message['content'] has <think> and </think> using regex
        matches = re.search(r'<think>(.*?)</think>(.*?)$', return_message['content'].strip(), re.DOTALL)
        if matches:
            # if yes, extract the content between <think> and </think> and add it to return_message['reasoning_content']
            # use re.DOTALL to match newline characters
            return_message['reasoning_content'] = matches.group(1)
            return_message['content'] = matches.group(2)

        return return_message

    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """
        Generate text using OpenAI-compatible API with streaming or non-streaming mode.
        """
        max_retries = self.config.get(self.client_type, {}).get('generation', {}).get('max_retries', 3)
        timeout = self.config.get(self.client_type, {}).get('generation', {}).get('timeout', 600)
        retry_count = 0
        while retry_count < max_retries:
            retry_count += 1
            try:
                return await self._chat_completion(messages, timeout=timeout, **kwargs)
            except Exception as e:
                traceback.print_exc()
                if retry_count < max_retries:
                    logger.warning(f"⚠️ [InferenceClient] [Chat completion] Failed: {e} [{retry_count}/{max_retries}], retrying in {2 ** retry_count} seconds...")
                    await asyncio.sleep(2 ** retry_count) # exponential backoff
                else:
                    logger.error(f"❌ [InferenceClient] [Chat completion] Failed: {e}, giving up...") # give up after max retries
        return None

    async def _completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text completion using OpenAI-compatible API with streaming or non-streaming mode.
        
        Args:
            prompt: Text prompt for completion
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'text' data
        """
        # Get defaults from config
        generation_config = self.config.get(self.client_type, {}).get('generation', {})
        temperature = temperature if temperature is not None else generation_config.get('temperature', 0.6)
        max_tokens = max_tokens or generation_config.get('max_tokens', 8192)

        # tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        
        generated_text = ""
        
        if self.streaming:
            # STREAMING MODE: Real-time token streaming using OpenAI client
            stream = await self.openai_client.completions.create(
                model=self.model_tag,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get('top_p', 1.0),
                stream=True,
                timeout=timeout,
                extra_body={"top_k": 40}
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
                model=self.model_tag,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get('top_p', 1.0),
                stream=False,
                timeout=timeout,
                extra_body={"top_k": 40}
            )
            
            # Extract text from response
            if response.choices:
                choice = response.choices[0]
                generated_text = choice.text or ""

        # tokenize the generated text
        # tokens = tokenizer.encode(generated_text) if generated_text else []
        
        # return {'text': generated_text, 'tokens': tokens}
        return {'text': generated_text} # return a ChatCompletionMessage object
    
    async def completion(self, prompt: str, **kwargs) -> Dict:
        """
        Generate text completion using OpenAI-compatible API with streaming or non-streaming mode.
        """
        max_retries = self.config.get(self.client_type, {}).get('generation', {}).get('max_retries', 3)
        timeout = self.config.get(self.client_type, {}).get('generation', {}).get('timeout', 600)
        retry_count = 0
        while retry_count < max_retries:
            retry_count += 1
            try:
                return await self._completion(prompt, timeout=timeout, **kwargs)
            except Exception as e:
                if retry_count < max_retries:
                    logger.warning(f"⚠️ [InferenceClient] [Completion] Failed: {e} [{retry_count}/{max_retries}], retrying in {2 ** retry_count} seconds...")
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
                response = await client.get(models_url, headers=headers, timeout=60)
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


def get_system_prompt(config: Dict) -> str:
    """Get system prompt from configuration."""
    prompts_config = config.get('prompts', {})
    reference_code = get_example_reference_code(config)
    generated_code = get_example_generated_code(config)
    return prompts_config.get('system_prompt', 'You are a helpful assistant.').format(reference_code=reference_code, generated_code=generated_code)

def get_user_prompt(config: Dict, source_code: str) -> str:
    """Get user prompt from configuration with source code substituted."""
    prompts_config = config.get('prompts', {})
    user_prompt_template = prompts_config.get('user_prompt', 'Analyze this code: {source_code}')
    return user_prompt_template.format(source_code=source_code)

def get_example_reference_code(config: Dict) -> str:
    """Get example reference code from configuration."""
    prompts_config = config.get('prompts', {}).get('examples', {})
    return prompts_config.get('reference_code', '')

def get_example_generated_code(config: Dict) -> str:
    """Get example generated code from configuration."""
    prompts_config = config.get('prompts', {}).get('examples', {})
    return prompts_config.get('generated_code', '')


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=str, default="fireworks-r1", help="Client type to use (vllm, sglang, deepseek, fireworks, together)")
    parser.add_argument("--source_code", type=str, default="level1/1_Square_matrix_multiplication_.py", help="Source code to generate")
    parser.add_argument("--api_type", type=str, default="chat", choices=["chat", "completion"], help="API type to use (chat or completion)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    filepath = os.path.join(os.path.dirname(__file__), 'kernel_bench', args.source_code)
    with open(filepath, "r") as f:
        source_code = f.read()

    client = InferenceClient(client_type=args.client, streaming=args.streaming)
    logger.info(f"[InferenceClient] Models: {await client.get_models()}")
    logger.info(f"[InferenceClient] Health check: {await client.health_check()}")

    system_prompt = get_system_prompt(client.config)
    user_prompt = get_user_prompt(client.config, source_code)

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
