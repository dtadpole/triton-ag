import os
import yaml
from string import Template
import argparse
import asyncio

from pydantic_ai.agent import Agent
from pydantic_ai.models import Model, ModelRequestParameters

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from pydantic_ai.providers.deepseek import DeepSeekProvider

from agent.logger import logger


def load_model_settings(provider: str, model: str) -> dict:
    # load yaml file from model.settings.yaml
    with open("model.settings.yaml", "r") as f:
        settings = yaml.safe_load(f)

    # get provider settings
    if provider not in settings:
        raise ValueError(f"Unknown provider: {provider}")

    provider_settings = settings[provider]

    # set env variables
    if "env" in provider_settings:
        for env_var in provider_settings["env"]:
            if "value" in env_var:
                os.environ[env_var["name"]] = env_var["value"]
            elif "file" in env_var:
                template = Template(env_var["file"])
                filename = template.safe_substitute(os.environ)
                with open(filename, "r") as f:
                    os.environ[env_var["name"]] = f.read().strip()
            else:
                raise ValueError(f"Invalid env variable: {env_var} for [{provider}]")

    if "models" not in provider_settings:
        raise ValueError(f"Invalid provider: {provider}, missing 'models'")

    # iterate (key, values) of provider_settings["models"]
    if model in provider_settings["models"]:
        return provider_settings["models"][model]
    else:
        raise ValueError(f"Unknown model: {model} for [{provider}]")


@logger.catch
def create_model(model_name: str, provider: str) -> Model:
    model_settings = load_model_settings(provider, model_name)
    # if "name" in model_settings, use the name as the model name
    if "name" in model_settings:
        model_name = model_settings["name"]

    if provider == "openai":
        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
        )
    elif provider == "anthropic":
        return AnthropicModel(
            model_name=model_name,
            provider=AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
        )
    elif provider == "gemini":
        return GeminiModel(
            model_name=model_name,
            provider=GoogleGLAProvider(api_key=os.environ["GEMINI_API_KEY"]),
        )
    elif provider == "deepseek":
        return OpenAIModel(
            model_name=model_name,
            provider=DeepSeekProvider(api_key=os.environ["DEEPSEEK_API_KEY"]),
        )
    elif provider == "fireworks":
        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ["FIREWORKS_API_KEY"],
            ),
        )
    elif provider == "together":
        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url="https://api.together.xyz/v1",
                api_key=os.environ["TOGETHER_API_KEY"],
            ),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    model_settings = load_model_settings(args.provider, args.model)
    settings = model_settings["settings"]
    model = create_model(args.model, args.provider)
    print(model)

    # response = await model.request(
    #     messages=[
    #         {"role": "user", "content": "Hello, world!"},
    #     ],
    #     model_settings=settings,
    #     model_request_parameters=ModelRequestParameters(
    #         function_tools=[],
    #         allow_text_result=True,
    #         result_tools=[],
    #     ),
    # )
    # print(response)

    agent = Agent(
        model=model,
    )
    response = await agent.run("Hello")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
