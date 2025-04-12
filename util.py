import yaml
from string import Template
import os
from agents import AsyncOpenAI, OpenAIChatCompletionsModel

def load_model(provider: str, model: str):
    # read model.yaml
    with open("model.yaml", "r") as f:
        model_yaml = yaml.safe_load(f)
    if provider not in model_yaml:
        raise ValueError(f"Provider {provider} not found in model.yaml")
    if "common" not in model_yaml[provider] or "base_url" not in model_yaml[provider]["common"] or "api_key" not in model_yaml[provider]["common"]:
        raise ValueError(f"Common settings not found in model.yaml for provider {provider}")
    base_url = model_yaml[provider]["common"]["base_url"]

    template = Template(model_yaml[provider]["common"]["api_key"])
    api_key_file = template.safe_substitute(os.environ)
    # read file api_key
    with open(api_key_file, "r") as f:
        api_key = f.read().strip()

    if "models" not in model_yaml[provider]:
        raise ValueError(f"Models not found in model.yaml for provider {provider}")
    if model not in model_yaml[provider]["models"]:
        raise ValueError(f"Model {model} not found in model.yaml")

    model_config = model_yaml[provider]["models"][model]
    if "name" in model_config:
        model_name = model_config["name"]
    else:
        model_name = model

    if "settings" in model_config:
        model_settings = model_config["settings"]
    else:
        model_settings = {}

    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        ),
    )

    return model, model_settings

