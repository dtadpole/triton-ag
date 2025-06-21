import os
import yaml
import dspy

def load_lm(provider: str, model: str) -> dspy.LM:
    with open("dspy_model.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if provider not in config:
        raise ValueError(f"Provider {provider} not found in config")

    if "common" not in config[provider]:
        raise ValueError(f"Common config not found for provider {provider}")
    
    if "provider" in config[provider]:
        provider = config[provider]["provider"]

    # Load API key
    if "api_key" not in config[provider]["common"]:
        raise ValueError(f"API key not found for provider {provider}")
    # substitute environment variable
    api_key_file = os.path.expandvars(config[provider]["common"]["api_key"])
    with open(api_key_file, "r") as f:
        api_key = f.read().strip()

    # Load base URL
    if "base_url" not in config[provider]["common"]:
        raise ValueError(f"Base URL not found for provider {provider}")
    api_base = config[provider]["common"]["base_url"]

    # Load model
    model_name = model
    if model not in config[provider]["models"]:
        raise ValueError(f"Model {model} not found in config")
    model_config = config[provider]["models"][model]
    if "name" in model_config:
        model_name = model_config["name"]

    lm = dspy.LM(f"{provider}/{model_name}", api_key=api_key, api_base=api_base)
    # lm.set_model_config(**model_config["settings"])
    return lm
