import os
from typing import Any, Dict, Optional
import yaml
import json
import dspy
from dspy.utils.callback import BaseCallback
from util import logger

def load_lm(provider: str, model: str) -> dspy.LM:
    with open("dspy_model.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if provider not in config:
        raise ValueError(f"Provider {provider} not found in config")

    if "common" not in config[provider]:
        raise ValueError(f"Common config not found for provider {provider}")
    
    if "dspy_provider" in config[provider]["common"]:
        dspy_provider = config[provider]["common"]["dspy_provider"]
    else:
        dspy_provider = provider

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

    # get temperature from model config, default to 0.6
    temperature = config[provider]["models"][model]["settings"].get("temperature", 0.6)
    max_tokens = config[provider]["models"][model]["settings"].get("max_tokens", 8192)

    lm = dspy.LM(f"{dspy_provider}/{model_name}",
                    api_key=api_key,
                    api_base=api_base,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
    logger.info(f"Loaded model: [provider={provider}], [model_name={model_name}], [api_base={api_base}], [temperature={temperature}], [max_tokens={max_tokens}]")
    # return the model
    return lm

instructions_cache = {}
def load_instructions(yaml_file: str = "dspy_instructions.yaml") -> dict:
    filepath = os.path.join(os.getcwd(), yaml_file)
    with open(filepath, "r") as f:
        instructions = yaml.load(f, Loader=yaml.FullLoader)
    return instructions

def load_instructions_for_module(module_name: str, yaml_file: str = "dspy_instructions.yaml") -> dict:
    global instructions_cache
    if module_name in instructions_cache:
        return instructions_cache[module_name]

    instructions_cache = load_instructions(yaml_file)
    if module_name not in instructions_cache:
        raise ValueError(f"Module {module_name} not found in {yaml_file}")
    instructions = instructions_cache[module_name]
    return instructions

class DSPyToolCallback(BaseCallback):
    def __init__(self):
        self._lookup = {}

    def on_tool_start(self, call_id: str, instance: Any, inputs: Dict[str, Any]):
        """A handler triggered when a tool is called.
        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Tool instance.
            inputs: The inputs to the Tool's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        self._lookup[call_id] = instance.name
        logger.info(f"Tool [{instance.name}] [{call_id}] called with inputs {inputs}")

    def on_tool_end(self, call_id: str, outputs: Optional[Dict[str, Any]], exception: Optional[Exception] = None):
        """A handler triggered after a tool is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Tool's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        logger.info(f"Tool [{self._lookup[call_id]}] [{call_id}] called with outputs {json.dumps(outputs)}")
