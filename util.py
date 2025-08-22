import json
import logging
import os
from datetime import datetime
from string import Template
from typing import Union

import boto3
import httpx
import mlflow
import yaml
from agents import (
    AsyncOpenAI,
    MessageOutputItem,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunHooks,
    RunResult,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.tool import Tool
from logger import logger
from pydantic.json_schema import to_jsonable_python


def is_devserver() -> bool:
    import socket

    hostname = socket.gethostname()
    return "facebook.com" in hostname

## Define the global data folders
if is_devserver():
    KB_EVAL_DIR = "shared/.kbeval"
else:
    KB_EVAL_DIR = os.path.join(os.path.expanduser("~"), ".kbeval")

if is_devserver():
    INFERENCE_DIR = "shared/.inference"
else:
    INFERENCE_DIR = os.path.join(os.path.expanduser("~"), ".inference")

if is_devserver():
    TRAINER_DIR = "shared/.trainer"
else:
    TRAINER_DIR = "~/.trainer"

if is_devserver():
    WORKFLOW_DIR = "shared/.workflow"
else:
    WORKFLOW_DIR = "~/.workflow"

def init_logging(agent_name: str):
    # enable_verbose_stdout_logging()
    # stdout_logger = logging.getLogger("agents")
    # stdout_logger.setLevel(logging.INFO)
    # stdout_logger.addHandler(logging.StreamHandler())

    mlflow.openai.autolog()
    mlflow.set_tracking_uri("http://localhost:5051")
    mlflow.set_experiment(f"Agent [{agent_name}]")

    # weave.init("openai-agents")
    # set_trace_processors([WeaveTracingProcessor()])


def is_subfolder(parent_folder: str, child_folder: str) -> bool:
    abs_parent_folder = os.path.abspath(parent_folder)
    abs_child_folder = os.path.abspath(child_folder)
    return abs_child_folder.startswith(abs_parent_folder)


def get_next_run_folder():
    i = 0
    while os.path.exists(os.path.join(os.getcwd(), f"_run_{i:03d}")):
        i += 1
    return os.path.join(os.getcwd(), f"_run_{i:03d}")


# function to find next available folder starting with _run_<number>
# def prepare_next_run_folder():
#     next_run_folder = get_next_run_folder()
#     os.makedirs(next_run_folder, exist_ok=True)
#     # create verifier folder in the new run folder
#     os.makedirs(os.path.join(next_run_folder, "verifier"), exist_ok=True)
#     # iterate over all files in verifier folder and copy them to the new run folder
#     for file in os.listdir(os.path.join(os.getcwd(), "verifier")):
#         # skipe anything that is not file, or not ends with .py
#        if not os.path.isfile(
#            os.path.join(os.getcwd(), "verifier", file)
#        ) or not file.endswith(".py"):
#            continue
#        # read file content, and write to corresponding file in the new verifier subfolder in the new run folder
#        with open(os.path.join(os.getcwd(), "verifier", file), "r") as f:
#            content = f.read()
#        with open(
#            os.path.join(os.getcwd(), f"_run_{i:03d}", "verifier", file), "w"
#        ) as f:
#            f.write(content)
#    # return the new run folder
#    folder = os.path.join(os.getcwd(), f"_run_{i:03d}")
#    return folder


def load_model(provider: str, model: str):
    # read model.yaml
    with open("model.yaml", "r") as f:
        model_yaml = yaml.safe_load(f)
    if provider not in model_yaml:
        raise ValueError(f"Provider {provider} not found in model.yaml")
    if (
        "common" not in model_yaml[provider]
        or "base_url" not in model_yaml[provider]["common"]
        or "api_key" not in model_yaml[provider]["common"]
    ):
        raise ValueError(
            f"Common settings not found in model.yaml for provider {provider}"
        )
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
        model_settings = ModelSettings(**model_config["settings"])
    else:
        model_settings = ModelSettings()

    # add proxy server if running on devserver
    if is_devserver() and api_key.lower().strip() != "empty":
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(proxy=httpx.Proxy("http://fwdproxy:8080")),
            timeout=240,
            max_retries=5,
        )
    else:
        client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=240, max_retries=5
        )
    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    )

    return model, model_settings


def load_agent_model(
    agent_name: str, provider: Union[str, None] = None, model: Union[str, None] = None
):
    # read from agent.yaml
    with open("agent.yaml", "r") as f:
        agent_yaml = yaml.safe_load(f)
    if agent_name not in agent_yaml:
        raise ValueError(f"Agent {agent_name} not found in agent.yaml")

    if "model" not in agent_yaml[agent_name]:
        raise ValueError(f"Model not found in agent.yaml for agent {agent_name}")

    model_config = agent_yaml[agent_name]["model"]
    if not provider and "provider" not in model_config:
        raise ValueError(f"Provider not found in agent.yaml for agent {agent_name}")
    if not model and "model" not in model_config:
        raise ValueError(f"Model not found in agent.yaml for agent {agent_name}")

    if "run_config" not in agent_yaml[agent_name]:
        raise ValueError(f"Run config not found in agent.yaml for agent {agent_name}")

    model, model_settings = load_model(
        provider or model_config["provider"], model or model_config["model"]
    )
    run_config = agent_yaml[agent_name]["run_config"]

    return model, model_settings, run_config, model_config


def get_run_hooks():
    run_hooks = RunHooks()

    async def on_tool_start(context, agent, tool):
        logger.info(f"Agent [{agent.name}] Tool [{tool.name}] started")

    async def on_tool_end(context, agent, tool, result):
        logger.info(
            f"Agent [{agent.name}] Tool [{tool.name}] ended with result:\n{result}\n"
        )

    run_hooks.on_tool_start = on_tool_start
    run_hooks.on_tool_end = on_tool_end

    return run_hooks


def log_result_items(
    tools: list[Tool],
    result: RunResult,
    name_tag: str,
    model_tag: str,
    task_tag: str,
    folder: str,
):
    if not result.new_items:
        raise ValueError("No items to log")

    # process tools
    functions = []
    for tool in tools:
        function = to_jsonable_python(tool)
        function["parameters"] = function["inputSchema"]
        del function["inputSchema"]
        functions.append(function)

    # process messages
    messages = []
    sys_msg = {
        "role": "system",
        "content": result.new_items[0].agent.instructions,
    }
    messages.append(sys_msg)
    user_msg = {
        "role": "user",
        "content": result.input,
    }
    function_calls = {}
    messages.append(user_msg)
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            json_item = {
                "role": "assistant",
                "content": to_jsonable_python(item.raw_item.content),
            }
            messages.append(json_item)
        elif isinstance(item, ToolCallItem):
            if "call_id" in item.raw_item:
                call_id = item.raw_item["call_id"]
            elif hasattr(item.raw_item, "call_id"):
                call_id = item.raw_item.call_id
            else:
                call_id = None
            if call_id:
                function_calls[call_id] = item.raw_item
            # final json item
            json_item = {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": item.raw_item.name,
                    "arguments": to_jsonable_python(item.raw_item.arguments),
                },
            }
            messages.append(json_item)
        elif isinstance(item, ToolCallOutputItem):
            if "call_id" in item.raw_item:
                call_id = item.raw_item["call_id"]
            elif hasattr(item.raw_item, "call_id"):
                call_id = item.raw_item.call_id
            else:
                call_id = None
                logger.error(f"Tool call item has no call_id: {item.raw_item}")
            if "output" in item.raw_item:
                output = item.raw_item["output"]
            elif hasattr(item.raw_item, "output"):
                output = item.raw_item.output
            else:
                messages = None
                logger.error(f"Tool call item has no output: {item.raw_item}")
            # final json item
            json_item = {
                "role": "function",
                "name": (
                    function_calls[call_id].name
                    if call_id and call_id in function_calls
                    else None
                ),
                "content": output,
            }
            messages.append(json_item)
        else:
            raise ValueError(f"Unknown item type: {type(item)}")

    json_output = json.dumps({"functions": functions, "messages": messages}, indent=2)
    try:
        # write to file
        with open(os.path.join(folder, f"{name_tag}_logger.json"), "w") as f:
            f.write(json_output)
    except Exception as e:
        logger.error(f"Error writing to file: {e}")

    if is_devserver() is not True:  # No s3 access on meta's devserver
        # push to s3
        try:
            s3_client = boto3.client("s3")
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            s3_client.put_object(
                Bucket="agent-xyz",
                Key=f"agent/{model_tag}/{task_tag}/{name_tag}_{datetime_str}.json",
                Body=json_output,
            )
        except Exception as e:
            logger.error(f"Error pushing to s3: {e}")

    return messages
