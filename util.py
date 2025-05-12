import yaml
import json
from string import Template
import os
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, RunHooks
import boto3
import mlflow
from datetime import datetime
import logging
from logger import logger
from pydantic.json_schema import to_jsonable_python
from agents import MessageOutputItem, ToolCallItem, ToolCallOutputItem
from agents import RunResult


def init_logging(agent_name: str):
    # enable_verbose_stdout_logging()
    # stdout_logger = logging.getLogger("agents")
    # stdout_logger.setLevel(logging.INFO)
    # stdout_logger.addHandler(logging.StreamHandler())

    mlflow.openai.autolog()
    mlflow.set_tracking_uri("http://localhost:5050")
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

    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        ),
    )

    return model, model_settings


def load_agent_model(agent_name: str):
    # read from agent.yaml
    with open("agent.yaml", "r") as f:
        agent_yaml = yaml.safe_load(f)
    if agent_name not in agent_yaml:
        raise ValueError(f"Agent {agent_name} not found in agent.yaml")

    if "model" not in agent_yaml[agent_name]:
        raise ValueError(f"Model not found in agent.yaml for agent {agent_name}")

    model_config = agent_yaml[agent_name]["model"]
    if "provider" not in model_config:
        raise ValueError(f"Provider not found in agent.yaml for agent {agent_name}")
    if "model" not in model_config:
        raise ValueError(f"Model not found in agent.yaml for agent {agent_name}")

    model, model_settings = load_model(model_config["provider"], model_config["model"])
    return model, model_settings


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


def log_result_items(result: RunResult, agent_name: str, folder: str):
    if not result.new_items:
        raise ValueError("No items to log")
    output = []
    sys_msg = {
        "role": "system",
        "content": result.new_items[0].agent.instructions,
    }
    output.append(sys_msg)
    user_msg = {
        "role": "user",
        "content": result.input,
    }
    output.append(user_msg)
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            json_item = {
                "role": "assistant",
                "content": to_jsonable_python(item.raw_item.content),
            }
            output.append(json_item)
        elif isinstance(item, ToolCallItem):
            json_item = {
                "role": "assistant",
                "content": to_jsonable_python(item.raw_item)
            }
            output.append(json_item)
        elif isinstance(item, ToolCallOutputItem):
            json_item = {
                "role": "user",
                "content": to_jsonable_python(item.raw_item),
            }
            output.append(json_item)
        else:
            raise ValueError(f"Unknown item type: {type(item)}")

    json_output = json.dumps(output, indent=2)
    # write to file
    with open(os.path.join(folder, "logger.json"), "w") as f:
        f.write(json_output)

    # push to s3
    s3_client = boto3.client("s3")
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    s3_client.put_object(Bucket="agent-xyz", Key=f"{agent_name}/{datetime_str}_logger.json", Body=json_output)

    return output
