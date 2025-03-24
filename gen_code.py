import os
import argparse
from pydantic_core import to_jsonable_python
from dataclasses import dataclass
from typing import List
from pydantic_ai import Agent, RunContext
import json
import yaml
from string import Template
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider

import logfire

logfire.configure()

@dataclass
class ManagerState:
    args: argparse.Namespace
    task: str
    wd: str

@dataclass
class CodePlan:
    plan: str
    functions: List[str]

@dataclass
class CodeFile:
    filename: str
    code: str

@dataclass
class TestResult:
    success: bool
    correct: bool
    stdout: str
    stderr: str


def _prep_working_directory(args):
    # get the next sequence id
    next_sequence_id = 0
    # check for next available subfolder name, starts with code_gen_, and ends with a number
    while os.path.exists(os.path.join(args.dir, f"_code_gen_{args.model}_{next_sequence_id:03d}")):
        next_sequence_id += 1
    folder_name = os.path.join(args.dir, f"_code_gen_{args.model}_{next_sequence_id:03d}")
    os.makedirs(folder_name, exist_ok=True)
    # for files with name "prompt.txt"  or ending with ".py" or ".stderr" or ".stdout" or ".failed" or ".success" or ".command " in the args.folder, read the content and write to subfolder_name
    for file in os.listdir(args.dir):
        if file == "prompt.txt" or file.endswith(".py") or file.endswith(".stderr") or file.endswith(".stdout") or file.endswith(".success") or file.endswith(".failed") or file.endswith(".command"):
            with open(os.path.join(args.dir, file), "r") as f:
                content = f.read()
            with open(os.path.join(folder_name, file), "w") as f:
                f.write(content)
    return folder_name

""" You will test the code, and based on the test result, fix the code until the code is correct.
"""

async def main(args, model_settings):

    # create model from args
    if args.provider == 'deepseek':
        model = OpenAIModel(
            args.model,
            provider=DeepSeekProvider(api_key=os.environ['DEEPSEEK_API_KEY'])
        )
    elif args.provider == 'fireworks':
        model = OpenAIModel(
            args.model,
            provider=OpenAIProvider(api_key=os.environ['FIREWORKS_API_KEY'], base_url="https://api.fireworks.ai/inference/v1"),
        )
    else:
        model = f"{args.provider}:{args.model}"

    agent_manager = Agent(
        model,
        deps_type=ManagerState,
        result_type=str,
        system_prompt='You are a manager, you oversee the code generation process. Use `establish_plan` function to establish a plan for the given task, then use `generate_code` function to generate code based on the plan.',
        instrument=True,
    )

    agent_planner = Agent(
        model,
        # deps_type=ManagerState,
        result_type=CodePlan,
        system_prompt='You are a GPU kernel optimization expert with deep knowledge in Triton kernel. Your job is to write a plan on how to implement a highly optimized Triton kernel for the given task.  You will first describe the overall plan in detail. Then you will write down a list of functions to organize the code, including both Triton kernel function(s), and python wrapper function(s), explain input (include autotune parameters), output, and description of execution flow, do not include actual code.',
        instrument=True,
    )
    agent_coder = Agent(
        model,
        # deps_type=CodePlan,
        result_type=CodeFile,
        system_prompt='You are a GPU kernel optimization expert with deep knowledge in Triton kernel. Your job is to write Triton and Python code based on the code plan to implement a highly optimized Triton kernel for the given task.  You will also generate a corresponding native PyTorch implementation, and compare the result between Triton and PyTorch.',
        instrument=True,
    )

    @agent_manager.tool
    async def establish_plan(ctx: RunContext[ManagerState]) -> CodePlan:
        """establish a plan for code generation"""
        return await agent_planner.run(ctx.deps.task, usage=ctx.usage, model_settings=model_settings)

    @agent_manager.tool
    async def generate_code(ctx: RunContext[ManagerState], plan: CodePlan) -> CodeFile:
        """generate code based on the plan"""
        prompt = [
            {
                'type': 'text',
                'text': ctx.deps.task,

            },
            {
                'type': 'text',
                'text': f"Plan:\n{plan.plan}\n\nFunctions:\n{'\n\n'.join(plan.functions)}\n\n",

            },
        ]
        # list working dir, read files ending with .py, and not starting with '_'
        files = [f for f in os.listdir(ctx.deps.wd) if f.endswith('.py') and not f.startswith('_')]
        for file in files:
            with open(os.path.join(ctx.deps.wd, file), 'r') as f:
                prompt.append({
                    'type': 'file',
                    'filename': file,
                    'file': f.read()
                })

        code_file = await agent_coder.run(json.dumps(prompt, indent=2), usage=ctx.usage, model_settings=model_settings)
        with open(os.path.join(ctx.deps.wd, code_file.data.filename), 'w') as f:
            f.write(code_file.data.code)

        return code_file


    """
    agent_tester = Agent(
        model_name,
        deps_type=CodeFile,
        result_type=TestResult,
        system_prompt='You are a tester with deep knowledge in Triton kernel. Your job is to test the code to implement a highly optimized Triton kernel for the given task.',
        instrument=True,
    )
    """

   # read from {args.dir}/prompt.txt
    with open(os.path.join(args.dir, 'prompt.txt'), 'r') as f:
        task = f.read()
    wd = _prep_working_directory(args)
    manager_state = ManagerState(args=args, task=task, wd=wd)
    result = await agent_manager.run(task, deps=manager_state, model_settings=model_settings)

    print(json.dumps(to_jsonable_python(result.data), indent=2))
    print(result.usage())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="./func_matmul")
    parser.add_argument("-p", "--provider", type=str, default="anthropic")
    parser.add_argument("-m", "--model", type=str, default="claude-3-7-sonnet-latest")
    args = parser.parse_args()

    # load yaml file from model.settings.yaml
    with open("model.settings.yaml", "r") as f:
        model_settings = yaml.safe_load(f)

    # get provider settings
    if args.provider not in model_settings:
        raise ValueError(f"Invalid provider: {args.provider}")
    provider_settings = model_settings[args.provider]
    # set env variables
    if 'env' in provider_settings:
        for env_var in provider_settings['env']:
            if 'value' in env_var:
                os.environ[env_var['name']] = env_var['value']
            elif 'file' in env_var:
                template = Template(env_var['file'])
                filename = template.safe_substitute(os.environ)
                with open(filename, 'r') as f:
                    os.environ[env_var['name']] = f.read().strip()
            else:
                raise ValueError(f"Invalid env variable: {env_var} for [{args.provider}]")

    if 'models' not in provider_settings:
        raise ValueError(f"Invalid provider: {args.provider}, missing 'models'")
    for model in provider_settings['models']:
        if model['name'] == args.model:
            if 'settings' not in model:
                raise ValueError(f"Invalid model: {args.model} for [{args.provider}], missing 'settings'")
            model_settings = model['settings']
            break
    else:
        raise ValueError(f"Invalid model: {args.model} for [{args.provider}]")

    import asyncio
    asyncio.run(main(args, model_settings))
