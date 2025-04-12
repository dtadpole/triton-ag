import os
import asyncio
import argparse
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
import yaml
from string import Template
from util import load_model
claude_key = os.environ["ANTHROPIC_API_KEY"]
openai_key = os.environ["OPENAI_API_KEY"]

model = OpenAIChatCompletionsModel(
    model="claude-3-7-sonnet-latest",
    openai_client=AsyncOpenAI(
        api_key=claude_key,
        base_url="https://api.anthropic.com/v1",
    ),
)

programmer = Agent(
    model=model,
    name="programmer",
    instructions="You are a programmer. You are given a task and you need to complete it.",
    tools=[],
)



async def main(args):
    model, model_settings = load_model(args.provider, args.model)
    result = await Runner.run(programmer, input="write a Triton kernel for matrix multiplication")
    print(result.final_output)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--provider", type=str, default="anthropic")
    parser.add_argument("-m", "--model", type=str, default="claude-3.7")
    args = parser.parse_args()
    asyncio.run(main(args))
