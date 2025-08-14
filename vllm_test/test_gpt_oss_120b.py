import asyncio

from agents import (
    Agent,
    function_tool,
    OpenAIResponsesModel,
    Runner,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

set_tracing_disabled(True)


@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main():

    # agent = Agent(
    #     name="Assistant",
    #     instructions="You only respond in haikus.",
    #     model=OpenAIResponsesModel(
    #         # model="openai/gpt-oss-120b",
    #         model="hub/models--openai--gpt-oss-120b/snapshots/bc75b44b8a2a116a0e4c6659bcd1b7969885f423",
    #         openai_client=AsyncOpenAI(
    #             base_url="http://localhost:8000/v1",
    #             api_key="EMPTY",
    #         ),
    #     ),
    #     tools=[get_weather],
    # )
    # result = await Runner.run(agent, "What's the weather in Tokyo?")
    # print(result.final_output)

    prompt = """
    Write a bash script that takes a matrix represented as a string with 
    format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.
    """

    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    response = client.responses.create(
        model="hub/models--openai--gpt-oss-120b/snapshots/bc75b44b8a2a116a0e4c6659bcd1b7969885f423",
        reasoning={"effort": "medium"},
        input=[{"role": "user", "content": prompt}],
    )

    print(response.output_text)


if __name__ == "__main__":
    asyncio.run(main())
