import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

logfire.configure()

joke_selection_agent = Agent(
    'openai:gpt-4o',
    system_prompt=(
        'Use the `joke_factory` to generate some jokes, then choose a random one. '
        'You must return just a single joke.'
    ),
    instrument=True,
)
joke_generation_agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    # 'google-gla:gemini-2.0-flash',
    result_type=list[str],
    instrument=True,
)


@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    r = await joke_generation_agent.run(
        f'Please generate {count} jokes.',
        usage=ctx.usage,
    )
    return r.data


result = joke_selection_agent.run_sync(
    'Tell me a joke.',
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=3_000),
)
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
print(result.usage())
"""
Usage(
    requests=3, request_tokens=204, response_tokens=24, total_tokens=228, details=None
)
"""
