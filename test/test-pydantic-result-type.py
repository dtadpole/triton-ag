from typing import Union

from pydantic import BaseModel

from pydantic_ai import Agent


class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str


agent: Agent[None, Union[Box, str]] = Agent(
    'openai:gpt-4o-mini',
    result_type=Union[Box, str],  # type: ignore
    system_prompt=(
        "Extract me the dimensions of a box, "
        "if you can't extract all data, ask the user to try again."
    ),
)

print('----------------')
request='The box is 10x20x30'
print(f"Request: {request}")
result = agent.run_sync(request)
print(f"Result: {result.data}")
print('----------------')
#> Please provide the units for the dimensions (e.g., cm, in, m).

request='The box is 10x20x30 cm'
print(f"Request: {request}")
result = agent.run_sync(request)
print(f"Result: {result.data}")
#> width=10 height=20 depth=30 units='cm'
print('----------------')
