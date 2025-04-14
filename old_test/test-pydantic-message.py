from pydantic_core import to_jsonable_python
import json
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

# agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    # 'openai:gpt-4o',
    system_prompt='Be a helpful assistant.',
    model_settings={'temperature': 1.0},
)

result1 = agent.run_sync('Tell me a small joke.')
history_step_1 = result1.all_messages()
as_python_objects = to_jsonable_python(history_step_1)
same_history_as_step_1 = ModelMessagesTypeAdapter.validate_python(as_python_objects)

print('----------------')
print(f"History: {json.dumps(as_python_objects, indent=2)}")
print('----------------')

result2 = agent.run_sync(
    'Tell me a different joke.', message_history=same_history_as_step_1
)

print('----------------')
print(f"Result: {result2.data}")
print('----------------')
