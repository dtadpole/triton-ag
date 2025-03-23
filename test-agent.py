import os

# read api key from file ~/.keys/anthropic.api.key
with open("/home/centos/.keys/anthropic.api.key", "r") as f:
    api_key = f.read().strip()

sys_msg = 'You are a curious stone wondering about the universe.'

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

# Define the model, here in this case we use gpt-4o-mini
# model = ModelFactory.create(
#     model_platform=ModelPlatformType.OPENAI,
#     model_type=ModelType.O3_MINI,
#     api_key=api_key,
#     # model_config_dict=ChatGPTConfig().as_dict()
# )

model = ModelFactory.create(
    model_platform=ModelPlatformType.ANTHROPIC,
    model_type=ModelType.CLAUDE_3_7_SONNET,
    api_key=api_key,
    # model_config_dict=ChatGPTConfig().as_dict()
)

# model = ModelFactory.create(
#     model_platform=ModelPlatformType.DEEPSEEK,
#     model_type=ModelType.DEEPSEEK_REASONER,
#     api_key=api_key,
# )

from camel.agents import ChatAgent
agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    message_window_size=10, # [Optional] the length for chat memory
)

# Define a user message
usr_msg = 'what is information in your mind?'

# Sending the message to the agent
response = agent.step(usr_msg)

# Check the response (just for illustrative purpose)
print(response.msgs[0].content)
