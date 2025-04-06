import os
import json
from typing import Optional, AsyncIterator, Any
from typing import Union
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import to_jsonable_python
from abc import ABC, abstractmethod
from pydantic_ai.models import Model, ModelRequestParameters, Usage
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import (
    SystemPromptPart,
    UserPromptPart,
    ToolReturnPart,
    ToolCallPart,
    TextPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from model_factory import create_model
from logger import logger
import yaml


@logger.catch
def get_agent_config(agent_name: str):
    with open("agent.yaml", "r") as f:
        agent_config = yaml.safe_load(f)

    if agent_name not in agent_config:
        raise ValueError(f"Agent {agent_name} not found in agent.yaml")

    return agent_config["agent_planner"]


class Memory(BaseModel):
    memory: list[
        Union[
            UserPromptPart,
            SystemPromptPart,
            ToolCallPart,
        ]
    ] = Field(default=[], description="The memory of the agent")


class BaseTool(BaseModel, ABC):
    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")

    def to_json(self) -> dict:
        """Convert the tool to a JSON object"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_json_schema(),
        }

    def outer_typed_dict_key(self) -> str | None:
        """The key in the outer TypedDict that wraps a result tool."""
        return None

    @abstractmethod
    def parameters_json_schema(self) -> dict:
        """The JSON schema for the tool's parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> None:
        """Run the tool"""
        pass


class BaseAgent(BaseModel, ABC):
    name: str = Field(description="A unique name for the agent")
    description: Optional[str] = Field(None, description="A description of the agent")

    class Config:
        arbitrary_types_allowed: bool = True

    # system prompt for the agent
    system_prompt: Optional[str] = Field(
        None, description="The system prompt for the agent"
    )
    next_prompt: Optional[str] = Field(
        None, description="The next prompt for the agent"
    )

    # model used by the agent
    model: Optional[Model] = Field(
        default=None,
        description="The model to use for the agent",
        exclude=True,
    )
    model_settings: dict = Field(
        default={},
        description="The settings for the model",
    )

    # memory of the agent
    memory: list[ModelMessage] = Field(
        default=[], description="The memory of the agent"
    )

    finished: bool = False

    current_step: int = 0
    max_steps: int = 50

    max_memory: int = 10

    # tools of the agent
    tools: list[BaseTool] = Field(default=[], description="The tools of the agent")

    tool_calls: list[ToolCallPart] = Field(
        default=[], description="The tool calls of the agent"
    )
    # result tools of the agent
    tool_returns: list[ToolReturnPart] = Field(
        default=[], description="The result tools of the agent"
    )

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        agent_config = get_agent_config("agent_planner")
        if "model" not in agent_config:
            raise ValueError("Model not provided in agent.yaml")
        else:
            self.model, self.model_settings = create_model(
                agent_config["model"]["provider"],
                agent_config["model"]["model"],
            )
        if "settings" not in agent_config:
            raise ValueError("Settings not provided in agent.yaml")
        else:
            if "max_memory" in agent_config["settings"]:
                self.max_memory = agent_config["settings"]["max_memory"]
            if "max_steps" in agent_config["settings"]:
                self.max_steps = agent_config["settings"]["max_steps"]
        return self

    async def request(self, messages: list[ModelMessage]) -> tuple[ModelMessage, Usage]:
        """Request the model"""
        request_messages = []
        if self.system_prompt:
            system_msg = ModelRequest(
                parts=[SystemPromptPart(content=self.system_prompt)],
            )
            request_messages.append(system_msg)
        request_messages.extend(messages)
        function_tools = []
        for tool in self.tools:
            function_tools.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters_json_schema(),
                    outer_typed_dict_key=tool.outer_typed_dict_key(),
                )
            )
        request_parameters = ModelRequestParameters(
            function_tools=function_tools,
            allow_text_result=True,
            result_tools=[],
        )
        try:
            result = await self.model.request(
                messages=request_messages,
                model_settings=self.model_settings,
                model_request_parameters=request_parameters,
            )
            logger.info(f"Response: {result[0]}")
            return result[0]
        except Exception as e:
            logger.error(f"Error requesting model: {e}")
            raise e

    async def request_stream(
        self, messages: list[ModelMessage]
    ) -> AsyncIterator[ModelMessage]:
        """Request the model"""
        request_messages = []
        if self.system_prompt:
            system_msg = ModelRequest(
                parts=[SystemPromptPart(content=self.system_prompt)],
            )
            request_messages.append(system_msg)
        request_messages.extend(messages)
        function_tools = []
        for tool in self.tools:
            function_tools.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters_json_schema(),
                    outer_typed_dict_key=tool.outer_typed_dict_key(),
                )
            )
        request_parameters = ModelRequestParameters(
            function_tools=function_tools,
            allow_text_result=True,
            result_tools=[],
        )
        try:
            result = await self.model.request(
                messages=request_messages,
                model_settings=self.model_settings,
                model_request_parameters=request_parameters,
            )
            logger.info(f"Response: {result[0]}")
            return result[0]
        except Exception as e:
            logger.error(f"Error requesting model: {e}")
            raise e

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously."""
        if request:
            user_msg = ModelRequest(
                parts=[UserPromptPart(content=request)],
            )
            self.memory.append(user_msg)

        results: list[str] = []
        self.finished = False
        while not self.finished:
            self.current_step += 1
            step_result = await self.step()
            # logger.info(f"Step {self.current_step}: {step_result}")
            results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
                break

        return "\n".join(results) if results else "No steps executed"

    def finish(self, finished: bool = True) -> None:
        """Finish the agent"""
        self.finished = finished

    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act, think_result = await self.think()
        if not should_act:
            callback_result = self.callback()
            if callback_result:
                return True, "\n\n".join([think_result, callback_result])

            else:
                return True, think_result
        else:
            act_result = await self.act()
            callback_result = self.callback()
            if callback_result:
                return False, "\n\n".join([think_result, act_result, callback_result])
            else:
                return False, "\n\n".join([think_result, act_result])

    async def think(self) -> bool:
        """Process current state and decide next actions using tools, return True if the agent should act"""
        if self.tool_returns:
            tool_msgs = ModelRequest(
                parts=self.tool_returns,
            )
            self.memory.append(tool_msgs)
            self.tool_returns = []

        if self.next_prompt:
            user_msg = ModelRequest(
                parts=[UserPromptPart(content=self.next_prompt)],
            )
            self.memory.append(user_msg)

        response = await self.request(messages=self.memory)

        if not response:
            raise RuntimeError("No response received from the LLM")

        self.memory.append(response)

        result = []
        self.tool_calls = []
        for part in response.parts:
            if part.part_kind == "tool-call":
                self.tool_calls.append(part)
            elif part.part_kind == "text":
                result.append(part.content)

        return bool(self.tool_calls), "\n\n".join(result)

    async def act(self) -> str:
        """Execute decided actions"""
        if not self.tool_calls:
            raise ValueError("No tool calls found")

        results = []
        for command in self.tool_calls:
            result = await self.execute_tool(command)
            logger.info(f"🎯 Tool '{command.tool_name}' completed! Result: {result}")

            # Add tool response to memory
            tool_msg = ToolReturnPart(
                content=result,
                tool_name=command.tool_name,
                tool_call_id=command.tool_call_id,
            )
            self.tool_returns.append(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCallPart) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.tool_name or not command.tool_call_id:
            return "Error: Invalid command format"

        name = command.tool_name

        try:
            # Parse arguments
            args = to_jsonable_python(command.args)

            # Execute the tool
            for tool in self.tools:
                if tool.name == name:
                    logger.info(f"🔧 Activating tool: '{name}'... {args}")
                    result = await tool.execute(agent=self, name=name, params=args)
                    break
            else:
                raise ValueError(f"Tool '{name}' not found")

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.args}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            logger.warning(f"Tool '{name}' encountered a problem: {str(e)}")
            return f"Error: {str(e)}"

    @abstractmethod
    async def callback(self) -> None:
        """Callback for the agent"""


class FinishTool(BaseTool):
    name: str = "finish"
    description: str = "Finish the task"

    def parameters_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "finished": {
                    "type": "boolean",
                    "description": "Whether the work has been finished",
                },
                "reason": {
                    "type": "string",
                    "description": "The reason for finishing the task",
                },
            },
            "required": ["finished"],
        }

    async def execute(self, agent: BaseAgent, name: str) -> None:
        if name != self.name:
            raise ValueError(f"Tool name {name} does not match {self.name}")
        agent.finish()
