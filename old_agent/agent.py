from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from enum import Enum
from pydantic_ai import Model
from old_agent.message import Memory, Message, ROLE_TYPE
from abc import ABC, abstractmethod
from old_agent.logger import logger


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class BaseAgent(BaseModel, ABC):
    name: str = Field(description="A unique name for the agent")
    description: Optional[str] = Field(None, description="A description of the agent")

    # system prompt for the agent
    system_prompt: Optional[str] = Field(
        None, description="The system prompt for the agent"
    )
    next_prompt: Optional[str] = Field(
        None, description="The next prompt for the agent"
    )

    # model used by the agent
    model: Model = Field(
        default=Model(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
        ),
        description="The model to use for the agent",
    )

    # memory of the agent
    memory: Memory = Field(default=Memory(), description="The memory of the agent")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )
    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.model is None or not isinstance(self.model, Model):
            self.model = Model(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.
        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).
        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        # Create message with appropriate parameters based on role
        kwargs = {**(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.
        Args:
            request: Optional initial user request to process.
        Returns:
            A string summarizing the execution results.
        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # Check for stuck state
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")

        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.
        Must be implemented by subclasses to define specific behavior.
        """

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
