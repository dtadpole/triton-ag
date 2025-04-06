import os
import json
import jsonref
import asyncio
import argparse
import yaml
from pydantic_core import to_jsonable_python
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.agent import Agent
from pydantic_ai.models import Model
from pydantic_ai import RunContext
from typing import List, Any
from model_factory import create_model
from pydantic import BaseModel, Field

from pydantic_core import core_schema as cs

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, TypeAdapter
from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.messages import TextPart
from agent import BaseAgent, BaseTool, FinishTool
from logger import logger

PLANNER_SYSTEM_PROMPT = """
You are an expert Planning Agent tasked with solving problems efficiently through structured plans.

1. Analyze requests to understand the task scope
2. Create a clear, actionable plan that makes meaningful progress with the `planning` tool
3. After each step of changing code, test and verify correctness using available tools, fix code until it is correct
4. Track progress and adapt plans when necessary
5. Use `finish` to conclude immediately when the task is complete

Available tools will vary by task but may include:
- `planning`: Create, update, and track plans (commands: create, update, mark_step, etc.)
- `finish`: End the task when complete
Break tasks into logical steps with clear outcomes. Avoid excessive detail or sub-steps.
Think about dependencies and verification methods.
Know when to conclude - don't continue thinking once objectives are met.
"""

PLANNING_NEXT_PROMPT = """
Goal: {{goal}}

Based on the current state, what's your next action?
Choose the most efficient path forward:
1. Is the plan sufficient, or does it need refinement?
2. Can you execute the next step immediately?
3. Is the task complete? If so, use `finish` right away.

Be concise in your reasoning, then select the appropriate tool or action.
"""


"""
{
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "The overall objective of the project"
        },
        "steps": {
            "type": "array",
            "description": "Ordered list of steps required to complete the project",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique identifier for the step, typically using a numeric prefix",
                        "pattern": "^[0-9]{2}_[a-z_]+$",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed explanation of the step's activities",
                    },
                    "depends_on": {
                        "type": "array",
                        "description": "List of step IDs that must be completed before this step",
                        "items": {
                            "type": "string",
                        },
                    },
                    "verification": {
                        "type": "string",
                        "description": "Criteria to determine if the step has been successfully completed",
                    },
                    "status": {
                        "type": "string",
                        "description": "The status of the step",
                        "enum": ["pending", "in_progress", "error", "completed"],
                        "default": "pending",
                    },
                    "required": [
                        "id",
                        "description",
                        "depends_on",
                        "verification",
                        "status",
                    ],
                },
                "minItems": 1,
            },
        },
        "required": [
        "goal",
        "steps"
    ]
}
"""


class PlanStep(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier for the step, typically using a numeric prefix",
        pattern=r"^[0-9]{2}_[a-z_]+$",
    )
    description: str = Field(
        ..., description="Detailed explanation of the step's activities"
    )
    depends_on: List[str] = Field(
        ...,
        description="List of step IDs that must be completed before this step",
    )
    verification: str = Field(
        ...,
        description="Criteria to determine if the step has been successfully completed",
    )
    status: str = Field(
        default="pending",
        description="The status of the step",
        enum=["pending", "in_progress", "error", "completed"],
    )


class Plan(BaseModel):
    goal: str = Field(..., description="The overall objective of the project")
    steps: list[PlanStep] = Field(
        ...,
        description="Ordered list of steps required to complete the project",
        min_items=1,
    )

    # @classmethod
    # def __get_pydantic_json_schema__(
    #     cls, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler
    # ) -> JsonSchemaValue:
    #     json_schema = handler(core_schema)
    #     json_schema = handler.resolve_ref_schema(json_schema)
    #     return json_schema


class CreatePlanTool(BaseTool):
    name: str = "create_plan"
    description: str = "Create a plan for the task"

    def parameters_json_schema(self) -> dict[str, Any]:
        return to_jsonable_python(Plan.model_json_schema(by_alias=False))

    @logger.catch
    async def execute(
        self, agent: BaseAgent, name: str, params: dict[str, Any]
    ) -> Plan:
        if name != self.name:
            raise ValueError(f"Tool name {name} does not match {self.name}")
        agent.plan = Plan.model_validate(params)
        return agent.plan


class UpdateStepStatusTool(BaseTool):
    name: str = "update_step_status"
    description: str = "Update the status of a step in the plan"

    def parameters_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "id": {
                "description": "Unique identifier for the step, typically using a numeric prefix",
                "pattern": "^[0-9]{2}_[a-z_]+$",
                "title": "Id",
                "type": "string",
            },
            "status": {
                "description": "The status of the step",
                "enum": ["pending", "in_progress", "error", "completed"],
                "title": "Status",
                "type": "string",
            },
            "required": ["id", "status"],
        }

    async def execute(
        self, agent: BaseAgent, name: str, params: dict[str, Any]
    ) -> PlanStep:
        if name != self.name:
            raise ValueError(f"Tool name {name} does not match {self.name}")
        for step in agent.plan.steps:
            if step.id == params["id"]:
                step.status = params["status"]
                return step
        raise ValueError(f"Step {params['id']} not found in plan")


class GetPlanTool(BaseTool):
    name: str = "get_plan"
    description: str = "Get the plan"

    def parameters_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The ID of the step to get",
                },
            },
            "required": ["id"],
        }

    async def execute(self, agent: BaseAgent, name: str) -> Plan:
        if name != self.name:
            raise ValueError(f"Tool name {name} does not match {self.name}")
        return agent.plan


class AgentPlanner(BaseAgent):
    name: str = "agent_planner"
    description: str = "An agent that plans and executes tasks"
    system_prompt: str = PLANNER_SYSTEM_PROMPT
    next_prompt: str = PLANNING_NEXT_PROMPT

    plan: Plan = Field(None, description="The plan")

    def __init__(self):
        super().__init__()
        self.tools = [
            CreatePlanTool(),
            UpdateStepStatusTool(),
            GetPlanTool(),
            FinishTool(),
        ]

    def callback(self) -> None:
        pass


async def main(args):

    agent_planner = AgentPlanner()
    await agent_planner.run(args.task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--task",
        type=str,
        default="implement Triton kernel for nn.Linear, no bias, both forward and backward pass, compare to PyTorch implementation, verify correctness, and benchmark performance",
    )
    args = parser.parse_args()
    agent_planner = AgentPlanner()
    asyncio.run(main(args))

    create_plan_tool = agent_planner.CreatePlanTool()
    tool_def = ToolDefinition(
        name=create_plan_tool.name,
        description=create_plan_tool.description,
        parameters_json_schema=create_plan_tool.parameters_json_schema(),
        outer_typed_dict_key=create_plan_tool.outer_typed_dict_key(),
    )
    # derefed = jsonref.loads(json.dumps(to_jsonable_python(create_plan_tool)))
    print(json.dumps(to_jsonable_python(tool_def), indent=4))

    params = {
        "goal": "Implement Triton kernel for nn.Linear (without bias) with forward and backward passes, validate against PyTorch implementation, and benchmark performance",
        "steps": [
            {
                "id": "01_setup_environment",
                "description": "Set up the necessary environment by importing required libraries (PyTorch, Triton, etc.) and defining test cases",
                "depends_on": [],
                "verification": "All necessary libraries are imported without errors and test data is properly initialized",
            },
            {
                "id": "02_implement_forward",
                "description": "Implement the forward pass of nn.Linear using Triton. This will involve matrix multiplication between input and weight tensors.",
                "depends_on": ["01_setup_environment"],
                "verification": "Forward function is implemented and can be called without errors",
            },
            {
                "id": "03_test_forward",
                "description": "Test the forward pass implementation by comparing results with PyTorch's nn.Linear. Verify numerical correctness within acceptable tolerance.",
                "depends_on": ["02_implement_forward"],
                "verification": "Triton forward pass produces results that match PyTorch's implementation (within numerical tolerance)",
            },
            {
                "id": "04_implement_backward",
                "description": "Implement the backward pass of nn.Linear using Triton. This involves computing gradients with respect to inputs and weights.",
                "depends_on": ["02_implement_forward"],
                "verification": "Backward function is implemented and can be called without errors",
            },
            {
                "id": "05_test_backward",
                "description": "Test the backward pass implementation by comparing gradients with PyTorch's autograd. Verify numerical correctness within acceptable tolerance.",
                "depends_on": ["04_implement_backward"],
                "verification": "Triton backward pass produces gradients that match PyTorch's implementation (within numerical tolerance)",
            },
            {
                "id": "06_benchmark_performance",
                "description": "Benchmark the performance of Triton kernels against PyTorch's implementation for both forward and backward passes. Test with various input sizes.",
                "depends_on": ["03_test_forward", "05_test_backward"],
                "verification": "Performance benchmarks are completed and compared",
            },
            {
                "id": "07_refine_implementation",
                "description": "Optimize the Triton kernels based on benchmark results, potentially adding tuning parameters or adjusting block sizes.",
                "depends_on": ["06_benchmark_performance"],
                "verification": "Optimized implementation shows improved performance over initial version",
            },
            {
                "id": "08_final_validation",
                "description": "Conduct final validation with comprehensive test cases and document the implementation's correctness and performance characteristics.",
                "depends_on": ["07_refine_implementation"],
                "verification": "Final implementation passes all tests and performance benchmarks are documented",
            },
        ],
    }
    plan = Plan.model_validate(params)
    print(plan)
