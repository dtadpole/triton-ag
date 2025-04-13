from mcp.server.fastmcp import FastMCP
import subprocess
import asyncio
from dataclasses import dataclass
from pydantic import Field
from typing import Any, List
import json

server = FastMCP("plan")

ctx: dict[str, Any] = {}


@dataclass
class PlanStep:
    step_id: str = Field(
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


@dataclass
class Plan:
    goal: str = Field(..., description="The overall objective of the project")
    steps: list[PlanStep] = Field(
        ...,
        description="Ordered list of steps required to complete the project",
        min_items=1,
    )


@server.tool(
    name="create_plan",
    description="Create a plan for the goal",
)
async def create_plan(plan: Plan) -> Plan:
    global ctx
    ctx["plan"] = plan
    return plan


@server.tool(
    name="update_plan_step",
    description="Update the status of a step in the plan",
)
async def update_plan_step(step_id: str, status: str) -> PlanStep:
    global ctx
    if "plan" not in ctx:
        raise ValueError("No plan found")
    for step in ctx["plan"].steps:
        if step.id == step_id:
            step.status = status
            return step
    raise ValueError(f"Step {step_id} not found in plan")


@server.tool(
    name="get_plan",
    description="Get the plan info",
)
async def get_plan() -> Plan:
    global ctx
    if "plan" not in ctx:
        raise ValueError("No plan found")
    return ctx["plan"]


if __name__ == "__main__":
    server.run(transport="stdio")
