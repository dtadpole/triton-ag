from mcp.server.fastmcp import FastMCP
import subprocess
import asyncio
import dataclasses
from dataclasses import dataclass
from pydantic import Field
from typing import Any, List
import json
import os


PLAN_FILE = "plan.json"

server = FastMCP("plan")


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
async def create_plan(
    working_dir: str = Field(..., description="The working directory"),
    plan: Plan = Field(..., description="The plan to create"),
) -> Plan:
    # write plan to file
    with open(os.path.join(working_dir, PLAN_FILE), "w") as f:
        json.dump(dataclasses.asdict(plan), f, indent=4)
    return plan


@server.tool(
    name="update_plan_step",
    description="Update the status of a step in the plan",
)
async def update_plan_step(
    working_dir: str = Field(..., description="The working directory"),
    step_id: str = Field(
        ..., description="The ID of the step to update", pattern=r"^[0-9]{2}_[a-z_]+$"
    ),
    status: str = Field(
        ...,
        description="The status of the step",
        enum=["pending", "in_progress", "error", "completed"],
    ),
) -> PlanStep:
    if not os.path.exists(os.path.join(working_dir, PLAN_FILE)):
        raise ValueError("No plan found")
    with open(os.path.join(working_dir, PLAN_FILE), "r") as f:
        plan = json.load(f)
    for step in plan["steps"]:
        if step["step_id"] == step_id:
            step["status"] = status
            with open(os.path.join(working_dir, PLAN_FILE), "w") as f:
                json.dump(plan, f, indent=4)
            return PlanStep(**step)
    raise ValueError(f"Step {step_id} not found in plan")


@server.tool(
    name="get_plan",
    description="Get the plan info",
)
async def get_plan(
    working_dir: str = Field(..., description="The working directory")
) -> Plan:
    if not os.path.exists(os.path.join(working_dir, PLAN_FILE)):
        raise ValueError("No plan found")
    with open(os.path.join(working_dir, PLAN_FILE), "r") as f:
        plan = json.load(f)
    return Plan(**plan)


if __name__ == "__main__":
    server.run(transport="stdio")
